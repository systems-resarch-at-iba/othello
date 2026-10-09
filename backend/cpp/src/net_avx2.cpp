// This translation unit is compiled with AVX2, FMA and F16C enabled. It should
// avoid standard library templates, because an instantiation emitted here
// could be picked by the linker for code that runs on CPUs without those
// instructions.

#include <immintrin.h>

#include <cmath>
#include <cstring>

#include "othello/net.hpp"

namespace othello {

    namespace {

        /** @brief Floats in one vector register. */
        constexpr int kLaneCount = 8;
        /** @brief Inputs ahead of the current one whose weights are prefetched. */
        constexpr int kPrefetchInputs = 192;
        /** @brief Output squares computed together by the convolution kernel. */
        constexpr int kTileSquares = 4;
        /** @brief Output channels computed together by the convolution kernel, as vectors. */
        constexpr int kTileVectors = 2;
        /** @brief Output channels computed together by the convolution kernel. */
        constexpr int kTileChannels = kTileVectors * kLaneCount;

        /**
         * @brief Leaky ReLU of eight values.
         *
         * @param x Input values.
         * @param slope Negative slope, broadcast to every lane.
         * @return `x` where positive, otherwise `x * slope`.
         */
        inline __m256 leaky(__m256 x, __m256 slope) {
            return _mm256_max_ps(x, _mm256_mul_ps(x, slope));
        }

        /**
         * @brief Sum of the eight lanes of a vector.
         *
         * @param v Vector to add up.
         * @return The sum of the lanes.
         */
        inline float horizontal_sum(__m256 v) {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            return _mm_cvtss_f32(s);
        }

        /**
         * @brief Offset of square (row, col) in a map with `Channels` channels
         *        and zero padding of `Pad` squares on every edge.
         *
         * @tparam Channels Channels of the map.
         * @tparam Pad Squares of zero padding on every edge.
         * @param row Row of the square, from 0 to 7.
         * @param col Column of the square, from 0 to 7.
         * @return Index of the square's first channel.
         */
        template <int Channels, int Pad> constexpr int padded_offset(int row, int col) {
            return ((row + Pad) * (kBoardSide + 2 * Pad) + (col + Pad)) * Channels;
        }

        /**
         * @brief First convolution: one input channel, 5x5 kernel, 16 output channels.
         *
         * @param input Board in a map padded by two squares.
         * @param w Weights, [tap][output channel].
         * @param bias Bias of each output channel.
         * @param out Output map padded by one square.
         */
        void conv_first(const float *input, const float *w, const float *bias, float *out) {
            constexpr int kKernel = 5;
            const __m256 bias_low = _mm256_loadu_ps(bias);
            const __m256 bias_high = _mm256_loadu_ps(bias + kLaneCount);

            for (int r = 0; r < kBoardSide; ++r) {
                for (int c = 0; c < kBoardSide; ++c) {
                    __m256 low = bias_low;
                    __m256 high = bias_high;

                    // One input channel, so every tap broadcasts a single board value.
                    for (int kh = 0; kh < kKernel; ++kh) {
                        for (int kw = 0; kw < kKernel; ++kw) {
                            const float value = input[(r + kh) * kInputPadSide + (c + kw)];
                            const __m256 x = _mm256_set1_ps(value);
                            const float *tap = w + (kh * kKernel + kw) * kSeqChannels;

                            low = _mm256_fmadd_ps(x, _mm256_loadu_ps(tap), low);
                            high = _mm256_fmadd_ps(x, _mm256_loadu_ps(tap + kLaneCount), high);
                        }
                    }

                    float *dst = out + padded_offset<kSeqChannels, 1>(r, c);
                    _mm256_storeu_ps(dst, low);
                    _mm256_storeu_ps(dst + kLaneCount, high);
                }
            }
        }

        /**
         * @brief Convolution with 64 output channels over a map padded by one square.
         *
         * Each step keeps a tile of 4 output squares by 16 output channels in
         * eight vector accumulators and loops over the kernel taps and input
         * channels. One pair of weight vectors serves the 4 squares. The
         * padding removes every bounds check.
         *
         * @tparam In Input channels.
         * @tparam Kernel Kernel side, 1 or 3.
         * @param in Input map padded by one square, [row][col][In].
         * @param w Weights, [tap][In][64].
         * @param bias Bias of each output channel.
         * @param out Output map.
         * @param out_offset Offset of square (0, 0) in `out`.
         * @param out_row Distance between output rows.
         */
        template <int In, int Kernel>
        void conv64(const float *in, const float *w, const float *bias, float *out, int out_offset,
                    int out_row) {
            constexpr int kPad = Kernel / 2;
            constexpr int kTaps = Kernel * Kernel;
            constexpr int kChannelBlocks = kResChannels / kTileChannels;

            // One pass over the board per block of 16 output channels, so that the
            // weights of the block stay in the L1 cache.
            for (int n = 0; n < kChannelBlocks; ++n) {
                const float *block_weights = w + n * kTileChannels;
                const __m256 bias_low = _mm256_loadu_ps(bias + n * kTileChannels);
                const __m256 bias_high = _mm256_loadu_ps(bias + n * kTileChannels + kLaneCount);

                for (int r = 0; r < kBoardSide; ++r) {
                    for (int c0 = 0; c0 < kBoardSide; c0 += kTileSquares) {
                        // Every accumulator starts from the bias.
                        __m256 acc[kTileSquares][kTileVectors];
#pragma GCC unroll 4
                        for (int j = 0; j < kTileSquares; ++j) {
                            acc[j][0] = bias_low;
                            acc[j][1] = bias_high;
                        }

                        // Add every tap and input channel. One pair of weight vectors serves
                        // all squares of the tile.
#pragma GCC unroll 9
                        for (int tap = 0; tap < kTaps; ++tap) {
                            const int kh = tap / Kernel;
                            const int kw = tap % Kernel;
                            const float *src =
                                in + padded_offset<In, 1>(r + kh - kPad, c0 + kw - kPad);
                            const float *tap_weights = block_weights + tap * In * kResChannels;

                            for (int i = 0; i < In; ++i) {
                                const float *row = tap_weights + i * kResChannels;
                                const __m256 w_low = _mm256_loadu_ps(row);
                                const __m256 w_high = _mm256_loadu_ps(row + kLaneCount);
#pragma GCC unroll 4
                                for (int j = 0; j < kTileSquares; ++j) {
                                    const __m256 x = _mm256_set1_ps(src[j * In + i]);
                                    acc[j][0] = _mm256_fmadd_ps(x, w_low, acc[j][0]);
                                    acc[j][1] = _mm256_fmadd_ps(x, w_high, acc[j][1]);
                                }
                            }
                        }

                        // Store the tile.
#pragma GCC unroll 4
                        for (int j = 0; j < kTileSquares; ++j) {
                            float *dst = out + out_offset + r * out_row + (c0 + j) * kResChannels +
                                         n * kTileChannels;
                            _mm256_storeu_ps(dst, acc[j][0]);
                            _mm256_storeu_ps(dst + kLaneCount, acc[j][1]);
                        }
                    }
                }
            }
        }

        /**
         * @brief Group normalization, affine transform and leaky ReLU over a feature map.
         *
         * Per-channel sums are reduced to group statistics, and the
         * normalization is folded into one scale and shift per channel. The
         * source and destination may be the same map. Positions outside the
         * 8x8 board are never read or written.
         *
         * @tparam Channels Channel count, a multiple of 8 and of kGroups.
         * @param src Source map.
         * @param src_offset Offset of square (0, 0) in `src`.
         * @param src_row Distance between source rows.
         * @param dst Destination map.
         * @param dst_offset Offset of square (0, 0) in `dst`.
         * @param dst_row Distance between destination rows.
         * @param gamma Scale of each channel.
         * @param beta Shift of each channel.
         * @param eps Epsilon added to the variance.
         * @param slope Negative slope of the leaky ReLU.
         * @param residual Flat [square][channel] map added after the activation, or null.
         */
        template <int Channels>
        void group_norm_act(const float *src, int src_offset, int src_row, float *dst,
                            int dst_offset, int dst_row, const float *gamma, const float *beta,
                            float eps, float slope, const float *residual) {
            constexpr int kVectors = Channels / kLaneCount;
            constexpr int kPerGroup = Channels / kGroups;
            constexpr float kCount = static_cast<float>(kSquares * kPerGroup);

            alignas(32) float channel_sum[Channels];
            alignas(32) float channel_mean[Channels];
            alignas(32) float channel_sq[Channels];
            float group_mean[kGroups];

            // Pass 1: sum of each channel over the board, reduced to one mean per group.
            __m256 sum[kVectors];
            for (int v = 0; v < kVectors; ++v) {
                sum[v] = _mm256_setzero_ps();
            }
            for (int r = 0; r < kBoardSide; ++r) {
                for (int c = 0; c < kBoardSide; ++c) {
                    const float *p = src + src_offset + r * src_row + c * Channels;
                    for (int v = 0; v < kVectors; ++v) {
                        sum[v] = _mm256_add_ps(sum[v], _mm256_loadu_ps(p + v * kLaneCount));
                    }
                }
            }
            for (int v = 0; v < kVectors; ++v) {
                _mm256_store_ps(channel_sum + v * kLaneCount, sum[v]);
            }
            for (int g = 0; g < kGroups; ++g) {
                float total = 0.0f;
                for (int j = 0; j < kPerGroup; ++j) {
                    total += channel_sum[g * kPerGroup + j];
                }
                group_mean[g] = total / kCount;
                for (int j = 0; j < kPerGroup; ++j) {
                    channel_mean[g * kPerGroup + j] = group_mean[g];
                }
            }

            // Pass 2: squared deviations from the group mean, reduced to one variance per group.
            __m256 sq[kVectors];
            for (int v = 0; v < kVectors; ++v) {
                sq[v] = _mm256_setzero_ps();
            }
            for (int r = 0; r < kBoardSide; ++r) {
                for (int c = 0; c < kBoardSide; ++c) {
                    const float *p = src + src_offset + r * src_row + c * Channels;
                    for (int v = 0; v < kVectors; ++v) {
                        const __m256 d =
                            _mm256_sub_ps(_mm256_loadu_ps(p + v * kLaneCount),
                                          _mm256_load_ps(channel_mean + v * kLaneCount));
                        sq[v] = _mm256_fmadd_ps(d, d, sq[v]);
                    }
                }
            }
            for (int v = 0; v < kVectors; ++v) {
                _mm256_store_ps(channel_sq + v * kLaneCount, sq[v]);
            }

            // Fold the normalization and the affine transform into one scale and shift
            // per channel.
            alignas(32) float scale[Channels];
            alignas(32) float shift[Channels];
            for (int g = 0; g < kGroups; ++g) {
                float total = 0.0f;
                for (int j = 0; j < kPerGroup; ++j) {
                    total += channel_sq[g * kPerGroup + j];
                }
                const float rstd = 1.0f / std::sqrt(total / kCount + eps);
                for (int j = 0; j < kPerGroup; ++j) {
                    const int c = g * kPerGroup + j;
                    scale[c] = gamma[c] * rstd;
                    shift[c] = beta[c] - group_mean[g] * scale[c];
                }
            }

            // Pass 3: apply the scale and shift, the activation and the residual.
            const __m256 slope_v = _mm256_set1_ps(slope);
            for (int r = 0; r < kBoardSide; ++r) {
                for (int c = 0; c < kBoardSide; ++c) {
                    const float *from = src + src_offset + r * src_row + c * Channels;
                    float *to = dst + dst_offset + r * dst_row + c * Channels;
                    const float *res =
                        residual != nullptr ? residual + (r * kBoardSide + c) * Channels : nullptr;
                    for (int v = 0; v < kVectors; ++v) {
                        __m256 y = _mm256_fmadd_ps(_mm256_loadu_ps(from + v * kLaneCount),
                                                   _mm256_load_ps(scale + v * kLaneCount),
                                                   _mm256_load_ps(shift + v * kLaneCount));
                        y = leaky(y, slope_v);
                        if (res != nullptr) {
                            y = _mm256_add_ps(y, _mm256_loadu_ps(res + v * kLaneCount));
                        }
                        _mm256_storeu_ps(to + v * kLaneCount, y);
                    }
                }
            }
        }

        /**
         * @brief Layer normalization, affine transform and leaky ReLU, in place.
         *
         * @param x Values to normalize, a multiple of 16 of them.
         * @param n Number of values.
         * @param gamma Scale of each value.
         * @param beta Shift of each value.
         * @param eps Epsilon added to the variance.
         * @param slope Negative slope of the leaky ReLU.
         */
        void layer_norm_act(float *x, int n, const float *gamma, const float *beta, float eps,
                            float slope) {
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            for (int i = 0; i < n; i += 2 * kLaneCount) {
                s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x + i));
                s1 = _mm256_add_ps(s1, _mm256_loadu_ps(x + i + kLaneCount));
            }
            const float mean = horizontal_sum(_mm256_add_ps(s0, s1)) / static_cast<float>(n);
            const __m256 mean_v = _mm256_set1_ps(mean);

            __m256 q0 = _mm256_setzero_ps();
            __m256 q1 = _mm256_setzero_ps();
            for (int i = 0; i < n; i += 2 * kLaneCount) {
                const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(x + i), mean_v);
                const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(x + i + kLaneCount), mean_v);
                q0 = _mm256_fmadd_ps(d0, d0, q0);
                q1 = _mm256_fmadd_ps(d1, d1, q1);
            }
            const float var = horizontal_sum(_mm256_add_ps(q0, q1)) / static_cast<float>(n);
            const __m256 rstd_v = _mm256_set1_ps(1.0f / std::sqrt(var + eps));
            const __m256 slope_v = _mm256_set1_ps(slope);
            for (int i = 0; i < n; i += kLaneCount) {
                const __m256 t =
                    _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), mean_v), rstd_v);
                const __m256 y =
                    _mm256_fmadd_ps(t, _mm256_loadu_ps(gamma + i), _mm256_loadu_ps(beta + i));
                _mm256_storeu_ps(x + i, leaky(y, slope_v));
            }
        }

        /**
         * @brief Loads 8 weights starting at element `index`, converting from fp16 when `Half`.
         *
         * @tparam Half Whether the weights are fp16.
         * @param w Weights, as fp32 or fp16 bits.
         * @param index Element of the first weight to load.
         * @return The weights as eight floats.
         */
        template <bool Half> inline __m256 load_weights(const void *w, size_t index) {
            if constexpr (Half) {
                const uint16_t *h = static_cast<const uint16_t *>(w) + index;
                return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(h)));
            } else {
                return _mm256_loadu_ps(static_cast<const float *>(w) + index);
            }
        }

        /**
         * @brief One tile of a fully connected layer: kFcTile outputs for `Boards` boards.
         *
         * Each weight vector is loaded once and used for every board. `Unroll`
         * consecutive inputs use separate accumulators, which are added at the
         * end, so that the fused multiply-adds do not wait on one another.
         *
         * @tparam Boards Boards in the tile.
         * @tparam Unroll Inputs handled per step. The input count should be a multiple of it.
         * @tparam Half Whether the weights are fp16.
         * @param w Weights of the tile, [input][kFcTile].
         * @param bias Bias of the tile's outputs.
         * @param x Inputs of the first board.
         * @param x_stride Distance between the inputs of consecutive boards.
         * @param in Input count.
         * @param y Outputs of the first board, starting at the tile's first output.
         * @param y_stride Distance between the outputs of consecutive boards.
         */
        template <int Boards, int Unroll, bool Half>
        void fc_tile(const void *w, const float *bias, const float *x, int x_stride, int in,
                     float *y, int y_stride) {
            static_assert(kFcTile == 2 * kLaneCount);
            __m256 acc[Unroll][Boards][2];
#pragma GCC unroll 4
            for (int u = 0; u < Unroll; ++u) {
#pragma GCC unroll 4
                for (int b = 0; b < Boards; ++b) {
                    acc[u][b][0] = _mm256_setzero_ps();
                    acc[u][b][1] = _mm256_setzero_ps();
                }
            }
            for (int i = 0; i < in; i += Unroll) {
#pragma GCC unroll 4
                for (int u = 0; u < Unroll; ++u) {
                    const size_t at = static_cast<size_t>(i + u) * kFcTile;
                    _mm_prefetch(static_cast<const char *>(w) +
                                     (at + kPrefetchInputs * kFcTile) * (Half ? 2 : 4),
                                 _MM_HINT_T0);
                    const __m256 w0 = load_weights<Half>(w, at);
                    const __m256 w1 = load_weights<Half>(w, at + kLaneCount);
#pragma GCC unroll 4
                    for (int b = 0; b < Boards; ++b) {
                        const __m256 xv = _mm256_set1_ps(x[b * x_stride + i + u]);
                        acc[u][b][0] = _mm256_fmadd_ps(xv, w0, acc[u][b][0]);
                        acc[u][b][1] = _mm256_fmadd_ps(xv, w1, acc[u][b][1]);
                    }
                }
            }
            const __m256 bias0 = _mm256_loadu_ps(bias);
            const __m256 bias1 = _mm256_loadu_ps(bias + kLaneCount);
#pragma GCC unroll 4
            for (int b = 0; b < Boards; ++b) {
                __m256 sum0 = bias0;
                __m256 sum1 = bias1;
#pragma GCC unroll 4
                for (int u = 0; u < Unroll; ++u) {
                    sum0 = _mm256_add_ps(sum0, acc[u][b][0]);
                    sum1 = _mm256_add_ps(sum1, acc[u][b][1]);
                }
                _mm256_storeu_ps(y + b * y_stride, sum0);
                _mm256_storeu_ps(y + b * y_stride + kLaneCount, sum1);
            }
        }

        /**
         * @brief Fully connected layer over weights packed in tiles of kFcTile outputs.
         *
         * @tparam Half Whether the weights are fp16.
         * @tparam Boards Boards in the batch.
         * @tparam Unroll Inputs per step of fc_tile.
         * @param w Weights of the layer, packed in tiles.
         * @param bias Bias of each output.
         * @param x Inputs of the first board.
         * @param x_stride Distance between the inputs of consecutive boards.
         * @param in Input count.
         * @param out Output count, a multiple of kFcTile.
         * @param y Outputs of the first board.
         * @param y_stride Distance between the outputs of consecutive boards.
         */
        template <bool Half, int Boards, int Unroll>
        void linear(const void *w, const float *bias, const float *x, int x_stride, int in, int out,
                    float *y, int y_stride) {
            for (int t = 0; t < out; t += kFcTile) {
                const size_t base = static_cast<size_t>(t) * in;
                const void *tile =
                    Half ? static_cast<const void *>(static_cast<const uint16_t *>(w) + base)
                         : static_cast<const void *>(static_cast<const float *>(w) + base);
                fc_tile<Boards, Unroll, Half>(tile, bias + t, x, x_stride, in, y + t, y_stride);
            }
        }

        /**
         * @brief Dispatches a fully connected layer on the weight type and the batch size.
         *
         * @param w Weights of the layer, packed in tiles.
         * @param half Whether the weights are fp16.
         * @param bias Bias of each output.
         * @param x Inputs of the first board.
         * @param x_stride Distance between the inputs of consecutive boards.
         * @param in Input count.
         * @param out Output count, a multiple of kFcTile.
         * @param y Outputs of the first board.
         * @param y_stride Distance between the outputs of consecutive boards.
         * @param boards Boards in the batch, from 1 to kMaxBatch.
         */
        void linear_dispatch(const void *w, bool half, const float *bias, const float *x,
                             int x_stride, int in, int out, float *y, int y_stride, int boards) {
            if (half) {
                switch (boards) {
                case 1:  linear<true, 1, 4>(w, bias, x, x_stride, in, out, y, y_stride); break;
                case 2:  linear<true, 2, 3>(w, bias, x, x_stride, in, out, y, y_stride); break;
                case 3:  linear<true, 3, 2>(w, bias, x, x_stride, in, out, y, y_stride); break;
                default: linear<true, 4, 1>(w, bias, x, x_stride, in, out, y, y_stride); break;
                }
            } else {
                switch (boards) {
                case 1:  linear<false, 1, 4>(w, bias, x, x_stride, in, out, y, y_stride); break;
                case 2:  linear<false, 2, 3>(w, bias, x, x_stride, in, out, y, y_stride); break;
                case 3:  linear<false, 3, 2>(w, bias, x, x_stride, in, out, y, y_stride); break;
                default: linear<false, 4, 1>(w, bias, x, x_stride, in, out, y, y_stride); break;
                }
            }
        }

        /**
         * @brief Group normalization in place over a padded map, with scale and shift tensors.
         *
         * @tparam Channels Channels of the map.
         * @param w Network weights.
         * @param map Feature map padded by one square.
         * @param scale Tensor of the per-channel scale.
         * @param shift Tensor of the per-channel shift.
         */
        template <int Channels>
        void normalize_padded(const Weights &w, float *map, Tensor scale, Tensor shift) {
            constexpr int kOffset = padded_offset<Channels, 1>(0, 0);
            constexpr int kRow = kPadSide * Channels;

            group_norm_act<Channels>(map, kOffset, kRow, map, kOffset, kRow, w.f32(scale),
                                     w.f32(shift), w.norm_eps(), w.leaky_slope(), nullptr);
        }

        /**
         * @brief Convolution with 64 output channels, with weights and bias tensors.
         *
         * @tparam In Input channels.
         * @tparam Kernel Kernel side, 1 or 3.
         * @param w Network weights.
         * @param weights Tensor of the convolution weights.
         * @param bias Tensor of the convolution bias.
         * @param in Input map padded by one square.
         * @param out Output map.
         * @param out_offset Offset of square (0, 0) in `out`.
         * @param out_row Distance between output rows.
         */
        template <int In, int Kernel>
        void convolve(const Weights &w, Tensor weights, Tensor bias, const float *in, float *out,
                      int out_offset, int out_row) {
            conv64<In, Kernel>(in, w.f32(weights), w.f32(bias), out, out_offset, out_row);
        }

        /**
         * @brief Layer normalization in place, with scale and shift tensors.
         *
         * @param w Network weights.
         * @param x Values to normalize.
         * @param n Number of values.
         * @param scale Tensor of the per-value scale.
         * @param shift Tensor of the per-value shift.
         */
        void layer_normalize(const Weights &w, float *x, int n, Tensor scale, Tensor shift) {
            layer_norm_act(x, n, w.f32(scale), w.f32(shift), w.norm_eps(), w.leaky_slope());
        }

        /**
         * @brief Fully connected layer over a batch, with weights and bias tensors.
         *
         * @param w Network weights.
         * @param weights Tensor of the packed weights.
         * @param bias Tensor of the bias.
         * @param x Inputs of the first board.
         * @param x_stride Distance between the inputs of consecutive boards.
         * @param in Input count.
         * @param out Output count.
         * @param y Outputs of the first board.
         * @param y_stride Distance between the outputs of consecutive boards.
         * @param boards Boards in the batch, from 1 to kMaxBatch.
         * @param stored_like_fc Whether the weights follow the storage type of the large
         *        fully connected matrices, which can be fp16.
         */
        void fully_connected(const Weights &w, Tensor weights, Tensor bias, const float *x,
                             int x_stride, int in, int out, float *y, int y_stride, int boards,
                             bool stored_like_fc) {
            const bool half = stored_like_fc && w.fc_type() == FcType::Fp16;
            const void *packed = half ? static_cast<const void *>(w.f16(weights)) : w.f32(weights);
            linear_dispatch(packed, half, w.f32(bias), x, x_stride, in, out, y, y_stride, boards);
        }

    }  // namespace

    void forward_batch_avx2(const Weights &w, Scratch &s, const float *boards, int count,
                            Evaluation *out) {
        constexpr int kResOffset = padded_offset<kResChannels, 1>(0, 0);
        constexpr int kResRow = kPadSide * kResChannels;
        constexpr int kFlatRow = kBoardSide * kResChannels;

        // The convolutions run one board at a time, and leave the flattened result of
        // each board in `s.flat` for the fully connected layers.
        for (int b = 0; b < count; ++b) {
            for (int r = 0; r < kBoardSide; ++r) {
                float *row = s.input_pad + (r + 2) * kInputPadSide + 2;
                std::memcpy(row, boards + b * kSquares + r * kBoardSide,
                            kBoardSide * sizeof(float));
            }

            // First convolution over the board.
            conv_first(s.input_pad, w.f32(Tensor::Conv0W), w.f32(Tensor::Conv0B), s.seq_pad);
            normalize_padded<kSeqChannels>(w, s.seq_pad, Tensor::Gn0W, Tensor::Gn0B);

            // Shortcut path, added after the last convolution of the residual block.
            convolve<kSeqChannels, 1>(w, Tensor::ShortW, Tensor::ShortB, s.seq_pad, s.shortcut_flat,
                                      0, kFlatRow);

            // Residual block of three convolutions.
            convolve<kSeqChannels, 3>(w, Tensor::Conv1W, Tensor::Conv1B, s.seq_pad, s.res_pad_a,
                                      kResOffset, kResRow);
            normalize_padded<kResChannels>(w, s.res_pad_a, Tensor::Gn1W, Tensor::Gn1B);

            convolve<kResChannels, 3>(w, Tensor::Conv2W, Tensor::Conv2B, s.res_pad_a, s.res_pad_b,
                                      kResOffset, kResRow);
            normalize_padded<kResChannels>(w, s.res_pad_b, Tensor::Gn2W, Tensor::Gn2B);

            convolve<kResChannels, 3>(w, Tensor::Conv3W, Tensor::Conv3B, s.res_pad_b, s.res_pad_a,
                                      kResOffset, kResRow);

            // The last normalization adds the shortcut and writes the flattened board.
            group_norm_act<kResChannels>(s.res_pad_a, kResOffset, kResRow, s.flat + b * kFc1In, 0,
                                         kFlatRow, w.f32(Tensor::Gn3W), w.f32(Tensor::Gn3B),
                                         w.norm_eps(), w.leaky_slope(), s.shortcut_flat);
        }

        // The fully connected layers read their weights once for the whole batch.
        fully_connected(w, Tensor::Fc1W, Tensor::Fc1B, s.flat, kFc1In, kFc1In, kFc1Out, s.fc1_batch,
                        kFc1Out, count, true);
        for (int b = 0; b < count; ++b) {
            layer_normalize(w, s.fc1_batch + b * kFc1Out, kFc1Out, Tensor::Ln1W, Tensor::Ln1B);
        }

        fully_connected(w, Tensor::Fc2W, Tensor::Fc2B, s.fc1_batch, kFc1Out, kFc1Out, kFc2Out,
                        s.fc2_batch, kFc2Out, count, true);
        for (int b = 0; b < count; ++b) {
            layer_normalize(w, s.fc2_batch + b * kFc2Out, kFc2Out, Tensor::Ln2W, Tensor::Ln2B);
        }

        fully_connected(w, Tensor::HeadW, Tensor::HeadB, s.fc2_batch, kFc2Out, kFc2Out, kHeadOut,
                        s.head_batch, kHeadOut, count, false);

        // Policy logits and value of each board.
        for (int b = 0; b < count; ++b) {
            const float *head = s.head_batch + b * kHeadOut;
            std::memcpy(out[b].logits, head, sizeof(out[b].logits));
            out[b].value = std::tanh(head[kHeadValue]);
        }
    }

    void forward_avx2(const Weights &w, Scratch &s, const float *board, Evaluation &out) {
        forward_batch_avx2(w, s, board, 1, &out);
    }

}  // namespace othello
