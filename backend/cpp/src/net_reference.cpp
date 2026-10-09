#include <cmath>
#include <cstring>

#include "othello/net.hpp"

namespace othello {

    namespace {

        /**
         * @brief Converts IEEE half bits to a float.
         *
         * @param h Bits of a half-precision value.
         * @return The same value as a float.
         */
        float half_to_float(uint16_t h) {
            const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
            const uint32_t exponent = (h >> 10) & 0x1Fu;
            const uint32_t mantissa = h & 0x3FFu;
            uint32_t bits;
            if (exponent == 0) {
                if (mantissa == 0) {
                    bits = sign;
                } else {
                    // Subnormal half: renormalize into a float.
                    int shift = 0;
                    uint32_t m = mantissa;
                    while ((m & 0x400u) == 0) {
                        m <<= 1;
                        ++shift;
                    }
                    bits = sign | (static_cast<uint32_t>(113 - shift) << 23) | ((m & 0x3FFu) << 13);
                }
            } else if (exponent == 0x1Fu) {
                bits = sign | 0x7F800000u | (mantissa << 13);
            } else {
                bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
            }
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }

        /**
         * @brief Leaky ReLU.
         *
         * @param x Input value.
         * @param slope Factor applied to negative inputs.
         * @return `x` when positive, otherwise `x * slope`.
         */
        float leaky(float x, float slope) {
            return x > 0.0f ? x : x * slope;
        }

        /**
         * @brief Convolution over a channels-last 8x8 map with zero padding.
         *
         * @param in Input map, [square][in_channels].
         * @param in_channels Channels of the input map.
         * @param w Weights, [tap][in_channels][out_channels].
         * @param bias Bias of each output channel.
         * @param out Output map, [square][out_channels].
         * @param out_channels Channels of the output map.
         * @param kernel Kernel side, which should be odd.
         */
        void conv(const float *in, int in_channels, const float *w, const float *bias, float *out,
                  int out_channels, int kernel) {
            const int pad = kernel / 2;
            for (int r = 0; r < kBoardSide; ++r) {
                for (int c = 0; c < kBoardSide; ++c) {
                    for (int o = 0; o < out_channels; ++o) {
                        float acc = bias[o];
                        for (int kh = 0; kh < kernel; ++kh) {
                            for (int kw = 0; kw < kernel; ++kw) {
                                const int rr = r + kh - pad;
                                const int cc = c + kw - pad;
                                if (rr < 0 || rr >= kBoardSide || cc < 0 || cc >= kBoardSide) {
                                    continue;
                                }
                                const float *src = in + (rr * kBoardSide + cc) * in_channels;
                                const float *tap =
                                    w + (kh * kernel + kw) * in_channels * out_channels;
                                for (int i = 0; i < in_channels; ++i) {
                                    acc += src[i] * tap[i * out_channels + o];
                                }
                            }
                        }
                        out[(r * kBoardSide + c) * out_channels + o] = acc;
                    }
                }
            }
        }

        /**
         * @brief Group normalization, affine transform and leaky ReLU, in place.
         *
         * @param x Feature map, [square][channel].
         * @param channels Channels of the map, a multiple of kGroups.
         * @param gamma Scale of each channel.
         * @param beta Shift of each channel.
         * @param eps Epsilon added to the variance.
         * @param slope Negative slope of the leaky ReLU.
         * @param residual Added after the activation, or null.
         */
        void group_norm_act(float *x, int channels, const float *gamma, const float *beta,
                            float eps, float slope, const float *residual) {
            const int per_group = channels / kGroups;
            const double count = static_cast<double>(kSquares) * per_group;

            for (int g = 0; g < kGroups; ++g) {
                const int first_channel = g * per_group;

                // Mean over the squares and channels of the group.
                double sum = 0.0;
                for (int s = 0; s < kSquares; ++s) {
                    for (int j = 0; j < per_group; ++j) {
                        sum += x[s * channels + first_channel + j];
                    }
                }
                const double mean = sum / count;

                // Variance over the same values.
                double squares = 0.0;
                for (int s = 0; s < kSquares; ++s) {
                    for (int j = 0; j < per_group; ++j) {
                        const double d = x[s * channels + first_channel + j] - mean;
                        squares += d * d;
                    }
                }
                const double inverse_deviation = 1.0 / std::sqrt(squares / count + eps);

                // Normalize, scale and shift, apply the activation and add the residual.
                for (int s = 0; s < kSquares; ++s) {
                    for (int j = 0; j < per_group; ++j) {
                        const int c = first_channel + j;
                        float &v = x[s * channels + c];

                        v = static_cast<float>((v - mean) * inverse_deviation * gamma[c] + beta[c]);
                        v = leaky(v, slope);
                        if (residual != nullptr) {
                            v += residual[s * channels + c];
                        }
                    }
                }
            }
        }

        /**
         * @brief Layer normalization, affine transform and leaky ReLU, in place.
         *
         * @param x Values to normalize.
         * @param n Number of values.
         * @param gamma Scale of each value.
         * @param beta Shift of each value.
         * @param eps Epsilon added to the variance.
         * @param slope Negative slope of the leaky ReLU.
         */
        void layer_norm_act(float *x, int n, const float *gamma, const float *beta, float eps,
                            float slope) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += x[i];
            }
            const double mean = sum / n;

            double squares = 0.0;
            for (int i = 0; i < n; ++i) {
                const double d = x[i] - mean;
                squares += d * d;
            }
            const double inverse_deviation = 1.0 / std::sqrt(squares / n + eps);

            for (int i = 0; i < n; ++i) {
                const double normalized = (x[i] - mean) * inverse_deviation;
                x[i] = leaky(static_cast<float>(normalized * gamma[i] + beta[i]), slope);
            }
        }

        /**
         * @brief Fully connected layer over weights packed in tiles of kFcTile outputs.
         *
         * @param w Weights, as fp32 or fp16 bits depending on `half`.
         * @param half Whether `w` holds fp16 bits.
         * @param bias Bias of each output.
         * @param x Input values.
         * @param in Number of inputs.
         * @param out Number of outputs, a multiple of kFcTile.
         * @param y Receives the outputs.
         */
        void linear(const void *w, bool half, const float *bias, const float *x, int in, int out,
                    float *y) {
            for (int o = 0; o < out; ++o) {
                // Each tile holds kFcTile consecutive outputs, stored as [input][output].
                const int tile = o / kFcTile;
                const int within_tile = o % kFcTile;
                const size_t tile_base = static_cast<size_t>(tile) * kFcTile * in;

                double acc = 0.0;
                for (int i = 0; i < in; ++i) {
                    const size_t index = tile_base + static_cast<size_t>(i) * kFcTile + within_tile;
                    const float weight =
                        half ? half_to_float(static_cast<const uint16_t *>(w)[index])
                             : static_cast<const float *>(w)[index];
                    acc += static_cast<double>(weight) * x[i];
                }
                y[o] = static_cast<float>(acc) + bias[o];
            }
        }

        /**
         * @brief Convolution whose weights and bias are tensors of the network.
         *
         * @param w Network weights.
         * @param weights Tensor of the convolution weights.
         * @param bias Tensor of the convolution bias.
         * @param in Input map, [square][in_channels].
         * @param in_channels Channels of the input map.
         * @param out Output map, [square][out_channels].
         * @param out_channels Channels of the output map.
         * @param kernel Kernel side, which should be odd.
         */
        void convolve(const Weights &w, Tensor weights, Tensor bias, const float *in,
                      int in_channels, float *out, int out_channels, int kernel) {
            conv(in, in_channels, w.f32(weights), w.f32(bias), out, out_channels, kernel);
        }

        /**
         * @brief Group normalization whose scale and shift are tensors of the network.
         *
         * @param w Network weights.
         * @param x Feature map, [square][channel], normalized in place.
         * @param channels Channels of the map.
         * @param scale Tensor of the per-channel scale.
         * @param shift Tensor of the per-channel shift.
         * @param residual Added after the activation, or null.
         */
        void normalize(const Weights &w, float *x, int channels, Tensor scale, Tensor shift,
                       const float *residual) {
            group_norm_act(x, channels, w.f32(scale), w.f32(shift), w.norm_eps(), w.leaky_slope(),
                           residual);
        }

        /**
         * @brief Layer normalization whose scale and shift are tensors of the network.
         *
         * @param w Network weights.
         * @param x Values to normalize in place.
         * @param n Number of values.
         * @param scale Tensor of the per-value scale.
         * @param shift Tensor of the per-value shift.
         */
        void layer_normalize(const Weights &w, float *x, int n, Tensor scale, Tensor shift) {
            layer_norm_act(x, n, w.f32(scale), w.f32(shift), w.norm_eps(), w.leaky_slope());
        }

        /**
         * @brief Fully connected layer whose weights and bias are tensors of the network.
         *
         * @param w Network weights.
         * @param weights Tensor of the packed weights.
         * @param bias Tensor of the bias.
         * @param x Input values.
         * @param in Number of inputs.
         * @param out Number of outputs.
         * @param y Receives the outputs.
         * @param stored_like_fc Whether the weights follow the storage type of the large
         *        fully connected matrices, which can be fp16.
         */
        void fully_connected(const Weights &w, Tensor weights, Tensor bias, const float *x, int in,
                             int out, float *y, bool stored_like_fc) {
            const bool half = stored_like_fc && w.fc_type() == FcType::Fp16;
            const void *packed = half ? static_cast<const void *>(w.f16(weights)) : w.f32(weights);
            linear(packed, half, w.f32(bias), x, in, out, y);
        }

    }  // namespace

    void forward_reference(const Weights &w, Scratch &s, const float *board, Evaluation &out) {
        std::memcpy(s.board, board, sizeof(s.board));

        // First convolution over the board.
        convolve(w, Tensor::Conv0W, Tensor::Conv0B, s.board, 1, s.seq, kSeqChannels, 5);
        normalize(w, s.seq, kSeqChannels, Tensor::Gn0W, Tensor::Gn0B, nullptr);

        // Shortcut path, added after the last convolution of the residual block.
        convolve(w, Tensor::ShortW, Tensor::ShortB, s.seq, kSeqChannels, s.shortcut, kResChannels,
                 1);

        // Residual block of three convolutions.
        convolve(w, Tensor::Conv1W, Tensor::Conv1B, s.seq, kSeqChannels, s.res_a, kResChannels, 3);
        normalize(w, s.res_a, kResChannels, Tensor::Gn1W, Tensor::Gn1B, nullptr);

        convolve(w, Tensor::Conv2W, Tensor::Conv2B, s.res_a, kResChannels, s.res_b, kResChannels,
                 3);
        normalize(w, s.res_b, kResChannels, Tensor::Gn2W, Tensor::Gn2B, nullptr);

        convolve(w, Tensor::Conv3W, Tensor::Conv3B, s.res_b, kResChannels, s.res_a, kResChannels,
                 3);
        normalize(w, s.res_a, kResChannels, Tensor::Gn3W, Tensor::Gn3B, s.shortcut);

        // Two fully connected layers, each followed by a layer normalization.
        fully_connected(w, Tensor::Fc1W, Tensor::Fc1B, s.res_a, kFc1In, kFc1Out, s.fc1, true);
        layer_normalize(w, s.fc1, kFc1Out, Tensor::Ln1W, Tensor::Ln1B);

        fully_connected(w, Tensor::Fc2W, Tensor::Fc2B, s.fc1, kFc1Out, kFc2Out, s.fc2, true);
        layer_normalize(w, s.fc2, kFc2Out, Tensor::Ln2W, Tensor::Ln2B);

        // Policy logits and value pre-activation.
        fully_connected(w, Tensor::HeadW, Tensor::HeadB, s.fc2, kFc2Out, kHeadOut, s.head, false);
        std::memcpy(out.logits, s.head, sizeof(out.logits));
        out.value = std::tanh(s.head[kHeadValue]);
    }

}  // namespace othello
