#include "othello/net.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace othello {

    namespace {

        /** @brief First bytes of every weight file. */
        constexpr char kMagic[8] = {'O', 'T', 'H', 'E', 'L', 'N', 'E', 'T'};
        /** @brief Version of the weight file format. */
        constexpr uint32_t kVersion = 2;
        /** @brief Alignment in bytes of every tensor and of the loaded file. */
        constexpr size_t kAlignment = 64;
        /** @brief Number of tensors of a weight file. */
        constexpr int kTensorCount = static_cast<int>(Tensor::Count);

        /** @brief On-disk header. All fields are little-endian. */
        struct FileHeader {
            char magic[8];          ///< Equals kMagic.
            uint32_t version;       ///< Equals kVersion.
            uint32_t fc_type;       ///< Storage type of the large matrices, a FcType value.
            uint32_t squares;       ///< Squares on the board.
            uint32_t seq_channels;  ///< Channels of the first convolution.
            uint32_t res_channels;  ///< Channels of the residual block.
            uint32_t groups;        ///< Groups of the group normalizations.
            uint32_t fc1_out;       ///< Outputs of the first fully connected layer.
            uint32_t fc2_out;       ///< Outputs of the second fully connected layer.
            uint32_t actions;       ///< Actions of the policy.
            uint32_t head_out;      ///< Outputs of the head, padded.
            float leaky_slope;      ///< Negative slope of the leaky ReLUs.
            float norm_eps;         ///< Variance epsilon of the normalizations.
            uint32_t tensor_count;  ///< Number of tensors.
            uint32_t reserved;      ///< Zero.
            uint64_t offsets[kTensorCount];  ///< Byte offset of each tensor.
        };
        static_assert(sizeof(FileHeader) == 64 + 8 * kTensorCount);

        /**
         * @brief Elements of a convolution weight tensor.
         *
         * @param taps Positions of the kernel.
         * @param in_channels Input channels.
         * @param out_channels Output channels.
         * @return The number of weights.
         */
        constexpr size_t conv_elements(size_t taps, size_t in_channels, size_t out_channels) {
            return taps * in_channels * out_channels;
        }

        /**
         * @brief Elements of a fully connected weight matrix.
         *
         * @param outputs Output count.
         * @param inputs Input count.
         * @return The number of weights.
         */
        constexpr size_t matrix_elements(size_t outputs, size_t inputs) {
            return outputs * inputs;
        }

        /** @brief Number of elements of each tensor, indexed by Tensor. */
        constexpr size_t kElementCounts[kTensorCount] = {
            conv_elements(5 * 5, 1, kSeqChannels),             // Conv0W
            kSeqChannels,                                      // Conv0B
            kSeqChannels,                                      // Gn0W
            kSeqChannels,                                      // Gn0B
            conv_elements(1, kSeqChannels, kResChannels),      // ShortW
            kResChannels,                                      // ShortB
            conv_elements(3 * 3, kSeqChannels, kResChannels),  // Conv1W
            kResChannels,                                      // Conv1B
            kResChannels,                                      // Gn1W
            kResChannels,                                      // Gn1B
            conv_elements(3 * 3, kResChannels, kResChannels),  // Conv2W
            kResChannels,                                      // Conv2B
            kResChannels,                                      // Gn2W
            kResChannels,                                      // Gn2B
            conv_elements(3 * 3, kResChannels, kResChannels),  // Conv3W
            kResChannels,                                      // Conv3B
            kResChannels,                                      // Gn3W
            kResChannels,                                      // Gn3B
            matrix_elements(kFc1Out, kFc1In),                  // Fc1W
            kFc1Out,                                           // Fc1B
            kFc1Out,                                           // Ln1W
            kFc1Out,                                           // Ln1B
            matrix_elements(kFc2Out, kFc1Out),                 // Fc2W
            kFc2Out,                                           // Fc2B
            kFc2Out,                                           // Ln2W
            kFc2Out,                                           // Ln2B
            matrix_elements(kHeadOut, kFc2Out),                // HeadW
            kHeadOut,                                          // HeadB
        };

        /**
         * @brief Bytes per element of a tensor under a given storage type.
         *
         * @param t Tensor to measure.
         * @param fc_type Storage type of the large fully connected matrices.
         * @return 2 for the large matrices stored as fp16, otherwise 4.
         */
        size_t element_bytes(Tensor t, FcType fc_type) {
            const bool is_fc = t == Tensor::Fc1W || t == Tensor::Fc2W;
            return is_fc && fc_type == FcType::Fp16 ? 2 : 4;
        }

        /**
         * @brief Reports an unusable weight file.
         *
         * @param path Path of the weight file.
         * @param why What is wrong with the file.
         * @throws std::runtime_error Always.
         */
        [[noreturn]] void fail(const std::string &path, const std::string &why) {
            throw std::runtime_error("weight file " + path + ": " + why);
        }

    }  // namespace

    void Weights::FreeDeleter::operator()(unsigned char *p) const {
        std::free(p);
    }

    Weights::Weights(const std::string &path) {
        // Read the whole file into memory aligned for the vector loads.
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            fail(path, "cannot be opened");
        }
        const size_t size = static_cast<size_t>(file.tellg());
        if (size < sizeof(FileHeader)) {
            fail(path, "is shorter than the header");
        }

        const size_t padded = (size + kAlignment - 1) / kAlignment * kAlignment;
        data_.reset(static_cast<unsigned char *>(std::aligned_alloc(kAlignment, padded)));
        if (!data_) {
            throw std::bad_alloc();
        }
        std::memset(data_.get(), 0, padded);

        file.seekg(0);
        file.read(reinterpret_cast<char *>(data_.get()), static_cast<std::streamsize>(size));
        if (!file) {
            fail(path, "cannot be read");
        }

        // Check the header against the architecture this build computes.
        FileHeader header;
        std::memcpy(&header, data_.get(), sizeof(header));
        if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
            fail(path, "has the wrong magic");
        }
        if (header.version != kVersion) {
            fail(path, "has an unsupported format version");
        }
        if (header.fc_type > static_cast<uint32_t>(FcType::Fp16)) {
            fail(path, "has an unknown fully connected weight type");
        }

        const uint32_t found[] = {
            header.squares, header.seq_channels, header.res_channels,
            header.groups,  header.fc1_out,      header.fc2_out,
            header.actions, header.head_out,     header.tensor_count,
        };
        const uint32_t expected[] = {
            kSquares, kSeqChannels, kResChannels, kGroups,      kFc1Out,
            kFc2Out,  kActions,     kHeadOut,     kTensorCount,
        };
        if (!std::equal(std::begin(found), std::end(found), std::begin(expected))) {
            fail(path, "describes a different architecture");
        }

        fc_type_ = static_cast<FcType>(header.fc_type);
        leaky_slope_ = header.leaky_slope;
        norm_eps_ = header.norm_eps;

        // Check that every tensor is aligned and lies inside the file.
        for (int i = 0; i < kTensorCount; ++i) {
            const uint64_t offset = header.offsets[i];
            const size_t bytes =
                kElementCounts[i] * element_bytes(static_cast<Tensor>(i), fc_type_);

            if (offset % kAlignment != 0) {
                fail(path, "has a misaligned tensor");
            }
            if (offset < sizeof(FileHeader) || offset > size || bytes > size - offset) {
                fail(path, "has a tensor outside the file");
            }
            offsets_[i] = offset;
        }
    }

    bool avx2_supported() {
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
               __builtin_cpu_supports("f16c");
    }

    void log_softmax(const float *logits, float *out) {
        const float max_logit = *std::max_element(logits, logits + kActions);

        float sum = 0.0f;
        for (int a = 0; a < kActions; ++a) {
            sum += std::exp(logits[a] - max_logit);
        }

        const float log_sum = std::log(sum) + max_logit;
        for (int a = 0; a < kActions; ++a) {
            out[a] = logits[a] - log_sum;
        }
    }

    void masked_softmax(const float *logits, uint64_t legal_squares, float *out) {
        const bool pass_legal = legal_squares == 0;
        const auto is_legal = [&](int action) {
            return action < kSquares ? ((legal_squares >> action) & 1) != 0 : pass_legal;
        };

        // Subtract the largest legal logit so that the exponentials cannot overflow.
        float max_logit = -INFINITY;
        for (int a = 0; a < kActions; ++a) {
            if (is_legal(a)) {
                max_logit = std::max(max_logit, logits[a]);
            }
        }

        float sum = 0.0f;
        for (int a = 0; a < kActions; ++a) {
            out[a] = is_legal(a) ? std::exp(logits[a] - max_logit) : 0.0f;
            sum += out[a];
        }

        const float inverse = 1.0f / sum;
        for (int a = 0; a < kActions; ++a) {
            out[a] *= inverse;
        }
    }

    Network::Network(const std::string &path) : weights_(path), use_avx2_(avx2_supported()) {
    }

    void Network::forward(Scratch &s, const float *board, Evaluation &out) const {
        if (use_avx2_) {
            forward_avx2(weights_, s, board, out);
        } else {
            othello::forward_reference(weights_, s, board, out);
        }
    }

    void Network::forward_batch(Scratch &s, const float *boards, int count, Evaluation *out) const {
        if (count < 1 || count > kMaxBatch) {
            throw std::invalid_argument("batch size must be in [1, kMaxBatch]");
        }
        if (use_avx2_) {
            forward_batch_avx2(weights_, s, boards, count, out);
        } else {
            for (int b = 0; b < count; ++b) {
                othello::forward_reference(weights_, s, boards + b * kSquares, out[b]);
            }
        }
    }

    void Network::forward_reference(Scratch &s, const float *board, Evaluation &out) const {
        othello::forward_reference(weights_, s, board, out);
    }

}  // namespace othello
