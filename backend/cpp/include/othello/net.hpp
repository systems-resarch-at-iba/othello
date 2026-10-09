#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace othello {

    /** Squares on the board. */
    constexpr int kSquares = 64;
    /** Actions of the policy: one per square and the pass. */
    constexpr int kActions = 65;
    /** Channels of the first convolution. */
    constexpr int kSeqChannels = 16;
    /** Channels of the residual block. */
    constexpr int kResChannels = 64;
    /** Groups of every group normalization. */
    constexpr int kGroups = 8;
    /** Outputs of the first fully connected layer. */
    constexpr int kFc1Out = 1024;
    /** Outputs of the second fully connected layer. */
    constexpr int kFc2Out = 512;
    /** Inputs of the first fully connected layer: every channel of every square. */
    constexpr int kFc1In = kSquares * kResChannels;
    /** Outputs per tile of a packed fully connected layer. */
    constexpr int kFcTile = 16;
    /** Policy logits followed by the value pre-activation, padded to a multiple of kFcTile. */
    constexpr int kHeadOut = 80;
    /** Index of the value pre-activation among the head outputs. */
    constexpr int kHeadValue = kActions;
    /** Most boards one batched forward pass takes. */
    constexpr int kMaxBatch = 4;
    /** Side of the board. */
    constexpr int kBoardSide = 8;
    /** Side of a feature map with one square of zero padding on every edge. */
    constexpr int kPadSide = kBoardSide + 2;
    /** Side of the input map, padded by two squares for the 5x5 convolution. */
    constexpr int kInputPadSide = kBoardSide + 4;

    /** @brief Tensors of a weight file, in file order. */
    enum class Tensor : int {
        Conv0W,  ///< Weights of the first convolution.
        Conv0B,  ///< Bias of the first convolution.
        Gn0W,    ///< Scale of the first group normalization.
        Gn0B,    ///< Shift of the first group normalization.
        ShortW,  ///< Weights of the shortcut convolution.
        ShortB,  ///< Bias of the shortcut convolution.
        Conv1W,  ///< Weights of the first residual convolution.
        Conv1B,  ///< Bias of the first residual convolution.
        Gn1W,    ///< Scale of the first residual normalization.
        Gn1B,    ///< Shift of the first residual normalization.
        Conv2W,  ///< Weights of the second residual convolution.
        Conv2B,  ///< Bias of the second residual convolution.
        Gn2W,    ///< Scale of the second residual normalization.
        Gn2B,    ///< Shift of the second residual normalization.
        Conv3W,  ///< Weights of the third residual convolution.
        Conv3B,  ///< Bias of the third residual convolution.
        Gn3W,    ///< Scale of the third residual normalization.
        Gn3B,    ///< Shift of the third residual normalization.
        Fc1W,    ///< Weights of the first fully connected layer.
        Fc1B,    ///< Bias of the first fully connected layer.
        Ln1W,    ///< Scale of the first layer normalization.
        Ln1B,    ///< Shift of the first layer normalization.
        Fc2W,    ///< Weights of the second fully connected layer.
        Fc2B,    ///< Bias of the second fully connected layer.
        Ln2W,    ///< Scale of the second layer normalization.
        Ln2B,    ///< Shift of the second layer normalization.
        HeadW,   ///< Weights of the policy and value head.
        HeadB,   ///< Bias of the policy and value head.
        Count    ///< Number of tensors.
    };

    /** @brief Storage type of the Fc1W and Fc2W tensors. All other tensors are fp32. */
    enum class FcType : uint32_t {
        Fp32 = 0,  ///< 32-bit floats.
        Fp16 = 1   ///< IEEE half-precision floats.
    };

    /**
     * @brief Network weights loaded from a weight file.
     *
     * Activations use a channels-last layout (square-major, channel-minor).
     * Convolution weights are laid out [tap][input channel][output channel],
     * and the first fully connected layer expects its input in the same
     * channels-last order. Fully connected weights are packed in tiles of
     * kFcTile outputs, each tile stored as [input][output within tile], so a
     * layer reads its weights as one sequential stream per tile.
     */
    class Weights {
    public:
        /**
         * @brief Reads and validates a weight file.
         *
         * @param path Path of the weight file.
         * @throws std::runtime_error When the file is missing, malformed or
         *         describes a different architecture.
         */
        explicit Weights(const std::string &path);

        /**
         * @brief Pointer to an fp32 tensor.
         *
         * @param t Tensor to locate.
         * @return Address of the tensor's first element, aligned to 64 bytes.
         */
        const float *f32(Tensor t) const {
            return reinterpret_cast<const float *>(data_.get() + offsets_[static_cast<int>(t)]);
        }

        /**
         * @brief Pointer to an fp16 tensor, as raw IEEE half bits.
         *
         * @param t Tensor to locate.
         * @return Address of the tensor's first element, aligned to 64 bytes.
         */
        const uint16_t *f16(Tensor t) const {
            return reinterpret_cast<const uint16_t *>(data_.get() + offsets_[static_cast<int>(t)]);
        }

        /**
         * @brief Storage type of the large fully connected matrices.
         *
         * @return The type recorded in the weight file.
         */
        FcType fc_type() const {
            return fc_type_;
        }

        /**
         * @brief Negative slope of every leaky ReLU.
         *
         * @return The slope recorded in the weight file.
         */
        float leaky_slope() const {
            return leaky_slope_;
        }

        /**
         * @brief Epsilon added to every variance in the normalizations.
         *
         * @return The value recorded in the weight file.
         */
        float norm_eps() const {
            return norm_eps_;
        }

    private:
        /** @brief Releases memory allocated with std::aligned_alloc. */
        struct FreeDeleter {
            /**
             * @brief Frees a buffer.
             *
             * @param p Buffer to free.
             */
            void operator()(unsigned char *p) const;
        };

        std::unique_ptr<unsigned char[], FreeDeleter> data_;    ///< Contents of the weight file.
        size_t offsets_[static_cast<int>(Tensor::Count)] = {};  ///< Byte offset of each tensor.
        FcType fc_type_ = FcType::Fp32;  ///< Storage type of the large matrices.
        float leaky_slope_ = 0.0f;       ///< Negative slope of the leaky ReLUs.
        float norm_eps_ = 0.0f;          ///< Variance epsilon of the normalizations.
    };

    /**
     * @brief Preallocated buffers for forward passes. One instance should serve one thread.
     *
     * The plain-loop pass uses the first group of arrays. The vectorized pass
     * uses the padded maps, whose borders stay zero because no pass writes
     * them, and the batch arrays.
     */
    struct alignas(64) Scratch {
        float board[kSquares];                    ///< Input board.
        float seq[kSquares * kSeqChannels];       ///< Output of the first convolution.
        float shortcut[kSquares * kResChannels];  ///< Output of the shortcut convolution.
        float res_a[kSquares * kResChannels];     ///< First residual feature map.
        float res_b[kSquares * kResChannels];     ///< Second residual feature map.
        float fc1[kFc1Out];                       ///< Output of the first fully connected layer.
        float fc2[kFc2Out];                       ///< Output of the second fully connected layer.
        float head[kHeadOut];                     ///< Output of the head.

        float input_pad[kInputPadSide * kInputPadSide] = {};  ///< Input padded by two squares.
        float seq_pad[kPadSide * kPadSide * kSeqChannels] =
            {};  ///< First feature map, padded by one square.
        float res_pad_a[kPadSide * kPadSide * kResChannels] = {};  ///< First residual map, padded.
        float res_pad_b[kPadSide * kPadSide * kResChannels] = {};  ///< Second residual map, padded.
        float shortcut_flat[kSquares * kResChannels];  ///< Shortcut output of the current board.
        float flat[kMaxBatch * kFc1In];                ///< Flattened residual output of each board.
        float fc1_batch[kMaxBatch * kFc1Out];    ///< First fully connected output of each board.
        float fc2_batch[kMaxBatch * kFc2Out];    ///< Second fully connected output of each board.
        float head_batch[kMaxBatch * kHeadOut];  ///< Head output of each board.
    };

    /** @brief Network output for one position. */
    struct Evaluation {
        float logits[kActions];  ///< Policy logits for the 64 squares and the pass action.
        float value;             ///< Expected outcome for the side to move, in [-1, 1].
    };

    /**
     * @brief Forward pass with plain loops, used as the reference for the
     *        vectorized pass.
     *
     * @param w Network weights.
     * @param s Scratch buffers.
     * @param board 64 squares seen from the side to move: 1 for own discs, -1 for opponent discs.
     * @param out Receives the evaluation.
     */
    void forward_reference(const Weights &w, Scratch &s, const float *board, Evaluation &out);

    /**
     * @brief Forward pass over several boards with AVX2, FMA and F16C instructions.
     *
     * The fully connected layers read their weights once for the whole batch.
     *
     * @param w Network weights.
     * @param s Scratch buffers.
     * @param boards `count` boards of 64 squares each, seen from the side to move.
     * @param count Number of boards, from 1 to kMaxBatch.
     * @param out Receives `count` evaluations.
     */
    void forward_batch_avx2(const Weights &w, Scratch &s, const float *boards, int count,
                            Evaluation *out);

    /**
     * @brief Forward pass over one board with AVX2, FMA and F16C instructions.
     *
     * @param w Network weights.
     * @param s Scratch buffers.
     * @param board 64 squares seen from the side to move: 1 for own discs, -1 for opponent discs.
     * @param out Receives the evaluation.
     */
    void forward_avx2(const Weights &w, Scratch &s, const float *board, Evaluation &out);

    /**
     * @brief Whether the running CPU supports the instructions forward_avx2 uses.
     *
     * @return True when AVX2, FMA and F16C are available.
     */
    bool avx2_supported();

    /**
     * @brief Log-softmax of the policy logits.
     *
     * @param logits Policy logits.
     * @param out Receives kActions log-probabilities.
     */
    void log_softmax(const float *logits, float *out);

    /**
     * @brief Softmax over the legal actions only.
     *
     * @param logits Policy logits.
     * @param legal_squares Bit mask of the legal squares.
     * @param out Receives kActions probabilities. Illegal actions get 0. When
     *        no square is legal the pass action gets probability 1.
     */
    void masked_softmax(const float *logits, uint64_t legal_squares, float *out);

    /** @brief A loaded network that runs the fastest forward pass the CPU supports. */
    class Network {
    public:
        /**
         * @brief Loads the weights.
         *
         * @param path Path of the weight file.
         * @throws std::runtime_error See Weights.
         */
        explicit Network(const std::string &path);

        /**
         * @brief Forward pass over one board with the fastest supported instruction set.
         *
         * @param s Scratch buffers.
         * @param board 64 squares seen from the side to move.
         * @param out Receives the evaluation.
         */
        void forward(Scratch &s, const float *board, Evaluation &out) const;

        /**
         * @brief Forward pass over several boards with the fastest supported instruction set.
         *
         * @param s Scratch buffers.
         * @param boards `count` boards of 64 squares each.
         * @param count Number of boards, from 1 to kMaxBatch.
         * @param out Receives `count` evaluations.
         */
        void forward_batch(Scratch &s, const float *boards, int count, Evaluation *out) const;

        /**
         * @brief Forward pass over one board with plain loops.
         *
         * @param s Scratch buffers.
         * @param board 64 squares seen from the side to move.
         * @param out Receives the evaluation.
         */
        void forward_reference(Scratch &s, const float *board, Evaluation &out) const;

        /**
         * @brief Name of the instruction set used by forward().
         *
         * @return "avx2" or "scalar".
         */
        const char *isa() const {
            return use_avx2_ ? "avx2" : "scalar";
        }

    private:
        Weights weights_;  ///< Loaded weights.
        bool use_avx2_;    ///< Whether forward() uses the vectorized pass.
    };

}  // namespace othello
