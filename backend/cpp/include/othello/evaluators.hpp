#pragma once

#include <bit>
#include <cstdint>
#include <memory>

#include "othello/net.hpp"
#include "othello/position.hpp"

namespace othello {

    /** @brief Evaluates positions with a Network. */
    class NetworkEvaluator {
    public:
        /**
         * @brief Wraps a network that outlives the evaluator.
         *
         * @param network Network to run.
         */
        explicit NetworkEvaluator(const Network &network)
            : network_(&network), scratch_(std::make_unique<Scratch>()) {
        }

        /**
         * @brief Policy over the legal actions and value of a position.
         *
         * @param p Position to evaluate.
         * @param legal Bit mask of the legal squares.
         * @param probs Receives kActions probabilities.
         * @param value Receives the value for the side to move.
         */
        void operator()(Position p, uint64_t legal, float *probs, float &value) {
            float board[kSquares];
            for (int i = 0; i < kSquares; ++i) {
                board[i] =
                    static_cast<float>((p.mine >> i) & 1) - static_cast<float>((p.theirs >> i) & 1);
            }
            Evaluation evaluation;
            network_->forward(*scratch_, board, evaluation);
            masked_softmax(evaluation.logits, legal, probs);
            value = evaluation.value;
        }

        /**
         * @brief Policies over the legal actions and values of several positions.
         *
         * The network evaluates up to kMaxBatch positions per pass.
         *
         * @param count Number of positions.
         * @param positions Positions to evaluate.
         * @param legal Bit masks of the legal squares, one per position.
         * @param probs Receives `count` rows of kActions probabilities.
         * @param values Receives `count` values for the sides to move.
         */
        void evaluate_batch(int count, const Position *positions, const uint64_t *legal,
                            float *probs, float *values) {
            for (int first = 0; first < count; first += kMaxBatch) {
                const int n = count - first < kMaxBatch ? count - first : kMaxBatch;
                float boards[kMaxBatch * kSquares];
                for (int k = 0; k < n; ++k) {
                    const Position &p = positions[first + k];
                    for (int i = 0; i < kSquares; ++i) {
                        boards[k * kSquares + i] = static_cast<float>((p.mine >> i) & 1) -
                                                   static_cast<float>((p.theirs >> i) & 1);
                    }
                }
                Evaluation evaluations[kMaxBatch];
                network_->forward_batch(*scratch_, boards, n, evaluations);
                for (int k = 0; k < n; ++k) {
                    masked_softmax(evaluations[k].logits, legal[first + k],
                                   probs + (first + k) * kActions);
                    values[first + k] = evaluations[k].value;
                }
            }
        }

    private:
        const Network *network_;            ///< Network to run, owned by the caller.
        std::unique_ptr<Scratch> scratch_;  ///< Buffers of the forward passes.
    };

    /**
     * @brief Deterministic pseudo-random policy and value, for tests.
     *
     * Policy weights are integers over 2^16 and values are integers over
     * 1024, so every sum the search takes over them is exact in float32.
     */
    class StubEvaluator {
    public:
        /**
         * @brief Policy over the legal actions and value of a position.
         *
         * @param p Position to evaluate.
         * @param legal Bit mask of the legal squares.
         * @param probs Receives kActions probabilities.
         * @param value Receives the value for the side to move.
         */
        void operator()(Position p, uint64_t legal, float *probs, float &value) const {
            const uint64_t h = position_hash(p);
            float sum = 0.0f;
            for (int a = 0; a < kActions; ++a) {
                const bool is_legal = a < kSquares ? ((legal >> a) & 1) != 0 : legal == 0;
                probs[a] =
                    is_legal ? static_cast<float>(1 + weight_hash(h, a) % 997) / 65536.0f : 0.0f;
                sum += probs[a];
            }
            for (int a = 0; a < kActions; ++a) {
                probs[a] /= sum;
            }
            value =
                static_cast<float>(static_cast<int>(detail::mix64(h ^ 0xABCDEFULL) % 2049) - 1024) /
                1024.0f;
        }

        /**
         * @brief Policies over the legal actions and values of several positions.
         *
         * @param count Number of positions.
         * @param positions Positions to evaluate.
         * @param legal Bit masks of the legal squares, one per position.
         * @param probs Receives `count` rows of kActions probabilities.
         * @param values Receives `count` values for the sides to move.
         */
        void evaluate_batch(int count, const Position *positions, const uint64_t *legal,
                            float *probs, float *values) const {
            for (int k = 0; k < count; ++k) {
                (*this)(positions[k], legal[k], probs + k * kActions, values[k]);
            }
        }

    private:
        /**
         * @brief Pseudo-random value that belongs to a position and an action.
         *
         * @param h Hash of the position.
         * @param action Action index.
         * @return A mixed 64-bit value.
         */
        static uint64_t weight_hash(uint64_t h, int action) {
            return detail::mix64(h + static_cast<uint64_t>(action + 1) * 0x9E3779B97F4A7C15ULL);
        }
    };

}  // namespace othello
