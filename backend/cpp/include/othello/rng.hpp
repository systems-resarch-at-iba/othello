#pragma once

#include <bit>
#include <cstdint>

namespace othello {

    /**
     * @brief 64-bit mixing step that also serves to expand a seed into generator state.
     *
     * @param state Generator state, advanced by the call.
     * @return The next mixed 64-bit value.
     */
    constexpr uint64_t splitmix64(uint64_t &state) noexcept {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    /** @brief xoshiro256** pseudo-random generator. */
    class Rng {
    public:
        /**
         * @brief Seeds the generator.
         *
         * @param seed Value expanded into the four state words.
         */
        explicit Rng(uint64_t seed) noexcept {
            for (uint64_t &word : state_) {
                word = splitmix64(seed);
            }
        }

        /**
         * @brief Next 64 random bits.
         *
         * @return The next value of the sequence.
         */
        uint64_t next() noexcept {
            const uint64_t result = std::rotl(state_[1] * 5, 7) * 9;
            const uint64_t t = state_[1] << 17;
            state_[2] ^= state_[0];
            state_[3] ^= state_[1];
            state_[1] ^= state_[2];
            state_[0] ^= state_[3];
            state_[2] ^= t;
            state_[3] = std::rotl(state_[3], 45);
            return result;
        }

        /**
         * @brief Uniform integer in [0, bound).
         *
         * @param bound Exclusive upper bound, at least 1.
         * @return A value below `bound`.
         */
        uint32_t below(uint32_t bound) noexcept {
            return static_cast<uint32_t>(((next() >> 32) * bound) >> 32);
        }

    private:
        uint64_t state_[4];  ///< Generator state words.
    };

}  // namespace othello
