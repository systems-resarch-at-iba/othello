/**
 * @file bench_rules.cpp
 * @brief Times the bitboard rules and prints JSON.
 *
 * Usage: bench_rules --reps R
 *
 * Reports perft throughput and the time of single legal-move and play calls
 * over a fixed set of positions.
 */

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "othello/position.hpp"

namespace {

    /**
     * @brief Clock used for all timings.
     */
    using Clock = std::chrono::steady_clock;
    /**
     * @brief Position type used by the benchmark.
     */
    using othello::Position;

    /**
     * @brief Counts the leaf nodes of the game tree.
     *
     * @param p Root position.
     * @param depth Number of plies to expand.
     * @return Number of leaf nodes.
     */
    uint64_t perft(Position p, int depth) {
        if (depth == 0) {
            return 1;
        }
        uint64_t moves = othello::legal_moves(p);
        if (moves == 0) {
            return othello::is_terminal(p)
                       ? 1
                       : perft(othello::play(p, othello::kPassAction), depth - 1);
        }
        uint64_t leaves = 0;
        while (moves != 0) {
            const int square = std::countr_zero(moves);
            moves &= moves - 1;
            leaves += perft(othello::play(p, square), depth - 1);
        }
        return leaves;
    }

    /**
     * @brief Next value of a xorshift generator.
     *
     * @param state Generator state, advanced by the call.
     * @return The next random value.
     */
    uint64_t next_random(uint64_t &state) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    /**
     * @brief Positions reached by random games, up to 4096 of them.
     *
     * @return The positions.
     */
    std::vector<Position> random_positions() {
        std::vector<Position> positions;
        uint64_t rng = 0x9E3779B97F4A7C15ULL;
        while (positions.size() < 4096) {
            Position p = othello::initial_position();
            while (!othello::is_terminal(p) && positions.size() < 4096) {
                uint64_t moves = othello::legal_moves(p);
                positions.push_back(p);
                if (moves == 0) {
                    p = othello::play(p, othello::kPassAction);
                    continue;
                }
                int skip = static_cast<int>(next_random(rng) % std::popcount(moves));
                while (skip-- > 0) {
                    moves &= moves - 1;
                }
                p = othello::play(p, std::countr_zero(moves));
            }
        }
        return positions;
    }

}  // namespace

/**
 * @brief Times perft, move generation and playing, and prints JSON.
 *
 * @param argc Argument count.
 * @param argv Arguments.
 * @return 0.
 */
int main(int argc, char **argv) {
    int reps = 5;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0) {
            reps = std::atoi(argv[i + 1]);
        }
    }

    constexpr int kPerftDepth = 10;
    std::printf("{\"perft_depth\": %d, \"perft_leaves\": ", kPerftDepth);
    uint64_t leaves = 0;
    std::vector<double> perft_seconds;
    for (int r = 0; r < reps; ++r) {
        const auto start = Clock::now();
        leaves = perft(othello::initial_position(), kPerftDepth);
        perft_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }
    std::printf("%llu, \"perft_seconds\": [", static_cast<unsigned long long>(leaves));
    for (int r = 0; r < reps; ++r) {
        std::printf("%s%.4f", r == 0 ? "" : ", ", perft_seconds[r]);
    }

    const std::vector<Position> positions = random_positions();
    constexpr int kPasses = 20000;
    uint64_t sink = 0;
    std::printf("], \"legal_moves_ns\": [");
    for (int r = 0; r < reps; ++r) {
        const auto start = Clock::now();
        for (int pass = 0; pass < kPasses; ++pass) {
            for (const Position &p : positions) {
                sink += othello::legal_moves(p);
            }
        }
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
                          (static_cast<double>(kPasses) * positions.size());
        std::printf("%s%.2f", r == 0 ? "" : ", ", ns);
    }
    std::printf("], \"play_ns\": [");
    std::vector<int> first_moves;
    for (const Position &p : positions) {
        const uint64_t moves = othello::legal_moves(p);
        first_moves.push_back(moves == 0 ? othello::kPassAction : std::countr_zero(moves));
    }
    for (int r = 0; r < reps; ++r) {
        const auto start = Clock::now();
        for (int pass = 0; pass < kPasses; ++pass) {
            for (size_t i = 0; i < positions.size(); ++i) {
                const Position q = othello::play(positions[i], first_moves[i]);
                sink += q.mine ^ q.theirs;
            }
        }
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
                          (static_cast<double>(kPasses) * positions.size());
        std::printf("%s%.2f", r == 0 ? "" : ", ", ns);
    }
    std::printf("], \"sink\": %llu}\n", static_cast<unsigned long long>(sink));
    return 0;
}
