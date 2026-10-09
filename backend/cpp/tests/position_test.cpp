#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "othello/position.hpp"

namespace {

    /**
     * @brief Position type used by the tests.
     */
    using othello::Position;

    /**
     * @brief Number of failed checks.
     */
    int failures = 0;

/**
 * @brief Records a failed condition and prints its location.
 *
 * @param cond Condition that should hold.
 */
#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                          \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

    /**
     * @brief Counts the leaf nodes of the game tree.
     *
     * A side without a legal move passes when the opponent can move, and a
     * finished game counts as one leaf.
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
            if (othello::is_terminal(p)) {
                return 1;
            }
            return perft(othello::play(p, othello::kPassAction), depth - 1);
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
     * @brief Collects every position reachable within a number of plies.
     *
     * The move invariants are checked along the way.
     *
     * @param p Root position.
     * @param depth Number of plies to expand.
     * @param out Receives the positions.
     */
    void collect(Position p, int depth, std::vector<Position> &out) {
        out.push_back(p);
        CHECK((p.mine & p.theirs) == 0);
        if (depth == 0) {
            return;
        }
        const int discs = std::popcount(p.mine | p.theirs);
        uint64_t moves = othello::legal_moves(p);
        CHECK((moves & (p.mine | p.theirs)) == 0);
        while (moves != 0) {
            const int square = std::countr_zero(moves);
            moves &= moves - 1;
            const uint64_t flips = othello::flipped_discs(p, square);
            CHECK(flips != 0);
            CHECK((flips & p.theirs) == flips);
            const Position next = othello::play(p, square);
            CHECK(std::popcount(next.mine | next.theirs) == discs + 1);
            CHECK(std::popcount(next.theirs) == std::popcount(p.mine) + 1 + std::popcount(flips));
            collect(next, depth - 1, out);
        }
    }

    /**
     * @brief Strict weak ordering of positions by their masks.
     *
     * @param a First position.
     * @param b Second position.
     * @return True when `a` sorts before `b`.
     */
    bool position_less(Position a, Position b) {
        return a.mine != b.mine ? a.mine < b.mine : a.theirs < b.theirs;
    }

    /** @brief Checks the initial position. */
    void test_initial_position() {
        const Position p = othello::initial_position();
        CHECK(std::popcount(p.mine) == 2);
        CHECK(std::popcount(p.theirs) == 2);
        CHECK(std::popcount(othello::legal_moves(p)) == 4);
        CHECK(!othello::is_terminal(p));
        CHECK(othello::terminal_value(p) == 0.0);
    }

    /** @brief Checks that a pass only swaps the sides. */
    void test_pass_swaps_sides() {
        const Position p = othello::initial_position();
        const Position q = othello::play(p, othello::kPassAction);
        CHECK(q.mine == p.theirs);
        CHECK(q.theirs == p.mine);
    }

    /** @brief Checks the outcome values of finished games. */
    void test_terminal_values() {
        CHECK(othello::terminal_value({0x1ULL, 0x0ULL}) == 1.0);
        CHECK(othello::terminal_value({0x0ULL, 0x1ULL}) == -1.0);
        CHECK(othello::terminal_value({0x0ULL, 0x0ULL}) == othello::kDrawValue);
        CHECK(othello::terminal_value({0x3ULL, 1ULL << 63}) == 1.0);
    }

    /** @brief Checks the perft counts for 1 to 10 plies. */
    void test_perft() {
        constexpr uint64_t expected[] = {4,    12,    56,     244,     1396,
                                         8200, 55092, 390216, 3005288, 24571284};
        for (int depth = 1; depth <= 10; ++depth) {
            const uint64_t leaves = perft(othello::initial_position(), depth);
            std::printf("perft(%d) = %llu\n", depth, static_cast<unsigned long long>(leaves));
            CHECK(leaves == expected[depth - 1]);
        }
    }

    /** @brief Checks that the hash separates positions and spreads across buckets. */
    void test_hash_separates_positions() {
        std::vector<Position> positions;
        collect(othello::initial_position(), 7, positions);
        std::sort(positions.begin(), positions.end(), position_less);
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());

        std::vector<uint64_t> hashes;
        hashes.reserve(positions.size());
        for (const Position &p : positions) {
            hashes.push_back(othello::position_hash(p));
        }
        std::sort(hashes.begin(), hashes.end());
        const bool distinct = std::adjacent_find(hashes.begin(), hashes.end()) == hashes.end();
        std::printf("%zu distinct positions within 7 plies\n", positions.size());
        CHECK(distinct);

        // The low bits index a hash table, so bucket counts should stay within
        // six standard deviations of a uniform (Poisson) spread.
        constexpr int kBucketBits = 12;
        std::vector<int> buckets(1 << kBucketBits, 0);
        for (const Position &p : positions) {
            ++buckets[othello::position_hash(p) & ((1ULL << kBucketBits) - 1)];
        }
        const double mean = static_cast<double>(positions.size()) / buckets.size();
        const int worst = *std::max_element(buckets.begin(), buckets.end());
        CHECK(worst < mean + 6.0 * std::sqrt(mean));
    }

}  // namespace

/**
 * @brief Runs every check.
 *
 * @return 0 when all checks pass, otherwise 1.
 */
int main() {
    test_initial_position();
    test_pass_swaps_sides();
    test_terminal_values();
    test_perft();
    test_hash_separates_positions();
    if (failures == 0) {
        std::printf("all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
