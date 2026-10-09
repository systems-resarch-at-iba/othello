#pragma once

#include <bit>
#include <cstdint>

namespace othello {

    /** @brief Number of squares on the board. */
    constexpr int kBoardSquares = 64;
    /** @brief Action index of a pass, one past the last square. */
    constexpr int kPassAction = 64;
    /** @brief Outcome value of a drawn game. */
    constexpr double kDrawValue = 1e-4;

    /**
     * @brief A board seen from the side to move.
     *
     * Bit i of each mask is the square at row i / 8, column i % 8. The masks do
     * not overlap.
     */
    struct Position {
        uint64_t mine;    ///< Discs of the side to move.
        uint64_t theirs;  ///< Discs of the opponent.

        /**
         * @brief Whether two positions hold the same discs.
         *
         * @param other Position to compare with.
         * @return True when both masks are equal.
         */
        constexpr bool operator==(const Position &other) const = default;
    };

    /**
     * @brief Initial position, with black (the first player) to move.
     *
     * @return The starting position.
     */
    constexpr Position initial_position() noexcept {
        return {0x0000000810000000ULL, 0x0000001008000000ULL};
    }

    /**
     * @brief The same board seen from the other side.
     *
     * @param p Position to flip.
     * @return The position with both masks exchanged.
     */
    constexpr Position swap_sides(Position p) noexcept {
        return {p.theirs, p.mine};
    }

    namespace detail {

        /** @brief Mask of every square outside file A, the leftmost column. */
        constexpr uint64_t kNotFileA = 0xFEFEFEFEFEFEFEFEULL;
        /** @brief Mask of every square outside file H, the rightmost column. */
        constexpr uint64_t kNotFileH = 0x7F7F7F7F7F7F7F7FULL;
        /** @brief Mask of every square. */
        constexpr uint64_t kAllSquares = 0xFFFFFFFFFFFFFFFFULL;

        /**
         * @brief One of the eight board directions.
         *
         * @tparam Shift Bit-index step. Positive steps left-shift and negative steps
         *         right-shift.
         * @tparam Mask Removes the bits that wrapped around a board edge.
         */
        template <int Shift, uint64_t Mask> struct Direction {
            /**
             * @brief Moves every disc one square in this direction.
             *
             * @param b Mask of squares to move.
             * @return The moved mask, without the bits that wrapped around an edge.
             */
            static constexpr uint64_t step(uint64_t b) noexcept {
                if constexpr (Shift > 0) {
                    return (b << Shift) & Mask;
                } else {
                    return (b >> -Shift) & Mask;
                }
            }
        };

        using South = Direction<8, kAllSquares>;
        using North = Direction<-8, kAllSquares>;
        using East = Direction<1, kNotFileA>;
        using West = Direction<-1, kNotFileH>;
        using SouthEast = Direction<9, kNotFileA>;
        using NorthWest = Direction<-9, kNotFileH>;
        using SouthWest = Direction<7, kNotFileH>;
        using NorthEast = Direction<-7, kNotFileA>;

        /**
         * @brief Empty squares that close a run of opponent discs in direction D.
         *
         * A run starts next to a disc of the side to move. It holds at most six
         * discs on an 8-square line, so six fill steps are enough.
         *
         * @tparam D Direction to search.
         * @param mine Discs of the side to move.
         * @param theirs Discs of the opponent.
         * @param empty Empty squares.
         * @return Mask of the squares where a disc can be placed along D.
         */
        template <typename D>
        constexpr uint64_t moves_along(uint64_t mine, uint64_t theirs, uint64_t empty) noexcept {
            uint64_t run = D::step(mine) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            return D::step(run) & empty;
        }

        /**
         * @brief Discs flipped in direction D by a disc placed on `move`.
         *
         * These are the run of opponent discs next to `move`, when a disc of the side
         * to move closes the run.
         *
         * @tparam D Direction to search.
         * @param move Single-bit mask of the placed disc.
         * @param mine Discs of the side to move.
         * @param theirs Discs of the opponent.
         * @return Mask of the flipped discs.
         */
        template <typename D>
        constexpr uint64_t flips_along(uint64_t move, uint64_t mine, uint64_t theirs) noexcept {
            uint64_t run = D::step(move) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            run |= D::step(run) & theirs;
            const uint64_t closed = (D::step(run) & mine) != 0;
            return run & (0 - closed);
        }

        /**
         * @brief 64-bit finalizer that spreads input bits across the result.
         *
         * @param x Value to mix.
         * @return Mixed value.
         */
        constexpr uint64_t mix64(uint64_t x) noexcept {
            x ^= x >> 30;
            x *= 0xBF58476D1CE4E5B9ULL;
            x ^= x >> 27;
            x *= 0x94D049BB133111EBULL;
            x ^= x >> 31;
            return x;
        }

    }  // namespace detail

    /**
     * @brief Squares where the side to move can legally play.
     *
     * @param p Position to inspect.
     * @return Bit mask of the legal squares.
     */
    constexpr uint64_t legal_moves(Position p) noexcept {
        using namespace detail;
        const uint64_t empty = ~(p.mine | p.theirs);
        return moves_along<South>(p.mine, p.theirs, empty) |
               moves_along<North>(p.mine, p.theirs, empty) |
               moves_along<East>(p.mine, p.theirs, empty) |
               moves_along<West>(p.mine, p.theirs, empty) |
               moves_along<SouthEast>(p.mine, p.theirs, empty) |
               moves_along<NorthWest>(p.mine, p.theirs, empty) |
               moves_along<SouthWest>(p.mine, p.theirs, empty) |
               moves_along<NorthEast>(p.mine, p.theirs, empty);
    }

    /**
     * @brief Discs flipped when the side to move plays on a square.
     *
     * @param p Position before the move.
     * @param square Bit index of the played square.
     * @return Bit mask of the flipped discs.
     */
    constexpr uint64_t flipped_discs(Position p, int square) noexcept {
        using namespace detail;
        const uint64_t move = 1ULL << square;
        return flips_along<South>(move, p.mine, p.theirs) |
               flips_along<North>(move, p.mine, p.theirs) |
               flips_along<East>(move, p.mine, p.theirs) |
               flips_along<West>(move, p.mine, p.theirs) |
               flips_along<SouthEast>(move, p.mine, p.theirs) |
               flips_along<NorthWest>(move, p.mine, p.theirs) |
               flips_along<SouthWest>(move, p.mine, p.theirs) |
               flips_along<NorthEast>(move, p.mine, p.theirs);
    }

    /**
     * @brief Position after an action, seen from the new side to move.
     *
     * @param p Position before the action.
     * @param action Square index of a legal move, or kPassAction, which only
     *        swaps the sides.
     * @return Position after the action.
     */
    constexpr Position play(Position p, int action) noexcept {
        if (action == kPassAction) {
            return swap_sides(p);
        }
        const uint64_t flips = flipped_discs(p, action);
        return {p.theirs & ~flips, p.mine | (1ULL << action) | flips};
    }

    /**
     * @brief Whether the game is over.
     *
     * @param p Position to inspect.
     * @return True when neither side has a legal move.
     */
    constexpr bool is_terminal(Position p) noexcept {
        return legal_moves(p) == 0 && legal_moves(swap_sides(p)) == 0;
    }

    /**
     * @brief Outcome of a finished game for the side to move.
     *
     * @param p Position to inspect.
     * @return 1 for a win, -1 for a loss, kDrawValue for a draw and 0 while the
     *         game continues.
     */
    constexpr double terminal_value(Position p) noexcept {
        if (!is_terminal(p)) {
            return 0.0;
        }
        const int diff = std::popcount(p.mine) - std::popcount(p.theirs);
        return diff > 0 ? 1.0 : diff < 0 ? -1.0 : kDrawValue;
    }

    /**
     * @brief 64-bit hash of a position.
     *
     * Distinct positions should be told apart by comparing the two masks.
     *
     * @param p Position to hash.
     * @return Hash value.
     */
    constexpr uint64_t position_hash(Position p) noexcept {
        return detail::mix64(p.mine ^ detail::mix64(p.theirs));
    }

}  // namespace othello
