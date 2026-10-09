#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>

#include "othello/evaluators.hpp"
#include "othello/mcts.hpp"

namespace {

    /**
     * @brief Position type used by the tests.
     */
    using othello::Position;
    /**
     * @brief Search over the deterministic stub evaluator.
     */
    using Search = othello::Mcts<othello::StubEvaluator>;

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
     * @brief Sum of the root visit counts.
     *
     * @param search Search to inspect.
     * @param root Searched position.
     * @return The total number of visits of the root's actions.
     */
    uint32_t total_visits(const Search &search, Position root) {
        uint32_t counts[othello::kActions];
        search.visit_counts(root, counts);
        return std::accumulate(counts, counts + othello::kActions, 0u);
    }

    /** @brief A fresh search adds one visit less than its simulation count, and only legal actions
     * are visited. */
    void test_visit_total_and_legality() {
        Search search(othello::StubEvaluator{}, 100000);
        const Position root = othello::initial_position();
        search.run(root, 500, {});
        CHECK(total_visits(search, root) == 499);
        uint32_t counts[othello::kActions];
        search.visit_counts(root, counts);
        const uint64_t legal = othello::legal_moves(root);
        for (int a = 0; a < othello::kActions; ++a) {
            if (a >= othello::kSquares || ((legal >> a) & 1) == 0) {
                CHECK(counts[a] == 0);
            }
        }
    }

    /** @brief The same search gives the same counts, and a second search keeps adding to the tree.
     */
    void test_determinism_and_persistence() {
        const Position root = othello::initial_position();
        Search a(othello::StubEvaluator{}, 100000);
        Search b(othello::StubEvaluator{}, 100000);
        a.run(root, 300, {});
        b.run(root, 300, {});
        uint32_t ca[othello::kActions];
        uint32_t cb[othello::kActions];
        a.visit_counts(root, ca);
        b.visit_counts(root, cb);
        for (int i = 0; i < othello::kActions; ++i) {
            CHECK(ca[i] == cb[i]);
        }
        a.run(root, 100, {});
        CHECK(total_visits(a, root) == 299 + 100);
    }

    /** @brief A root with no legal move searches through the pass action. */
    void test_pass_root() {
        // A disc of the side to move sits next to an opponent disc on the board edge with no empty
        // square behind it.
        const Position root{0x2ULL, 0x1ULL};
        CHECK(othello::legal_moves(root) == 0);
        CHECK(!othello::is_terminal(root));
        Search search(othello::StubEvaluator{}, 1000);
        search.run(root, 20, {});
        uint32_t counts[othello::kActions];
        search.visit_counts(root, counts);
        CHECK(counts[othello::kPassAction] == 19);
    }

    /** @brief A finished game gives no visits. */
    void test_terminal_root() {
        const Position root{0x1ULL, 0x0ULL};
        Search search(othello::StubEvaluator{}, 1000);
        search.run(root, 10, {});
        CHECK(total_visits(search, root) == 0);
    }

    /** @brief The tree is cleared when the node limit would be exceeded, and the search still
     * works. */
    void test_node_limit() {
        Search search(othello::StubEvaluator{}, 1000);
        const Position root = othello::initial_position();
        search.run(root, 600, {});
        search.run(root, 600, {});
        CHECK(search.node_count() <= 1000);
        CHECK(total_visits(search, root) == 599);
        search.run(root, 5000, {});
        CHECK(total_visits(search, root) == 4999);
    }

    /** @brief Temperature 0 gives a one-hot distribution on a most visited action, temperature 1
     * the normalized counts. */
    void test_action_probs() {
        uint32_t counts[othello::kActions] = {};
        counts[3] = 10;
        counts[7] = 30;
        counts[9] = 30;
        othello::Rng rng(1);
        double probs[othello::kActions];
        int seen7 = 0;
        for (int i = 0; i < 200; ++i) {
            Search::action_probs(counts, 0.0, rng, probs);
            CHECK(std::accumulate(probs, probs + othello::kActions, 0.0) == 1.0);
            CHECK(probs[7] == 1.0 || probs[9] == 1.0);
            seen7 += probs[7] == 1.0;
        }
        CHECK(seen7 > 50 && seen7 < 150);
        Search::action_probs(counts, 1.0, rng, probs);
        CHECK(std::abs(probs[7] - 30.0 / 70.0) < 1e-6);
    }

    /** @brief Batched searches add one visit per simulation, keep every node consistent and leave
     * no waiting leaf. */
    void test_batched_search() {
        for (int batch : {2, 3, 4, 8, 16}) {
            Search search(othello::StubEvaluator{}, 100000);
            const Position root = othello::initial_position();
            othello::SearchConfig config;
            config.batch_size = batch;
            search.run(root, 1000, config);
            CHECK(total_visits(search, root) == 999);
            search.run(root, 500, config);
            CHECK(total_visits(search, root) == 1499);
            for (const auto &node : search.export_tree(root, 100, 0)) {
                if (node.terminal) {
                    continue;
                }
                uint32_t edge_visits = 0;
                for (const auto &edge : node.edges) {
                    edge_visits += edge.visits;
                    CHECK(std::isfinite(edge.value) && edge.value >= -1.001f &&
                          edge.value <= 1.001f);
                }
                CHECK(edge_visits == node.visits);
            }
        }
    }

    /** @brief A batch size outside the supported range is rejected. */
    void test_bad_batch_size() {
        Search search(othello::StubEvaluator{}, 1000);
        othello::SearchConfig config;
        config.batch_size = 0;
        bool threw = false;
        try {
            search.run(othello::initial_position(), 10, config);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    /** @brief Clearing leaves an empty tree that searches identically to a new one. */
    void test_clear_restores_empty_tree() {
        const Position root = othello::initial_position();
        Search fresh(othello::StubEvaluator{}, 100000);
        fresh.run(root, 400, {});
        uint32_t expected[othello::kActions];
        fresh.visit_counts(root, expected);
        Search reused(othello::StubEvaluator{}, 100000);
        reused.run(root, 700, {});
        reused.clear();
        CHECK(reused.node_count() == 0);
        reused.run(root, 400, {});
        uint32_t actual[othello::kActions];
        reused.visit_counts(root, actual);
        for (int i = 0; i < othello::kActions; ++i) {
            CHECK(expected[i] == actual[i]);
        }
    }

}  // namespace

/**
 * @brief Runs every check.
 *
 * @return 0 when all checks pass, otherwise 1.
 */
int main() {
    test_visit_total_and_legality();
    test_determinism_and_persistence();
    test_pass_root();
    test_terminal_root();
    test_node_limit();
    test_action_probs();
    test_batched_search();
    test_bad_batch_size();
    test_clear_restores_empty_tree();
    if (failures == 0) {
        std::printf("all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
