/**
 * @file bench_mcts.cpp
 * @brief Times full searches and prints JSON.
 *
 * Usage: bench_mcts --boards FILE --sims N --searches S --reps R [--weights FILE] [--keep-tree]
 * [--batch B]
 *
 * With --weights the search uses the network. Without it the search uses the
 * cheap deterministic StubEvaluator, so that the time per simulation is
 * dominated by the tree. Each repetition runs `searches` searches, cycling
 * through the boards, from an empty tree unless --keep-tree is given, and
 * reports the mean nanoseconds per simulation.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "othello/evaluators.hpp"
#include "othello/mcts.hpp"

namespace {

    /**
     * @brief Reads canonical boards of float32 values, 64 per board, as positions.
     *
     * @param path File to read.
     * @return One position per board.
     * @throws std::runtime_error When the file cannot be opened.
     */
    std::vector<othello::Position> read_positions(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open " + path);
        }
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
        std::vector<float> values(bytes.size() / sizeof(float));
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
        std::vector<othello::Position> positions;
        for (size_t b = 0; b + othello::kSquares <= values.size(); b += othello::kSquares) {
            othello::Position p{0, 0};
            for (int i = 0; i < othello::kSquares; ++i) {
                p.mine |= static_cast<uint64_t>(values[b + i] == 1.0f) << i;
                p.theirs |= static_cast<uint64_t>(values[b + i] == -1.0f) << i;
            }
            positions.push_back(p);
        }
        return positions;
    }

    /**
     * @brief Value of a command-line option.
     *
     * @param argc Argument count.
     * @param argv Arguments.
     * @param name Option name, such as "--sims".
     * @return The argument after the option, or an empty string when absent.
     */
    std::string option(int argc, char **argv, const char *name) {
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], name) == 0) {
                return argv[i + 1];
            }
        }
        return "";
    }

    /**
     * @brief Whether a flag is present.
     *
     * @param argc Argument count.
     * @param argv Arguments.
     * @param name Flag name, such as "--keep-tree".
     * @return True when the flag appears among the arguments.
     */
    bool flag(int argc, char **argv, const char *name) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], name) == 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Runs the repetitions and prints the JSON line.
     *
     * @tparam Evaluator Policy and value source of the search.
     * @param evaluator Policy and value source.
     * @param positions Positions to search, used in turn.
     * @param sims Simulations per search.
     * @param searches Searches per repetition.
     * @param reps Repetitions.
     * @param keep_tree Whether the tree stays between searches.
     * @param batch Simulations whose leaves are evaluated together.
     */
    template <typename Evaluator>
    void run(Evaluator evaluator, const std::vector<othello::Position> &positions, int sims,
             int searches, int reps, bool keep_tree, int batch) {
        othello::Mcts<Evaluator> search(std::move(evaluator), 500000);
        othello::SearchConfig config;
        config.batch_size = batch;
        uint64_t sink = 0;
        const auto one_search = [&](int i) {
            if (!keep_tree) {
                search.clear();
            }
            search.run(positions[i % positions.size()], sims, config);
            sink += search.node_count();
        };
        for (int i = 0; i < 3; ++i) {
            one_search(i);
        }
        std::printf("{\"sims\": %d, \"batch\": %d, \"searches\": %d, \"ns_per_sim\": [", sims,
                    batch, searches);
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < searches; ++i) {
                one_search(i);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                              (static_cast<double>(searches) * sims);
            std::printf("%s%.1f", r == 0 ? "" : ", ", ns);
        }
        std::printf("], \"sink\": %llu}\n", static_cast<unsigned long long>(sink));
    }

}  // namespace

/**
 * @brief Times full searches and prints JSON.
 *
 * @param argc Argument count.
 * @param argv Arguments.
 * @return 0 on success, 2 for invalid arguments and 1 for an error.
 */
int main(int argc, char **argv) {
    try {
        const std::string boards = option(argc, argv, "--boards");
        const std::string weights = option(argc, argv, "--weights");
        const int sims = std::atoi(option(argc, argv, "--sims").c_str());
        const int searches = std::atoi(option(argc, argv, "--searches").c_str());
        const int reps = std::atoi(option(argc, argv, "--reps").c_str());
        if (boards.empty() || sims <= 0 || searches <= 0 || reps <= 0) {
            std::fprintf(stderr, "usage: bench_mcts --boards F --sims N --searches S --reps R "
                                 "[--weights F] [--keep-tree]\n");
            return 2;
        }
        const std::vector<othello::Position> positions = read_positions(boards);
        const bool keep_tree = flag(argc, argv, "--keep-tree");
        const int batch = std::max(1, std::atoi(option(argc, argv, "--batch").c_str()));
        if (weights.empty()) {
            run(othello::StubEvaluator{}, positions, sims, searches, reps, keep_tree, batch);
        } else {
            const othello::Network network(weights);
            run(othello::NetworkEvaluator(network), positions, sims, searches, reps, keep_tree,
                batch);
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
