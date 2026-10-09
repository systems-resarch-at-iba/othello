#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "othello/evaluators.hpp"
#include "othello/mcts.hpp"
#include "othello/net.hpp"
#include "othello/position.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

    /** @brief Board argument converted to a contiguous int8 array. */
    using Board = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;

    /**
     * @brief Checks the player value.
     *
     * @param player Value to check.
     * @throws py::value_error When `player` is not 1 or -1.
     */
    void check_player(int player) {
        if (player != 1 && player != -1) {
            throw py::value_error("player must be 1 or -1");
        }
    }

    /**
     * @brief Reads a 64-square board of -1, 0 and 1 into a position.
     *
     * @param board Board with 64 squares.
     * @param player Side to move, 1 or -1.
     * @return Position seen from `player`.
     * @throws py::value_error When the player, the size or a square value is invalid.
     */
    othello::Position to_position(const Board &board, int player) {
        check_player(player);
        if (board.size() != othello::kBoardSquares) {
            throw py::value_error("board must have 64 squares");
        }
        const int8_t *squares = board.data();
        uint64_t mine = 0;
        uint64_t theirs = 0;
        for (int i = 0; i < othello::kBoardSquares; ++i) {
            const uint64_t bit = 1ULL << i;
            if (squares[i] == player) {
                mine |= bit;
            } else if (squares[i] == -player) {
                theirs |= bit;
            } else if (squares[i] != 0) {
                throw py::value_error("board squares must be -1, 0 or 1");
            }
        }
        return {mine, theirs};
    }

    /**
     * @brief Writes a position into an 8x8 board of -1, 0 and 1.
     *
     * @param p Position to write.
     * @param player Color value of the side that owns `p.mine`, 1 or -1.
     * @return The 8x8 board.
     */
    py::array_t<int8_t> to_board(othello::Position p, int player) {
        py::array_t<int8_t> board({8, 8});
        int8_t *squares = board.mutable_data();
        for (int i = 0; i < othello::kBoardSquares; ++i) {
            const uint64_t bit = 1ULL << i;
            squares[i] = (p.mine & bit) ? player : (p.theirs & bit) ? -player : 0;
        }
        return board;
    }

    /**
     * @brief Initial 8x8 board, with black (-1) to move.
     *
     * @return The starting board.
     */
    py::array_t<int8_t> get_init_board() {
        return to_board(othello::initial_position(), -1);
    }

    /**
     * @brief Legal moves of a player.
     *
     * @param board Board with 64 squares.
     * @param player Side to move, 1 or -1.
     * @return 65 flags: squares 0 to 63 that are legal, and index 64 set when the player must pass.
     */
    py::array_t<uint8_t> get_valid_moves(const Board &board, int player) {
        const uint64_t moves = othello::legal_moves(to_position(board, player));
        py::array_t<uint8_t> valid(othello::kPassAction + 1);
        uint8_t *flags = valid.mutable_data();
        for (int i = 0; i < othello::kBoardSquares; ++i) {
            flags[i] = (moves >> i) & 1;
        }
        flags[othello::kPassAction] = moves == 0;
        return valid;
    }

    /**
     * @brief Whether a player has a legal move.
     *
     * @param board Board with 64 squares.
     * @param player Side to move, 1 or -1.
     * @return True when a legal move exists.
     */
    bool has_valid_moves(const Board &board, int player) {
        return othello::legal_moves(to_position(board, player)) != 0;
    }

    /**
     * @brief Plays an action.
     *
     * @param board Board with 64 squares.
     * @param player Side to move, 1 or -1.
     * @param action Legal square index, or 64 to pass.
     * @return The new 8x8 board and the next player.
     * @throws py::value_error When the action is out of range or not legal.
     */
    py::tuple get_next_state(const Board &board, int player, int action) {
        const othello::Position p = to_position(board, player);
        if (action < 0 || action > othello::kPassAction) {
            throw py::value_error("action must be in [0, 64]");
        }
        if (action != othello::kPassAction && !((othello::legal_moves(p) >> action) & 1)) {
            throw py::value_error("action is not a legal move");
        }
        // play() returns the position from the next mover's side, so the board is
        // written from the opponent's point of view: -player owns the new `mine`.
        return py::make_tuple(to_board(othello::play(p, action), -player), -player);
    }

    /**
     * @brief Outcome of the game for a player.
     *
     * @param board Board with 64 squares.
     * @param player Side to evaluate, 1 or -1.
     * @return 0 while the game continues, otherwise 1 (win), -1 (loss) or 1e-4 (draw).
     */
    double get_game_ended(const Board &board, int player) {
        return othello::terminal_value(to_position(board, player));
    }

    /** @brief Canonical board argument converted to a contiguous float32 array. */
    using FloatBoard = py::array_t<float, py::array::c_style | py::array::forcecast>;

    /** @brief A network together with the scratch buffers of its forward passes. */
    class PyNetwork {
    public:
        /**
         * @brief Loads a network.
         *
         * @param path Path of the weight file.
         * @throws std::runtime_error When the weight file cannot be used.
         */
        explicit PyNetwork(const std::string &path)
            : network_(path), scratch_(std::make_unique<othello::Scratch>()) {
        }

        /**
         * @brief Log-probabilities of all 65 actions and the value.
         *
         * @param board 64 squares seen from the side to move: 1 own, -1 opponent.
         * @param reference Whether to use the plain-loop forward pass.
         * @return A tuple of the 65 log-probabilities and the value.
         * @throws py::value_error When the board does not have 64 squares.
         */
        py::tuple forward(const FloatBoard &board, bool reference) {
            othello::Evaluation e = evaluate(board, reference);
            py::array_t<float> log_policy(othello::kActions);
            othello::log_softmax(e.logits, log_policy.mutable_data());
            return py::make_tuple(log_policy, e.value);
        }

        /**
         * @brief Probabilities over the legal actions only, and the value.
         *
         * @param board 64 squares seen from the side to move: 1 own, -1 opponent.
         * @param valid 65 flags: legal squares, then the pass flag.
         * @return A tuple of the 65 probabilities and the value.
         * @throws py::value_error When `valid` does not have 65 flags or the board does not
         *         have 64 squares.
         */
        py::tuple masked_policy(const FloatBoard &board, const py::array_t<uint8_t> &valid) {
            if (valid.size() != othello::kActions) {
                throw py::value_error("valid must have 65 flags");
            }
            uint64_t legal = 0;
            for (int i = 0; i < othello::kSquares; ++i) {
                legal |= static_cast<uint64_t>(valid.data()[i] != 0) << i;
            }
            const othello::Evaluation e = evaluate(board, false);
            py::array_t<float> probs(othello::kActions);
            othello::masked_softmax(e.logits, legal, probs.mutable_data());
            return py::make_tuple(probs, e.value);
        }

        /**
         * @brief Name of the instruction set the network uses.
         *
         * @return "avx2" or "scalar".
         */
        const char *isa() const {
            return network_.isa();
        }

    private:
        /**
         * @brief Runs the network on a board.
         *
         * @param board 64 squares seen from the side to move.
         * @param reference Whether to use the plain-loop forward pass.
         * @return The network output.
         * @throws py::value_error When the board does not have 64 squares.
         */
        othello::Evaluation evaluate(const FloatBoard &board, bool reference) {
            if (board.size() != othello::kSquares) {
                throw py::value_error("board must have 64 squares");
            }
            othello::Evaluation e;
            if (reference) {
                network_.forward_reference(*scratch_, board.data(), e);
            } else {
                network_.forward(*scratch_, board.data(), e);
            }
            return e;
        }

        othello::Network network_;                   ///< The loaded network.
        std::unique_ptr<othello::Scratch> scratch_;  ///< Buffers of the forward passes.
    };

    /** @brief Largest simulation count of one search. */
    constexpr int kMaxSimulations = 1 << 24;

    /**
     * @brief A search tree behind a lock, with arguments converted from Python.
     *
     * Searches run without the GIL and one at a time.
     *
     * @tparam Evaluator Policy and value source of the search.
     */
    template <typename Evaluator> class PySearch {
    public:
        /**
         * @brief Creates an empty search tree.
         *
         * @param evaluator Policy and value source.
         * @param max_nodes Node count above which the tree is cleared before a search.
         */
        PySearch(Evaluator evaluator, size_t max_nodes) : mcts_(std::move(evaluator), max_nodes) {
        }

        /**
         * @brief Runs a search and returns the action distribution.
         *
         * @param board Board with 64 squares.
         * @param player Side to move, 1 or -1.
         * @param num_sims Number of simulations.
         * @param temperature 0 for a one-hot most visited action, otherwise the
         *        visit counts raised to 1 / temperature, normalized.
         * @param c_puct Exploration constant.
         * @param cpuct_type "static", "increment" or "decrement".
         * @param seed Seed of the tie-break at temperature 0.
         * @param batch_size Simulations whose leaves the network evaluates together, from 1 to
         *        16. One gives the sequential search.
         * @return 65 probabilities, one per action.
         * @throws py::value_error When an argument is out of range or the board is invalid.
         */
        py::array_t<double> get_action_probs(const Board &board, int player, int num_sims,
                                             double temperature, double c_puct,
                                             const std::string &cpuct_type, uint64_t seed,
                                             int batch_size) {
            if (num_sims < 1 || num_sims > kMaxSimulations) {
                throw py::value_error("num_sims must be in [1, 16777216]");
            }
            if (temperature < 0.0 || !(c_puct > 0.0)) {
                throw py::value_error("temperature must be non-negative and c_puct positive");
            }
            othello::SearchConfig config;
            config.c_puct = c_puct;
            config.cpuct_type = parse_cpuct_type(cpuct_type);
            if (batch_size < 1 || batch_size > othello::kMaxSimBatch) {
                throw py::value_error("batch_size must be in [1, 16]");
            }
            config.batch_size = batch_size;
            const othello::Position root = to_position(board, player);
            py::array_t<double> probs(othello::kActions);
            double *out = probs.mutable_data();
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(mutex_);
                mcts_.run(root, num_sims, config);
                uint32_t counts[othello::kActions];
                mcts_.visit_counts(root, counts);
                othello::Rng rng(seed);
                Mcts::action_probs(counts, temperature, rng, out);
            }
            return probs;
        }

        /**
         * @brief Copies the search tree below a position.
         *
         * @param board Board with 64 squares.
         * @param player Side to move, 1 or -1.
         * @param max_depth Deepest level whose nodes are listed.
         * @param min_visits Edges with fewer visits are listed but not followed.
         * @return A list of node dicts in breadth-first order. Each has `id`,
         *         `mine`, `theirs` (bitboards of the side to move and of the
         *         opponent), `visits`, `terminal`, `terminal_value`, `depth` and
         *         `edges`, a list of dicts with `action`, `prior`, `value`,
         *         `visits` and `child` (a node id or -1).
         */
        py::list get_tree(const Board &board, int player, int max_depth, uint32_t min_visits) {
            const othello::Position root = to_position(board, player);
            std::lock_guard<std::mutex> lock(mutex_);
            py::list nodes;
            for (const othello::ExportedNode &node :
                 mcts_.export_tree(root, max_depth, min_visits)) {
                py::list edges;
                for (const othello::ExportedEdge &edge : node.edges) {
                    edges.append(py::dict("action"_a = edge.action, "prior"_a = edge.prior,
                                          "value"_a = edge.value, "visits"_a = edge.visits,
                                          "child"_a = edge.child));
                }
                nodes.append(py::dict("id"_a = node.id, "mine"_a = node.pos.mine,
                                      "theirs"_a = node.pos.theirs, "visits"_a = node.visits,
                                      "terminal"_a = node.terminal,
                                      "terminal_value"_a = node.terminal_value,
                                      "depth"_a = node.depth, "edges"_a = edges));
            }
            return nodes;
        }

        /** @brief Drops the search tree. */
        void clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            mcts_.clear();
        }

        /**
         * @brief Number of positions in the search tree.
         *
         * @return The node count.
         */
        size_t node_count() {
            std::lock_guard<std::mutex> lock(mutex_);
            return mcts_.node_count();
        }

    private:
        /** @brief The search tree type. */
        using Mcts = othello::Mcts<Evaluator>;

        /**
         * @brief Converts the name of an exploration scaling.
         *
         * @param name "static", "increment" or "decrement".
         * @return The matching scaling.
         * @throws py::value_error When the name is not one of the three.
         */
        static othello::CpuctType parse_cpuct_type(const std::string &name) {
            if (name == "static") {
                return othello::CpuctType::Static;
            }
            if (name == "increment") {
                return othello::CpuctType::Increment;
            }
            if (name == "decrement") {
                return othello::CpuctType::Decrement;
            }
            throw py::value_error("cpuct_type must be 'static', 'increment' or 'decrement'");
        }

        Mcts mcts_;         ///< The search tree.
        std::mutex mutex_;  ///< Serializes searches and tree access.
    };

    /** @brief Holds a network so that it outlives the evaluator that points at it. */
    struct NetworkOwner {
        /**
         * @brief Loads a network.
         *
         * @param path Path of the weight file.
         * @throws std::runtime_error When the weight file cannot be used.
         */
        explicit NetworkOwner(const std::string &path)
            : network(std::make_unique<othello::Network>(path)) {
        }

        std::unique_ptr<othello::Network> network;  ///< The loaded network.
    };

    /** @brief Search guided by a network. */
    class PyEngine : private NetworkOwner, public PySearch<othello::NetworkEvaluator> {
    public:
        /**
         * @brief Loads a network and creates an empty search tree.
         *
         * @param path Path of the weight file.
         * @param max_nodes Node count above which the tree is cleared before a search.
         * @throws std::runtime_error When the weight file cannot be used.
         */
        PyEngine(const std::string &path, size_t max_nodes)
            : NetworkOwner(path),
              PySearch<othello::NetworkEvaluator>(othello::NetworkEvaluator(*network), max_nodes) {
        }

        /**
         * @brief Name of the instruction set the network uses.
         *
         * @return "avx2" or "scalar".
         */
        const char *isa() const {
            return network->isa();
        }
    };

    /** @brief Search guided by the pseudo-random StubEvaluator. */
    class PyStubEngine : public PySearch<othello::StubEvaluator> {
    public:
        /**
         * @brief Creates an empty search tree.
         *
         * @param max_nodes Node count above which the tree is cleared before a search.
         */
        explicit PyStubEngine(size_t max_nodes)
            : PySearch<othello::StubEvaluator>(othello::StubEvaluator{}, max_nodes) {
        }
    };

    /**
     * @brief Registers the search methods on a Python class.
     *
     * @tparam Class The pybind11 class type.
     * @tparam Evaluator Policy and value source of the searches.
     * @param cls Class that receives the methods.
     */
    template <typename Class, typename Evaluator> void def_search_methods(Class &cls) {
        using Search = PySearch<Evaluator>;
        cls.def("get_action_probs", &Search::get_action_probs, py::arg("board"), py::arg("player"),
                py::arg("num_sims"), py::arg("temperature") = 1.0, py::arg("c_puct") = 1.5,
                py::arg("cpuct_type") = "static", py::arg("seed") = 0, py::arg("batch_size") = 1,
                "Runs a search for the player to move and returns 65 action probabilities.")
            .def("get_tree", &Search::get_tree, py::arg("board"), py::arg("player"),
                 py::arg("max_depth") = 3, py::arg("min_visits") = 1,
                 "Copies the search tree below a position as a list of node dicts.")
            .def("clear", &Search::clear, "Drops the search tree.")
            .def_property_readonly("node_count", &Search::node_count,
                                   "Number of positions in the search tree.");
    }

}  // namespace

PYBIND11_MODULE(othello_cpp, m) {
    m.doc() = "Othello rules on 64-bit bitboards.";
    m.def("get_init_board", &get_init_board,
          "Initial 8x8 board (black is -1, white is 1, black moves first).");
    m.def("get_valid_moves", &get_valid_moves, py::arg("board"), py::arg("player"),
          "uint8 array of 65 flags: squares 0 to 63 that are legal, and index 64 set when the "
          "player must pass.");
    m.def("has_valid_moves", &has_valid_moves, py::arg("board"), py::arg("player"),
          "True when the player has a legal move.");
    m.def("get_next_state", &get_next_state, py::arg("board"), py::arg("player"), py::arg("action"),
          "Board and next player after playing a legal square or passing with action 64.");
    m.def(
        "get_game_ended", &get_game_ended, py::arg("board"), py::arg("player"),
        "0 while the game continues, otherwise 1 (win), -1 (loss) or 1e-4 (draw) for the player.");

    py::class_<PyNetwork>(m, "Network", "Policy and value network loaded from a weight file.")
        .def(py::init<const std::string &>(), py::arg("path"))
        .def("forward", &PyNetwork::forward, py::arg("board"), py::arg("reference") = false,
             "Log-probabilities of the 65 actions and the value for a canonical board.")
        .def("masked_policy", &PyNetwork::masked_policy, py::arg("board"), py::arg("valid"),
             "Probabilities over the legal actions and the value for a canonical board.")
        .def_property_readonly("isa", &PyNetwork::isa,
                               "Instruction set used by forward: 'avx2' or 'scalar'.");

    /** Node count above which a search tree clears itself. */
    constexpr size_t kDefaultMaxNodes = 500000;
    auto engine =
        py::class_<PyEngine>(m, "Engine", "Monte Carlo tree search guided by a network.")
            .def(py::init<const std::string &, size_t>(), py::arg("path"),
                 py::arg("max_nodes") = kDefaultMaxNodes)
            .def_property_readonly("isa", &PyEngine::isa,
                                   "Instruction set the network uses: 'avx2' or 'scalar'.");
    def_search_methods<decltype(engine), othello::NetworkEvaluator>(engine);

    auto stub =
        py::class_<PyStubEngine>(
            m, "StubEngine", "Search guided by a deterministic pseudo-random evaluator, for tests.")
            .def(py::init<size_t>(), py::arg("max_nodes") = kDefaultMaxNodes);
    def_search_methods<decltype(stub), othello::StubEvaluator>(stub);
}
