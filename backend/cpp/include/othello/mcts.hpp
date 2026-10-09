#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "othello/position.hpp"
#include "othello/rng.hpp"

namespace othello {

    /** @brief How the exploration constant changes with the visit count of a node. */
    enum class CpuctType {
        Static,     ///< The constant does not change.
        Increment,  ///< The constant grows with the visits of a node, from 0.5 towards c_puct.
        Decrement   ///< The constant shrinks with the visits of a node, from c_puct towards 0.5.
    };

    /** Most simulations one batch can hold. */
    constexpr int kMaxSimBatch = 16;

    /** @brief Exploration settings of one search. */
    struct SearchConfig {
        double c_puct = 1.5;                       ///< Exploration constant.
        CpuctType cpuct_type = CpuctType::Static;  ///< Scaling of the constant with node visits.
        /**
         * Simulations whose leaves are evaluated together, from 1 to kMaxSimBatch. One
         * reproduces the sequential search exactly. Larger batches mark each selected path
         * with a virtual loss until its leaf is evaluated, so the search differs slightly.
         */
        int batch_size = 1;
    };

    /** @brief An edge of an exported node. */
    struct ExportedEdge {
        int action;       ///< Square index, or kPassAction.
        float prior;      ///< Policy probability of the action.
        float value;      ///< Mean outcome for the node's side to move.
        uint32_t visits;  ///< Simulations through the edge.
        int64_t child;    ///< Node id of the resulting position, or -1 when never followed.
    };

    /** @brief A node of an exported tree. */
    struct ExportedNode {
        uint32_t id;                      ///< Index of the node in the tree.
        Position pos;                     ///< Position of the node, seen from its side to move.
        uint32_t visits;                  ///< Selections through the node.
        bool terminal;                    ///< Whether the game is over.
        float terminal_value;             ///< Outcome for the side to move when `terminal`.
        int depth;                        ///< Fewest moves from the root.
        std::vector<ExportedEdge> edges;  ///< Legal actions of the node, empty when `terminal`.
    };

    /**
     * @brief Monte Carlo tree search with a transposition table.
     *
     * Nodes are keyed by position, so transpositions share their statistics.
     * The tree persists between searches until clear() or until the node
     * limit would be exceeded.
     *
     * @tparam Evaluator Provides `void operator()(Position p, uint64_t legal, float *probs,
     *         float &value)`, which writes kActions policy probabilities that are zero outside
     *         the legal actions and sum to 1, and the value of `p` for its side to move. It also
     *         provides `void evaluate_batch(int count, const Position *positions, const uint64_t
     *         *legal, float *probs, float *values)`, which does the same for several positions,
     *         with `probs` holding `count` rows of kActions entries.
     */
    template <typename Evaluator> class Mcts {
    public:
        /**
         * @brief Creates an empty tree.
         *
         * @param evaluator Policy and value source.
         * @param max_nodes Node count above which the tree is cleared before a search.
         * @throws std::invalid_argument When `max_nodes` is below 2.
         */
        Mcts(Evaluator evaluator, size_t max_nodes)
            : evaluator_(std::move(evaluator)), max_nodes_(max_nodes) {
            if (max_nodes_ < 2) {
                throw std::invalid_argument("max_nodes must be at least 2");
            }
            resize_table();
        }

        /**
         * @brief Runs simulations from a root position.
         *
         * The first simulation from an unseen root only expands it, so
         * `num_sims` simulations add `num_sims - 1` visits.
         *
         * @param root Position to search from.
         * @param num_sims Number of simulations.
         * @param config Exploration settings.
         * @throws std::invalid_argument When the batch size is outside [1, kMaxSimBatch].
         */
        void run(Position root, int num_sims, const SearchConfig &config) {
            const size_t needed = static_cast<size_t>(num_sims) + 1;
            if (config.batch_size < 1 || config.batch_size > kMaxSimBatch) {
                throw std::invalid_argument("batch_size must be in [1, kMaxSimBatch]");
            }
            if (nodes_.size() + needed > max_nodes_) {
                clear();
                if (needed > max_nodes_) {
                    max_nodes_ = needed;
                    resize_table();
                }
            }
            int remaining = num_sims;
            while (remaining > 0) {
                if (config.batch_size == 1) {
                    simulate(root, config);
                    --remaining;
                } else {
                    remaining -=
                        simulate_batch(root, config, std::min(config.batch_size, remaining));
                }
            }
        }

        /**
         * @brief Visit counts of the root's actions.
         *
         * @param root Searched position.
         * @param counts Receives kActions counts, all zero when the root is unseen or finished.
         */
        void visit_counts(Position root, uint32_t *counts) const {
            std::fill(counts, counts + kActions, 0u);
            size_t slot;
            const uint32_t index = lookup(root, slot);
            if (index == kEmpty || nodes_[index].terminal) {
                return;
            }
            const Node &node = nodes_[index];
            for (uint32_t i = 0; i < node.edge_count; ++i) {
                counts[edge_actions_[node.edge_begin + i]] = edges_[node.edge_begin + i].visits;
            }
        }

        /**
         * @brief Picks an action from the root's visit counts.
         *
         * @param counts kActions visit counts.
         * @param temperature 0 picks a most visited action, ties broken at
         *        random. Otherwise the result is distributed as counts^(1 / temperature).
         * @param rng Source of the tie-break.
         * @param probs Receives kActions probabilities.
         */
        static void action_probs(const uint32_t *counts, double temperature, Rng &rng,
                                 double *probs) {
            constexpr double kEps = 1e-8;
            if (temperature == 0.0) {
                const uint32_t best = *std::max_element(counts, counts + kActions);
                int ties = 0;
                for (int a = 0; a < kActions; ++a) {
                    ties += counts[a] == best;
                }
                int pick = static_cast<int>(rng.below(static_cast<uint32_t>(ties)));
                std::fill(probs, probs + kActions, 0.0);
                for (int a = 0; a < kActions; ++a) {
                    if (counts[a] == best && pick-- == 0) {
                        probs[a] = 1.0;
                        break;
                    }
                }
                return;
            }
            double sum = 0.0;
            for (int a = 0; a < kActions; ++a) {
                probs[a] = std::pow(static_cast<double>(counts[a]), 1.0 / temperature);
                sum += probs[a];
            }
            sum += kEps;
            for (int a = 0; a < kActions; ++a) {
                probs[a] /= sum;
            }
        }

        /**
         * @brief Copies the part of the tree below a root, breadth first.
         *
         * Positions reached by several paths appear once, and edges to them
         * share the node id.
         *
         * @param root Position to start from.
         * @param max_depth Deepest level whose nodes are listed.
         * @param min_visits Edges with fewer visits are listed but not followed.
         * @return Nodes in breadth-first order, empty when the root is unseen.
         */
        std::vector<ExportedNode> export_tree(Position root, int max_depth,
                                              uint32_t min_visits) const {
            std::vector<ExportedNode> out;
            size_t slot;
            const uint32_t root_index = lookup(root, slot);
            if (root_index == kEmpty) {
                return out;
            }
            std::vector<uint8_t> queued(nodes_.size(), 0);
            std::vector<std::pair<uint32_t, int>> queue{{root_index, 0}};
            queued[root_index] = 1;
            for (size_t q = 0; q < queue.size(); ++q) {
                const auto [index, depth] = queue[q];
                const Node &node = nodes_[index];
                ExportedNode exported{
                    index, node.pos, node.visits, node.terminal, node.terminal_value, depth, {}};
                if (!node.terminal) {
                    for (uint32_t i = node.edge_begin; i < node.edge_begin + node.edge_count; ++i) {
                        const Edge &edge = edges_[i];
                        const bool followed = edge.child != kNoChild;
                        exported.edges.push_back(
                            {edge_actions_[i], edge.prior, edge.value, edge.visits,
                             followed ? static_cast<int64_t>(edge.child) : -1});
                        if (followed && depth < max_depth && edge.visits >= min_visits &&
                            !queued[edge.child]) {
                            queued[edge.child] = 1;
                            queue.push_back({edge.child, depth + 1});
                        }
                    }
                }
                out.push_back(std::move(exported));
            }
            return out;
        }

        /** @brief Drops the whole tree. The cost follows the number of nodes, not the table size.
         */
        void clear() {
            for (const Node &node : nodes_) {
                slots_[node.slot] = kEmpty;
            }
            nodes_.clear();
            edges_.clear();
            edge_actions_.clear();
        }

        /**
         * @brief Number of positions in the tree.
         *
         * @return The node count.
         */
        size_t node_count() const {
            return nodes_.size();
        }

    private:
        /** @brief Marks an empty table slot. */
        static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
        /** @brief Marks an edge whose child has not been looked up yet. */
        static constexpr uint32_t kNoChild = 0xFFFFFFFFu;
        /** @brief Longest path a simulation can follow. */
        static constexpr int kMaxDepth = 256;

        /** @brief A position of the tree. */
        struct Node {
            Position pos;          ///< Position, seen from its side to move.
            uint32_t visits;       ///< Selections through this node.
            uint32_t edge_begin;   ///< Index of the first edge.
            uint32_t edge_count;   ///< Number of edges.
            float terminal_value;  ///< Outcome for the side to move when `terminal`.
            uint32_t slot;         ///< Table slot that refers to this node.
            bool terminal;         ///< Whether the game is over.
            bool pending;          ///< Whether the node waits for its evaluation inside a batch.
        };

        /** @brief A legal action of a node. */
        struct Edge {
            float prior;      ///< Policy probability of the action.
            float value;      ///< Mean outcome of the simulations through this edge.
            uint32_t visits;  ///< Simulations through this edge, virtual ones included.
            uint32_t child;   ///< Node index, or kNoChild until first followed.
        };

        /** @brief One move of a path. */
        struct Step {
            uint32_t node;  ///< Node the move starts from.
            uint32_t edge;  ///< Edge that was chosen.
        };

        /** @brief A leaf of a batch that waits for its evaluation, with the path to it. */
        struct Pending {
            uint32_t node;         ///< The leaf.
            uint64_t legal;        ///< Legal squares of the leaf.
            int length;            ///< Number of steps of the path.
            Step path[kMaxDepth];  ///< Steps from the root to the leaf.
        };

        /** @brief Sizes the table for `max_nodes_` nodes and empties it. */
        void resize_table() {
            size_t capacity = 16;
            while (capacity < 2 * max_nodes_) {
                capacity <<= 1;
            }
            slots_.assign(capacity, kEmpty);
            mask_ = capacity - 1;
        }

        /**
         * @brief Finds the node of a position.
         *
         * @param p Position to look up.
         * @param slot Receives the table slot where the position belongs when it is absent.
         * @return The node index, or kEmpty when the position is absent.
         */
        uint32_t lookup(Position p, size_t &slot) const {
            size_t i = position_hash(p) & mask_;
            while (true) {
                const uint32_t index = slots_[i];
                if (index == kEmpty) {
                    slot = i;
                    return kEmpty;
                }
                if (nodes_[index].pos == p) {
                    return index;
                }
                i = (i + 1) & mask_;
            }
        }

        /**
         * @brief Adds the edges of an evaluated node.
         *
         * @param index Node to expand.
         * @param legal Bit mask of the node's legal squares.
         * @param probs kActions policy probabilities of the node.
         */
        void expand(uint32_t index, uint64_t legal, const float *probs) {
            Node &node = nodes_[index];
            node.edge_begin = static_cast<uint32_t>(edges_.size());
            node.edge_count = 0;
            node.pending = false;
            if (legal == 0) {
                edges_.push_back({probs[kPassAction], 0.0f, 0, kNoChild});
                edge_actions_.push_back(kPassAction);
                node.edge_count = 1;
                return;
            }
            for (uint64_t moves = legal; moves != 0; moves &= moves - 1) {
                const int square = std::countr_zero(moves);
                edges_.push_back({probs[square], 0.0f, 0, kNoChild});
                edge_actions_.push_back(static_cast<uint8_t>(square));
                ++node.edge_count;
            }
        }

        /**
         * @brief Adds a node for an unseen position without evaluating it.
         *
         * A finished game becomes a terminal node. Any other position becomes a
         * pending node that expand() completes.
         *
         * @param p Position of the node.
         * @param slot Table slot returned by lookup().
         * @param legal Receives the position's legal squares.
         * @return Index of the new node.
         */
        uint32_t add_node(Position p, size_t slot, uint64_t &legal) {
            const uint32_t index = static_cast<uint32_t>(nodes_.size());
            slots_[slot] = index;
            legal = legal_moves(p);
            Node node{p, 0, 0, 0, 0.0f, static_cast<uint32_t>(slot), false, true};
            if (legal == 0 && legal_moves(swap_sides(p)) == 0) {
                node.terminal = true;
                node.pending = false;
                node.terminal_value = static_cast<float>(terminal_value(p));
            }
            nodes_.push_back(node);
            return index;
        }

        /**
         * @brief Adds a node for an unseen position and evaluates it.
         *
         * @param p Position of the node.
         * @param slot Table slot returned by lookup().
         * @param index Receives the new node's index.
         * @return Value passed up to the parent: the negated evaluation or finished-game outcome.
         */
        float create_node(Position p, size_t slot, uint32_t &index) {
            uint64_t legal;
            index = add_node(p, slot, legal);
            if (nodes_[index].terminal) {
                return -nodes_[index].terminal_value;
            }
            float probs[kActions];
            float value;
            evaluator_(p, legal, probs, value);
            expand(index, legal, probs);
            return -value;
        }

        /**
         * @brief Exploration constant for a node with `visits` selections.
         *
         * @param config Exploration settings.
         * @param visits Selections through the node.
         * @return The constant to use for the node.
         */
        static double exploration(const SearchConfig &config, uint32_t visits) {
            constexpr double kFloor = 0.5;
            constexpr double kRate = 0.001;
            switch (config.cpuct_type) {
            case CpuctType::Decrement:
                return kFloor + (config.c_puct - kFloor) * std::exp(-kRate * visits);
            case CpuctType::Increment:
                return config.c_puct + (kFloor - config.c_puct) * std::exp(-kRate * visits);
            default: return config.c_puct;
            }
        }

        /**
         * @brief Chooses the edge with the highest upper confidence bound.
         *
         * The lowest action wins ties.
         *
         * @param node Node to select from, which has at least one edge.
         * @param config Exploration settings.
         * @return Index of the chosen edge.
         */
        uint32_t select_edge(const Node &node, const SearchConfig &config) const {
            constexpr double kEps = 1e-8;
            const float c = static_cast<float>(exploration(config, node.visits));
            const float sqrt_visits =
                static_cast<float>(std::sqrt(static_cast<double>(node.visits)));
            const float sqrt_unvisited = static_cast<float>(std::sqrt(node.visits + kEps));
            float best_score = -std::numeric_limits<float>::infinity();
            uint32_t best = node.edge_begin;
            for (uint32_t i = node.edge_begin; i < node.edge_begin + node.edge_count; ++i) {
                const Edge &edge = edges_[i];
                const float explore = c * edge.prior;
                const float score =
                    edge.visits == 0
                        ? explore * sqrt_unvisited
                        : edge.value + explore * sqrt_visits / static_cast<float>(1 + edge.visits);
                if (score > best_score) {
                    best_score = score;
                    best = i;
                }
            }
            return best;
        }

        /**
         * @brief Runs one simulation that evaluates its leaf at once.
         *
         * @param root Position to search from.
         * @param config Exploration settings.
         */
        void simulate(Position root, const SearchConfig &config) {
            size_t slot;
            uint32_t current = lookup(root, slot);
            if (current == kEmpty) {
                create_node(root, slot, current);
                return;
            }
            Step path[kMaxDepth];
            int length = 0;
            float result;
            while (true) {
                const Node &node = nodes_[current];
                if (node.terminal) {
                    result = -node.terminal_value;
                    break;
                }
                const uint32_t e = select_edge(node, config);
                path[length++] = {current, e};
                uint32_t child = edges_[e].child;
                if (child == kNoChild) {
                    const Position next = play(node.pos, edge_actions_[e]);
                    child = lookup(next, slot);
                    if (child == kEmpty) {
                        result = create_node(next, slot, child);
                        edges_[e].child = child;
                        break;
                    }
                    edges_[e].child = child;
                }
                current = child;
            }
            backup(path, length, result);
        }

        /**
         * @brief Adds a path's result to its edges and nodes.
         *
         * @param path Steps from the root.
         * @param length Number of steps.
         * @param result Outcome for the side that chose the last step.
         */
        void backup(const Step *path, int length, float result) {
            for (int i = length - 1; i >= 0; --i) {
                Edge &edge = edges_[path[i].edge];
                const float n = static_cast<float>(edge.visits);
                edge.value = (n * edge.value + result) / (n + 1.0f);
                ++edge.visits;
                ++nodes_[path[i].node].visits;
                result = -result;
            }
        }

        /**
         * @brief Counts a path as a lost simulation for every side that chose a step on it.
         *
         * @param path Steps from the root.
         * @param length Number of steps.
         */
        void apply_virtual_loss(const Step *path, int length) {
            for (int i = 0; i < length; ++i) {
                Edge &edge = edges_[path[i].edge];
                const float n = static_cast<float>(edge.visits);
                edge.value = (n * edge.value - 1.0f) / (n + 1.0f);
                ++edge.visits;
                ++nodes_[path[i].node].visits;
            }
        }

        /**
         * @brief Replaces the virtual loss on a path with the real result.
         *
         * @param path Steps from the root.
         * @param length Number of steps.
         * @param result Outcome for the side that chose the last step.
         */
        void settle_virtual_loss(const Step *path, int length, float result) {
            for (int i = length - 1; i >= 0; --i) {
                Edge &edge = edges_[path[i].edge];
                edge.value += (1.0f + result) / static_cast<float>(edge.visits);
                result = -result;
            }
        }

        /**
         * @brief Evaluates the waiting leaves together, expands them and settles their paths.
         *
         * @param pending Waiting leaves.
         * @param count Number of waiting leaves.
         */
        void flush(Pending *pending, int count) {
            Position positions[kMaxSimBatch];
            uint64_t legal[kMaxSimBatch];
            float probs[kMaxSimBatch * kActions];
            float values[kMaxSimBatch];
            for (int k = 0; k < count; ++k) {
                positions[k] = nodes_[pending[k].node].pos;
                legal[k] = pending[k].legal;
            }
            evaluator_.evaluate_batch(count, positions, legal, probs, values);
            for (int k = 0; k < count; ++k) {
                expand(pending[k].node, legal[k], probs + k * kActions);
                settle_virtual_loss(pending[k].path, pending[k].length, -values[k]);
            }
        }

        /**
         * @brief Runs simulations whose leaves are evaluated together.
         *
         * Each simulation that reaches an unseen position marks its path with a
         * virtual loss, so that the next simulation of the batch prefers another
         * path. The batch ends when `target` simulations are collected or when a
         * simulation would reach a leaf that is still waiting.
         *
         * @param root Position to search from.
         * @param config Exploration settings.
         * @param target Simulations to collect, at most kMaxSimBatch.
         * @return Number of simulations run, at least 1.
         */
        int simulate_batch(Position root, const SearchConfig &config, int target) {
            size_t slot;
            uint32_t root_index = lookup(root, slot);
            if (root_index == kEmpty) {
                create_node(root, slot, root_index);
                return 1;
            }
            Pending pending[kMaxSimBatch];
            int waiting = 0;
            int done = 0;
            while (done < target) {
                Step path[kMaxDepth];
                int length = 0;
                uint32_t current = root_index;
                uint32_t leaf = kEmpty;
                float result = 0.0f;
                bool blocked = false;
                while (true) {
                    const Node &node = nodes_[current];
                    if (node.pending) {
                        blocked = true;
                        break;
                    }
                    if (node.terminal) {
                        result = -node.terminal_value;
                        break;
                    }
                    const uint32_t e = select_edge(node, config);
                    path[length++] = {current, e};
                    uint32_t child = edges_[e].child;
                    if (child == kNoChild) {
                        const Position next = play(node.pos, edge_actions_[e]);
                        child = lookup(next, slot);
                        if (child == kEmpty) {
                            uint64_t legal;
                            child = add_node(next, slot, legal);
                            edges_[e].child = child;
                            if (nodes_[child].terminal) {
                                result = -nodes_[child].terminal_value;
                            } else {
                                leaf = child;
                                pending[waiting].node = child;
                                pending[waiting].legal = legal;
                            }
                            break;
                        }
                        edges_[e].child = child;
                    }
                    current = child;
                }
                if (blocked) {
                    break;
                }
                if (leaf == kEmpty) {
                    backup(path, length, result);
                } else {
                    apply_virtual_loss(path, length);
                    pending[waiting].length = length;
                    std::copy(path, path + length, pending[waiting].path);
                    ++waiting;
                }
                ++done;
            }
            flush(pending, waiting);
            return done;
        }

        Evaluator evaluator_;      ///< Policy and value source.
        size_t max_nodes_;         ///< Node count above which the tree is cleared before a search.
        std::vector<Node> nodes_;  ///< Every position of the tree.
        std::vector<Edge> edges_;  ///< Every edge, grouped by node.
        std::vector<uint8_t> edge_actions_;  ///< Action of each edge.
        std::vector<uint32_t> slots_;  ///< Open-addressing table from positions to node indices.
        size_t mask_ = 0;  ///< Table size minus one, the table size being a power of two.
    };

}  // namespace othello
