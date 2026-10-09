"""
Parity tests between the Python search (engine.MCTS) and the C++ search
(othello_cpp.Engine and othello_cpp.StubEngine). Skipped when the extension
has not been built.
"""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
othello_cpp = pytest.importorskip("othello_cpp")

import export_weights  # noqa: E402
from arch import CNN_vAlpha3, args  # noqa: E402
from board import Utils as BU  # noqa: E402
from engine import MCTS, OthelloCNN  # noqa: E402
from utils import dotdict  # noqa: E402

CHECKPOINT = BACKEND_DIR / "models" / f"model-{args.version}.pth.tar"
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
ACTIONS = 65


def mix64(x):
    x &= MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


class StubCNN:
    """
    Network stand-in that gives the same pseudo-random policy and value as
    othello_cpp.StubEngine, for any canonical board.
    """

    def predict(self, canonical_board):
        flat = canonical_board.reshape(-1)
        mine = sum(1 << int(i) for i in np.flatnonzero(flat == 1))
        theirs = sum(1 << int(i) for i in np.flatnonzero(flat == -1))
        h = mix64(mine ^ mix64(theirs))
        weights = np.array(
            [1 + mix64((h + (a + 1) * GOLDEN) & MASK64) % 997 for a in range(ACTIONS)], dtype=np.float32
        ) / np.float32(65536)
        value = np.float32(((mix64(h ^ 0xABCDEF) % 2049) - 1024) / 1024)
        return weights, np.array([value], dtype=np.float32)


def python_search(cnn, sims, cpuct_type="static", c_puct=1.5):
    search_args = dotdict({**args, "num_mcts_sims": sims, "c_puct": c_puct, "cuda": False})
    return MCTS(cnn, search_args, cpuct_type)


def random_positions(count, seed, max_plies=40):
    """
    Plays random games and returns boards with the player to move.

    Parameters
    ----------
    count : int
        Number of positions.
    seed : int
        Seed of the random games.
    max_plies : int, optional
        Plies after which a game stops.

    Returns
    -------
    list
        Tuples of an 8x8 board and the player to move.
    """
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        board, player = BU.get_init_board(), -1
        for _ in range(int(rng.integers(0, max_plies))):
            if BU.get_game_ended(board, player) != 0:
                break
            board, player = BU.get_next_state(board, player, int(rng.choice(np.flatnonzero(BU.get_valid_moves(board, player)))))
        if BU.get_game_ended(board, player) == 0:
            positions.append((board, player))
    return positions


@pytest.mark.parametrize("cpuct_type", ["static", "increment", "decrement"])
@pytest.mark.parametrize("sims", [30, 300])
def test_stub_visit_distributions_match_python(cpuct_type, sims):
    cpp = othello_cpp.StubEngine()
    for board, player in random_positions(8, seed=sims):
        python = python_search(StubCNN(), sims, cpuct_type)
        expected = np.array(python.get_action_probs(BU.get_canonical_form(board, player), temperature=1))
        actual = cpp.get_action_probs(board, player, sims, 1.0, 1.5, cpuct_type)
        cpp.clear()
        assert np.array_equal(expected, actual)


def test_stub_search_matches_python_across_a_game():
    sims = 150
    cpp = othello_cpp.StubEngine()
    python = python_search(StubCNN(), sims)
    board, player = BU.get_init_board(), -1
    for _ in range(12):
        if BU.get_game_ended(board, player) != 0:
            break
        expected = np.array(python.get_action_probs(BU.get_canonical_form(board, player), temperature=1))
        actual = cpp.get_action_probs(board, player, sims)
        assert np.array_equal(expected, actual)
        board, player = BU.get_next_state(board, player, int(np.argmax(expected)))


def test_exported_tree_is_consistent_and_matches_python_tables():
    sims = 200
    board, player = BU.get_init_board(), -1
    cpp = othello_cpp.StubEngine()
    cpp.get_action_probs(board, player, sims)
    python = python_search(StubCNN(), sims)
    canonical = BU.get_canonical_form(board, player)
    python.get_action_probs(canonical, temperature=1)

    nodes = cpp.get_tree(board, player, max_depth=4, min_visits=1)
    ids = {node["id"] for node in nodes}
    assert len(ids) == len(nodes) and nodes[0]["depth"] == 0
    for node in nodes:
        if node["terminal"]:
            assert not node["edges"]
            continue
        assert sum(e["visits"] for e in node["edges"]) == node["visits"]
        for edge in node["edges"]:
            if node["depth"] < 4 and edge["visits"] >= 1:
                assert edge["child"] in ids

    root = nodes[0]
    state = python.hasher.hash(canonical)
    assert root["visits"] == python.Ns[state]
    for edge in root["edges"]:
        key = (state, edge["action"])
        assert edge["visits"] == python.Nsa.get(key, 0)
        assert edge["prior"] == python.Ps[state][edge["action"]]
        if edge["visits"]:
            assert np.isclose(edge["value"], float(np.asarray(python.Qsa[key]).reshape(-1)[0]), atol=1e-6)
    assert cpp.get_tree(board, 1, 2, 1) == []


@pytest.mark.parametrize("batch_size", [2, 4, 16])
def test_batched_search_gives_a_valid_distribution(batch_size):
    cpp = othello_cpp.StubEngine()
    board, player = BU.get_init_board(), -1
    probs = cpp.get_action_probs(board, player, 500, batch_size=batch_size)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    valid = BU.get_valid_moves(board, player)
    assert all(p == 0 for p, ok in zip(probs, valid) if not ok)
    tree = cpp.get_tree(board, player, max_depth=100, min_visits=0)
    assert tree[0]["visits"] == 499
    for node in tree:
        if not node["terminal"]:
            assert sum(e["visits"] for e in node["edges"]) == node["visits"]


def test_batch_size_out_of_range_raises():
    cpp = othello_cpp.StubEngine()
    board = BU.get_init_board()
    for batch_size in (0, 17):
        with pytest.raises(ValueError):
            cpp.get_action_probs(board, -1, 10, batch_size=batch_size)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint not available")
def test_batched_trained_search_picks_the_same_move_as_the_sequential_search(tmp_path):
    state_dict = torch.load(CHECKPOINT, map_location="cpu")["state_dict"]
    weights = tmp_path / "weights.bin"
    weights.write_bytes(export_weights.pack_state_dict(state_dict))
    cpp = othello_cpp.Engine(str(weights))
    positions = random_positions(20, seed=5)
    agree = 0
    for board, player in positions:
        sequential = cpp.get_action_probs(board, player, 800)
        cpp.clear()
        batched = cpp.get_action_probs(board, player, 800, batch_size=4)
        cpp.clear()
        agree += int(np.argmax(sequential)) == int(np.argmax(batched))
    assert agree / len(positions) >= 0.9


def test_temperature_zero_is_one_hot_and_seeded():
    cpp = othello_cpp.StubEngine()
    board, player = BU.get_init_board(), -1
    first = cpp.get_action_probs(board, player, 100, temperature=0.0, seed=5)
    assert first.sum() == 1.0 and first.max() == 1.0
    assert BU.get_valid_moves(board, player)[int(np.argmax(first))] == 1
    cpp.clear()
    assert np.array_equal(first, cpp.get_action_probs(board, player, 100, temperature=0.0, seed=5))


def test_pass_and_finished_positions():
    cpp = othello_cpp.StubEngine()
    must_pass = np.zeros(64, dtype=np.int8)
    must_pass[1], must_pass[0] = 1, -1
    probs = cpp.get_action_probs(must_pass, 1, 30)
    assert probs[64] > 0.99
    finished = np.zeros(64, dtype=np.int8)
    finished[0] = 1
    assert not cpp.get_action_probs(finished, 1, 10).any()
    assert cpp.get_action_probs(finished, 1, 10, temperature=0.0).sum() == 1.0


def test_node_limit_clears_the_tree():
    cpp = othello_cpp.StubEngine(max_nodes=500)
    board, player = BU.get_init_board(), -1
    for _ in range(3):
        cpp.get_action_probs(board, player, 300)
        assert cpp.node_count <= 500
    probs = cpp.get_action_probs(board, player, 2000)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_invalid_arguments_raise():
    cpp = othello_cpp.StubEngine()
    board = BU.get_init_board()
    with pytest.raises(ValueError):
        cpp.get_action_probs(board, -1, 0)
    with pytest.raises(ValueError):
        cpp.get_action_probs(board, -1, 10, cpuct_type="other")
    with pytest.raises(ValueError):
        cpp.get_action_probs(board, -1, 10, temperature=-1.0)
    with pytest.raises(ValueError):
        cpp.get_action_probs(board, 0, 10)


def test_concurrent_searches_are_serialized():
    cpp = othello_cpp.StubEngine()
    board, player = BU.get_init_board(), -1
    results = []

    def work():
        results.append(cpp.get_action_probs(board, player, 200))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4
    assert all(np.isclose(r.sum(), 1.0, atol=1e-6) for r in results)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint not available")
def test_trained_network_search_agrees_with_python(tmp_path):
    sims = 100
    state_dict = torch.load(CHECKPOINT, map_location="cpu")["state_dict"]
    weights = tmp_path / "weights.bin"
    weights.write_bytes(export_weights.pack_state_dict(state_dict))
    cpp = othello_cpp.Engine(str(weights))

    cnn = OthelloCNN(CNN_vAlpha3, dotdict({**args, "cuda": False}))
    cnn.load_checkpoint(str(CHECKPOINT.parent), CHECKPOINT.name)
    positions = random_positions(20, seed=3)
    top_matches, distances = 0, []
    for board, player in positions:
        python = python_search(cnn, sims)
        expected = np.array(python.get_action_probs(BU.get_canonical_form(board, player), temperature=1))
        actual = cpp.get_action_probs(board, player, sims)
        cpp.clear()
        top_matches += int(np.argmax(expected)) == int(np.argmax(actual))
        distances.append(np.abs(expected - actual).sum())
    assert top_matches / len(positions) >= 0.9
    assert np.mean(distances) < 0.2
