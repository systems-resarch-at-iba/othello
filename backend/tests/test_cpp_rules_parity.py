"""
Parity tests between the Python rules (board.Utils) and the C++ rules
(othello_cpp). Skipped when the extension has not been built.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpp" / "build"))
othello_cpp = pytest.importorskip("othello_cpp")

from board import Utils as BU  # noqa: E402

GAMES_PER_SEED = 200
PASS_ACTION = 64


def assert_same_next_state(board, player, action):
    expected_board, expected_player = BU.get_next_state(board, player, action)
    actual_board, actual_player = othello_cpp.get_next_state(board, player, action)
    assert np.array_equal(expected_board, actual_board)
    assert expected_player == actual_player


def assert_same_position_queries(board, player):
    for side in (player, -player):
        assert np.array_equal(BU.get_valid_moves(board, side), othello_cpp.get_valid_moves(board, side))
        assert BU.has_valid_moves(board, side) == othello_cpp.has_valid_moves(board, side)
        assert BU.get_game_ended(board, side) == othello_cpp.get_game_ended(board, side)


def test_init_board_matches():
    assert np.array_equal(BU.get_init_board(), othello_cpp.get_init_board())


@pytest.mark.parametrize("seed", range(10))
def test_random_games_match(seed):
    rng = np.random.default_rng(seed)
    for _ in range(GAMES_PER_SEED):
        board = BU.get_init_board()
        player = -1
        while True:
            assert_same_position_queries(board, player)
            if BU.get_game_ended(board, player) != 0:
                break
            legal = np.flatnonzero(BU.get_valid_moves(board, player))
            chosen = int(rng.choice(legal))
            other = int(rng.choice(legal))
            assert_same_next_state(board, player, other)
            assert_same_next_state(board, player, chosen)
            board, player = BU.get_next_state(board, player, chosen)


def test_pass_action_swaps_player_and_keeps_board():
    board = BU.get_init_board()
    next_board, next_player = othello_cpp.get_next_state(board, -1, PASS_ACTION)
    assert np.array_equal(board, next_board)
    assert next_player == 1


def test_illegal_action_raises():
    board = BU.get_init_board()
    with pytest.raises(ValueError):
        othello_cpp.get_next_state(board, -1, 0)
    with pytest.raises(ValueError):
        othello_cpp.get_next_state(board, -1, 65)


def test_invalid_inputs_raise():
    board = BU.get_init_board()
    with pytest.raises(ValueError):
        othello_cpp.get_valid_moves(board, 0)
    with pytest.raises(ValueError):
        othello_cpp.get_valid_moves(np.zeros(10, dtype=np.int8), 1)
    with pytest.raises(ValueError):
        othello_cpp.get_valid_moves(np.full(64, 2, dtype=np.int8), 1)
