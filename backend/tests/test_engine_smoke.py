"""
Smoke tests for the board/MCTS/CNN logic. Deliberately avoid OthelloAI's
constructor (which loads a real trained checkpoint via load_checkpoint) so
these run in CI without needing the actual model weights: OthelloCNN and
MCTS can be constructed directly with a randomly-initialized network, which
is enough to check the search produces a structurally valid result.
"""
import numpy as np

from board import OthelloBitBoard, Utils as BU
from arch import CNN_vAlpha3, args
from engine.othello_cnn import OthelloCNN
from engine.mcts import MCTS
from utils import dotdict


def test_initial_board_has_four_valid_moves():
    board = OthelloBitBoard()
    assert len(board.get_valid_moves(-1)) == 4  # black moves first


def test_apply_move_flips_captured_pieces():
    board = OthelloBitBoard()
    black_before, white_before = board.score()
    moves = board.get_valid_moves(-1)
    board.apply_move(moves[0], -1)
    black_after, white_after = board.score()
    # Black gains at least the placed piece plus one flipped white piece.
    assert black_after > black_before
    assert white_after < white_before


def test_get_game_ended_is_zero_on_fresh_board():
    board = BU.get_init_board()
    assert BU.get_game_ended(board, 1) == 0


def test_mcts_returns_valid_action_distribution():
    # dotdict(...), not `args | {...}`: dict's own `|` returns a
    # plain dict, not a dotdict, which would break the attribute access
    # (args.cuda, args.log_level, etc.) that OthelloCNN/MCTS rely on.
    fast_args = dotdict({**args, "num_mcts_sims": 8})
    cnn = OthelloCNN(CNN_vAlpha3, fast_args)
    mcts = MCTS(cnn, fast_args)

    board = BU.get_init_board()
    canonical = BU.get_canonical_form(board, 1)
    probs = mcts.get_action_probs(canonical, temperature=1)

    assert len(probs) == 65  # 64 squares + pass
    assert all(p >= 0 for p in probs)
    assert np.isclose(sum(probs), 1.0, atol=1e-3)
