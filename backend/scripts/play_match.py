"""
Plays games between two configurations of the C++ search and reports the result.

Both sides use the same weight file. Every opening of --random-plies random
moves is played twice with the colors swapped, and each side keeps its search
tree only within a game.

Usage:
    python scripts/play_match.py --help
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))

import othello_cpp  # noqa: E402

PASS_ACTION = 64


def play_game(first, second, board: np.ndarray, player: int) -> tuple[int, float, float]:
    """
    Plays a game from a position until it ends.

    Parameters
    ----------
    first : dict
        Settings of the side playing black (-1): `engine`, `sims` and `batch`.
    second : dict
        Settings of the side playing white (1).
    board : numpy.ndarray
        Starting 8x8 board.
    player : int
        Side to move at the start, 1 or -1.

    Returns
    -------
    tuple
        The disc difference from black's view, and the seconds each side spent searching.
    """
    sides = {-1: first, 1: second}
    seconds = {-1: 0.0, 1: 0.0}
    for side in sides.values():
        side["engine"].clear()
    while othello_cpp.get_game_ended(board, player) == 0:
        valid = othello_cpp.get_valid_moves(board, player)
        if valid[PASS_ACTION]:
            action = PASS_ACTION
        else:
            side = sides[player]
            start = time.perf_counter()
            probs = side["engine"].get_action_probs(board, player, side["sims"], temperature=0.0, batch_size=side["batch"])
            seconds[player] += time.perf_counter() - start
            action = int(np.argmax(probs))
        board, player = othello_cpp.get_next_state(board, player, action)
    return int((board == -1).sum()) - int((board == 1).sum()), seconds[-1], seconds[1]


def random_opening(plies: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """
    Plays random legal moves from the initial position.

    Parameters
    ----------
    plies : int
        Number of moves.
    rng : numpy.random.Generator
        Source of the moves.

    Returns
    -------
    tuple
        The board and the player to move.
    """
    board, player = othello_cpp.get_init_board(), -1
    for _ in range(plies):
        valid = othello_cpp.get_valid_moves(board, player)
        action = PASS_ACTION if valid[PASS_ACTION] else int(rng.choice(np.flatnonzero(valid[:64])))
        board, player = othello_cpp.get_next_state(board, player, action)
    return board, player


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-a", type=Path, required=True)
    parser.add_argument("--weights-b", type=Path, help="defaults to --weights-a")
    parser.add_argument("--sims-a", type=int, default=200)
    parser.add_argument("--sims-b", type=int, default=200)
    parser.add_argument("--batch-a", type=int, default=1)
    parser.add_argument("--batch-b", type=int, default=1)
    parser.add_argument("--openings", type=int, default=20, help="random openings, each played with both colors")
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    a = {"engine": othello_cpp.Engine(str(ns.weights_a)), "sims": ns.sims_a, "batch": ns.batch_a}
    b = {"engine": othello_cpp.Engine(str(ns.weights_b or ns.weights_a)), "sims": ns.sims_b, "batch": ns.batch_b}
    rng = np.random.default_rng(ns.seed)
    points = {"A": 0.0, "B": 0.0}
    results = {"A": 0, "B": 0, "draw": 0}
    seconds = {"A": 0.0, "B": 0.0}
    games = 2 * ns.openings
    progress = tqdm(total=games, desc="games", unit="game")
    for _ in range(ns.openings):
        board, player = random_opening(ns.random_plies, rng)
        for a_is_black in (True, False):
            first, second = (a, b) if a_is_black else (b, a)
            diff, black_seconds, white_seconds = play_game(first, second, board.copy(), player)
            a_diff = diff if a_is_black else -diff
            seconds["A"] += black_seconds if a_is_black else white_seconds
            seconds["B"] += white_seconds if a_is_black else black_seconds
            winner = "A" if a_diff > 0 else "B" if a_diff < 0 else "draw"
            results[winner] += 1
            points["A"] += 1.0 if winner == "A" else 0.5 if winner == "draw" else 0.0
            progress.update(1)
            progress.set_postfix(A=f"{results['A']}W", B=f"{results['B']}W", draws=results["draw"])
    progress.close()
    score = points["A"] / games
    margin = 1.96 * math.sqrt(max(score * (1 - score), 1e-9) / games)
    print(f"A: batch {ns.batch_a}, {ns.sims_a} sims   B: batch {ns.batch_b}, {ns.sims_b} sims")
    print(f"games {games}: A wins {results['A']}, B wins {results['B']}, draws {results['draw']}")
    print(f"A score {score:.3f} (about +/- {margin:.3f}), search time A {seconds['A']:.1f} s, B {seconds['B']:.1f} s")


if __name__ == "__main__":
    main()
