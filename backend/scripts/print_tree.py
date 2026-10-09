"""
Runs a C++ search and writes the search tree below the root as text.

Each node lists its edges with the policy probability (P), the mean outcome
for the side to move (Q), the visit count (N) and the child node. A position
reached by several paths is listed once, under the first path that reached it,
and later paths reference its node id.

Usage:
    python scripts/print_tree.py --help
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
# arch.py, board/ and configs.py are top-level modules under backend/ with no
# package init. This script lives in scripts/, where sys.path[0] is scripts/, so
# backend/ is added to sys.path explicitly.
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))

import export_weights  # noqa: E402
import othello_cpp  # noqa: E402
from arch import args as model_args  # noqa: E402
from board import Utils as BU  # noqa: E402

PASS_ACTION = 64
COLOR_NAMES = {-1: "black", 1: "white"}
COLOR_SYMBOLS = {-1: "B", 1: "W"}


def notation(action: int) -> str:
    """
    Converts an action to board notation.

    Parameters
    ----------
    action : int
        Square index, or 64 for a pass.

    Returns
    -------
    str
        Column letter and row number such as "d3", or "pass".
    """
    if action == PASS_ACTION:
        return "pass"
    return chr(ord("a") + action % 8) + str(action // 8 + 1)


def action_from_notation(text: str) -> int:
    """
    Converts board notation to an action.

    Parameters
    ----------
    text : str
        Notation such as "d3", or "pass".

    Returns
    -------
    int
        Square index, or 64 for a pass.
    """
    if text == "pass":
        return PASS_ACTION
    return (int(text[1]) - 1) * 8 + ord(text[0]) - ord("a")


def board_lines(board: np.ndarray) -> list[str]:
    """
    Draws a board.

    Parameters
    ----------
    board : numpy.ndarray
        Board of -1 (black), 0 and 1 (white).

    Returns
    -------
    list
        Text lines, a column header followed by one line per row.
    """
    lines = ["    a b c d e f g h"]
    for row in range(8):
        cells = " ".join(COLOR_SYMBOLS.get(int(v), ".") for v in board.reshape(8, 8)[row])
        lines.append(f"  {row + 1} {cells}")
    return lines


def assign_paths(nodes: list[dict], root_player: int) -> tuple[dict, dict, dict]:
    """
    Assigns each listed node its side to move, its first move path and its first parent edge.

    A position reached by several paths gets the values of the first path that
    reached it in breadth-first order.

    Parameters
    ----------
    nodes : list
        Output of get_tree.
    root_player : int
        Side to move at the root, 1 or -1.

    Returns
    -------
    tuple
        Three dicts keyed by node id: the side to move, the list of move
        notations from the root, and the (parent id, action) pair of the first
        edge that reached the node (absent for the root).
    """
    by_id = {node["id"]: node for node in nodes}
    mover = {nodes[0]["id"]: root_player}
    path = {nodes[0]["id"]: []}
    parent = {}
    for node in nodes:
        for edge in node["edges"]:
            child = edge["child"]
            if child in by_id and child not in mover:
                mover[child] = -mover[node["id"]]
                path[child] = path[node["id"]] + [notation(edge["action"])]
                parent[child] = (node["id"], edge["action"])
    return mover, path, parent


def render(nodes: list[dict], root_player: int, header: list[str], root_board: np.ndarray) -> str:
    """
    Formats exported tree nodes as text.

    Parameters
    ----------
    nodes : list
        Output of get_tree.
    root_player : int
        Side to move at the root, 1 or -1.
    header : list
        Lines printed before the tree.
    root_board : numpy.ndarray
        Board at the root.

    Returns
    -------
    str
        The text of the tree.
    """
    by_id = {node["id"]: node for node in nodes}
    mover, path, _ = assign_paths(nodes, root_player)

    lines = list(header) + [""] + board_lines(root_board) + [""]
    for node in nodes:
        label = " ".join(path[node["id"]]) or "root"
        side = COLOR_NAMES[mover[node["id"]]]
        if node["terminal"]:
            lines.append(f"#{node['id']} [{label}] {side} to move, game over, outcome {node['terminal_value']:+.4f}")
            lines.append("")
            continue
        lines.append(f"#{node['id']} [{label}] {side} to move, N = {node['visits']}")
        for edge in sorted(node["edges"], key=lambda e: (-e["visits"], e["action"])):
            if edge["child"] < 0:
                target = "never followed"
            elif edge["child"] in by_id:
                here = path[node["id"]] + [notation(edge["action"])]
                target = f"-> #{edge['child']}" + ("" if path[edge["child"]] == here else f" (transposition, first listed as [{' '.join(path[edge['child']])}])")
            else:
                target = f"-> #{edge['child']} (below the printed depth)"
            lines.append(
                f"    {notation(edge['action']):>4s}  P {edge['prior']:.4f}  Q {edge['value']:+.4f}  N {edge['visits']:5d}  {target}"
            )
        lines.append("")
    return "\n".join(lines)


def search_tree(checkpoint: Path, moves: list[str], sims: int, max_depth: int, min_visits: int) -> dict:
    """
    Runs a C++ search with the trained network and exports its tree.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Trained checkpoint to export and load.
    moves : list
        Moves in board notation from the initial position to the searched position.
    sims : int
        Number of simulations.
    max_depth : int
        Deepest level whose nodes are listed.
    min_visits : int
        Edges with fewer visits are listed but not followed.

    Returns
    -------
    dict
        The searched `board` and `player`, the chosen `move`, the exported
        `nodes` and the `node_count` of the whole tree.
    """
    board, player = BU.get_init_board(), -1
    for text in moves:
        board, player = BU.get_next_state(board, player, action_from_notation(text))

    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    with tempfile.TemporaryDirectory() as directory:
        weights = Path(directory) / "weights.bin"
        weights.write_bytes(export_weights.pack_state_dict(state_dict))
        engine = othello_cpp.Engine(str(weights))

    probs = engine.get_action_probs(board, player, sims, temperature=0.0)
    return {
        "board": board,
        "player": player,
        "move": int(np.argmax(probs)),
        "nodes": engine.get_tree(board, player, max_depth, min_visits),
        "node_count": engine.node_count,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="text file to write")
    parser.add_argument("--moves", nargs="*", default=[], help="moves from the initial position, e.g. d3 c5")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-visits", type=int, default=5, help="edges with fewer visits are listed but not followed")
    parser.add_argument("--checkpoint", type=Path, default=BACKEND_DIR / "models" / f"model-{model_args.version}.pth.tar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    result = search_tree(ns.checkpoint, ns.moves, ns.sims, ns.max_depth, ns.min_visits)
    board, player, nodes = result["board"], result["player"], result["nodes"]
    header = [
        f"Search tree: {ns.sims} simulations, {COLOR_NAMES[player]} to move, network {model_args.version}",
        f"Position after: {' '.join(ns.moves) if ns.moves else 'no moves (initial position)'}",
        f"Chosen move: {notation(result['move'])}",
        f"Nodes listed: {len(nodes)} (depth up to {ns.max_depth}, edges followed from {ns.min_visits} visits), nodes in the whole tree: {result['node_count']}",
        "Q is the mean outcome for the side to move at that node, from -1 (loss) to +1 (win).",
        "P is the network's policy probability among the legal moves.",
    ]
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(render(nodes, player, header, board) + "\n")


if __name__ == "__main__":
    main()
