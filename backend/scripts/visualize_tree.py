"""
Runs a C++ search and draws the search tree below the root as an SVG image
and an interactive HTML page.

Every position is a card with a small board, the number of visits and the
mean outcome for black. The card border has the color of that outcome, and a
red ring marks the move that led to the position. Edges are moves. The best
line, which follows the most visited move at every level, is drawn in orange,
and dashed purple edges lead to a position first reached by another path. Only
the most visited moves of each position are drawn, and a card says how many it
leaves out. The HTML page adds panning, zooming and a larger board with the
move table of the hovered position. The page has a grid behind the tree, which
G switches off. The root is on top, or on the left with --layout horizontal.

Usage:
    python scripts/visualize_tree.py --help
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from print_tree import (
    BACKEND_DIR,
    COLOR_NAMES,
    model_args,
    notation,
    search_tree,
)

CARD_WIDTH = 92.0
BOARD_SIZE = 76.0
HEADER_HEIGHT = 26.0
FOOTER_HEIGHT = 46.0
CARD_HEIGHT = HEADER_HEIGHT + BOARD_SIZE + FOOTER_HEIGHT
COLUMN_GAP = 236.0
LEVEL_GAP = CARD_HEIGHT + 96.0
SIBLING_GAP = 28.0
ROW_GAP = 16.0
MARGIN = 36.0
TITLE_HEIGHT = 150.0

WHITE_ADVANTAGE = (217, 130, 43)
NEUTRAL = (244, 241, 236)
BLACK_ADVANTAGE = (47, 111, 176)
BEST_LINE = "#e8590c"
MAIN_EDGE = "#4a5568"
SIDE_EDGE = "#a0aec0"
TRANSPOSITION = "#8b5cf6"
BOARD_GREEN = "#2f855a"
GRID_GREEN = "#276749"
LAST_MOVE = "#e53e3e"
INK = "#1a202c"
MUTED = "#5a6677"


def mix(a: tuple, b: tuple, t: float) -> tuple:
    """
    Interpolates between two RGB colors.

    Parameters
    ----------
    a : tuple
        Color at t = 0.
    b : tuple
        Color at t = 1.
    t : float
        Position between the colors.

    Returns
    -------
    tuple
        The mixed color.
    """
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def outcome_color(black_outcome: float, scale: float) -> str:
    """
    Maps an outcome for black to a color.

    Parameters
    ----------
    black_outcome : float
        Mean outcome, from -1 (black loses) to +1 (black wins).
    scale : float
        Outcome that gets the strongest color.

    Returns
    -------
    str
        Hex color from orange through a neutral tone to blue.
    """
    t = max(-1.0, min(1.0, black_outcome / scale))
    rgb = mix(NEUTRAL, BLACK_ADVANTAGE, t) if t >= 0 else mix(NEUTRAL, WHITE_ADVANTAGE, -t)
    return "#" + "".join(f"{round(c):02x}" for c in rgb)


def board_string(node: dict, mover: int) -> str:
    """
    Writes a node's board as 64 characters.

    Parameters
    ----------
    node : dict
        Exported node.
    mover : int
        Side to move at the node, -1 (black) or 1 (white).

    Returns
    -------
    str
        "B" for black, "W" for white and "." for empty squares, row by row.
    """
    own = "B" if mover == -1 else "W"
    other = "W" if mover == -1 else "B"
    return "".join(own if (node["mine"] >> i) & 1 else other if (node["theirs"] >> i) & 1 else "." for i in range(64))


def select_tree(nodes: list[dict], root_player: int, top_children: int, max_depth: int, min_visits: int) -> dict:
    """
    Chooses the part of the exported tree that is drawn.

    Starting from the root, every position keeps its most visited moves, up to
    `top_children` of them, that have at least `min_visits` visits. A position
    reached by several kept moves belongs to the first one in breadth-first
    order, and the other moves are kept as transpositions.

    Parameters
    ----------
    nodes : list
        Output of get_tree.
    root_player : int
        Side to move at the root, 1 or -1.
    top_children : int
        Most moves kept per position.
    max_depth : int
        Deepest level drawn.
    min_visits : int
        Moves with fewer visits are dropped.

    Returns
    -------
    dict
        `order` (node ids in breadth-first order), `mover`, `depth`, `parent`
        (id and action of the first move that reached the node), `path`
        (move notations from the root), `children` (kept tree moves of each
        node), `transpositions` (kept moves to nodes that are drawn elsewhere)
        and `hidden` (number of visited moves each node leaves out).
    """
    by_id = {node["id"]: node for node in nodes}
    root = nodes[0]["id"]
    mover = {root: root_player}
    depth = {root: 0}
    parent: dict = {}
    path = {root: []}
    children = {root: []}
    transpositions = {root: []}
    hidden = {}
    order = [root]

    for node_id in order:
        node = by_id[node_id]
        children.setdefault(node_id, [])
        transpositions.setdefault(node_id, [])
        visited = sorted((e for e in node["edges"] if e["visits"] > 0), key=lambda e: (-e["visits"], e["action"]))
        kept = [e for e in visited if e["child"] in by_id and e["visits"] >= min_visits][:top_children]
        hidden[node_id] = len(visited) - len(kept)
        if depth[node_id] >= max_depth:
            hidden[node_id] = len(visited)
            continue
        for edge in kept:
            child = edge["child"]
            if child not in depth:
                depth[child] = depth[node_id] + 1
                mover[child] = -mover[node_id]
                parent[child] = (node_id, edge["action"])
                path[child] = path[node_id] + [notation(edge["action"])]
                children[node_id].append(edge)
                order.append(child)
            else:
                transpositions[node_id].append(edge)
    return {
        "order": order,
        "mover": mover,
        "depth": depth,
        "parent": parent,
        "path": path,
        "children": children,
        "transpositions": transpositions,
        "hidden": hidden,
    }


def build_graph(
    nodes: list[dict],
    root_player: int,
    top_children: int = 3,
    max_depth: int = 99,
    min_visits: int = 1,
    layout: str = "vertical",
    header: bool = True,
) -> dict:
    """
    Computes the layout and the display values of every card and edge.

    Parameters
    ----------
    nodes : list
        Output of get_tree.
    root_player : int
        Side to move at the root, 1 or -1.
    top_children : int, optional
        Most moves drawn per position.
    max_depth : int, optional
        Deepest level drawn.
    min_visits : int, optional
        Moves with fewer visits are not drawn.
    layout : str, optional
        "vertical" draws the root on top and each level below the previous one.
        "horizontal" draws the root on the left and each level to the right of
        the previous one.
    header : bool, optional
        Whether to leave room above the tree for the title and the legend of the
        standalone image. A page that shows them elsewhere should pass False.

    Returns
    -------
    dict
        `nodes`, `edges`, `width`, `height`, `scale` (outcome of the strongest
        card color), `pv` (the best line as move notations), `max_visits` and
        `layout`.
    """
    vertical = layout == "vertical"
    by_id = {node["id"]: node for node in nodes}
    chosen = select_tree(nodes, root_player, top_children, max_depth, min_visits)
    order, mover, depth, parent = chosen["order"], chosen["mover"], chosen["depth"], chosen["parent"]
    root = order[0]
    max_visits = max(by_id[root]["visits"], 1)

    # The legend sits beside the root in the vertical layout, so that layout starts lower.
    layout_top = (TITLE_HEIGHT + (40.0 if vertical else 0.0)) if header else MARGIN

    # Every subtree gets the room its children need across the layout, and a leaf the room of a card.
    # Across means vertically in the horizontal layout and horizontally in the vertical layout.
    leaf_room = CARD_WIDTH + SIBLING_GAP if vertical else CARD_HEIGHT + ROW_GAP
    rooms: dict = {}

    def subtree_room(node_id: int) -> float:
        stacked = sum(subtree_room(e["child"]) for e in chosen["children"][node_id])
        rooms[node_id] = max(leaf_room, stacked)
        return rooms[node_id]

    position: dict = {}

    def place(node_id: int, start: float) -> None:
        position[node_id] = start + rooms[node_id] / 2
        stacked = sum(rooms[e["child"]] for e in chosen["children"][node_id])
        cursor = start + (rooms[node_id] - stacked) / 2
        for edge in chosen["children"][node_id]:
            place(edge["child"], cursor)
            cursor += rooms[edge["child"]]

    total_room = subtree_room(root)
    place(root, 0.0)

    # Mean outcome for black: from the move that first reached the node, or from the game result.
    black_outcome: dict = {}
    for node_id in order:
        node = by_id[node_id]
        sign = 1 if mover[node_id] == -1 else -1
        if node["terminal"]:
            black_outcome[node_id] = sign * node["terminal_value"]
        elif node_id == root:
            visits = sum(e["visits"] for e in node["edges"])
            total = sum(e["visits"] * e["value"] for e in node["edges"])
            black_outcome[node_id] = sign * total / max(visits, 1)
        else:
            parent_id, action = parent[node_id]
            edge = next(e for e in by_id[parent_id]["edges"] if e["action"] == action)
            black_outcome[node_id] = (1 if mover[parent_id] == -1 else -1) * edge["value"]
    below_root = [abs(v) for k, v in black_outcome.items() if k != root]
    # The strongest color goes to the 85th percentile, so that a few extreme cards do not wash out the rest.
    ranked = sorted(below_root)
    typical = ranked[int(0.85 * (len(ranked) - 1))] if ranked else 0.1
    scale = max(0.1, math.ceil(typical * 20) / 20)

    # The best line follows the most visited drawn move of every position.
    pv_nodes = [root]
    while chosen["children"][pv_nodes[-1]]:
        top = chosen["children"][pv_nodes[-1]][0]
        pv_nodes.append(top["child"])
    pv_set = set(pv_nodes)

    # A card counts the simulations that passed through its position. A search tree counts only
    # the selections through a position, which leaves out the simulation that expanded it, so that
    # count is one lower. A finished game is never selected through, so it shows the visits of the
    # move that reached it.
    arrival_visits = {
        child: next(e for e in by_id[parent_id]["edges"] if e["action"] == action)["visits"]
        for child, (parent_id, action) in parent.items()
    }

    out_nodes = {}
    for node_id in order:
        node = by_id[node_id]
        legal_moves = sorted(node["edges"], key=lambda e: (-e["visits"], e["action"]))[:8]
        out_nodes[node_id] = {
            "id": node_id,
            "x": MARGIN + position[node_id] if vertical else MARGIN + CARD_WIDTH / 2 + depth[node_id] * COLUMN_GAP,
            "y": layout_top + CARD_HEIGHT / 2 + depth[node_id] * LEVEL_GAP if vertical else layout_top + position[node_id],
            "w": CARD_WIDTH,
            "h": CARD_HEIGHT,
            "depth": depth[node_id],
            "mover": mover[node_id],
            "visits": arrival_visits[node_id] if node["terminal"] and node_id in arrival_visits else node["visits"] + 1,
            "terminal": node["terminal"],
            "black": black_outcome[node_id],
            "color": outcome_color(black_outcome[node_id], scale),
            "label": " ".join(chosen["path"][node_id]) or "root",
            "board": board_string(node, mover[node_id]),
            "last_move": parent[node_id][1] if node_id in parent else None,
            "hidden": chosen["hidden"][node_id],
            "on_best_line": node_id in pv_set,
            "edges": [{"move": notation(e["action"]), "prior": e["prior"], "q": e["value"], "n": e["visits"]} for e in legal_moves],
            "parent": parent[node_id][0] if node_id in parent else None,
        }

    out_edges = []
    for node_id in order:
        source = out_nodes[node_id]
        drawn = [("tree", e) for e in chosen["children"][node_id]] + [("transposition", e) for e in chosen["transpositions"][node_id]]
        for kind, edge in drawn:
            target = out_nodes[edge["child"]]
            out_edges.append(
                {
                    "from": node_id,
                    "to": edge["child"],
                    "kind": kind,
                    "vertical": vertical,
                    "x1": source["x"] if vertical else source["x"] + CARD_WIDTH / 2,
                    "y1": source["y"] + CARD_HEIGHT / 2 if vertical else source["y"],
                    "x2": target["x"] if vertical else target["x"] - CARD_WIDTH / 2,
                    "y2": target["y"] - CARD_HEIGHT / 2 if vertical else target["y"],
                    "move": notation(edge["action"]),
                    "prior": edge["prior"],
                    "q": edge["value"],
                    "n": edge["visits"],
                    "width": 1.4 + 6.0 * math.sqrt(edge["visits"] / max_visits),
                    "best_line": kind == "tree" and node_id in pv_set and edge["child"] in pv_set,
                    "most_visited": kind == "tree" and edge is chosen["children"][node_id][0],
                    "action": edge["action"],
                }
            )

    levels = max(depth.values())
    if vertical:
        width = MARGIN * 2 + total_room
        height = layout_top + CARD_HEIGHT + levels * LEVEL_GAP + MARGIN
    else:
        width = MARGIN * 2 + CARD_WIDTH + levels * COLUMN_GAP
        height = layout_top + total_room + MARGIN
    return {
        "nodes": list(out_nodes.values()),
        "edges": out_edges,
        "width": width,
        "height": height,
        "scale": scale,
        "pv": [notation(parent[n][1]) for n in pv_nodes[1:]],
        "max_visits": max_visits,
        "layout": layout,
    }


def bezier(edge: dict, t: float) -> tuple:
    """
    Evaluates the horizontal cubic curve of an edge.

    Parameters
    ----------
    edge : dict
        Edge with its end points.
    t : float
        Position along the curve, from 0 to 1.

    Returns
    -------
    tuple
        The x and y coordinates of the point.
    """
    if edge["vertical"]:
        mid = (edge["y1"] + edge["y2"]) / 2
        xs = (edge["x1"], edge["x1"], edge["x2"], edge["x2"])
        ys = (edge["y1"], mid, mid, edge["y2"])
    else:
        mid = (edge["x1"] + edge["x2"]) / 2
        xs = (edge["x1"], mid, mid, edge["x2"])
        ys = (edge["y1"], edge["y1"], edge["y2"], edge["y2"])
    weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3)
    return sum(w * x for w, x in zip(weights, xs)), sum(w * y for w, y in zip(weights, ys))


def edge_path(edge: dict) -> str:
    """
    Builds the SVG path of an edge as a cubic curve that leaves and enters horizontally, or vertically in the vertical layout.

    Parameters
    ----------
    edge : dict
        Edge with its end points.

    Returns
    -------
    str
        Value of the path's `d` attribute.
    """
    if edge["vertical"]:
        mid = (edge["y1"] + edge["y2"]) / 2
        return f"M{edge['x1']:.1f},{edge['y1']:.1f} C{edge['x1']:.1f},{mid:.1f} {edge['x2']:.1f},{mid:.1f} {edge['x2']:.1f},{edge['y2']:.1f}"
    mid = (edge["x1"] + edge["x2"]) / 2
    return f"M{edge['x1']:.1f},{edge['y1']:.1f} C{mid:.1f},{edge['y1']:.1f} {mid:.1f},{edge['y2']:.1f} {edge['x2']:.1f},{edge['y2']:.1f}"


def board_svg(board: str, last_move: int | None, size: float) -> str:
    """
    Draws a board as SVG elements with its origin at (0, 0).

    Parameters
    ----------
    board : str
        64 characters: "B", "W" or ".", row by row.
    last_move : int or None
        Square to mark with a ring, or None.
    size : float
        Side of the board.

    Returns
    -------
    str
        SVG elements.
    """
    cell = size / 8
    grid = "".join(f"M{i * cell:.2f},0V{size:.2f}M0,{i * cell:.2f}H{size:.2f}" for i in range(1, 8))
    parts = [
        f'<rect width="{size:.1f}" height="{size:.1f}" rx="3" fill="{BOARD_GREEN}"/>',
        f'<path d="{grid}" stroke="{GRID_GREEN}" stroke-width="0.8" fill="none"/>',
    ]
    for i, square in enumerate(board):
        if square == ".":
            continue
        cx, cy = (i % 8 + 0.5) * cell, (i // 8 + 0.5) * cell
        fill = "#161616" if square == "B" else "#f7f7f7"
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{cell * 0.38:.2f}" fill="{fill}" stroke="#101010" stroke-width="0.5"/>')
    if last_move is not None and last_move < 64:
        cx, cy = (last_move % 8 + 0.5) * cell, (last_move // 8 + 0.5) * cell
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{cell * 0.47:.2f}" fill="none" stroke="{LAST_MOVE}" stroke-width="1.6"/>')
    return "".join(parts)


def legend_svg(scale: float) -> str:
    """
    Draws the legend that sits at the top right of the title area.

    Parameters
    ----------
    scale : float
        Outcome that gets the strongest card color.

    Returns
    -------
    str
        SVG elements.
    """
    steps = 11
    swatches = "".join(
        f'<rect x="{i * 26}" y="0" width="26" height="14" fill="{outcome_color(-scale + 2 * scale * i / (steps - 1), scale)}"/>'
        for i in range(steps)
    )
    return (
        f'<g class="legend" transform="translate({MARGIN + 560},58)" font-size="12" fill="{INK}">'
        '<text x="0" y="-10" font-weight="600">Card border and score: mean outcome for black</text>'
        f"{swatches}"
        f'<text x="0" y="32">{-scale:+.2f} white ahead</text>'
        f'<text x="143" y="32" text-anchor="middle">0</text>'
        f'<text x="286" y="32" text-anchor="end">{scale:+.2f} black ahead</text>'
        f'<circle cx="320" cy="7" r="6" fill="none" stroke="{LAST_MOVE}" stroke-width="1.8"/>'
        '<text x="334" y="11">last move</text>'
        f'<line x1="0" y1="62" x2="30" y2="62" stroke="{BEST_LINE}" stroke-width="5"/>'
        '<text x="38" y="66">best line (most visited move at every level)</text>'
        f'<line x1="0" y1="84" x2="30" y2="84" stroke="{TRANSPOSITION}" stroke-width="2" stroke-dasharray="5 4"/>'
        '<text x="38" y="88">position first reached by another path</text>'
        "</g>"
    )


def place_label(edge: dict, width: float, placed: list) -> tuple:
    """
    Chooses where an edge's label goes along the edge so that it does not cover another label.

    Parameters
    ----------
    edge : dict
        Edge with its end points.
    width : float
        Width of the label.
    placed : list
        Labels already placed, as (x, y, width) tuples. The new label is added.

    Returns
    -------
    tuple
        The x and y coordinates of the label's center. When every position along the
        edge is taken, the label goes to the first one.
    """
    height = 20.0
    candidates = [bezier(edge, t) for t in (0.68, 0.55, 0.8, 0.45, 0.9, 0.35)]
    for x, y in candidates:
        if all(abs(x - px) >= (width + pw) / 2 or abs(y - py) >= height for px, py, pw in placed):
            placed.append((x, y, width))
            return x, y
    placed.append((*candidates[0], width))
    return candidates[0]


def card_svg(n: dict) -> str:
    """
    Draws one position as a card.

    Parameters
    ----------
    n : dict
        Card from build_graph.

    Returns
    -------
    str
        An SVG group.
    """
    x0, y0 = n["x"] - n["w"] / 2, n["y"] - n["h"] / 2
    stroke_width = 4 if n["terminal"] else 3
    side = "black" if n["mover"] == -1 else "white"
    if n["terminal"]:
        score_text = "black wins" if n["black"] > 0.5 else "white wins" if n["black"] < -0.5 else "draw"
    else:
        score_text = f"black {n['black']:+.2f}"
    disc = "#161616" if n["mover"] == -1 else "#f7f7f7"
    header = "game over" if n["terminal"] else f"{side} to move"
    hidden = (
        f'<text x="{n["w"] / 2:.0f}" y="{HEADER_HEIGHT + BOARD_SIZE + 41:.0f}" font-size="9" fill="{MUTED}" text-anchor="middle">+{n["hidden"]} more moves</text>'
        if n["hidden"]
        else ""
    )
    tip = html.escape(
        f"#{n['id']} [{n['label']}]\n{side} to move, {n['visits']} visits\nmean outcome for black {n['black']:+.3f}"
        + ("\ngame over" if n["terminal"] else "")
    )
    board_x = (n["w"] - BOARD_SIZE) / 2
    return (
        f'<g class="node" id="n{n["id"]}" data-id="{n["id"]}" transform="translate({x0:.1f},{y0:.1f})">'
        f'<rect class="card" width="{n["w"]:.0f}" height="{n["h"]:.0f}" rx="10" fill="#ffffff" stroke="{n["color"]}" stroke-width="{stroke_width}"/>'
        f'<circle cx="12" cy="13" r="5" fill="{disc}" stroke="#101010" stroke-width="1"/>'
        f'<text x="21" y="17" font-size="10" fill="{INK}">{header}</text>'
        f'<g transform="translate({board_x:.1f},{HEADER_HEIGHT:.0f})">{board_svg(n["board"], n["last_move"], BOARD_SIZE)}</g>'
        f'<text x="{n["w"] / 2:.0f}" y="{HEADER_HEIGHT + BOARD_SIZE + 15:.0f}" font-size="11" font-weight="600" fill="{INK}" text-anchor="middle">{n["visits"]:,} visits</text>'
        f'<text x="{n["w"] / 2:.0f}" y="{HEADER_HEIGHT + BOARD_SIZE + 29:.0f}" font-size="10" fill="{MUTED}" text-anchor="middle">{score_text}</text>'
        f"{hidden}<title>{tip}</title></g>"
    )


def grid_svg(width: float, height: float) -> str:
    """
    Draws a grid with a fine line every 40 units and a stronger one every 200.

    Parameters
    ----------
    width : float
        Width of the drawing.
    height : float
        Height of the drawing.

    Returns
    -------
    str
        SVG elements. The grid extends 4000 units past the drawing on every side.
    """
    reach = 4000
    return (
        "<defs>"
        '<pattern id="grid-minor" width="40" height="40" patternUnits="userSpaceOnUse">'
        '<path d="M40,0H0V40" fill="none" stroke="#edf0f4" stroke-width="1"/></pattern>'
        '<pattern id="grid-major" width="200" height="200" patternUnits="userSpaceOnUse">'
        '<rect width="200" height="200" fill="url(#grid-minor)"/>'
        '<path d="M200,0H0V200" fill="none" stroke="#dbe0e8" stroke-width="1.4"/></pattern></defs>'
        f'<rect id="grid" x="{-reach}" y="{-reach}" width="{width + 2 * reach:.0f}" height="{height + 2 * reach:.0f}" fill="url(#grid-major)"/>'
    )


def svg_markup(graph: dict, title: str, subtitle: str = "", grid: bool = False, header: bool = True) -> str:
    """
    Draws the tree as SVG elements.

    Parameters
    ----------
    graph : dict
        Output of build_graph.
    title : str
        Heading drawn at the top.
    subtitle : str, optional
        Lines of text below the heading, separated by newlines.
    grid : bool, optional
        Whether to draw a grid behind the tree. The grid reaches well past the
        drawing, so it stays visible while the page is panned.
    header : bool, optional
        Whether to draw the title, the subtitle and the legend. A page that shows
        them elsewhere should pass False, together with False to build_graph.

    Returns
    -------
    str
        An `<svg>` element.
    """
    width = max(graph["width"], MARGIN * 2 + 1040) if header else graph["width"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {graph["height"]:.0f}" '
        f'width="{width:.0f}" height="{graph["height"]:.0f}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        *([grid_svg(width, graph["height"])] if grid else []),
        *(
            [
                f'<text x="{MARGIN}" y="44" font-size="20" font-weight="700" fill="{INK}">{html.escape(title)}</text>',
                *[
                    f'<text x="{MARGIN}" y="{68 + 19 * i}" font-size="13" fill="{MUTED}">{html.escape(line)}</text>'
                    for i, line in enumerate(subtitle.split("\n"))
                ],
                legend_svg(graph["scale"]),
            ]
            if header
            else []
        ),
        '<g class="edges" fill="none">',
    ]
    for e in graph["edges"]:
        if e["kind"] == "transposition":
            color, dash, opacity = TRANSPOSITION, ' stroke-dasharray="6 4"', 0.9
        elif e["best_line"]:
            color, dash, opacity = BEST_LINE, "", 1.0
        elif e["most_visited"]:
            color, dash, opacity = MAIN_EDGE, "", 0.9
        else:
            color, dash, opacity = SIDE_EDGE, "", 0.85
        tip = html.escape(f"{e['move']}: {e['n']} visits, prior {e['prior']:.3f}, mean outcome {e['q']:+.3f}")
        parts.append(
            f'<path class="edge" id="e{e["from"]}-{e["action"]}" data-from="{e["from"]}" data-to="{e["to"]}" d="{edge_path(e)}" '
            f'stroke="{color}" stroke-width="{e["width"]:.2f}" stroke-opacity="{opacity}"{dash}><title>{tip}</title></path>'
        )
    parts.append("</g>")
    parts.append('<g class="edge-labels" font-size="11" text-anchor="middle">')
    placed: list = []
    for e in graph["edges"]:
        label = f"{e['move']}  {e['n']:,}"
        w = 6.6 * len(label) + 12
        x, y = place_label(e, w, placed)
        fill, text, stroke = (BEST_LINE, "#ffffff", BEST_LINE) if e["best_line"] else ("#ffffff", INK, "#cbd5e0")
        parts.append(
            f'<g class="pill" transform="translate({x:.1f},{y:.1f})">'
            f'<rect x="{-w / 2:.1f}" y="-9" width="{w:.1f}" height="18" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            f'<text y="4" fill="{text}">{html.escape(label)}</text></g>'
        )
    parts.append("</g>")
    parts.append('<g class="nodes">')
    parts.extend(card_svg(n) for n in graph["nodes"])
    parts.append("</g></svg>")
    return "\n".join(parts)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #ffffff; --panel: #f7f8fa; --line: #e2e8f0; --ink: #1a202c; --muted: #5a6677; }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--ink); font-family: system-ui, sans-serif; }
  body { display: flex; flex-direction: column; }
  #top { flex: none; display: flex; align-items: center; justify-content: space-between; gap: 32px; padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--line); }
  #top h1 { font-size: 18px; margin: 0 0 4px; }
  #top p { margin: 0; font-size: 12px; line-height: 1.5; color: var(--muted); }
  .legend { display: flex; align-items: center; gap: 36px; font-size: 12px; }
  .legend .caption { font-weight: 600; margin-bottom: 6px; }
  .ramp { width: 330px; }
  .ramp .caption { white-space: nowrap; }
  .ramp .bar { height: 12px; border-radius: 3px; background: linear-gradient(to right, rgb(217,130,43), rgb(244,241,236), rgb(47,111,176)); }
  .ramp .ticks { display: flex; justify-content: space-between; gap: 12px; margin-top: 4px; color: var(--muted); white-space: nowrap; }
  .keys { display: flex; flex-direction: column; gap: 5px; }
  .key { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
  .key i { display: inline-block; width: 26px; flex: none; }
  .key .ring { width: 12px; height: 12px; margin: 0 7px; border: 2px solid #e53e3e; border-radius: 50%; box-sizing: border-box; }
  .key .best { height: 5px; background: #e8590c; border-radius: 3px; }
  .key .dash { height: 0; border-top: 2px dashed #8b5cf6; }
  .layout { display: flex; flex: 1; min-height: 0; }
  #canvas { flex: 1; min-width: 0; overflow: hidden; cursor: grab; background: #ffffff; }
  #canvas.dragging { cursor: grabbing; }
  #canvas svg { width: 100%; height: 100%; display: block; }
  aside { width: 330px; flex: none; padding: 16px; background: var(--panel); border-left: 1px solid var(--line); overflow: auto; box-sizing: border-box; }
  aside h2 { font-size: 15px; margin: 0 0 4px; }
  aside p { color: var(--muted); font-size: 13px; margin: 4px 0 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 3px 6px; text-align: right; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  #board { width: 100%; max-width: 298px; display: block; margin: 8px 0 12px; border-radius: 4px; }
  .node { cursor: pointer; }
  .node:hover > .card { stroke: #e53e3e; stroke-width: 4; }
  svg.focus .edge:not(.hl), svg.focus .pill:not(.hl) { opacity: 0.12; }
  svg.focus .node:not(.hl) { opacity: 0.4; }
  .edge.hl { stroke-opacity: 1 !important; }
  .node.pinned > .card { stroke: #e53e3e; stroke-width: 4; }
  .hint { font-size: 12px; color: var(--muted); margin-top: 12px; }
</style>
</head>
<body>
<header id="top">
  <div>
    <h1>__TITLE__</h1>
    <p>__SUBTITLE__</p>
  </div>
  __LEGEND__
</header>
<div class="layout">
  <div id="canvas">__SVG__</div>
  <aside>
    <h2 id="title"></h2>
    <p id="subtitle"></p>
    <svg id="board" viewBox="0 0 80 80"></svg>
    <table><thead><tr><th>Move</th><th>P</th><th>Q</th><th>N</th></tr></thead><tbody id="moves"></tbody></table>
    <p class="hint">Hover a card to see its position and moves. Click to pin it. Drag to pan, scroll to zoom, double-click to reset, press F to fit the whole tree and G to toggle the grid.</p>
  </aside>
</div>
<script id="tree-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('tree-data').textContent);
  const byId = new Map(data.nodes.map(n => [n.id, n]));
  const canvas = document.getElementById('canvas');
  const svg = canvas.querySelector('svg');
  const full = svg.viewBox.baseVal;
  const home = { x: full.x, y: full.y, width: full.width, height: full.height };
  let view = { ...home };
  const applyView = () => svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.width} ${view.height}`);
  svg.removeAttribute('width'); svg.removeAttribute('height');
  const rootNode = data.nodes[0];
  const fitAll = () => {
    const box = canvas.getBoundingClientRect();
    const scale = Math.min(box.width / home.width, box.height / home.height);
    view = { x: 0, y: 0, width: box.width / scale, height: box.height / scale };
    applyView();
  };
  // Start at a readable zoom, centered on the root, so that a tall tree does not shrink to a thumbnail.
  const fit = () => {
    const box = canvas.getBoundingClientRect();
    const scale = Math.max(Math.min(box.width / home.width, box.height / home.height), 0.55);
    const width = box.width / scale;
    const height = box.height / scale;
    if (data.layout === 'vertical') {
      const x = Math.max(0, Math.min(rootNode.x - width / 2, home.width - width));
      view = { x: width >= home.width ? 0 : x, y: 0, width, height };
    } else {
      const y = Math.max(0, Math.min(rootNode.y - height / 2, home.height - height));
      view = { x: 0, y: height >= home.height ? 0 : y, width, height };
    }
    applyView();
  };
  fit(); window.addEventListener('resize', fit);
  const grid = svg.getElementById('grid');
  window.addEventListener('keydown', (event) => {
    if (event.key === 'f' || event.key === 'F') fitAll();
    if ((event.key === 'g' || event.key === 'G') && grid) grid.style.display = grid.style.display === 'none' ? '' : 'none';
  });

  const toSvg = (event) => {
    const box = svg.getBoundingClientRect();
    return { x: view.x + (event.clientX - box.left) / box.width * view.width,
             y: view.y + (event.clientY - box.top) / box.height * view.height };
  };
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const point = toSvg(event);
    const factor = Math.exp(event.deltaY * 0.0015);
    view = { x: point.x - (point.x - view.x) * factor, y: point.y - (point.y - view.y) * factor,
             width: view.width * factor, height: view.height * factor };
    applyView();
  }, { passive: false });
  let drag = null;
  canvas.addEventListener('mousedown', (event) => { drag = { x: event.clientX, y: event.clientY, view: { ...view } }; canvas.classList.add('dragging'); });
  window.addEventListener('mouseup', () => { drag = null; canvas.classList.remove('dragging'); });
  window.addEventListener('mousemove', (event) => {
    if (!drag) return;
    const box = svg.getBoundingClientRect();
    view.x = drag.view.x - (event.clientX - drag.x) / box.width * view.width;
    view.y = drag.view.y - (event.clientY - drag.y) / box.height * view.height;
    applyView();
  });
  canvas.addEventListener('dblclick', fit);

  const boardSvg = document.getElementById('board');
  const drawBoard = (node) => {
    let markup = '<rect width="80" height="80" rx="3" fill="#2f855a"/>';
    for (let i = 1; i < 8; i++) markup += `<path d="M${i * 10},0V80M0,${i * 10}H80" stroke="#276749" stroke-width="0.6"/>`;
    for (let i = 0; i < 64; i++) {
      const c = node.board[i];
      if (c === '.') continue;
      markup += `<circle cx="${(i % 8) * 10 + 5}" cy="${Math.floor(i / 8) * 10 + 5}" r="3.9" fill="${c === 'B' ? '#161616' : '#f7f7f7'}" stroke="#101010" stroke-width="0.4"/>`;
    }
    if (node.last_move !== null && node.last_move < 64) {
      markup += `<circle cx="${(node.last_move % 8) * 10 + 5}" cy="${Math.floor(node.last_move / 8) * 10 + 5}" r="4.7" fill="none" stroke="#e53e3e" stroke-width="1.1"/>`;
    }
    boardSvg.innerHTML = markup;
  };
  const sides = { '-1': 'black', '1': 'white' };
  const show = (id) => {
    const node = byId.get(id);
    document.getElementById('title').textContent = `#${node.id} [${node.label}]`;
    document.getElementById('subtitle').textContent =
      `${sides[node.mover]} to move, ${node.visits} visits, mean outcome for black ${node.black >= 0 ? '+' : ''}${node.black.toFixed(3)}` + (node.terminal ? ', game over' : '');
    drawBoard(node);
    document.getElementById('moves').innerHTML = node.edges.map(e =>
      `<tr><td>${e.move}</td><td>${e.prior.toFixed(3)}</td><td>${e.q >= 0 ? '+' : ''}${e.q.toFixed(3)}</td><td>${e.n}</td></tr>`).join('');
  };
  const clearHighlight = () => {
    svg.classList.remove('focus');
    svg.querySelectorAll('.hl').forEach(el => el.classList.remove('hl'));
  };
  const highlight = (id) => {
    clearHighlight();
    svg.classList.add('focus');
    let current = id;
    while (current !== null && current !== undefined) {
      const node = byId.get(current);
      svg.getElementById('n' + current).classList.add('hl');
      if (node.parent === null) break;
      svg.querySelectorAll(`.edge[data-from="${node.parent}"][data-to="${current}"]`).forEach(edge => edge.classList.add('hl'));
      current = node.parent;
    }
  };

  let pinned = null;
  const root = data.nodes[0].id;
  show(root);
  svg.addEventListener('mouseover', (event) => {
    const group = event.target.closest('.node');
    if (!group) return;
    const id = Number(group.dataset.id);
    show(id); highlight(id);
  });
  svg.addEventListener('mouseout', (event) => {
    if (!event.target.closest('.node')) return;
    if (pinned === null) { clearHighlight(); show(root); } else { show(pinned); highlight(pinned); }
  });
  svg.addEventListener('click', (event) => {
    const group = event.target.closest('.node');
    svg.querySelectorAll('.pinned').forEach(el => el.classList.remove('pinned'));
    if (!group || Number(group.dataset.id) === pinned) { pinned = null; return; }
    pinned = Number(group.dataset.id);
    group.classList.add('pinned');
  });
})();
</script>
</body>
</html>
"""


def legend_html(scale: float) -> str:
    """
    Builds the legend of the page's top bar.

    Parameters
    ----------
    scale : float
        Outcome that gets the strongest card color.

    Returns
    -------
    str
        HTML element with the color scale and the meaning of the marks.
    """
    return (
        '<div class="legend">'
        '<div class="ramp"><div class="caption">Card border and score: mean outcome for black</div>'
        '<div class="bar"></div>'
        f'<div class="ticks"><span>{-scale:+.2f} white ahead</span><span>0</span><span>{scale:+.2f} black ahead</span></div></div>'
        '<div class="keys">'
        '<span class="key"><i class="ring"></i>last move</span>'
        '<span class="key"><i class="best"></i>best line, the most visited move at every level</span>'
        '<span class="key"><i class="dash"></i>position first reached by another path</span>'
        "</div></div>"
    )


def html_page(graph: dict, svg: str, title: str, subtitle: str = "") -> str:
    """
    Builds the interactive page around the SVG.

    The title, the subtitle and the legend sit in a bar above the drawing, so
    they stay in place while the drawing is panned and zoomed.

    Parameters
    ----------
    graph : dict
        Output of build_graph.
    svg : str
        The drawing from svg_markup, made without a header.
    title : str
        Page title.
    subtitle : str, optional
        Lines of text below the title, separated by newlines.

    Returns
    -------
    str
        HTML document.
    """
    data = {"layout": graph["layout"], "nodes": graph["nodes"]}
    payload = json.dumps(data).replace("</", "<\\/")
    return (
        PAGE.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", "<br>".join(html.escape(line) for line in subtitle.split("\n")))
        .replace("__LEGEND__", legend_html(graph["scale"]))
        .replace("__SVG__", svg)
        .replace("__DATA__", payload)
    )


def describe_position(moves: list[str]) -> str:
    """
    Describes the searched position by the moves that led to it.

    Parameters
    ----------
    moves : list
        Moves in board notation from the initial position.

    Returns
    -------
    str
        The moves themselves when there are few, otherwise their count and the last ones.
    """
    if not moves:
        return "the initial position"
    if len(moves) <= 8:
        return " ".join(moves)
    return f"{len(moves)} moves, ending {' '.join(moves[-6:])}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="path without extension, writes <output>.svg and <output>.html")
    parser.add_argument("--moves", nargs="*", default=[], help="moves from the initial position, e.g. d3 c5")
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=4, help="deepest level drawn")
    parser.add_argument("--top-children", type=int, default=3, help="most visited moves drawn per position")
    parser.add_argument("--min-visits", type=int, default=5, help="moves with fewer visits are not drawn")
    parser.add_argument(
        "--layout", choices=["vertical", "horizontal"], default="vertical",
        help="vertical puts the root on top, horizontal puts it on the left",
    )
    parser.add_argument("--checkpoint", type=Path, default=BACKEND_DIR / "models" / f"model-{model_args.version}.pth.tar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    # The search lists every visited node, and the drawing keeps the top moves of each.
    result = search_tree(ns.checkpoint, ns.moves, ns.sims, ns.max_depth + 1, 1)
    nodes, player = result["nodes"], result["player"]
    graph = build_graph(nodes, player, ns.top_children, ns.max_depth, ns.min_visits, ns.layout)
    page_graph = build_graph(nodes, player, ns.top_children, ns.max_depth, ns.min_visits, ns.layout, header=False)
    title = f"MCTS tree after {describe_position(ns.moves)}"
    subtitle = (
        f"{ns.sims} simulations, {COLOR_NAMES[result['player']]} to move, chosen move {notation(result['move'])}\n"
        f"Best line: {' '.join(graph['pv']) or 'none'}\n"
        f"Top {ns.top_children} moves per position, down to depth {ns.max_depth}"
    )
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    svg = svg_markup(graph, title, subtitle)
    Path(f"{ns.output}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n")
    page_svg = svg_markup(page_graph, title, subtitle, grid=True, header=False)
    Path(f"{ns.output}.html").write_text(html_page(page_graph, page_svg, title, subtitle))


if __name__ == "__main__":
    main()
