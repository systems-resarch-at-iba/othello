"""
Tests for scripts/visualize_tree.py on a tree searched with the stub
evaluator. Skipped when the C++ extension has not been built.
"""
import json
import re
import sys
import xml.dom.minidom
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
othello_cpp = pytest.importorskip("othello_cpp")

import visualize_tree  # noqa: E402
from board import Utils as BU  # noqa: E402


@pytest.fixture(scope="module")
def searched():
    engine = othello_cpp.StubEngine()
    board, player = BU.get_init_board(), -1
    engine.get_action_probs(board, player, 800)
    return engine.get_tree(board, player, max_depth=5, min_visits=1), player


def test_the_drawing_keeps_the_most_visited_moves_of_each_position(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=2, max_depth=3, min_visits=1, layout="horizontal")
    by_id = {n["id"]: n for n in nodes}
    drawn = {n["id"] for n in graph["nodes"]}
    assert nodes[0]["id"] in drawn
    for node in graph["nodes"]:
        assert node["depth"] <= 3
        outgoing = [e for e in graph["edges"] if e["from"] == node["id"] and e["kind"] == "tree"]
        assert len(outgoing) <= 2
        visits = sorted((e["visits"] for e in by_id[node["id"]]["edges"] if e["visits"] > 0), reverse=True)
        assert node["hidden"] >= len(visits) - 2 if node["depth"] < 3 else node["hidden"] == len(visits)
    assert all(e["from"] in drawn and e["to"] in drawn for e in graph["edges"])


def test_the_best_line_follows_the_most_visited_move(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=3, max_depth=4, min_visits=1)
    best = [e for e in graph["edges"] if e["best_line"]]
    assert len(best) == len(graph["pv"])
    root_edges = sorted((e for e in nodes[0]["edges"] if e["visits"] > 0), key=lambda e: (-e["visits"], e["action"]))
    assert best[0]["n"] == root_edges[0]["visits"]


def test_cards_do_not_overlap_in_a_column_of_the_horizontal_layout(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=3, max_depth=4, min_visits=1, layout="horizontal")
    columns = {}
    for node in graph["nodes"]:
        columns.setdefault(node["x"], []).append(node)
    for column in columns.values():
        column.sort(key=lambda n: n["y"])
        for upper, lower in zip(column, column[1:]):
            assert lower["y"] - upper["y"] >= upper["h"]


def test_card_colors_use_the_outcome_scale():
    scale = 0.2
    assert visualize_tree.outcome_color(0.0, scale) == "#f4f1ec"
    assert visualize_tree.outcome_color(scale, scale) == visualize_tree.outcome_color(5.0, scale)
    assert visualize_tree.outcome_color(-scale, scale) != visualize_tree.outcome_color(scale, scale)


def test_svg_is_well_formed_and_html_embeds_the_data(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=2, max_depth=3, min_visits=1)
    svg = visualize_tree.svg_markup(graph, "test tree", "first line\nsecond line")
    document = xml.dom.minidom.parseString(svg)
    assert len(document.getElementsByTagName("g")) > len(graph["nodes"])
    page = visualize_tree.html_page(graph, svg, "test tree")
    payload = re.search(r'<script id="tree-data" type="application/json">(.*?)</script>', page, re.S).group(1)
    data = json.loads(payload.replace("<\\/", "</"))
    assert len(data["nodes"]) == len(graph["nodes"])
    assert page.count("<svg") >= 1


def test_vertical_layout_is_the_default_and_puts_levels_below_each_other(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=3, max_depth=3, min_visits=1)
    assert graph["layout"] == "vertical"
    levels = {}
    for node in graph["nodes"]:
        levels.setdefault(node["depth"], []).append(node)
    ys = [levels[d][0]["y"] for d in sorted(levels)]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    for level in levels.values():
        assert len({n["y"] for n in level}) == 1
        level.sort(key=lambda n: n["x"])
        for left, right in zip(level, level[1:]):
            assert right["x"] - left["x"] >= left["w"]
    assert all(e["vertical"] and e["y2"] > e["y1"] for e in graph["edges"] if e["kind"] == "tree")


def test_page_has_a_top_bar_with_the_legend_and_a_grid(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=2, max_depth=3, min_visits=1, header=False)
    svg = visualize_tree.svg_markup(graph, "test tree", "line one\nline two", grid=True, header=False)
    assert "test tree" not in svg and 'id="grid"' in svg and "legend" not in svg
    page = visualize_tree.html_page(graph, svg, "test tree", "line one\nline two")
    assert '<header id="top">' in page
    assert "<h1>test tree</h1>" in page and "line one<br>line two" in page
    assert "white ahead" in page and "black ahead" in page
    assert page.index('<header id="top">') < page.index('<div id="canvas">')


def test_standalone_svg_keeps_the_title_and_the_legend(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=2, max_depth=3, min_visits=1)
    svg = visualize_tree.svg_markup(graph, "test tree", "line one")
    assert "test tree" in svg and "legend" in svg and 'id="grid"' not in svg


def test_the_svg_has_a_transparent_background(searched):
    nodes, player = searched
    graph = visualize_tree.build_graph(nodes, player, top_children=2, max_depth=3, min_visits=1)
    svg = visualize_tree.svg_markup(graph, "test tree", "line one")
    assert 'width="100%"' not in svg
    page = visualize_tree.html_page(graph, visualize_tree.svg_markup(graph, "t", header=False, grid=True), "t")
    assert "#canvas { flex: 1; min-width: 0; overflow: hidden; cursor: grab; background: #ffffff; }" in page
