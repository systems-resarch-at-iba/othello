"""
Tests for the C++-backed engine behind the HTTP endpoints, using a randomly
initialized network. Skipped when the C++ extension has not been built.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))
pytest.importorskip("othello_cpp")

import main  # noqa: E402
from arch import CNN_vAlpha3, args  # noqa: E402
from board import Utils as BU  # noqa: E402
from engine import CppAI  # noqa: E402
from main import app  # noqa: E402
from utils import dotdict  # noqa: E402

MID_GAME = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28
LEGAL_SQUARES = {21, 29, 33, 41}


@pytest.fixture(scope="module")
def cpp_ai():
    torch.manual_seed(0)
    model = CNN_vAlpha3(dotdict({**args, "cuda": False}))
    return CppAI.from_state_dict(model.state_dict(), dotdict({**args, "num_mcts_sims": 24}))


@pytest.fixture
def client(cpp_ai):
    with TestClient(app) as c:
        main.ai = cpp_ai
        yield c


def test_select_move_returns_a_legal_move(cpp_ai):
    board = np.array(MID_GAME, dtype=np.int8).reshape(8, 8)
    for player in (1, -1):
        move = cpp_ai.select_move(board, player)
        assert BU.get_valid_moves(board, player)[move] == 1


def test_parameter_count_matches_the_network(cpp_ai):
    assert cpp_ai.parameter_count == sum(p.numel() for p in CNN_vAlpha3(dotdict({**args, "cuda": False})).parameters())


def test_health_reports_ok(client, cpp_ai):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["parameter_count"] == cpp_ai.parameter_count


def test_move_and_hints_endpoints(client):
    response = client.post("/move", json={"board": MID_GAME, "player": 1, "num_mcts_sims": 16})
    assert response.status_code == 200
    assert response.json()["move"] in LEGAL_SQUARES
    response = client.post("/hints", json={"board": MID_GAME, "player": 1, "num_mcts_sims": 16})
    probs = response.json()["probs"]
    assert len(probs) == 65 and abs(sum(probs) - 1.0) < 1e-6
    assert all(p == 0 for i, p in enumerate(probs) if i not in LEGAL_SQUARES)


@pytest.mark.parametrize("cpuct_type", ["static", "increment", "decrement"])
def test_each_cpuct_type_is_accepted(client, cpuct_type):
    response = client.post(
        "/move", json={"board": MID_GAME, "player": 1, "num_mcts_sims": 8, "cpuct_type": cpuct_type}
    )
    assert response.status_code == 200


def test_a_board_with_an_invalid_square_value_is_rejected(client):
    response = client.post("/move", json={"board": [3] + [0] * 63, "player": 1})
    assert response.status_code == 422


def test_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        CppAI(dotdict(dict(args)), version="none", model_dir=str(tmp_path))
