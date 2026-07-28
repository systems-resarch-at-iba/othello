"""
Endpoint tests exercise the request/response contract (shapes, status
codes, field types) against a randomly-initialized network, not the real
trained checkpoint: main.ai is overridden right after the app's own
lifespan finishes starting up. Whether the model plays well isn't in scope
here, whether the endpoint plumbing is right, is -- a fast, untrained
network is enough for that, and it means these run the same in CI as they
do with the real weights installed. Only the "not loaded" tests below care
about main.ai actually being None.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from arch import CNN_vAlpha3, args
from board import Utils as BU
from engine.mcts import MCTS
from engine.othello_cnn import OthelloCNN
from main import app


class _RandomAI:
    """Same select_move contract as OthelloAI, backed by a randomly
    initialized network instead of a loaded checkpoint."""

    def __init__(self) -> None:
        cnn = OthelloCNN(CNN_vAlpha3, args)
        self.mcts = MCTS(cnn, args)

    def select_move(self, board_state: np.ndarray, player: int = 1) -> int:
        canonical_board = BU.get_canonical_form(board_state, player)
        action_probs = self.mcts.get_action_probs(canonical_board, temperature=0)
        return int(np.argmax(action_probs))


@pytest.fixture
def client():
    with TestClient(app) as c:
        main.ai = _RandomAI()
        yield c


def test_health_reports_ok_when_a_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == args.version
    assert body["architecture"] == args.model_name
    assert body["parameter_count"] > 0


def test_health_reports_not_loaded_when_no_model_is_loaded():
    with TestClient(app) as c:
        main.ai = None
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "model not loaded"


def test_move_returns_a_legal_looking_move(client):
    board = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28  # a real mid-game-ish position
    response = client.post("/move", json={"board": board, "player": 1})
    assert response.status_code == 200
    move = response.json()["move"]
    assert 0 <= move <= 64  # 0-63 a board square, 64 = pass


def test_move_rejects_a_board_of_the_wrong_length(client):
    response = client.post("/move", json={"board": [0] * 63, "player": 1})
    assert response.status_code == 422


def test_move_rejects_an_invalid_player_value(client):
    response = client.post("/move", json={"board": [0] * 64, "player": 0})
    assert response.status_code == 422


def test_move_accepts_a_low_sim_count_and_still_returns_a_legal_looking_move(client):
    board = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28
    response = client.post("/move", json={"board": board, "player": 1, "num_mcts_sims": 1})
    assert response.status_code == 200
    assert 0 <= response.json()["move"] <= 64


def test_move_accepts_each_cpuct_type(client):
    board = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28
    for cpuct_type in ["static", "increment", "decrement"]:
        response = client.post(
            "/move",
            json={"board": board, "player": 1, "num_mcts_sims": 1, "cpuct_type": cpuct_type},
        )
        assert response.status_code == 200, cpuct_type


def test_move_rejects_an_out_of_range_sim_count(client):
    response = client.post("/move", json={"board": [0] * 64, "player": 1, "num_mcts_sims": 0})
    assert response.status_code == 422
    response = client.post("/move", json={"board": [0] * 64, "player": 1, "num_mcts_sims": 5000})
    assert response.status_code == 422


def test_move_rejects_an_invalid_cpuct_type(client):
    response = client.post("/move", json={"board": [0] * 64, "player": 1, "cpuct_type": "not-a-real-type"})
    assert response.status_code == 422


def test_move_reports_elapsed_ms(client):
    board = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28
    response = client.post("/move", json={"board": board, "player": 1, "num_mcts_sims": 1})
    assert response.status_code == 200
    assert response.json()["elapsed_ms"] >= 0


def test_move_rejects_an_out_of_range_c_puct(client):
    response = client.post("/move", json={"board": [0] * 64, "player": 1, "c_puct": 0})
    assert response.status_code == 422
    response = client.post("/move", json={"board": [0] * 64, "player": 1, "c_puct": 5.1})
    assert response.status_code == 422


def test_hints_returns_a_probability_distribution_over_all_actions(client):
    board = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28
    response = client.post("/hints", json={"board": board, "player": 1, "num_mcts_sims": 4})
    assert response.status_code == 200
    body = response.json()
    assert len(body["probs"]) == 65  # 64 board squares + pass
    assert body["probs"][40] == 0  # empty but not a legal move here, so masked out
    assert abs(sum(body["probs"]) - 1.0) < 1e-6
    assert body["elapsed_ms"] >= 0


def test_hints_puts_all_probability_mass_on_legal_squares(client):
    board = [0] * 27 + [1, -1, 0, 0, 0, 0, 0, -1, 1] + [0] * 28
    response = client.post("/hints", json={"board": board, "player": 1, "num_mcts_sims": 4})
    legal_indices = {21, 29, 33, 41}  # verified via BU.get_valid_moves on this exact board
    probs = response.json()["probs"]
    for i, p in enumerate(probs):
        if i not in legal_indices:
            assert p == 0, f"index {i} should have zero probability"


def test_move_returns_503_when_no_model_is_loaded():
    with TestClient(app) as c:
        main.ai = None
        response = c.post("/move", json={"board": [0] * 64, "player": 1})
        assert response.status_code == 503


def test_hints_returns_503_when_no_model_is_loaded():
    with TestClient(app) as c:
        main.ai = None
        response = c.post("/hints", json={"board": [0] * 64, "player": 1})
        assert response.status_code == 503
