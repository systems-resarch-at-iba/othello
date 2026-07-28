from typing import Literal
from pydantic import BaseModel, Field


class _EngineParams(BaseModel):
    # Capped well above the default (500) so a user cranking this up from the
    # UI can't turn one request into a multi-minute search on a shared,
    # unauthenticated, free-tier CPU backend.
    num_mcts_sims: int = Field(default=500, ge=1, le=2000)
    # See MCTS.get_dynamic_cpuct: 'static' uses a fixed exploration constant,
    # 'increment'/'decrement' scale it based on how many times the current
    # position has been visited, approaching/leaving c_puct as a bound.
    cpuct_type: Literal["static", "increment", "decrement"] = "static"
    # arch.py's own default; bounded to a range that stays a real exploration
    # constant rather than degenerating into pure exploitation (~0) or search
    # that never commits to a line (>5).
    c_puct: float = Field(default=1.5, gt=0, le=5)


class MoveRequest(_EngineParams):
    # Row-major flat board, index = row*8 + col (matches OthelloBitBoard's own
    # index convention). Values: -1 = black, 0 = empty, 1 = white.
    board: list[int] = Field(min_length=64, max_length=64)
    # 1 = white, -1 = black, matching OthelloAI.select_move's own contract.
    player: Literal[1, -1]


class MoveResponse(BaseModel):
    # 0-63 for a board square, or 64 for pass.
    move: int
    # Wall-clock time spent inside the MCTS search itself, not counting
    # request/response transport, so "how long did the AI take" reflects
    # think time even over a slow network.
    elapsed_ms: float


class HintRequest(_EngineParams):
    board: list[int] = Field(min_length=64, max_length=64)
    player: Literal[1, -1]


class HintResponse(BaseModel):
    # MCTS visit-count distribution over all 65 actions (0-63 board squares,
    # row-major, plus 64 for pass) from a fresh search at the given position:
    # how much of its search budget the engine spent on each candidate move,
    # not the policy network's raw prior. Same indexing as MoveRequest.board,
    # so the frontend can zip probs[0:64] directly against board squares.
    probs: list[float]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    # e.g. "Alpha3S": arch.py's model_name for whichever args built the
    # currently-loaded network, not a hardcoded label, so this stays correct
    # if the served checkpoint changes.
    architecture: str
    # Real parameter count of the loaded network (sum of tensor.numel() over
    # net.parameters()), not a documented/assumed figure.
    parameter_count: int
