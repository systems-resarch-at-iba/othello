import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from arch import CNN_vAlpha3, args
from board import Utils as BU
from engine import CppAI, OthelloAI
from engine.cpp_ai import MAX_TREE_NODES
from schemas import HealthResponse, HintRequest, HintResponse, MoveRequest, MoveResponse

ai: OthelloAI | CppAI | None = None

# Uvicorn's own logger, so that these lines reach the server log (Cloud
# Logging on Cloud Run) at INFO level without extra logging configuration.
logger = logging.getLogger("uvicorn.error")


def _log_engine(engine: OthelloAI | CppAI | None) -> None:
    """
    Logs which engine serves requests and which CPU path it uses.

    Parameters
    ----------
    engine : OthelloAI or CppAI or None
        The engine chosen at startup, or None when no model is loaded.
    """
    if engine is None:
        logger.error("engine: none, the model checkpoint was not found, so /move and /hints return 503")
    elif isinstance(engine, CppAI):
        logger.info(
            "engine: C++ (instruction set %s, batch size %d, tree limit %d positions)",
            engine.isa,
            engine.mcts.batch_size,
            MAX_TREE_NODES,
        )
        if engine.isa != "avx2":
            logger.warning(
                "the C++ engine runs its plain-loop network because this CPU lacks AVX2, FMA or F16C; "
                "searches are much slower than with AVX2"
            )
    else:
        logger.info("engine: Python (single-threaded, the slower fallback)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup and shared across all requests. MCTS's own
    # Qsa/Nsa/Ns/Ps/Es/Vs dicts are keyed by Zobrist hash of the position, so
    # sharing them across concurrent games behaves like a shared transposition
    # table, not a correctness bug, and avoids one CNN copy in memory per game.
    #
    # The C++ engine serves requests when its extension is built, and the
    # Python engine serves them otherwise. Both load the same checkpoint.
    global ai
    try:
        ai = CppAI(args=args, version=args.version)
    except ImportError as error:
        logger.warning("the C++ engine is unavailable (%s), falling back to the Python engine", error)
        try:
            ai = OthelloAI(cnn=CNN_vAlpha3, args=args, version=args.version)
        except FileNotFoundError:
            ai = None
    except FileNotFoundError:
        ai = None
    _log_engine(ai)
    yield


app = FastAPI(lifespan=lifespan)

# Site's production domain isn't finalized yet (see site/app/layout.tsx's own
# metadataBase TODO), so this reads from an env var rather than hardcoding it.
# Defaults to the Next.js dev server's own default port so `npm run dev` in
# site/ works against a locally-running backend with zero extra config;
# override ALLOWED_ORIGINS explicitly for any real deployment.
_default_origins = "http://localhost:3000"
allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> HealthResponse:
    if ai is None:
        return HealthResponse(
            status="model not loaded",
            model_version=args.version,
            architecture=args.model_name,
            parameter_count=0,
        )

    # ai.mcts.cnn is the OthelloCNN wrapper; .cnn on *that* is the actual
    # nn.Module (CNN_vAlpha3) -- counts the real loaded network's parameters
    # rather than assuming a figure.
    parameter_count = (
        ai.parameter_count
        if isinstance(ai, CppAI)
        else sum(p.numel() for p in ai.mcts.cnn.cnn.parameters())
    )
    return HealthResponse(
        status="ok",
        model_version=args.version,
        architecture=args.model_name,
        parameter_count=parameter_count,
    )


def _apply_engine_params(request: MoveRequest | HintRequest) -> None:
    # Mutating the shared instance's settings per-request (rather than
    # constructing a fresh MCTS per call) keeps the Qsa/Nsa/... memory
    # tables shared across requests; num_mcts_sims, cpuct_type and c_puct are
    # plain attributes MCTS reads at the start of each search, not baked in
    # at construction time. This is a simplification for a low-traffic demo,
    # not safe under genuinely concurrent requests with different settings
    # racing each other (FastAPI's sync endpoints run in a thread pool, so
    # two requests could interleave); acceptable here, but worth knowing if
    # this backend ever needs to handle real concurrent load.
    ai.mcts.args.num_mcts_sims = request.num_mcts_sims
    ai.mcts.args.c_puct = request.c_puct
    ai.mcts.cpuct_type = request.cpuct_type
    ai.mcts.use_dcpuct = request.cpuct_type != "static"


@app.post("/move")
def move(request: MoveRequest) -> MoveResponse:
    if ai is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _apply_engine_params(request)

    board = np.array(request.board, dtype=np.int8).reshape(8, 8)
    start = time.perf_counter()
    try:
        selected = ai.select_move(board, request.player)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return MoveResponse(move=selected, elapsed_ms=elapsed_ms)


@app.post("/hints")
def hints(request: HintRequest) -> HintResponse:
    if ai is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _apply_engine_params(request)

    board = np.array(request.board, dtype=np.int8).reshape(8, 8)
    # get_canonical_form is just player * board (no spatial transform, see
    # board/utils.py), so action indices line up 1:1 with real board squares
    # regardless of which player is asking -- no remapping needed below.
    canonical_board = BU.get_canonical_form(board, request.player)
    start = time.perf_counter()
    # temperature=1 returns the raw visit-count distribution (how much
    # search budget each move actually got), not the deterministic argmax
    # select_move uses -- that's what makes a probability gradient possible.
    try:
        probs = ai.mcts.get_action_probs(canonical_board, temperature=1)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return HintResponse(probs=probs, elapsed_ms=elapsed_ms)
