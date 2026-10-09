"""
Latency benchmarks for the Othello engine.

Measures move-selection latency (MCTS and CNN together) across simulation
counts and game phases, plus board logic operations and the CNN forward
pass in isolation. With --engine cpp, only move selection is measured, using
the C++ search.

Game phase (start, mid, end) is broken out for get_valid_moves,
get_next_state and get_game_ended. get_canonical_form and the CNN forward
pass have a fixed cost, so each is pooled across all boards into one number.

Writes three CSV files to --output-dir: move_selection.csv, board_logic.csv
(the three phase-dependent operations) and flat_ops.csv (get_canonical_form
and cnn_inference, pooled).

Usage:
    python scripts/benchmark_latency.py --help
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

BACKEND_DIR = Path(__file__).resolve().parent.parent
# arch.py, board/, engine/, configs.py and utils.py are top-level modules under
# backend/ with no package init. This script lives in scripts/, where sys.path[0]
# is scripts/, so backend/ is added to sys.path explicitly.
sys.path.insert(0, str(BACKEND_DIR))

from arch import CNN_vAlpha3, args as model_args  # noqa: E402
from board import Utils as BU  # noqa: E402
from configs import BOARD_SIZE  # noqa: E402
from engine import MCTS, OthelloCNN  # noqa: E402
from utils import LoggerSetup, dotdict  # noqa: E402

log = LoggerSetup(log_name="benchmark_latency", log_level="INFO", filename="benchmark_latency.log").get_logger()

MODEL_DIR = str(BACKEND_DIR / "models")
PASS_ACTION = BOARD_SIZE * BOARD_SIZE

BOARD_REGIONS = ("start", "mid", "end")
# Ply ranges sampled via random self-play from the initial position. Random
# Othello games rarely end before about 50 plies, since one side or the other
# nearly always has a legal move until the board is almost full, so the end
# region is reachable by ply count in nearly all samples. generate_board()
# stops early on the rare game that ends sooner.
REGION_PLY_RANGE = {
    "start": (4, 12),
    "mid": (24, 36),
    "end": (46, 58),
}

DEFAULT_SIMS = [10, 50, 100, 500, 1000, 2500, 5000]
DEFAULT_BOARDS_PER_REGION = 100
DEFAULT_SEED = 0
DEFAULT_BOARD_OP_ITERATIONS = 2000
DEFAULT_CNN_ITERATIONS = 50
DEFAULT_OUTPUT_DIR = Path("results")

Board = tuple  # (np.ndarray board, int player_to_move)


# --- Board generation ------------------------------------------------------


def _random_legal_action(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    valid = BU.get_valid_moves(board, player)
    legal = np.flatnonzero(valid)
    if legal.size == 0:
        return PASS_ACTION
    return int(rng.choice(legal))


def generate_board(region: str, rng: np.random.Generator) -> Board:
    """
    Plays random legal moves from the initial position until the ply count
    drawn from the region's ply range is reached or the game ends.

    Parameters
    ----------
    region : str
        One of BOARD_REGIONS.
    rng : numpy.random.Generator
        Source of the target ply count and the random moves.

    Returns
    -------
    tuple
        The board and the player to move.
    """
    lo, hi = REGION_PLY_RANGE[region]
    target_plies = int(rng.integers(lo, hi + 1))

    board = BU.get_init_board()
    player = 1
    for _ in range(target_plies):
        if BU.get_game_ended(board, player) != 0:
            break
        action = _random_legal_action(board, player, rng)
        board, player = BU.get_next_state(board, player, action)
    return board, player


def generate_boards(boards_per_region: int, seed: int) -> dict[str, list[Board]]:
    log.info(f"generating boards: {boards_per_region}/region, seed={seed}")
    rng = np.random.default_rng(seed)
    boards_by_region = {
        region: [
            generate_board(region, rng)
            for _ in tqdm(range(boards_per_region), desc=f"generate {region} boards", leave=False)
        ]
        for region in BOARD_REGIONS
    }
    log.info("boards generated")
    return boards_by_region


# --- Engine setup ------------------------------------------------------


@dataclass
class DeviceContext:
    device: str  # "cpu" or "gpu"
    cnn: OthelloCNN
    mcts: MCTS


def build_context(device: str) -> DeviceContext:
    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU requested but torch.cuda.is_available() is False. This "
            "environment's torch build may be CPU-only (e.g. the "
            "requirements.txt / Dockerfile both install the CPU wheel). "
            "Install a CUDA build of torch matching your driver to get a "
            "real GPU number, or run with --device cpu."
        )

    log.info(f"loading model onto device={device}")
    # Cloned rather than mutated in place: model_args is arch.py's own
    # module-level singleton.
    device_args = dotdict({**model_args, "cuda": device == "gpu"})

    cnn = OthelloCNN(CNN_vAlpha3, device_args)
    cnn.load_checkpoint(MODEL_DIR, f"model-{model_args.version}.pth.tar")
    mcts = MCTS(cnn, device_args)
    log.info(f"model loaded: version={model_args.version}")
    return DeviceContext(device=device, cnn=cnn, mcts=mcts)


def warm_up(ctx: DeviceContext, boards_by_region: dict[str, list[Board]]) -> None:
    """
    Runs untimed CNN and MCTS calls so one-time initialization costs (for
    example the first CUDA call) do not land on the first timed data point.

    Parameters
    ----------
    ctx : DeviceContext
        The CNN and MCTS to warm up.
    boards_by_region : dict
        Boards from which the first start-region board is used.
    """
    log.info("warming up")
    board, player = boards_by_region["start"][0]
    canonical = BU.get_canonical_form(board, player)
    ctx.cnn.predict(canonical)
    ctx.mcts.args.num_mcts_sims = 1
    ctx.mcts.get_action_probs(canonical, temperature=0)
    ctx.mcts.clear_memory()
    log.info("warm-up done")


# --- Benchmarks ------------------------------------------------------


def build_cpp_engine():
    """
    Builds the C++ search with the exported weights of the trained checkpoint.

    Returns
    -------
    othello_cpp.Engine
        Search ready for get_action_probs.
    """
    sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))
    import othello_cpp

    import export_weights

    checkpoint = torch.load(Path(MODEL_DIR) / f"model-{model_args.version}.pth.tar", map_location="cpu")
    with tempfile.TemporaryDirectory() as directory:
        weights = Path(directory) / "weights.bin"
        weights.write_bytes(export_weights.pack_state_dict(checkpoint["state_dict"]))
        return othello_cpp.Engine(str(weights))


def time_cpp_move_selection(engine, board: Board, sims: int, batch_size: int) -> float:
    """
    Times one C++ search from an empty tree, so results cached by earlier
    calls do not shorten later ones.

    Parameters
    ----------
    engine : othello_cpp.Engine
        The search to run.
    board : Board
        The board and the player to move.
    sims : int
        Number of MCTS simulations.
    batch_size : int
        Simulations whose leaves the network evaluates together.

    Returns
    -------
    float
        Elapsed time in milliseconds.
    """
    board_state, player = board
    engine.clear()
    start = time.perf_counter()
    engine.get_action_probs(board_state, player, sims, temperature=0.0, batch_size=batch_size)
    return (time.perf_counter() - start) * 1000


def time_move_selection(ctx: DeviceContext, board: Board, sims: int) -> float:
    """
    Times one MCTS search from an empty transposition table, so results
    cached by earlier calls do not shorten later ones.

    Parameters
    ----------
    ctx : DeviceContext
        The MCTS to run.
    board : Board
        The board and the player to move.
    sims : int
        Number of MCTS simulations.

    Returns
    -------
    float
        Elapsed time in milliseconds.
    """
    board_state, player = board
    ctx.mcts.clear_memory()
    ctx.mcts.args.num_mcts_sims = sims
    canonical = BU.get_canonical_form(board_state, player)
    start = time.perf_counter()
    ctx.mcts.get_action_probs(canonical, temperature=0)
    return (time.perf_counter() - start) * 1000


def _summarize(timings: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(timings),
        "stdev_ms": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "n": len(timings),
    }


def run_move_selection_benchmark(
    time_search: Callable[[Board, int], float], boards_by_region: dict[str, list[Board]], sims_list: list[int]
) -> list[dict]:
    """
    Times move selection for every region and simulation count. Each region
    reuses the same boards for every simulation count.

    Parameters
    ----------
    time_search : callable
        Takes a board and a simulation count and returns the elapsed
        milliseconds of one search.
    boards_by_region : dict
        Boards for each region in BOARD_REGIONS.
    sims_list : list
        Simulation counts to time.

    Returns
    -------
    list
        One row per region and simulation count with timing statistics.
    """
    log.info(f"move-selection benchmark: sims={sims_list}")
    rows = []
    for region in BOARD_REGIONS:
        boards = boards_by_region[region]
        for sims in sims_list:
            timings = [
                time_search(board, sims)
                for board in tqdm(boards, desc=f"move_selection region={region} sims={sims}", leave=False)
            ]
            summary = _summarize(timings)
            rows.append({"region": region, "sims": sims, **summary})
            log.info(f"move_selection region={region} sims={sims} mean_ms={summary['mean_ms']:.1f}")
    return rows


def run_board_logic_benchmark(
    boards_by_region: dict[str, list[Board]], iterations: int
) -> list[dict]:
    """
    Times get_valid_moves, get_next_state and get_game_ended in isolation
    for each region.

    Parameters
    ----------
    boards_by_region : dict
        Boards for each region in BOARD_REGIONS.
    iterations : int
        Repeats per operation and board.

    Returns
    -------
    list
        One row per region and operation with the mean time in milliseconds.
    """
    log.info(f"board-logic benchmark: {iterations} iterations/op/board")
    rows = []
    for region, boards in boards_by_region.items():
        op_timings: dict[str, list[float]] = {
            "get_valid_moves": [],
            "get_next_state": [],
            "get_game_ended": [],
        }
        for board, player in tqdm(boards, desc=f"board_logic region={region}", leave=False):
            valid = BU.get_valid_moves(board, player)
            legal = np.flatnonzero(valid)
            action = int(legal[0]) if legal.size else PASS_ACTION

            start = time.perf_counter()
            for _ in range(iterations):
                BU.get_valid_moves(board, player)
            op_timings["get_valid_moves"].append((time.perf_counter() - start) / iterations * 1000)

            start = time.perf_counter()
            for _ in range(iterations):
                BU.get_next_state(board, player, action)
            op_timings["get_next_state"].append((time.perf_counter() - start) / iterations * 1000)

            start = time.perf_counter()
            for _ in range(iterations):
                BU.get_game_ended(board, player)
            op_timings["get_game_ended"].append((time.perf_counter() - start) / iterations * 1000)

        for op, times in op_timings.items():
            mean_ms = statistics.fmean(times)
            rows.append({"region": region, "operation": op, "mean_ms": mean_ms, "n_boards": len(times)})
            log.info(f"board_logic region={region} op={op} mean_ms={mean_ms:.4f}")
    return rows


def run_flat_ops_benchmark(
    ctx: DeviceContext, boards_by_region: dict[str, list[Board]], board_op_iterations: int, cnn_iterations: int
) -> list[dict]:
    """
    Times get_canonical_form and the CNN forward pass, pooling boards from
    all regions into one number per operation.

    Parameters
    ----------
    ctx : DeviceContext
        The CNN to run.
    boards_by_region : dict
        Boards for each region in BOARD_REGIONS.
    board_op_iterations : int
        Repeats per board for get_canonical_form.
    cnn_iterations : int
        Repeats per board for the CNN forward pass.

    Returns
    -------
    list
        One row per operation with the mean time in milliseconds.
    """
    log.info("flat-ops benchmark: get_canonical_form, cnn_inference (pooled across regions)")
    all_boards = [board for boards in boards_by_region.values() for board in boards]

    canonical_timings = []
    for board, player in tqdm(all_boards, desc="get_canonical_form", leave=False):
        start = time.perf_counter()
        for _ in range(board_op_iterations):
            BU.get_canonical_form(board, player)
        canonical_timings.append((time.perf_counter() - start) / board_op_iterations * 1000)

    cnn_timings = []
    for board, player in tqdm(all_boards, desc="cnn_inference", leave=False):
        canonical = BU.get_canonical_form(board, player)
        ctx.cnn.predict(canonical)  # warm-up for this position
        start = time.perf_counter()
        for _ in range(cnn_iterations):
            ctx.cnn.predict(canonical)
        cnn_timings.append((time.perf_counter() - start) / cnn_iterations * 1000)

    rows = [
        {"operation": "get_canonical_form", "mean_ms": statistics.fmean(canonical_timings), "n": len(canonical_timings)},
        {"operation": "cnn_inference", "mean_ms": statistics.fmean(cnn_timings), "n": len(cnn_timings)},
    ]
    for row in rows:
        log.info(f"flat_op op={row['operation']} mean_ms={row['mean_ms']:.4f}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {path}")


# --- CLI ------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["python", "cpp"], default="python")
    parser.add_argument("--batch-size", type=int, default=1, help="with --engine cpp, simulations whose leaves are evaluated together")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--sims", type=int, nargs="+", default=DEFAULT_SIMS)
    parser.add_argument("--boards-per-region", type=int, default=DEFAULT_BOARDS_PER_REGION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--board-op-iterations", type=int, default=DEFAULT_BOARD_OP_ITERATIONS,
        help="repeats per board for the isolated board-logic microbenchmark",
    )
    parser.add_argument(
        "--cnn-iterations", type=int, default=DEFAULT_CNN_ITERATIONS,
        help="repeats per board for the isolated CNN forward-pass microbenchmark",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    log.info(
        f"starting: device={ns.device} seed={ns.seed} boards_per_region={ns.boards_per_region} "
        f"sims={ns.sims} output_dir={ns.output_dir}"
    )

    boards_by_region = generate_boards(ns.boards_per_region, ns.seed)
    if ns.engine == "cpp":
        engine = build_cpp_engine()
        engine.get_action_probs(*boards_by_region["start"][0], 1)
        rows = run_move_selection_benchmark(
            lambda board, sims: time_cpp_move_selection(engine, board, sims, ns.batch_size), boards_by_region, ns.sims
        )
        write_csv(ns.output_dir / "move_selection.csv", rows)
        log.info("done")
        return

    ctx = build_context(ns.device)
    warm_up(ctx, boards_by_region)

    move_selection_rows = run_move_selection_benchmark(
        lambda board, sims: time_move_selection(ctx, board, sims), boards_by_region, ns.sims
    )
    board_logic_rows = run_board_logic_benchmark(boards_by_region, ns.board_op_iterations)
    flat_ops_rows = run_flat_ops_benchmark(ctx, boards_by_region, ns.board_op_iterations, ns.cnn_iterations)

    write_csv(ns.output_dir / "move_selection.csv", move_selection_rows)
    write_csv(ns.output_dir / "board_logic.csv", board_logic_rows)
    write_csv(ns.output_dir / "flat_ops.csv", flat_ops_rows)

    log.info("done")


if __name__ == "__main__":
    main()
