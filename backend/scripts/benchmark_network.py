"""
Benchmarks the policy and value network and the bitboard rules, and records
the machine they ran on.

Each variant runs in its own process pinned to one CPU. Rounds interleave the
variants so that drift in clock speed or temperature affects all of them
equally. Hardware counters come from `perf stat`, and each per-evaluation
counter is the difference between two runs with different iteration counts,
so process start-up cancels out.

Writes environment.json, peaks.json, rules.json, network.json and
network.csv into <output-dir>/<run-id>/, and optionally the Python engine
benchmarks into python_engine/, cpp_engine/ and cpp_batch4_engine/.

Usage:
    python scripts/benchmark_network.py --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
# arch.py, board/, engine/, configs.py and utils.py are top-level modules under
# backend/ with no package init. This script lives in scripts/, where sys.path[0]
# is scripts/, so backend/ is added to sys.path explicitly.
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))

import export_weights  # noqa: E402
from arch import CNN_vAlpha3, args as model_args  # noqa: E402
from utils import dotdict  # noqa: E402

BUILD_DIR = BACKEND_DIR / "cpp" / "build"
DEFAULT_CHECKPOINT = BACKEND_DIR / "models" / f"model-{model_args.version}.pth.tar"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "results" / "baselines"
DEFAULT_CPU = 8
BOARD_COUNT = 64
SEED = 0

# Iterations of one repetition and of the two perf-stat runs, per variant.
# Slow variants run fewer iterations so that every repetition takes a similar time.
ITERS = {
    "cpp_reference_fp32": (20, 20, 100),
    "cpp_reference_fp16": (20, 20, 100),
    "cpp_avx2_fp32": (2000, 500, 2500),
    "cpp_avx2_fp16": (2000, 500, 2500),
    "cpp_avx2_fp32_batch4": (2000, 500, 2500),
    "cpp_avx2_fp16_batch4": (2000, 500, 2500),
    "torch_forward_1thread": (400, 100, 500),
    "torch_predict_1thread": (400, 100, 500),
    "torch_predict_default_threads": (400, 0, 0),
}
PINNED_VARIANTS = [name for name in ITERS if name != "torch_predict_default_threads"]
PERF_EVENTS = "cycles,instructions,cache-misses,task-clock"


# --- Analytic work ------------------------------------------------------


def multiply_adds() -> dict[str, int]:
    """
    Counts the multiply-adds of one forward pass per layer.

    Returns
    -------
    dict
        Multiply-adds by layer name.
    """
    s, seq, res = export_weights.SQUARES, export_weights.SEQ_CHANNELS, export_weights.RES_CHANNELS
    return {
        "conv_first": s * seq * 25,
        "shortcut": s * res * seq,
        "conv_1": s * res * seq * 9,
        "conv_2": s * res * res * 9,
        "conv_3": s * res * res * 9,
        "fc_1": s * res * export_weights.FC1_OUT,
        "fc_2": export_weights.FC1_OUT * export_weights.FC2_OUT,
        "head": export_weights.FC2_OUT * (export_weights.ACTIONS + 1),
    }


def layer_weight_bytes(fc_type: str) -> dict[str, int]:
    """
    Counts the weight bytes of each layer that one forward pass reads.

    Parameters
    ----------
    fc_type : str
        "fp32" or "fp16", the storage type of the two large fully connected matrices.

    Returns
    -------
    dict
        Bytes by layer name.
    """
    macs = multiply_adds()
    fc_bytes_per_weight = 2 if fc_type == "fp16" else 4
    sizes = {name: macs[name] // export_weights.SQUARES * 4 for name in ("conv_first", "shortcut", "conv_1", "conv_2", "conv_3")}
    sizes["fc_1"] = macs["fc_1"] * fc_bytes_per_weight
    sizes["fc_2"] = macs["fc_2"] * fc_bytes_per_weight
    sizes["head"] = export_weights.HEAD_OUT * export_weights.FC2_OUT * 4
    return sizes


# --- Environment ------------------------------------------------------


def run_text(cmd: list[str]) -> str:
    """
    Runs a command and returns its standard output, or an empty string on failure.

    Parameters
    ----------
    cmd : list
        Command and arguments.

    Returns
    -------
    str
        Standard output without trailing whitespace.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def read_text(path: str) -> str:
    """
    Reads a small text file, or returns an empty string when it cannot be read.

    Parameters
    ----------
    path : str
        File to read.

    Returns
    -------
    str
        Contents without trailing whitespace.
    """
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def sha256(path: Path) -> str:
    """
    Computes the SHA-256 digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_environment(cpu: int, files: dict[str, Path]) -> dict:
    """
    Records the machine, toolchain and source state of a run.

    Parameters
    ----------
    cpu : int
        CPU the pinned variants run on.
    files : dict
        Files to hash, by label.

    Returns
    -------
    dict
        JSON-serializable description of the environment.
    """
    cpuinfo = read_text("/proc/cpuinfo")
    model = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    flags = re.search(r"flags\s*:\s*(.+)", cpuinfo)
    interesting_flags = {"avx", "avx2", "fma", "f16c", "bmi2", "popcnt", "avx512f", "avx_vnni"}
    cache = BUILD_DIR / "CMakeCache.txt"
    cmake_cache = cache.read_text() if cache.exists() else ""

    def cmake_value(name: str) -> str:
        match = re.search(rf"^{name}:[A-Z]+=(.*)$", cmake_cache, re.MULTILINE)
        return match.group(1) if match else ""

    repo = BACKEND_DIR.parent
    compiler = run_text(["g++", "--version"])
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu_model": model.group(1) if model else "",
        "cpu_flags": sorted(interesting_flags & set(flags.group(1).split())) if flags else [],
        "pinned_cpu": cpu,
        "pinned_cpu_max_mhz": read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"),
        "pinned_cpu_cur_khz_at_start": read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"),
        "governor": read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"),
        "perf_event_paranoid": read_text("/proc/sys/kernel/perf_event_paranoid"),
        "load_average": os.getloadavg(),
        "logical_cpus": os.cpu_count(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads_default": torch.get_num_threads(),
        "numpy": np.__version__,
        "compiler": compiler.splitlines()[0] if compiler else "",
        "cmake_build_type": cmake_value("CMAKE_BUILD_TYPE"),
        "cmake_cxx_flags_release": cmake_value("CMAKE_CXX_FLAGS_RELEASE"),
        "git_head": run_text(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_dirty": bool(run_text(["git", "-C", str(repo), "status", "--porcelain"])),
        "sha256": {label: sha256(path) for label, path in files.items()},
    }


# --- Variants ------------------------------------------------------


def parse_json_line(text: str) -> dict:
    """
    Parses the last line of a command's output that is a JSON object.

    Parameters
    ----------
    text : str
        Command output.

    Returns
    -------
    dict
        Parsed object.
    """
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON in output: {text[-500:]}")


def variant_command(name: str, ctx: dict, iters: int, reps: int) -> list[str]:
    """
    Builds the command that times one variant.

    Parameters
    ----------
    name : str
        Variant name, a key of ITERS.
    ctx : dict
        Paths of the weight files, the boards file and the checkpoint.
    iters : int
        Evaluations per repetition.
    reps : int
        Repetitions.

    Returns
    -------
    list
        Command and arguments.
    """
    if name.startswith("cpp_"):
        _, variant, fc_type, *rest = name.split("_")
        batch = int(rest[0].removeprefix("batch")) if rest else 1
        return [
            str(BUILD_DIR / "bench_net"),
            "--weights", str(ctx["weights"][fc_type]),
            "--boards", str(ctx["boards"]),
            "--variant", variant,
            "--iters", str(iters),
            "--reps", str(reps),
            "--batch", str(batch),
        ]
    kind = "forward" if name.startswith("torch_forward") else "predict"
    threads = 1 if name.endswith("1thread") else torch.get_num_threads()
    return [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", kind,
        "--threads", str(threads),
        "--boards", str(ctx["boards"]),
        "--checkpoint", str(ctx["checkpoint"]),
        "--iters", str(iters),
        "--reps", str(reps),
    ]


def run_variant(name: str, ctx: dict, iters: int, reps: int, cpu: int | None) -> list[float]:
    """
    Times one variant in a fresh process.

    Parameters
    ----------
    name : str
        Variant name, a key of ITERS.
    ctx : dict
        Paths of the weight files, the boards file and the checkpoint.
    iters : int
        Evaluations per repetition.
    reps : int
        Repetitions.
    cpu : int or None
        CPU to pin the process to, or None to leave it unpinned.

    Returns
    -------
    list
        Mean nanoseconds per evaluation for each repetition.
    """
    cmd = variant_command(name, ctx, iters, reps)
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu)] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return parse_json_line(out)["ns_per_eval"]


def perf_totals(cmd: list[str], cpu: int) -> dict[str, float]:
    """
    Runs a command under perf stat and returns its totals.

    Parameters
    ----------
    cmd : list
        Command and arguments.
    cpu : int
        CPU to pin the process to.

    Returns
    -------
    dict
        Totals for cycles, instructions, cache_misses and task_clock_ms,
        counted on the performance cores.
    """
    full = ["taskset", "-c", str(cpu), "perf", "stat", "-x,", "-e", PERF_EVENTS, "--"] + cmd
    result = subprocess.run(full, capture_output=True, text=True, check=True)
    totals: dict[str, float] = {}
    for line in result.stderr.splitlines():
        fields = line.split(",")
        if len(fields) < 3 or fields[0].startswith("<"):
            continue
        event = fields[2]
        if "cpu_atom" in event:
            continue
        for label, key in (("cycles", "cycles"), ("instructions", "instructions"),
                           ("cache-misses", "cache_misses"), ("task-clock", "task_clock_ms")):
            if label in event:
                totals[key] = float(fields[0])
    return totals


def per_evaluation_counters(name: str, ctx: dict, cpu: int) -> dict[str, float]:
    """
    Measures hardware counters per evaluation from two runs of different length.

    Parameters
    ----------
    name : str
        Variant name, a key of ITERS.
    ctx : dict
        Paths of the weight files, the boards file and the checkpoint.
    cpu : int
        CPU to pin the process to.

    Returns
    -------
    dict
        Cycles, instructions, cache misses and ns per evaluation, plus IPC and
        the average clock in GHz.
    """
    _, short, long = ITERS[name]
    low = perf_totals(variant_command(name, ctx, short, 1), cpu)
    high = perf_totals(variant_command(name, ctx, long, 1), cpu)
    n = long - short
    counters = {key: (high[key] - low[key]) / n for key in ("cycles", "instructions", "cache_misses", "task_clock_ms")}
    counters["ns"] = counters.pop("task_clock_ms") * 1e6
    counters["ipc"] = counters["instructions"] / counters["cycles"]
    counters["ghz"] = counters["cycles"] / counters["ns"]
    return counters


def summarize(samples: list[float]) -> dict:
    """
    Summarizes repetition times.

    Parameters
    ----------
    samples : list
        Nanoseconds per evaluation, one entry per repetition.

    Returns
    -------
    dict
        Median, minimum, quartiles, mean and coefficient of variation in percent.
    """
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    mean = statistics.fmean(ordered)
    return {
        "n": len(ordered),
        "median_ns": statistics.median(ordered),
        "min_ns": ordered[0],
        "p25_ns": quartiles[0],
        "p75_ns": quartiles[2],
        "mean_ns": mean,
        "cv_percent": statistics.pstdev(ordered) / mean * 100 if len(ordered) > 1 else 0.0,
    }


def roofline(name: str, median_ns: float, peaks: dict) -> dict:
    """
    Compares a variant's time with the machine's compute and bandwidth limits.

    Each layer is limited by the slower of its multiply-add time at peak
    throughput and its weight read time at the measured bandwidth for a buffer
    of its size. The layer limits add up to the time floor of one evaluation.

    Parameters
    ----------
    name : str
        Variant name.
    median_ns : float
        Median time per evaluation in nanoseconds.
    peaks : dict
        Output of bench_peaks.

    Returns
    -------
    dict
        Work, achieved rates, the time floor and the share of that floor the
        variant reaches.
    """
    macs = multiply_adds()
    fc_type = "fp16" if name.endswith("fp16") else "fp32"
    layer_bytes = layer_weight_bytes(fc_type)
    sizes = {int(k): v for k, v in peaks["read_gbps"].items()}
    floors = {}
    for layer, layer_macs in macs.items():
        nearest = min(sizes, key=lambda size: abs(np.log(size) - np.log(layer_bytes[layer])))
        compute_ns = 2 * layer_macs / peaks["peak_gflops_fp32"]
        memory_ns = layer_bytes[layer] / sizes[nearest]
        floors[layer] = {"compute_ns": compute_ns, "memory_ns": memory_ns, "bound": "memory" if memory_ns > compute_ns else "compute"}
    floor_ns = sum(max(f["compute_ns"], f["memory_ns"]) for f in floors.values())
    flops = 2 * sum(macs.values())
    total_bytes = sum(layer_bytes.values())
    return {
        "flops": flops,
        "weight_bytes": total_bytes,
        "achieved_gflops": flops / median_ns,
        "achieved_gbps": total_bytes / median_ns,
        "floor_ns": floor_ns,
        "floor_share_percent": floor_ns / median_ns * 100,
        "layer_floors": floors,
    }


# --- Workers ------------------------------------------------------


def run_torch_worker(ns: argparse.Namespace) -> None:
    """
    Times the PyTorch network and prints a JSON line.

    Parameters
    ----------
    ns : argparse.Namespace
        Parsed worker arguments.
    """
    from engine import OthelloCNN

    torch.set_num_threads(ns.threads)
    boards = np.fromfile(ns.boards, dtype=np.float32).reshape(-1, 64)
    device_args = dotdict({**model_args, "cuda": False})
    cnn = OthelloCNN(CNN_vAlpha3, device_args)
    cnn.load_checkpoint(str(ns.checkpoint.parent), ns.checkpoint.name)
    cnn.cnn.eval()

    if ns.worker == "forward":
        inputs = [torch.from_numpy(b).view(1, 8, 8) for b in boards]

        def evaluate(i: int) -> None:
            with torch.no_grad():
                cnn.cnn(inputs[i % len(inputs)])
    else:
        inputs = [b.reshape(8, 8) for b in boards]

        def evaluate(i: int) -> None:
            cnn.predict(inputs[i % len(inputs)])

    for i in range(50):
        evaluate(i)
    samples = []
    for _ in range(ns.reps):
        start = time.perf_counter()
        for i in range(ns.iters):
            evaluate(i)
        samples.append((time.perf_counter() - start) / ns.iters * 1e9)
    print(json.dumps({"variant": f"torch_{ns.worker}", "iters": ns.iters, "ns_per_eval": samples}))


# --- Orchestration ------------------------------------------------------


def make_boards(path: Path) -> None:
    """
    Writes a fixed set of canonical boards from random games.

    Parameters
    ----------
    path : pathlib.Path
        Output file of float32 values, 64 per board.
    """
    import othello_cpp as cpp

    rng = np.random.default_rng(SEED)
    boards = []
    while len(boards) < BOARD_COUNT:
        board, player = cpp.get_init_board(), -1
        while len(boards) < BOARD_COUNT and cpp.get_game_ended(board, player) == 0:
            valid = cpp.get_valid_moves(board, player)
            boards.append((player * board).astype(np.float32).reshape(-1))
            board, player = cpp.get_next_state(board, player, int(rng.choice(np.flatnonzero(valid))))
    np.stack(boards).astype(np.float32).tofile(path)


def run_engine_benchmark(engine: str, out_dir: Path, cpu: int, boards_per_region: int, batch_size: int = 1) -> None:
    """
    Runs the move-selection benchmark of an engine pinned to one CPU with one torch thread.

    Parameters
    ----------
    engine : str
        "python" or "cpp".
    out_dir : pathlib.Path
        Directory that receives the CSV files.
    cpu : int
        CPU to pin the process to.
    boards_per_region : int
        Boards per game phase.
    batch_size : int, optional
        With the C++ engine, simulations whose leaves are evaluated together.
    """
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    cmd = [
        "taskset", "-c", str(cpu), sys.executable, str(BACKEND_DIR / "scripts" / "benchmark_latency.py"),
        "--engine", engine,
        "--batch-size", str(batch_size),
        "--sims", "10", "100", "500", "1000", "2500",
        "--boards-per-region", str(boards_per_region),
        "--output-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=BACKEND_DIR)


def print_table(rows: list[dict]) -> None:
    """
    Prints the network summary as a table.

    Parameters
    ----------
    rows : list
        Summary rows, one per variant.
    """
    print(f"{'variant':32s} {'median us':>10s} {'min us':>9s} {'cv %':>6s} {'cycles':>11s} {'instr':>11s} {'IPC':>5s} {'GHz':>5s} {'roofline %':>10s}")
    for r in rows:
        print(
            f"{r['variant']:32s} {r['median_ns'] / 1000:10.1f} {r['min_ns'] / 1000:9.1f} {r['cv_percent']:6.1f} "
            f"{r.get('cycles', float('nan')):11.0f} {r.get('instructions', float('nan')):11.0f} "
            f"{r.get('ipc', float('nan')):5.2f} {r.get('ghz', float('nan')):5.2f} {r.get('floor_share_percent', float('nan')):10.1f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=DEFAULT_CPU, help="CPU the pinned variants run on")
    parser.add_argument("--rounds", type=int, default=6, help="interleaved rounds over all variants")
    parser.add_argument("--reps", type=int, default=5, help="repetitions per variant and round")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--engine-boards", type=int, default=0,
        help="boards per game phase for the Python and C++ engine benchmarks, 0 to skip",
    )
    parser.add_argument("--worker", choices=["forward", "predict"], help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--boards", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--iters", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    if ns.worker:
        run_torch_worker(ns)
        return

    out_dir = ns.output_dir / ns.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="benchmark_network_"))
    ctx = {"checkpoint": ns.checkpoint, "boards": work / "boards.f32", "weights": {}}
    state_dict = torch.load(ns.checkpoint, map_location="cpu")["state_dict"]
    for fc_type in export_weights.FC_TYPES:
        ctx["weights"][fc_type] = work / f"weights_{fc_type}.bin"
        ctx["weights"][fc_type].write_bytes(export_weights.pack_state_dict(state_dict, fc_type))
    make_boards(ctx["boards"])

    files = {
        "checkpoint": ns.checkpoint,
        "weights_fp32": ctx["weights"]["fp32"],
        "weights_fp16": ctx["weights"]["fp16"],
        "bench_net": BUILD_DIR / "bench_net",
        "bench_peaks": BUILD_DIR / "bench_peaks",
        "bench_rules": BUILD_DIR / "bench_rules",
    }
    files.update({p.name: p for p in BUILD_DIR.glob("othello_cpp*.so")})
    environment = capture_environment(ns.cpu, files)
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2))

    pin = ["taskset", "-c", str(ns.cpu)]
    print("measuring machine peaks")
    peaks = parse_json_line(subprocess.run(pin + [str(BUILD_DIR / "bench_peaks")], capture_output=True, text=True, check=True).stdout)
    (out_dir / "peaks.json").write_text(json.dumps(peaks, indent=2))

    print("measuring rules")
    rules = parse_json_line(
        subprocess.run(pin + [str(BUILD_DIR / "bench_rules"), "--reps", "7"], capture_output=True, text=True, check=True).stdout
    )
    rules["perft_leaves_per_second_median"] = rules["perft_leaves"] / statistics.median(rules["perft_seconds"])
    (out_dir / "rules.json").write_text(json.dumps(rules, indent=2))

    samples: dict[str, list[float]] = {name: [] for name in ITERS}
    for round_index in range(ns.rounds):
        print(f"round {round_index + 1}/{ns.rounds}")
        for name, (iters, _, _) in ITERS.items():
            cpu = None if name == "torch_predict_default_threads" else ns.cpu
            samples[name] += run_variant(name, ctx, iters, ns.reps, cpu)

    counters = {}
    for name in PINNED_VARIANTS:
        print(f"counters {name}")
        counters[name] = per_evaluation_counters(name, ctx, ns.cpu)

    rows = []
    for name in ITERS:
        row = {"variant": name, **summarize(samples[name]), **counters.get(name, {})}
        if name.startswith(("cpp_avx2", "cpp_reference")) and "batch" not in name:
            row.update(roofline(name, row["median_ns"], peaks))
        rows.append(row)
    by_name = {r["variant"]: r for r in rows}
    baselines = ("torch_predict_default_threads", "torch_predict_1thread", "torch_forward_1thread")
    for row in rows:
        for base in baselines:
            row[f"speedup_vs_{base}"] = by_name[base]["median_ns"] / row["median_ns"]

    (out_dir / "network.json").write_text(
        json.dumps({"rows": rows, "samples_ns_per_eval": samples, "rounds": ns.rounds, "reps": ns.reps}, indent=2)
    )
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "network.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"peaks: {json.dumps(peaks)}")
    print(f"rules: perft {rules['perft_leaves_per_second_median'] / 1e6:.1f} M leaves/s, "
          f"legal_moves {statistics.median(rules['legal_moves_ns']):.1f} ns, play {statistics.median(rules['play_ns']):.1f} ns")

    if ns.engine_boards > 0:
        for label, engine, batch_size in (("python", "python", 1), ("cpp", "cpp", 1), ("cpp_batch4", "cpp", 4)):
            print(f"running the {label} engine benchmark")
            run_engine_benchmark(engine, out_dir / f"{label}_engine", ns.cpu, ns.engine_boards, batch_size)
    shutil.rmtree(work, ignore_errors=True)
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
