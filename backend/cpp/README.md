# C++ engine

Othello rules on 64-bit bitboards ([include/othello/position.hpp](include/othello/position.hpp)), the policy
and value network ([include/othello/net.hpp](include/othello/net.hpp), [src/](src/)), a C++ test executable
([tests/position_test.cpp](tests/position_test.cpp)) and the `othello_cpp` Python module
([bindings/module.cpp](bindings/module.cpp)).

## Build

```bash
cmake -S cpp -B cpp/build -DPython3_EXECUTABLE=$(which python)
cmake --build cpp/build -j
```

`pybind11` comes from `requirements-dev.txt`.

## Network weights

`scripts/export_weights.py` converts a trained checkpoint to the weight file
that `othello_cpp.Network` loads:

```bash
python scripts/export_weights.py models/model-alpha3S-v2.1.pth.tar weights.bin
python scripts/export_weights.py models/model-alpha3S-v2.1.pth.tar weights_fp16.bin --fc-type fp16
```

`--fc-type fp16` stores the two large fully connected matrices as half
precision, which halves the memory they read per forward pass. The weight format
packs fully connected layers in tiles of 16 outputs, and the loader rejects files
of another format version.

```python
import numpy as np
import othello_cpp

net = othello_cpp.Network("weights.bin")
board = np.zeros(64, dtype=np.float32)  # own discs 1, opponent discs -1
log_policy, value = net.forward(board)
```

## Search

`othello_cpp.Engine` runs the Monte Carlo tree search with the network. It
keeps its tree between calls, so transpositions and earlier searches carry
over until `clear()` or until the node limit (`max_nodes`) would be exceeded.

```python
engine = othello_cpp.Engine("weights.bin")
probs = engine.get_action_probs(board, player, num_sims=500, temperature=0.0)
```

`engine.isa` is `"avx2"` or `"scalar"`: the instruction set the network uses. The
server logs it at startup, with a warning for `"scalar"`.

`board` holds 64 squares (-1 black, 0 empty, 1 white) and `player` is the side
to move. The 65 probabilities cover the squares and the pass action.

`batch_size` (1 to 16, default 1) sets how many simulations evaluate their
leaves together. One reproduces the sequential search exactly. Larger values
mark each selected path with a virtual loss until its leaf is evaluated, and the
network reads its weights once per pass of up to 4 boards, which makes a search
about twice as fast with batch sizes of 4 or more. The server uses 4.
`temperature=0` gives a one-hot most visited action with ties broken by `seed`;
otherwise the result is the visit counts raised to `1 / temperature` and
normalized. Searches run without the GIL and one at a time.

## Inspecting a search

`engine.get_tree(board, player, max_depth, min_visits)` copies the search tree as
a list of node dicts. Two scripts render a search from the initial position or
after given moves:

```bash
python scripts/print_tree.py tree.txt --sims 400 --max-depth 3 --min-visits 5
python scripts/visualize_tree.py tree --sims 800 --max-depth 4 --top-children 3   # tree.svg and tree.html
python scripts/visualize_tree.py tree --layout horizontal                         # root on the left instead of on top
```

## Benchmark

```bash
python scripts/benchmark_network.py --rounds 6 --reps 5 --engine-boards 5
```

Each network variant runs in its own process pinned to one performance core
(`--cpu`, default 8), with the variants interleaved over rounds. The script
also measures the machine's peak multiply-add throughput and read bandwidth
(`bench_peaks`), the bitboard rules (`bench_rules`), hardware counters per
evaluation through `perf stat`, and the Python and C++ engines through
`benchmark_latency.py` with one torch thread. Results go to
`results/baselines/<run-id>/`. Counters need `kernel.perf_event_paranoid` of 1
or lower.

## Matches

`scripts/play_match.py` plays two configurations of the search against each
other from random openings, with both colors, and reports the score:

```bash
python scripts/play_match.py --weights-a weights.bin --batch-a 4 --sims-a 400 --batch-b 1 --sims-b 200
```

## Test

```bash
ctest --test-dir cpp/build --output-on-failure
python -m pytest tests/test_cpp_rules_parity.py tests/test_cpp_network_parity.py tests/test_cpp_mcts_parity.py
```

`position_test` and `mcts_test` run under `ctest`. `position_test` checks perft counts for 1 to 10 plies, move invariants and hash
separation. `test_cpp_rules_parity.py` plays random games through both
`board.Utils` and `othello_cpp` and compares legal moves, next states and game
results. `test_cpp_network_parity.py` compares the C++ network (vectorized and
plain-loop passes, fp32 and fp16 weights) with `arch.CNN_vAlpha3`, and runs
against the trained checkpoint when `models/` holds it. `test_cpp_mcts_parity.py`
runs `engine.MCTS` and `othello_cpp.Engine` on the same positions, once with a
deterministic pseudo-random evaluator (`StubEngine`, exact visit-count equality,
all three c_puct modes, trees kept across a game) and once with the trained
network.
