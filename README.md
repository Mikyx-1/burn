# Flex CPU CTC benchmark evidence

This package records the final old-versus-optimized CTC measurements: 16 shared scenarios for each
of F16, BF16, F32, and F64 with I32 indices, plus S04–S07 repeated with six workers. The 80 matched
comparisons contain 1,120 timing-round records and 480 heap-count records. All final rounds/calls,
including varying peaks and BF16 runtime regressions, are retained without trimming.

The implementation is published at [commit 04338b610](https://github.com/Mikyx-1/burn/commit/04338b610b0b48e0ce07506ef3f24b2990efe230),
based on upstream `5b95516d6140693f3945520ebaa9a69f96079ffc`. `native-ctc.rs` snapshots the measured implementation. `environment.json` identifies the CPU,
compiler, pre-change base commit, features, and affinity. `runner/fixtures.rs` is the shared fixture
source. The public runner changes only the two module paths to relative files; **no relocated build
or timing run is claimed**. Rust sources are supplied under the repository's MIT OR Apache-2.0
licenses, included here.

## Metrics

- Timing compares direct Flex CPU backend calls with `ctc_loss_default::<Flex>` on identical retained
  inputs. Input creation is excluded; cloning, eligibility checks, layout conversion, output access,
  and disposal are included. Public masking/reductions, backward, whole models, and GPU execution are
  outside the measurements.
- Rayon workers are initialized and allowed to park before measurements. Each case is checked and
  warmed three times per path. Each of seven alternating old/new rounds runs at least 80 ms and three
  calls. The reported latency is the median round mean; speedup is old latency divided by optimized latency.
- A separate executable measures peak extra live requested Rust heap, including the output and
  excluding pre-existing inputs. The output remains alive through the end of each counting window.
  Tables use the maximum peak from three calls and retain observed ranges. This is requested heap,
  not RSS, allocator overhead, or total model memory. Parallel scratch belongs to tasks.
- The allocator counter validates known-size allocation, realloc growth, and ignored frees of
  pre-existing allocations. Recorded windows had no counter overflow or untracked reallocations.
  `false` in the CSV policy column means the old decomposition; `true` means the optimized hook.

## Reproduce in a Burn checkout

Use a checkout containing the measured implementation and its pinned dependencies. Commands below
run from the Burn repository root. Set the evidence path and a fresh results directory first; keep
this package unchanged. The `--offline` commands require cached dependencies. Adapt CPU affinity on
other machines. Run builds and measurements sequentially without concurrent benchmark/build work.

```sh
ctc_evidence=/path/to/burn-ctc-public-evidence
ctc_results=/path/to/fresh-ctc-results
mkdir -p "$ctc_results"
cp "$ctc_evidence/catalog.csv" "$ctc_results/catalog.csv"
cp "$ctc_evidence/summarize.py" "$ctc_results/summarize.py"
mkdir -p crates/burn-flex/examples/ctc_matrix_report
cp "$ctc_evidence/runner/"*.rs crates/burn-flex/examples/ctc_matrix_report/

cargo build -p burn-flex --example ctc_matrix_report --release --locked --offline -j 2
cp target/release/examples/ctc_matrix_report "$ctc_results/ctc-matrix-time"
cargo rustc -p burn-flex --example ctc_matrix_report --release --locked --offline -j 2 -- --cfg ctc_count_alloc
cp target/release/examples/ctc_matrix_report "$ctc_results/ctc-matrix-memory"

RAYON_NUM_THREADS=1 taskset -c 6 "$ctc_results/ctc-matrix-time" > "$ctc_results/time-1.csv"
RAYON_NUM_THREADS=6 taskset -c 0,2,4,6,8,10 "$ctc_results/ctc-matrix-time" --parallel > "$ctc_results/time-6.csv"
RAYON_NUM_THREADS=1 taskset -c 6 "$ctc_results/ctc-matrix-memory" > "$ctc_results/memory-1.csv"
RAYON_NUM_THREADS=6 taskset -c 0,2,4,6,8,10 "$ctc_results/ctc-matrix-memory" --parallel > "$ctc_results/memory-6.csv"
python3 "$ctc_results/summarize.py" > "$ctc_results/summary.txt"
```

The summarizer checks record counts, all rounds/calls, and untracked reallocations before calculating
latencies, speedups, heap peaks, and reductions. `summary.json` and `summary.txt` can also be regenerated
from the supplied raw CSV files with `python3 summarize.py > summary.txt` in this package.

`SHA256.json` covers every other file using paths relative to this directory.
