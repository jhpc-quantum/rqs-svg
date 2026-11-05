# Pauli expectation performance

[API and implementation](pauli-expectation.md) · [Validation](pauli-validation.md) · [Performance](pauli-performance.md) · [Acceptance report](pauli-acceptance.md)

The benchmark is implemented in `tests/pauli_benchmark.cu`, with a runner in
[`scripts/benchmark_pauli.py`](../scripts/benchmark_pauli.py). Production kernels
are used directly for generated-pair stage timing. A benchmark-only kernel scans
the full local index domain and selects `i < (i ^ flip_mask)` before applying
the same product/phase arithmetic and measurement block reduction. Both variants
use the same grid, block size, scratch region, and Stage 2 kernel. The source
branch may be predicated by the compiler; no particular machine-code branch is
assumed. The baseline is not linked into the production library.

```sh
python3 scripts/benchmark_pauli.py --output /tmp/pauli-timing
# Choose sizes that fit the available GPUs; vendor flags/config.mk apply:
python3 scripts/benchmark_pauli.py --output /tmp/pauli-single-timing --ranks 1 --qubits 12 20
python3 scripts/benchmark_pauli.py --output /tmp/pauli-timing-plan --dry-run
# Direct launch on a configured machine:
make benchmark-pauli RANKS=2 BENCH_QUBITS=20 BENCH_WARMUPS=5 BENCH_REPETITIONS=30
```

The default sweep uses 12/20/24 total qubits and 1/2/4 GPUs. It includes diagonal
strings, low/middle/high flip pivots, sparse masks, and global X targets on
multi-GPU runs. Kernel rows record CUDA-event Stage 1 and Stage 2 durations and
host duration including the two launches, NCCL, scalar readback, and final stream
synchronization. API rows record the full production call including validation,
projection checks, localization/SWAP when required, NCCL, and readback. Stage
fields are blank for API rows; their block field is zero (not a launch count).
Global-X rows require SWAP; other workloads start fully local. Their operator
layouts differ, so the difference is not an isolated estimate of SWAP latency.

Before **every** warmup and retained sample, the identity mapping is restored
and deterministic logical amplitudes are regenerated on-device, followed by
an untimed stream synchronization and MPI barrier. This deliberately restores
equivalent input and required exchange costs rather than allowing repeated API
calls to eliminate SWAP. No full-vector temporary buffer is allocated. Kernel
methods are interleaved per sample to limit drift. Each measured duration is the
maximum across ranks; maxima for separate stages may belong to different ranks
and must not be added to estimate an end-to-end maximum. Event instrumentation
is present in kernel rows but absent from API rows.

The harness rejects non-finite results and requires generated/baseline/API
expectations to agree within `1e-12 + 5e-12 * abs(reference)`. These comparisons
check benchmark consistency; independent numerical coverage remains in the
regression suite. Run that suite before interpreting timings.

The runner retains raw per-repetition CSV files and per-run logs, including
rank/host/GPU placement, PCI IDs, SM counts, compute capability, CUDA driver and
runtime versions, NVCC/host compiler versions, and NCCL/MPI versions. Its manifest
records the source revision, working-tree status, build flags/commands, and run
statuses. `summary.csv` reports p10/median/p90 per metric; p10/p90 select the
sorted sample at `floor(p * (count-1))`. `ratios.csv` gives scan/generated ratios
for median Stage 1 and host kernel-pipeline time: values above one favor generated
pairs. Retain raw variability and slower results. Scratch bytes report the
reserved shared measurement slice, not additional memory per Pauli call.
Dry runs remain `planned` and contain no timing results.

No GPU measurements were obtained here: the actual benchmark build stops at
missing `nvcc`. The CUDA harness is uncompiled and unexecuted locally. No speedup,
bandwidth, stage latency, multi-GPU scaling, or optimization benefit is claimed.
Production memory-access tuning is deferred until measurements justify it.
