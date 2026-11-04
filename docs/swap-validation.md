# Dedicated SWAP regression and comparison report

Status: tooling added; tests, compilation, symbol inspection, sanitizers and
benchmarks have **not been run**, as requested. There are no measured correctness
or performance results in this report. Toolchain and GPU availability were not
probed. The commands below are deferred work for a configured CUDA/NCCL/MPI host.

## Source observations and expected behavior

The implementation under assessment is commit `9f833b5`; its parent is the
original generic implementation. `src/qcs.cu` retains the live
`qcs::gate::swap<num_swaps>` definition and its `apply()` body. The common public
SWAP entry point calls `sim->core->operate_swap`, which launches `cuda_swap`.
SWAP1 through SWAP4 wrappers call that common entry point. These are source
observations, not evidence of successful compilation or emitted device code.

The dedicated kernel enumerates target bits and non-operand bits, fixing controls
through the existing index helper. It exchanges each adjacent pair in the supplied
target list to construct a partner index. Disjoint transpositions compose to an
involution: each amplitude belongs to a fixed point or a two-element cycle.
Only the thread with `index < partner` reads and writes a cycle, and it reads both
amplitudes before either write. This gives each nontrivial cycle one writer,
assuming the shared index helper enumerates eligible states uniquely. Regression
and sanitizer execution are still required to assess that assumption in practice.

## Added coverage, pending execution

`tests/swap_harness.cu` includes the production translation unit to access private
state amplitudes. It introduces no production hooks or legacy template calls.
Its CPU oracle applies each transposition sequentially to a copied vector, in
logical coordinates. Snapshots convert physical storage back to logical indices.
Every comparison checks finite amplitudes, a maximum amplitude error below
`2e-12`, and a global squared-norm error below `2e-11`.

The matrix covers all widths, ascending, descending and interleaved distant target
lists, identity and reversed mappings, no/negative/positive/mixed controls,
normalized unequal complex amplitudes, and basis states that satisfy or reject
controls. It checks five successive operations per case, alternating the generic
C API and width-specific wrappers. Dense states contain all target patterns,
including patterns changed by several disjoint transpositions at once.

Measurement cases cover a measured target with deferred and already materialized
projection, a measured control that permits or rejects execution, and an unrelated
measured qubit whose projection stays pending across SWAP. Effective snapshots
mask pending branches on the CPU without materializing GPU state prematurely.
A final materialized snapshot also checks the stored state. Projection intentionally
retains the selected branch's raw norm, matching the simulator; SWAP must preserve
that norm rather than implicitly renormalizing it.

With multiple ranks, reversed mappings put targets across the local/rank boundary,
while high logical controls in the identity mapping exercise rank conditions.
The same CPU oracle runs on each rank, with collective error/norm checks and
measurement-outcome agreement. Use a power-of-two rank count and at least eight
local qubits for SWAP4. Standard comparison runs support 1, 2 and 4 ranks.

```sh
make swap-harness
make test-swap RANKS=1
make test-swap RANKS=2
make test-swap RANKS=4
# On a supported single-GPU CUDA installation:
compute-sanitizer --tool memcheck ./bin/swap_harness regression 12
compute-sanitizer --tool racecheck ./bin/swap_harness regression 12
```

CUDA racecheck primarily diagnoses shared-memory hazards; its success alone does
not prove the absence of global-memory races. The ownership argument and repeated
full-vector comparisons provide complementary evidence. Larger randomized target
matrices, exhaustive target permutations, and sanitizer/MPI combinations remain
uncovered. Optional NVSHMEM runs require a separately configured build, sufficient
symmetric heap, and the appropriate cluster launcher; they are untested here.

## Legacy instantiation audit, pending execution

```sh
make sharedlibrary
python3 scripts/check_swap_instantiations.py --source . \
  --binary lib/libqcs.so --require-dedicated
```

The audit records the retained definition, references to `gate::swap<...>` under
`src`, each public wrapper's common dispatch, and the dedicated kernel/host path.
When supplied a binary it records demangled `nm` and, when available,
`cuobjdump --dump-elf-symbols` output alongside that binary. Dedicated mode rejects
legacy symbol matches and requires a dedicated symbol to be visible in at least
one available tool. Inspect both symbol logs, particularly device symbols, before
concluding that no `cuda_gate<qcs::gate::swap<N>>` was emitted. The source scanner
uses lexical patterns for this implementation, not a C++ parser: aliases, macros,
stripped binaries, and tool limitations can limit the evidence. No automated
source audit or symbol inspection has been executed in this task.

## Before/after builds and runtime, pending execution

Configure `config.mk` as described in [Building](building.md), then run:

```sh
python3 scripts/compare_swap.py --output /tmp/swap-comparison-r1 \
  --ranks 1 --qubits 12 20 24 --build-repetitions 3 --nvcc-time
python3 scripts/compare_swap.py --output /tmp/swap-comparison-r4 \
  --ranks 4 --qubits 12 20 24 --build-repetitions 3 --nvcc-time
```

Omit `--nvcc-time` if that NVCC version does not support `--time`. Use
`--make-arg 'GENCODE_FLAGS=...'` and other make assignments as needed. NVSHMEM
builds need both `--make-arg 'NVSHMEM=1'` and `--make-arg 'NCCL_BUFFER_MODE=0'`;
every assignment is applied identically to both variants. Disable
compiler caches in the supplied compiler command/environment. Use fixed GPU clocks
and an otherwise idle machine when comparing runtime. Review `config.mk` for
external cache wrappers. The driver forces serial rebuilds with `make -B -j1`,
uses separate retained `git archive` trees, and never modifies production sources.
It copies the current configuration and one identical harness into both trees.
Working-tree source edits are excluded: commit them first or select explicit
committed revisions with `--baseline` and `--current`.

The manifest records exact commits, configuration, selected build/runtime
environment, NVCC/GPU/MPI information, expanded build logs, commands, raw repeated
build measurements and symbol audit results. Production shared-library and private
harness compilation are measured separately. `build-comparison.csv` contains
median build wall time, peak RSS and binary size for both. Peak RSS comes from GNU
`/usr/bin/time` and describes the largest observed process footprint, not the sum
of simultaneously resident compiler subprocesses. NVCC phase CSVs, when enabled,
retain frontend, device optimization and PTX assembly timing for manual comparison;
phase naming depends on the toolkit. Forced rebuilds prevent Make timestamp reuse,
but do not flush OS filesystem caches. The baseline-first ordering can introduce
thermal/cache bias: repeat with the order reversed in a controlled experiment if
small differences matter.

Both variants must pass regression before their runtime comparison proceeds. Each
runtime size has 48 rows (four widths, three layouts, four control modes). The
benchmark initializes complex amplitudes and times completed steady-state public
API operations, including host preparation and stream synchronization, using
`MPI_Wtime` and the maximum elapsed time across ranks. Localization is settled
before sampling, so initial exchange cost and isolated kernel time are not
measured. It reports median, p10 and p90 per operation. Identity-mapped targets
stay below the top two control bits; distributed benchmark controls can reside in
rank bits. Regression covers target localization under reversed mapping.

`comparison.csv` gives current/baseline median ratios and flags ratios above 1.10
(configurable with `--regression-threshold`). Flags require investigation and an
application-specific acceptance decision; they are not an automatic performance
pass/fail. Raw runtime logs and extracted CSVs are retained. Run
`make benchmark-swap BENCH_QUBITS=24 RANKS=1` for just the current implementation.

| Requested evidence | Result in this task |
| --- | --- |
| SWAP1–SWAP4 correctness and norm | Harness added; not run |
| Distributed and pending-projection correctness | Cases added; not run |
| No legacy template instantiations | Source path described; automated/symbol audit not run |
| Original/current compilation wall time and peak memory | Driver added; no measurements |
| Original/current NVCC phase timings and binary size | Driver added; no measurements |
| Original/current runtime performance | Driver added; no measurements |

Compilation improvement and runtime acceptability remain open. No successful
build, GPU validation, or performance improvement is claimed.
