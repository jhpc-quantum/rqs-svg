# Measurement reducer and reproducible validation

Production measurement uses the custom shared-memory, bin-major two-stage
reducer in `src/measurement_reduce.cuh`. There is no production CUB backend.
Double precision and the public API are unchanged; groups contain at most four
qubits. The actual pre-migration baseline recorded before task 1 and the reviewed
baseline are both `1ce55aff510934fd1502abf933e8c26a1db79133`.
The reducer was introduced in `2d15882` and integrated in `7f12af8`.
Comparison runs record the full current and baseline revisions in `manifest.json`.

## Algorithm and ownership

The reducer adapts custom warp/block sums, shared first stage, second stage,
and resource checks from
[naoto-aoki-fy/cuda-reduce at 15bb3cd08e72a423e2be7d517003567f681b4f26](https://github.com/naoto-aoki-fy/cuda-reduce/tree/15bb3cd08e72a423e2be7d517003567f681b4f26).
That pinned tree supplies no license file or source notices; attribution does not
assert an upstream license grant.

For width N=1..4, the descriptor specifies local physical targets, their output
bit positions, fixed outcome bits, and **all** prior local control mask/value bits,
including controls outside the request. Controls and targets must be disjoint.
A packed domain of `num_states_local >> popcount(control_mask)` inserts control
bits to enumerate eligible physical indices in ascending order. Each amplitude
contributes its squared norm to one of `2^N` raw probability bins. L=0 is valid;
impossible bins and excluded ranks contribute exact zeros.

The launch policy remains S=256 threads and G=8*SM_count blocks in both stages.
Stage one writes `partials[bin*G + block]`; stage two reads using that actual G.
Dynamic shared bytes are `256 * 2^N * sizeof(double)` (32 KiB at N=4), plus
compiled static scratch. Exact kernel limits, opt-in shared capacity, and nonzero
occupancy are checked on first use. There is no fallback or measured tuning.

Each simulator owns one device/stream-bound workspace. It grows on demand and
reuses storage; all used elements are overwritten. `enqueue` launches both stages
without synchronization and returns device bins valid until the next call or
close. Results are copied and the stream is synchronized before MPI aggregation.
The owning stream must outlive the workspace; explicit `close()` reports release
errors. Descriptor shifts, allocation products, and indexing use checked 64-bit
operations. The reducer receives metadata by value and does not depend on the
gate constant/global metadata storage selector.

## Regression harnesses

Use the production vendor include/link/compiler flags, including MPI, NCCL,
cuRAND and atlc. One GPU per rank is required; simulator initialization assigns
GPUs by node-local MPI rank. Launch these commands from the repository root:

```sh
make sharedlibrary bin/measurement_reduce_test bin/measurement_integration_test
./bin/measurement_reduce_test
mpiexec -n 1 ./bin/measurement_integration_test
mpiexec -n 2 ./bin/measurement_integration_test
mpiexec -n 4 ./bin/measurement_integration_test
make test-measurement-distributed RANKS=1
make test-measurement-distributed RANKS=2
make test-measurement-distributed RANKS=4
make test-measurement-metadata RANKS=1 MEASUREMENT_METADATA=constant
make test-measurement-metadata RANKS=2 MEASUREMENT_METADATA=constant
make test-measurement-metadata RANKS=4 MEASUREMENT_METADATA=constant
make test-measurement-metadata RANKS=1 MEASUREMENT_METADATA=global
make test-measurement-metadata RANKS=2 MEASUREMENT_METADATA=global
make test-measurement-metadata RANKS=4 MEASUREMENT_METADATA=global
```

The existing reducer executable covers every N/L pair, poisoned buffers,
reuse/growth, zero/sparse/nonuniform complex inputs, very small active domains,
noncontiguous and reversed target order, both control polarities, teardown,
nondefault streams, rejection cases, and bounded indices above 32 bits without
huge allocations. Its independent compensated long-double CPU oracle enumerates
physical indices. Analytic amplitudes 1..8 with targets [0,2] give bins
[10,20,74,100]; reversed targets give [10,74,20,100].

The new internal harness compiles a private instrumented copy of the existing
source; it adds no public probability interface. After each real measurement
Allreduce, it independently enumerates the physical state, applies the recorded
logical measurement mask, and constructs every bin in caller order. It compares
both local and MPI-summed bins, including exact-zero excluded ranks, then checks
sample agreement on all ranks. It exercises all feasible N/L pairs for the rank
count, unequal masses, a reverse permutation crossing the local/global boundary,
low/high/noncontiguous targets, prior local/global controls, controls inside and
outside the current request with pending projection, repeated measurements, and
requests of lengths 5..11 covering all remainder widths. All-global groups wider
than log2(ranks) are impossible at 1/2/4 ranks; the direct reducer test covers
synthetic L=0 descriptors at every width.

The public integration executable covers basis ordering, classical destinations,
empty/all-already-measured requests, invalid inputs, fresh shots/remainders,
Bell and biased GHZ correlations, gates after measurement, reset, and projected
statevector output. Both CPU oracles require
`abs(error) <= 1e-12 + 5e-12*abs(expected)` with finite values and exact zero for
impossible bins. The internal harness prints the observed maximum absolute error.
RNG calls in production are unchanged. Floating-point boundary bitstrings and
bitwise CUB sums are not acceptance criteria.

The harness source copier supports `constant` and `global` metadata variants in
separate build directories, leaving the production selector untouched. The
`test-measurement-metadata` target compiles the existing public API cases together
with the private source copy, then runs both public and internal coverage. This
avoids changing the installed library or its runtime search path. The standalone
reducer has no metadata selector.

For small sanitizer runs (use a launcher that propagates the sanitizer exit code):

```sh
make measurement-harness
compute-sanitizer --error-exitcode 1 --tool memcheck ./bin/measurement_reduce_test
compute-sanitizer --error-exitcode 1 --tool racecheck ./bin/measurement_reduce_test
compute-sanitizer --error-exitcode 1 --tool synccheck ./bin/measurement_reduce_test
mpiexec -n 2 compute-sanitizer --error-exitcode 1 --tool memcheck ./bin/measurement-harness-constant/measurement_harness regression 12
mpiexec -n 2 compute-sanitizer --error-exitcode 1 --tool racecheck ./bin/measurement-harness-constant/measurement_harness regression 12
mpiexec -n 2 compute-sanitizer --error-exitcode 1 --tool synccheck ./bin/measurement-harness-constant/measurement_harness regression 12
```

## Paired performance runs

Python 3.12+ is needed for the archive extraction safety filter. Each command
below builds both probes with the current Makefile/config.mk and identical vendor
flags, extracts the recorded baseline into a separate temporary tree, and keeps
build logs, raw CSV, device/runtime/compiler/MPI logs, commands, configuration,
environment overrides, working-tree status, and revision IDs. Output directories
must be new. Only tracked baseline files are extracted. The current custom
source includes any working-tree changes recorded in the manifest; use a clean
committed tree for publishable comparisons.

```sh
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-1 --ranks 1 --qubits 12 24 --warmups 5 --repetitions 30
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-2 --ranks 2 --qubits 13 25 --warmups 5 --repetitions 30
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-4 --ranks 4 --qubits 14 26 --warmups 5 --repetitions 30
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-global --ranks 2 --qubits 13 25 --metadata global
```

These state sizes hold local qubit counts constant across ranks; choose smaller
sizes if memory is limited. The driver reports GPU model, rank/GPU count, CUDA
runtime/driver, compiler, and MPI implementation. Gencode/toolchain/link flags
are in build logs and the manifest. It sweeps N=1..4, physical low `[0..N-1]`,
high `[local_qubits-1..local_qubits-N]`, noncontiguous `[0,2,...,2*(N-1)]`, and
mixed `[first_global,2,4,...]` positions. The logical permutation is reverse.
Controls are absent, deterministic local physical bit 5=0, or deterministic first
global bit=0. Global control cases use local layouts to avoid overlap. State
amplitudes are deterministic functions of the full physical index; conditioning
zeros the discarded branch before upload. For a local control, active states per
rank halve; for a global control, half the ranks have zero contributions and
eligible ranks retain the full local active domain. CSV active_states reports
the eligible rank domain. GPU count assumes one GPU per rank on homogeneous GPUs;
record heterogeneous node/device configurations separately.

Every fresh timed measurement restores identical state and discards measurement
metadata outside timing. Prior controls are measured outside timing. Stream
synchronization completes uploads and metadata preparation before an untimed
MPI barrier. An extra discarded priming iteration precedes the requested warmups
and repetitions, so reported samples reuse allocated workspace. No gate or
cached measurement substitutes for the fresh measurement operation.

CUDA events bracket both custom kernel launches or the single whole CUB Reduce
call, after resource/allocation setup and before any result download. This
includes normal launch submission gaps but excludes state preparation and
validation downloads. Host wall-clock intervals cover the public fresh group
measurement: reduction, host result transfer/synchronization, MPI aggregation,
sampling and broadcast. They also include event-record instrumentation overhead;
compare identically instrumented builds. The regression observer is disabled
in benchmark mode. Both metrics use the **maximum rank duration per repetition**
and report p10/median/p90 milliseconds separately.

CSV includes state/active sizes, layout, controls, N, S/G, first-stage dynamic and
static shared bytes, workspace capacity, warmups/repetitions. Custom static
scratch is queried from the compiled N=4 kernel (same scratch declaration for
all widths); the second stage uses static scratch only. CUB S/G/shared fields
are zero meaning unspecified library internals; workspace includes CUB scratch
and result bytes. Capacity can reflect earlier larger cases in the sweep.
The script writes all median ratios `CUB/custom` to `comparison.csv`, flagging
values below one as regressions; retain both raw variability intervals when
reporting them. Nonpositive timing rows cause comparison failure. Keep raw rows;
do not promise a speedup.
For a single custom run, use `make benchmark-measurement RANKS=2 BENCH_QUBITS=13`.

## Execution report and remaining gaps

Tests, builds, validation, sanitizer runs, environment probes, and benchmarks
were **not executed**, at the user's explicit request. Hardware/tool availability
was not probed; no absence or availability is inferred. Numerical tolerances above
are acceptance thresholds, not observed errors. Observed errors, MPI/projection
results, sanitizer results, and CUB-versus-custom timing/regressions are all
**unmeasured**. No runtime failure was diagnosed, no production fix or launch
geometry tuning was made, and no post-fix rerun is claimed.

The commands above are the follow-up matrix. Complete them on matching CUDA/MPI
hardware, attach manifests and raw CSV, report actual device/compiler details and
maximum numerical errors, and investigate failures before claiming acceptance.
The harness and added coverage themselves remain uncompiled and unverified.
