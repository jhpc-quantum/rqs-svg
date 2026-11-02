# Measurement reducer and reproducible validation

Production measurement uses the custom shared-memory, bin-major two-stage
reducer in `src/measurement_reduce.cuh`. CUB is used for the subsequent
small cumulative scan, rather than state-vector reduction.
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
occupancy are checked when reserving workspace (at simulator allocation time,
or on first use for the standalone owning workspace). There is no fallback or measured tuning.

Each simulator owns one device/stream-bound measurement workspace borrowing a
fixed slice of the shared work buffer. After state allocation, the work buffer
reserves 95% of remaining CUDA memory. The measurement slice is sized once for
the largest requirement across N=1..4, before the swap slice is carved out;
measurements do not grow or allocate storage. All used elements are overwritten.
The standalone reducer test also supports an owning workspace that grows on
demand. `enqueue` launches both stages
without synchronization and returns device bins valid until the next call or
close. NCCL AllReduce sums the output bins in place on the same stream, followed
by the device sampler and an eight-byte outcome/status copy. Only that final
readback is followed by stream synchronization; weights and prefixes remain on
the device throughout production sampling.
The owning stream and backing work buffer must outlive the measurement workspace.
For a borrowed slice, `close()` only detaches it; the simulator frees the backing
buffer after all uses. For a standalone owning workspace, `close()` reports
release errors. Descriptor shifts, allocation products, and indexing use checked 64-bit
operations. The reducer receives metadata by value and does not depend on the
gate constant/global metadata storage selector.

## Device sampling and measurement RNG

After NCCL, `sample_outcome<N>` launches exactly one block of 32 threads for
N=1..4 (2..16 bins). It uses `cub::BlockScan<double, 32,
cub::BLOCK_SCAN_RAKING>` with one item per lane and zero padding. Original
weights remain available beside the shared cumulative array. The scan uses an
explicit round-to-nearest double addition operator (`__dadd_rn`). It takes its
total from cumulative entry `2^N-1`; no separate total reduction is performed.
Thread zero checks every original weight for finiteness and nonnegativity and
the total for finiteness and strict positivity before drawing. Invalid weights
or totals return distinct compact status codes and do not advance the counter.

Reproducible selection assumes NCCL supplies bitwise identical reduced bins to
all ranks, and all participating GPUs execute the same compiled CUB scan
implementation with the same CUDA/CUB version, compiler flags, target compute
capability, and IEEE double arithmetic. Fixed geometry, the explicit raking
algorithm and rounded additions keep the prefix addition order independent of
local state size, SM count and reducer launch geometry. A default DeviceScan
policy is not used. Cross-architecture or mixed-toolchain runs are outside these
assumptions until agreement is established. There is no broadcast to repair
prefix, random-value or outcome disagreement. Agreement is asserted by the
private harness, but has **not been validated**
in this change because execution was explicitly skipped. Equal seed/counter
inputs imply equal raw draws; agreement within a supported execution additionally
requires the numerical assumptions above. Exact outcome sequences across GPU
counts, decompositions, architectures, compiler flags or toolkit versions are
not promised: reduction order and rounding may change the bins.

The device creates a local `curandStatePhilox4_32_10_t` using
`curand_init(common_rng_seed, measurement_counter, 0, &state)` and calls
`curand4` once. It forms the 53-bit integer `(uint64_t(bits.x) << 21) |
(uint64_t(bits.y) >> 11)` and multiplies by `2^-53` to obtain `u` in `[0,1)`.
The existing setup seed is unsigned 32-bit and is promoted to 64 bits for
this mapping; the internal sampler also accepts full 64-bit seeds in tests.
No cuRAND uniform endpoint convention is involved. No rank, GPU identity,
thread identity, local mass, timing, or local domain size enters this mapping.
The existing one-time seed broadcast remains; no RNG state, counter, draw or
outcome is communicated. State-vector initialization retains its independent,
rank-dependent cuRAND generator and seeds.

Each simulator initializes its replicated 64-bit counter to zero through its
member initializer, once per simulator lifetime. A successfully sampled group
with any previously unmeasured target consumes its current value, including a
deterministic distribution and ranks contributing zero weights. After successful
status readback the counter advances, before updating measurement bookkeeping.
Empty requests and entirely cached groups consume nothing. Rejected arguments,
invalid distributions and failed sampler/readback operations do not advance it.
State initialization, mapping resets, measurement/classical bookkeeping resets
and resets between shots never reset the cursor. Sampling counter `UINT64_MAX`
sets a separate exhausted flag; the next fresh measurement throws before
reduction instead of wrapping. Cached measurements still work after exhaustion.

The threshold is the rounded product `u * total`, clamped with
`nextafter(total, 0)` if rounding reaches the upper endpoint. Selection scans bins
in request order and takes the first original positive-weight bin whose prefix
is strictly greater than the threshold. An endpoint fallback chooses the last
positive-weight bin, so a trailing zero bin is never selected. This preserves
outcome bit `i` as requested qubit `qubits[i]`.

The sampler reuses the dead partial-reduction storage for its eight-byte result
(two 32-bit fields: outcome and status), without another allocation or teardown
resource. The synchronous API reads back only this result and retains its
existing measured masks, value lists, pending projections and classical-bit
updates. No state-vector exchange is introduced. Production host validation,
cumulative summation, random draws and outcome broadcasts have been removed.
The old `std::mt19937_64` measurement sequence is intentionally not preserved.

## Regression harnesses

Use the production vendor include/link/compiler flags, including MPI, NCCL,
cuRAND and atlc. One GPU per rank is required; initialization assigns GPUs by
node-local MPI rank. Run from the repository root:

```sh
make sharedlibrary bin/measurement_reduce_test bin/measurement_integration_test
./bin/measurement_reduce_test
mpiexec -n 1 ./bin/measurement_integration_test
mpiexec -n 2 ./bin/measurement_integration_test
mpiexec -n 4 ./bin/measurement_integration_test
make test-measurement-distributed RANKS=1
make test-measurement-distributed RANKS=2
make test-measurement-distributed RANKS=4
make test-measurement-metadata RANKS=2 MEASUREMENT_METADATA=constant
make test-measurement-metadata RANKS=2 MEASUREMENT_METADATA=global
# Repeat metadata variants with RANKS=1 and RANKS=4 where available.
# Example Open MPI two-node launch; use the site's scheduler/launcher as needed.
make measurement-harness
mpiexec --host nodeA:1,nodeB:1 -n 2 ./bin/measurement-harness-constant/measurement_harness regression 12
mpiexec --host nodeA:1,nodeB:1 -n 2 ./bin/measurement_integration_test
```

The standalone reducer executable covers every N/L pair, poisoned buffers,
reuse/growth, sparse/nonuniform inputs, small active domains, noncontiguous and
reversed target order, controls, teardown, streams, invalid descriptors and
bounded indices above 32 bits. Its independent compensated long-double oracle
enumerates physical indices.

`tests/measurement_sampling_cases.hpp` runs inside the private distributed
harness. An internal `Inspect=true` kernel specialization records prefixes,
raw Philox words and the uniform, and can override the selection uniform.
The production specialization compiles out these diagnostics and overrides.
Tests use synthetic device weight buffers at every width 1–4, every one-hot bin,
nonuniform `[0,2,0,6]` (or `[2,6]` for width one), all interior boundaries of a
uniform distribution, leading/interior/trailing zeros, unnormalized totals,
widely separated powers of two, and a subnormal total whose upper-end product
rounds to the endpoint. For `[0,2,0,6]`, uniforms 0, 0.25 and nextafter(1,0)
expect indices 1, 3 and 3. Tests check values on both sides of boundaries and
reject negative/NaN/infinite weights, zero totals and overflowed totals.
Finite prefixes are compared with a long-double serial oracle at relative
error at most `5e-15`; selected bins must have positive original weights.
Raw draws are repeated with two seeds and counters 0, 2^32, 2^32+17 and
UINT64_MAX, compared across ranks and against their 32-bit-truncated inputs.
The 53-bit uniform mapping is also checked explicitly.

The counter fixture calls the real public measurement path, starting above
2^32, and asserts the cursor on every rank after deterministic, empty, cached,
mixed cached/fresh, full and remainder groups. With reverse mapping, prior
global controls exclude ranks while those ranks continue incrementing. State
preparation, measurement reset and the exact between-shot reset sequence from
`qcs_main.cpp` must preserve the cursor. All-zero and NaN statevectors must
fail through the C API without advancement. UINT64_MAX is consumed once;
further fresh requests fail while cached/empty requests remain usable. A
warmed workspace pointer and capacity must survive subsequent measurement.
Only the private fixture sets/restores counters; the public API is unchanged.

The source copier instruments the NCCL boundary in a private source copy,
reading local bins before NCCL and global bins afterward. These extra copies
and test-only MPI reductions are enabled only when the regression observer is
installed, never in production or benchmark mode. Independent physical state
enumeration applies the recorded measurement controls and compares both local
and global bins, requiring finite values and
`abs(error) <= 1e-12 + 5e-12*abs(expected)` and exact zeros for excluded bins.
The harness compares diagnostic prefixes, raw draws, uniforms, statuses and
outcomes byte-for-byte across ranks, then compares the actual production
specialization's returned outcome/status with the diagnostic sample using the
same bins, seed and counter. No agreement is repaired by these assertions.

Distributed cases cover all feasible local/global mixes, reverse mappings,
reordered targets, prior controls inside/outside a request, excluded ranks,
pending projections, cached measurements and lengths 5–11 covering all
remainders. All-global groups wider than log2(ranks) cannot occur with 1/2/4
ranks; synthetic sampler/reducer cases cover all widths independently.

The public integration executable covers single measurement, both multiple
measurement APIs, basis states, classical destinations, empty/repeated requests,
invalid inputs, Bell/biased GHZ correlations, gates after measurement, reset and
projected statevector output. An additional 2048 independent biased Bell shots
check correlations and per-shot rank agreement. For p=sin(0.7/2)^2, the count
of ones must differ from 2048*p by at most
`7*sqrt(2048*p*(1-p)) + 2`. This binomial tolerance is deliberately conservative;
no legacy CPU sequence is required. These are acceptance criteria, not observed
results.

The copier retains support for the recorded CPU/CUB baseline used by the paired
benchmark driver. Constant/global metadata variants build in separate private
directories. Neither changes the installed library or public API.

## Production data path and profiling

The source path is `simulator_core::measure_qubits` -> local reduction or
`enqueue_zero` -> same-stream `ncclAllReduce` -> `enqueue_sample` ->
`sample_outcome` -> one `sizeof(sample_result)` device-to-host copy -> stream
synchronization. The result is eight bytes. Cumulative selection and Philox run
on device. No measurement probability MPI reduction, CPU draw, CPU cumulative
selection or sampling broadcast occurs on this path. Setup still broadcasts the
common seed. The workspace reuses dead partial storage for the result and
retains its preallocated capacity. The regression source copy deliberately adds
array downloads; it must not be used to claim production transfer behavior.
This description comes from implementation inspection, not a runtime trace.

For runtime verification, profile the uninstrumented production executable with
a circuit containing repeated fresh measurements after warm-up. Record the
actual circuit, flags, GPU/node topology, CUDA/CUB/NCCL/MPI versions and revision:

```sh
nsys profile --trace=cuda,mpi,nvtx -o /tmp/qcs-measurement ./bin/qcs -s 100 user_circuit.so
# Distributed example (one trace per rank, shared output directory):
mpiexec -n 2 nsys profile --trace=cuda,mpi,nvtx -o '/tmp/qcs-measurement-%h-%p' ./bin/qcs -s 100 user_circuit.so
```

Inspect CUDA transfers, sampler launches, NCCL kernels and allocation calls.
Require eight-byte sampling readbacks and no weight/prefix downloads or
sampling MPI reductions/broadcasts. Exclude initialization seed/NCCL setup,
state preparation and explicit statevector output when attributing transfers.
Check warmed measurements for allocation reuse; pool allocation calls still
count even if they avoid a device allocation. No trace was captured here.

Optional sanitizer runs, after building the executables:

```sh
compute-sanitizer --error-exitcode 1 --tool memcheck ./bin/measurement_reduce_test
mpiexec -n 2 compute-sanitizer --error-exitcode 1 --tool memcheck ./bin/measurement-harness-constant/measurement_harness regression 12
mpiexec -n 2 compute-sanitizer --error-exitcode 1 --tool racecheck ./bin/measurement-harness-constant/measurement_harness regression 12
mpiexec -n 2 compute-sanitizer --error-exitcode 1 --tool synccheck ./bin/measurement-harness-constant/measurement_harness regression 12
```

## Paired performance runs

Python 3.12+ is required for the archive extraction filter. The driver builds
isolated current and historical CUB probes with identical vendor flags and
records manifests, commands, build/environment logs, raw CSV and median ratios:

```sh
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-1 --ranks 1 --qubits 12 24 --warmups 5 --repetitions 30
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-2 --ranks 2 --qubits 13 25 --warmups 5 --repetitions 30
python3 scripts/compare_measurement.py --output /tmp/qcs-pair-4 --ranks 4 --qubits 14 26 --warmups 5 --repetitions 30
make benchmark-measurement RANKS=2 BENCH_QUBITS=13
```

Use new output directories and matching workloads/environments. The default
baseline is the historical whole pipeline, including CPU sampling and MPI
probability reduction; this comparison does not isolate the effect of removing
host sampling. The source copier supports that baseline and the current GPU
path, not arbitrary intermediate revisions. CUDA events bracket only statevector
reduction. Host timing covers the fresh public measurement including reduction,
NCCL, scan, sampling and scalar readback in the current pipeline (host sampling
and MPI in the baseline). Both include equivalent event instrumentation. The
regression observer and sampler diagnostics are disabled during timing.

The sweep covers widths 1–4, low/high/noncontiguous/mixed layouts, absent/local/
global controls, and a reverse mapping. Identical prepared amplitudes and prior
controls are restored outside timing; an untimed barrier precedes each sample.
An extra priming iteration and warmups precede retained samples. Report maximum
rank duration per repetition, p10/median/p90, GPU/node topology, software versions,
flags and both revisions. CSV workspace capacity reflects reserved measurement storage.
Retain variability and regressions; removing host sampling does not establish
a speedup for every workload.

## Completion report

Implementation includes controlled sampler diagnostics, numerical/RNG tests,
real-path counter/error/reuse coverage, a repaired NCCL-aware private harness,
and statistical public integration coverage. The maximum/default group width
remains four. No public API change or group-size expansion was made.

Tests and validation were explicitly skipped by user request. No build, test,
script syntax check, sanitizer, runtime trace, device/tool probe or timing command
above was executed. CUDA availability, GPU count, multi-node support and tool
versions were not probed; none is claimed unavailable. Consequently all single-
GPU, two-/four-GPU and multi-node runs remain skipped, and all added coverage
is uncompiled and unverified. No latency result or measured numerical error is
reported.

Work was performed in `/root/Projects/20261031-rqs-svg` using Bash. Commands
actually executed were source/document reads and searches (`cat`, `sed`, `rg`,
`find`, `ls`, `pwd`), Python file edits, and Git status/log/diff/staging/commit
operations. Source inspection and final diff review were needed to implement
and commit the requested changes; they are not execution validation. Use the
matrix above on a configured CUDA/MPI installation before claiming runtime
agreement or performance.
