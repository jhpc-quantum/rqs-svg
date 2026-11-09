# Adaptive measurement reduction launches

The old first stage always launched 8 blocks per SM, even for one active
amplitude. The second stage launched the same grid although only 2–16 blocks
could process bins. Its partial stride was implicitly its own grid size.

The first stage now uses min(8*SMs, ceil(active_count/256)) blocks. Bins do not
multiply useful first-stage parallelism: each amplitude contributes to exactly
one bin, and each block reduces every bin. The second stage uses one block per
bin and an explicit first-stage partial count. Block size remains 256 and
shared storage is unchanged. No specialized kernel or tuning threshold is added
without measurements. Large domains keep the previous first-stage launch cap.

Allocation remains at maximum capacity, with output after the maximum partial
region. Live partials use the actual first-stage grid as their bin-major stride.
Every live partial and output bin is overwritten. Second-stage reads stop at the
live count, so stale values in the unused capacity cannot contribute. The tile
calculation divides before rounding to avoid uint64 overflow. Existing checked
allocation products and device grid checks remain in force.

No grouping, descriptor mapping, sampler, rank exclusion, NCCL participation,
collective ordering, host synchronization, or public API changes are made.
Excluded ranks retain the existing output-zeroing path.

## Validation in this environment

Passed: `make test-measurement-launch test-work-buffer test-swap-launch`.
The new host geometry test sweeps tile boundaries and caps and checks uint64
limits and invalid inputs. Both constant/global private harness source copies
were generated successfully with `scripts/prepare_measurement_harness.py`.

`make test-measurement-reduce` failed to build: nvcc was not found. Neither
nvcc nor nvidia-smi is on PATH and no CUDA installation appears in /usr/local.
CUDA tests, sanitizers, distributed execution and benchmarks were not run.
There are no before/after latency results and no established performance gain.

The reducer's existing independent compensated long-double oracle checks every
width/local-target mix, tiny domains, fixed global outcome bits, noncontiguous
controls and NaN-poisoned storage. Added consecutive-group cases at widths 1–4
shrink a 4096-element domain to one active element with alternating control
values, retaining prior partials between groups. Existing distributed regression
covers local/global targets, excluded ranks, masks and groups with remainders.
Run on CUDA/MPI hardware:

```sh
make test-measurement-reduce
make test-measurement-distributed RANKS=1
make test-measurement-distributed RANKS=4
make test-measurement-metadata RANKS=4 MEASUREMENT_METADATA=global
compute-sanitizer --error-exitcode 1 --tool memcheck ./bin/measurement_reduce_test
```

## Paired target-system procedure

Obtain the baseline with `git rev-parse <candidate>^`. Export both revisions with
`git archive` into separate directories (no branch switching needed). Build each
with identical site config.mk/vendor flags, CUDA, MPI, NCCL and architecture
flags for GB200. In each exported directory:

```sh
make target
./compile_circuit.sh examples/standalone/measure-only.c -o /tmp/measure-only.so
# Example Open MPI mapping; allocate NODES nodes with four GPUs each first.
export NUM_QUBITS=33
mpiexec -n "$RANKS" --map-by ppr:4:node ./bin/qcs -s 40 /tmp/measure-only.so > run.jsonl
```

Run (NODES,RANKS) = (4,16), (8,32), (16,64), (32,128). Use a distinct
output per revision, rank count and repetition. Discard the first 10 samples,
retain 30, and repeat paired jobs at least three times with alternating order.
Report p10/median/p90 of JSON `elapsed_time`, units seconds, and paired ratios;
this is the existing event interval around circuit_run, including host allocation
and measurement API work, rather than whole-job initialization time. Record
revisions, full flags, topology, versions, and variability/regressions.

For kernel costs, repeat a separate profiling run on each revision:

```sh
mpiexec -n "$RANKS" --map-by ppr:4:node nsys profile --trace=cuda,nvtx \
  -o '/tmp/measure-%h-%p' ./bin/qcs -s 40 /tmp/measure-only.so
```

Use unique trace directories for each revision/run. Compare first_stage and
second_stage GPU durations and grid sizes for each successive group, excluding
warmups. Derive active_count from local state size and prior local controls;
global controls can exclude whole ranks. Report rank maxima as well as group
sizes and distributions. Do not mix profiled latency with unprofiled results.
The existing measurement harness benchmark provides width/mask sweeps on smaller
machines (`make benchmark-measurement RANKS=4 BENCH_QUBITS=16`). Its source
copier supports both pre-change and adaptive launch forms.

Potential remaining costs include the unchanged per-bin shared-memory sums,
256-thread blocks for tiny inputs, two kernel launches, NCCL and synchronous
sampling readback. A fused single-block path or smaller block size needs
measurement before adding specialization; numerical bins may differ slightly
because adaptive geometry changes summation order.

## Single-block final output

For `measurement_blocks(active_count, maximum) == 1`, grouped reduction now
passes the existing final-output pointer to the first stage and omits stage two.
Its bin-major stride is one, so every bin (including impossible zero bins) is
already final. Inactive threads still initialize their shared bins and join all
block reductions. Multi-block geometry, output offsets, allocation sizes and
stream ownership are unchanged. No larger single-block threshold is selected
without timing evidence.

`measurement_reduce_test` captures warmed enqueue calls and checks one versus
two kernel nodes across active sizes 1 through 1024, all widths and local-target
counts, controls, fixed bits and noncontiguous mappings. It checks the reserved
output offset, compares against the old two-stage kernels and the independent
physical-index oracle, and poisons reused buffers.

Build with `make bin/measurement_reduce_test`, then run the test normally.
For paired timing, run `./bin/measurement_reduce_test --benchmark > timings.csv`.
The optional benchmark compares explicit old two-stage dispatch with enqueue
for N=1..4 and active sizes 1, 16, 256, 512 and 1048576. Each path receives 20
warmups and 200 samples. GPU event batch time includes launch gaps; host time
includes metadata packing, workspace checks and stream completion per call.
These are reducer timings, not full public measurement API timings or isolated
GPU kernel durations. Repeat runs and retain variability; the default benchmark
order is baseline followed by optimized. Use the existing public measurement
harness for full API timing including NCCL, sampling and readback.

Validation in the implementation environment: `git diff --check` passed, but
`make test-measurement-reduce` failed before compilation because `nvcc` was not
on PATH. `nvidia-smi` was also absent. CUDA tests, capture checks, sanitizer runs,
and both reducer and full API latency measurements remain unexecuted; no runtime
speedup or numerical agreement is claimed.
