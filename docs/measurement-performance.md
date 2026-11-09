# Distributed measurement performance

Current source audit and host baseline: [measurement baseline](measurement-baseline.md).

The paired benchmark covers `grouped`, `rank_sampled`, `tiled_cdf`, and `auto`.
Production backend implementations and automatic dispatch policy are preserved.

## Reproduction

Configure CUDA, MPI and NCCL using [building](building.md), then run:

```sh
make measurement-backend-benchmark
python3 scripts/benchmark_measurement.py --ranks 4 --qubits 24 \
  --backends grouped rank_sampled tiled_cdf auto --warmups 5 --repetitions 30 \
  --mpiexec 'mpirun --map-by ppr:4:node --bind-to none' \
  --no-build --output results/measurement-4-q24
```

Use one MPI rank per GPU. All ranks must see the local GPUs for the simulator's
local-rank device assignment; avoid per-task GPU masking. Launcher options are
site dependent. `make benchmark-measurement-backends RANKS=4 BENCH_QUBITS=24
BENCH_OUTPUT=results/example` provides the equivalent Makefile entry point;
`BENCH_BACKENDS` selects the ordered backend list.

Run separate output directories for qubits 20, 24, and 28 on available 1/2/4-GPU
single-node allocations and 8/16-GPU multi-node allocations. For larger runs:

| Qubits | GPUs | Nodes with four GPUs each | Local amplitudes per rank |
| --- | --- | --- | --- |
| 33 | 32 | 8 | 2^28 |
| 33 | 64 | 16 | 2^27 |
| 33 | 128 | 32 | 2^26 |

For example allocate 16 nodes with four GPUs and four tasks per node, then use
`--ranks 64 --qubits 33 --mpiexec 'srun --mpi=pmix --gpu-bind=none'`.
A 33-qubit complex-double vector uses 128 GiB aggregate before workspace/runtime
storage. Check memory capacity before running. These are pending experiments,
not completed hardware tests. Repeat backend lists in reverse order in another
directory to examine thermal/order bias; keep binary, allocation and clocks fixed.

Every backend retains raw rank-maximum samples, first-call latency and logs.
`manifest.json` records revision, working-tree status/diff, configuration, selected
build/runtime environment and commands. Archive the binary and untracked build
inputs alongside results if present. `comparison.csv` retains grouped versus
rank_sampled output when both are selected; `pairwise.csv` compares every selected
pair, including auto. Duplicate labels, duplicate workloads and mismatched case
sets fail rather than silently comparing different workloads.

## Workloads and timing

Full-register requests run first within each state/mapping combination. Small
and partial requests (1/2/3/4/5/7/8/9/15/16/17/24/32 targets when valid) cover local, global,
mixed and full layouts. States include zero, uniform, bounded nonuniform
amplitudes `(1 + global_index % 17)/17`, sparse support at multiples of 257,
and skewed rank masses (rank zero has 16 times the amplitude of other ranks).
Weights need not be normalized. Identity, reversed and rotated physical-to-logical mappings
include cross-rank-boundary mapping changes; target layouts describe physical
locations before conversion to logical API targets.

Each iteration discards metadata, resets mapping and prepares the state outside
timing. Fresh calls have no cached targets; partial calls install zero outcomes
for the first max(1, floor(targets/2)) targets; fully cached calls install zero outcomes for all targets.
The private benchmark installs both measured lists and deferred projection masks,
without sampling or materializing projection. Index zero has positive mass in
every distribution, so this is a valid branch. Backends receive identical effective
states and conditioning metadata. Single-target partial cases are fully cached. An additional `local_low_high` case
conditions on physical local bits 0 and local_qubits-1 (one bit when only one local
qubit exists), including controls outside the request. Both API paths run each
case; classical destinations are reversed. The CSV `api` dimension is mandatory
for comparison, so older CSVs need regeneration.

A device synchronization and MPI barrier precede the timer. The measured interval
covers the public multi-measurement API through GPU completion. MPI MAX is outside
the timer and outside the API NVTX range. The first iteration is reported separately
as `first_call_ms`, followed by discarded warmups and independent prepared samples
for steady-state p10/median/p90 (floor(p*(N-1))). Cached calls cannot become fresh
samples accidentally. First-call means first call for that workload in this process:
only the first workload includes cold process-wide workspace allocation. For a
cold-process latency study use a fresh process and inspect its first row.

## Trace collection and structural acceptance

```sh
make measurement-backend-profile
QCS_MEASUREMENT_BACKEND=tiled_cdf mpirun -np 4 \
  nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi \
  --output=tiled-%q{OMPI_COMM_WORLD_RANK} \
  ./bin/measurement_backend_profile 24 1 3
```

Repeat for grouped, rank_sampled and auto with distinct prefixes. Use the site's
rank variable for Slurm output names. Profile separately from timing runs.
NVTX labels distinguish tile reduction, device CDF construction, scalar readback
and wait, rank exchange, rank selection, GPU tile lookup/amplitude sampling,
owner readback/wait, result broadcast and metadata updates. Workspace reservation
is separately labeled. Normal builds compile ranges away.

Use correlated CUDA kernels/copies and MPI events for execution times, not NVTX
host enqueue widths. For each iteration report each rank's computation, transfers,
collectives and waits, along with the rank with maximum API duration. Do not add
overlapping intervals or stage maxima from different ranks into a critical path.
The benchmark's barrier and MAX reduction are harness collectives, not measurement
collectives; constructor/setup collectives must also be excluded.

Source inspection establishes the following expectations; trace validation remains
pending. Let L be local amplitudes and T=ceil(L/4096):

| Property | tiled_cdf | rank_sampled |
| --- | --- | --- |
| Device mass metadata | Two T-element double arrays | Chunk weights, double per chunk |
| Per-rank D2H mass exchange | ranks*sizeof(rank_summary) | 8*(T+1) + 16*ranks bytes |
| Result D2H on every rank | sizeof(sample_result), constant | uint64 index |
| Fresh API collectives | One NCCL AllGather + one NCCL Broadcast | One NCCL AllGather + one NCCL Broadcast |
| Fully cached API collectives | Zero | Zero |

Tiled workspace requested bytes are `16*T + sizeof(rank_summary) +
sizeof(sample_result)` per rank (allocator overhead excluded). Host summary storage
is O(ranks), with a constant-size contribution per rank. For 33 qubits on 32/64/128
GPUs, tiled arrays occupy respectively 1 MiB/512 KiB/256 KiB per rank, compared
with rank_sampled chunk-summary readback of about 512/256/128 KiB per fresh call.
Tiled readback is the gathered summaries and one result on every rank; use actual
ABI sizes and trace copy sizes for exact byte counts. Neither backend allocates
an amplitude-sized probability/CDF array. Tiled arrays stay on device, and capacity
is retained across calls. No tile array is copied to host in the production path.
Grouped performs one small NCCL AllReduce per fresh group of up to four targets;
count actual calls in traces, including mixed cached/fresh requests.

For acceptance, inspect allocation events after warmup for workspace reuse, copy
sizes for absence of tile-array D2H transfer, kernel allocation sizes for absence
of full-amplitude intermediates, and MPI/NCCL event counts for each selected call.
Record measured workspace/high-water allocation separately from requested sizes.
No tile size or launch tuning is justified without timing evidence.

## Results and pending validation

Executed locally: host backend-selection, measurement-launch and work-buffer tests,
Python syntax compilation, and synthetic four-backend orchestration verifying six
pairwise comparisons plus the legacy comparison and manifest. Synthetic timings
are test fixtures only and are not performance results.
CUDA compiler and GPU utilities are unavailable in this environment. MPI is
available outside PATH; see the current comparison below. GPU builds, tiled correctness tests, single/multi-node timings and
CUDA/MPI traces have not been executed. All four backends' latency percentiles,
transfer timings, workspace reuse in traces, collective counts in traces and
crossovers remain unverified. No speedup or scaling conclusion is claimed.

On GPU hardware run:

```sh
make test-measurement-tiled-cdf
make bin/measurement-harness-constant/measurement_integration_test
for backend in grouped rank_sampled tiled_cdf auto; do
  for ranks in 1 2 4 8; do
    QCS_MEASUREMENT_BACKEND=$backend mpirun -np "$ranks" \
      ./bin/measurement-harness-constant/measurement_integration_test
  done
done
```

Only run rank counts supported by the allocation. Populate comparisons from actual
CSV results and stage costs from actual traces before selecting a backend or
claiming a crossover. Auto retains its existing provisional policy.

## NCCL exponential race experiment

The opt-in `tiled_race_nccl` and MPI comparison `tiled_race` are accepted by the
paired benchmark via `--backends tiled_race_nccl tiled_race tiled_cdf rank_sampled`.
See [NCCL tiled race](tiled-race-nccl-measurement.md) for separate GPU candidate,
communication, argmin, synchronization, and final readback tracing, numerical
limits, and hardware validation status. No measured speedup is currently claimed.

## Comparison audit (2026-11-09)

Base revision: `134e10c3b0b35a98cbda74716b05201739ceb13a`, current branch retained.
Only benchmark infrastructure, its host regression test and this report changed;
production algorithms and dispatch thresholds are unchanged. Pre-existing
untracked `bin/` artifacts were excluded from commits.

| Workload category (both APIs) | Best observed backend | Regressions |
| --- | --- | --- |
| Boundary widths, local/global/mixed/full | Unresolved | No GPU samples |
| Fresh, partial, fully cached, low/high local controls | Unresolved | No GPU samples |
| Identity/reversed/rotated, all five state families | Unresolved | No GPU samples |
| Single GPU, single-node multiple GPUs, multiple nodes | Unresolved | No GPU allocation |

No first-call or steady-state p10/median/p90, speedup, or complete API latency
claim is possible. There are no hardware raw samples or profiles. Synthetic runner
timings only test CSV pairing and must never be used to rank backends. Auto's
policy remains provisional (see [baseline contract](measurement-baseline.md));
small states cannot exercise its rank-sampled branch. q24 and above with sufficient
fresh bits are needed, including after conditioning on cached local bits.

Executed checks on `z170-1`: Intel i7-6700K, 4 cores/8 threads, x86_64 Linux
5.14.0-687.30.1.el9_8. Hardware, compiler defaults and flags otherwise match the
[baseline environment](measurement-baseline.md). No `config.mk` exists.

```sh
make -B test-measurement-backend test-measurement-launch test-work-buffer test-measurement-race test-swap-launch test-pauli-string
python3 tests/test_nccl_buffer_setup.py
python3 tests/test_benchmark_measurement.py
/usr/lib64/openmpi/bin/mpiexec --version
/usr/lib64/openmpi/bin/mpiexec -n 2 /bin/hostname
make measurement-backend-benchmark
make test-measurement-tiled-cdf
make bin/measurement-harness-constant/measurement_integration_test
python3 scripts/prepare_measurement_harness.py --source . --output /tmp/rqs-comparison-global --metadata global
```

Six rebuilt C++ tests passed, five NCCL setup unittest methods passed, and one
new runner regression passed (six backend pairs with two distinct API workloads,
binary digest and manifest checks). Measurement Python scripts passed syntax
parsing. OpenRTE 4.1.1 launched two host processes successfully; this provides no
GPU communication evidence. Benchmark and tiled-CDF builds failed because `nvcc`
is missing. Integration preparation failed for both constant/global variants:
`unsupported source: expected exactly one occurrence of
'        first_stage<N><<<blocks, block_size, config.dynamic_shared, stream>>>(state, packed, data_);'`.
This source-copier mismatch precedes compilation and needs repair before the
private integration commands above can succeed. It is an infrastructure blocker,
not an observed production backend regression. No GPU correctness or numerical
edge-case tests executed, including zero totals, NaN/Inf, underflow, CDF endpoints,
correlated distributions, projection/coherence, counter exhaustion and rank failure.
Compare statistical distributions and documented semantics, not exact trajectories
across reduction orders. GPU source changes in this benchmark remain uncompiled.

Future runs now record host/tool probes, topology, forced build-command preview,
binary SHA-256, per-backend environment overrides, and per-rank host/local rank/GPU
placement. Benchmark logs report actual CUDA driver/runtime, NCCL and MPI versions.
For `--no-build`, the command preview describes requested flags, not proof of the
binary's actual flags: retain its original build log. Multi-node topology/version
probes must also be collected on each allocated node; runner probes are local.

Use the timing and Nsight Systems commands above on a configured GPU allocation.
Additionally run `ncu --set full --target-processes all` with the benchmark command
under the site launcher in a separate, short profiling run for kernel memory and
compute metrics, and `compute-sanitizer --tool memcheck` on primitive/integration
tests. Do not use instrumented durations as latency measurements. Correlate CUDA
kernels, copies, NCCL kernels/host calls, stream waits and MPI with API NVTX ranges;
attribute host residuals without adding overlapping intervals. Nsight Systems,
Nsight Compute and sanitizer are unavailable here, so all stage costs, allocation
reuse, collective counts, crossovers and performance regressions remain unresolved.
