# Distributed measurement performance

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
and partial requests (1/2/4/8/16/24/32 targets when valid) cover local, global,
mixed and full layouts. States include zero, uniform, bounded nonuniform
amplitudes `(1 + global_index % 17)/17`, sparse support at multiples of 257,
and skewed rank masses (rank zero has 16 times the amplitude of other ranks).
Weights need not be normalized. Identity and rotated physical-to-logical mappings
include cross-rank-boundary mapping changes; target layouts describe physical
locations before conversion to logical API targets.

Each iteration discards metadata, resets mapping and prepares the state outside
timing. Fresh calls have no cached targets; partial calls install a zero outcome
for the first target; fully cached calls install zero outcomes for all targets.
The private benchmark installs both measured lists and deferred projection masks,
without sampling or materializing projection. Index zero has positive mass in
every distribution, so this is a valid branch. Backends receive identical effective
states and conditioning metadata. Single-target partial cases are fully cached.

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
| Per-rank D2H mass summary | sizeof(rank_summary), constant | 8*(T+1) bytes |
| Owner D2H result | sizeof(sample_result), constant | uint64 index |
| Fresh API collectives | One MPI Allgather + one MPI Bcast | One MPI Allgather + one MPI Bcast |
| Fully cached API collectives | Zero | Zero |

Tiled workspace requested bytes are `16*T + sizeof(rank_summary) +
sizeof(sample_result)` per rank (allocator overhead excluded). Host summary storage
is O(ranks), with a constant-size contribution per rank. For 33 qubits on 32/64/128
GPUs, tiled arrays occupy respectively 1 MiB/512 KiB/256 KiB per rank, compared
with rank_sampled chunk-summary readback of about 512/256/128 KiB per fresh call.
Tiled readback is one summary on each rank plus one result on the owner; use actual
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
CUDA compiler, GPU utilities and MPI compiler/launcher are unavailable in this
environment. GPU builds, tiled correctness tests, single/multi-node timings and
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
