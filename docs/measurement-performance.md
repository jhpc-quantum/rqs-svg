# Distributed measurement performance

The paired benchmark extends the existing private CUDA benchmark and Makefile
infrastructure. It does not change backend selection or deferred projections.
No builds, tests, validation, profiling, or benchmark runs were performed for
this change, at the user's request. There are no measured results yet.

## Reproduce on GB200 NVL4

Configure CUDA, MPI, NCCL and vendor flags as described in [building](building.md).
Build once on a compatible node with `make measurement-backend-benchmark`.
Use one MPI rank per GPU, four ranks per node, with unique GPU assignment per
local rank (the simulator uses its existing local-rank device selection).
Allocate the following resources with the site's scheduler:

| GPUs/ranks | Nodes | Ranks per node |
| --- | --- | --- |
| 1 | 1 | 1 |
| 2 | 1 | 2 |
| 4 | 1 | 4 |
| 8 | 2 | 4 |
| 16 | 4 | 4 |
| 32 | 8 | 4 |
| 64 | 16 | 4 |
| 128 | 32 | 4 |

Within each allocation, use this command with G set to the GPU count above:

```sh
G=64
python3 scripts/benchmark_measurement.py --ranks "$G" --qubits 33 \
  --warmups 5 --repetitions 30 --no-build \
  --mpiexec 'srun --mpi=pmix --gpus-per-task=1 --gpu-bind=none' \
  --output "results/measurement-${G}"
```

For example, the 64-GPU allocation can be requested with
`salloc --nodes=16 --ntasks=64 --ntasks-per-node=4 --gpus-per-node=4 --exclusive`.
Repeat with node/task counts from the table. Scheduler MPI options depend on the
installed MPI integration. All ranks must see all four local GPUs for the
simulator's local-rank assignment; avoid scheduler per-task device masking.
For Open MPI use `--mpiexec 'mpirun --map-by ppr:4:node --bind-to none'` for G>=4;
use `ppr:1:node` or `ppr:2:node` for one or two GPUs.
A 33-qubit complex-double vector alone uses 128 GiB in aggregate; allow additional
workspace and runtime memory. Smaller development runs can use `--qubits 20`.

Outputs are raw CSV per backend (including every rank-maximum sample), paired
`comparison.csv`, build/runtime logs, and a revision/command/config manifest.
Keep the same binary, allocation, clocks, environment, and idle GPUs for a pair.
Repeat pairs in reversed order manually to investigate thermal/order bias.

## Workloads and timing

Each backend runs zero, uniform, and bounded nonuniform real amplitudes with
identity mapping. Nonuniform amplitudes follow `(1 + global_index % 17) / 17`
times the uniform amplitude; these weights need not have unit norm, consistent
with measurement of unnormalized deferred branches. Targets have even/odd
physical ordering: local-only, global-only, alternating mixed, and full register.
Counts are 1, 2, 4, 8, 16, 24, 32 and the full register when valid. Impossible
layouts are omitted, including global-only counts exceeding log2(ranks).
Deferred cases premeasure one requested target or all requested targets; the
latter isolates the cached API endpoint. A single-target partial case is fully
cached too. No projection is forced at the endpoint.

Each repetition discards measurement metadata and prepares the state on GPU
outside timing. Deferred setup also occurs outside timing. Five complete API
warmups per case are discarded by default. A device synchronization and MPI
barrier precede the timer. The steady-clock interval covers the public C++
multi-measurement API and a final device synchronization. MPI MAX reduction is
outside timing; p10, median, p90 use floor(p*(samples-1)) of sorted rank maxima.
Result allocations are outside timing. Temporary allocations inside production
measurement remain timed. Node counts use MPI shared-memory communicators.

## Profiling and cost breakdown

Build a separate binary with `make measurement-backend-profile` (requires CUDA
NVTX3 headers). Normal builds compile tracing away. Run profiling separately:

```sh
QCS_MEASUREMENT_BACKEND=grouped mpirun -np 4 \
  nsys profile --trace=cuda,nvtx,mpi --mpi-impl=openmpi \
  --output=grouped-%q{OMPI_COMM_WORLD_RANK} \
  ./bin/measurement_backend_profile 33 1 3
```

Repeat with `rank_sampled` and a distinct output prefix. Use the scheduler's rank
environment variable in the filename for Slurm. This profiles all cases; use
Nsight's timeline to select the case/iteration of interest. Do not use profile
binary timings as performance results. No stage barriers or CUDA waits were
added to production measurement.

NVTX ranges label grouped reduction, NCCL AllReduce, GPU sampling, readback,
host synchronization, metadata, and per-group host setup. Rank-sampled labels
cover allocation/setup, local mass, readback/wait, host mass sum, rank-mass
collection, selection/draw broadcast, owner sampling, result broadcast and
metadata. These ranges measure host enqueue regions; GPU durations come from
correlated CUDA kernels/copies and NCCL activity, not NVTX width. CUDA stream
waits expose synchronization costs; MPI traces expose collective waits. Owner
sampling includes its copy/wait. Selection includes the authoritative-draw
broadcast; separate that MPI event in the trace. Inter-group gaps are the time
between group ranges. Allocation/free and API overhead remain visible as well.

For each selected iteration report GPU computation, GPU copies, NCCL activity,
MPI calls, and host waits separately for each rank, plus the critical-rank
wall time. Overlapping intervals must not be added; do not sum stage maxima
from different ranks as if they represented a single critical path.

## Comparison and scaling interpretation

| Backend | Median/p10/p90 | Compute | Communication | Synchronization |
| --- | --- | --- | --- | --- |
| grouped | Not measured | Not measured | Not measured | Not measured |
| rank_sampled | Not measured | Not measured | Not measured | Not measured |

`comparison.csv` fills the timing comparison from actual runs; traces supply
the stage breakdown. No observed crossover or numeric speedup is available.
Expected behavior, to be tested: grouped scans state once per fresh group of
up to four targets and performs a small NCCL reduction per group. Rank-sampled
scans local amplitudes once, copies chunk weights to host, and uses three MPI
collectives regardless of target count. Its per-call allocation, host sum and
weight transfer may dominate small target sets. Strong scaling shrinks local
scans while communication latency remains; beyond 32 GPUs either implementation
may become latency-bound, so GPU count alone does not predict a crossover.

Choose using paired medians and p10/p90 for the intended target layout and
state. Grouped is a candidate for small fresh target counts; rank-sampled is a
candidate for many fresh targets. Cached cases test metadata overhead rather
than state scanning. Treat these as hypotheses until measurements establish a
stable advantage, particularly across nodes. No automatic dispatch is added.
