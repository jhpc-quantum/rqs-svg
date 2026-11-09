# Experimental NCCL tiled race measurement

Use `QCS_MEASUREMENT_BACKEND=tiled_race_nccl` consistently on all ranks before
simulator creation. This opt-in backend preserves `tiled_race` (the MPI race
comparison), `tiled_cdf`, `rank_sampled`, `grouped`, and the existing `auto` policy.

## Sampling and communication

The existing tile hierarchy computes each rank's effective mass and a local
amplitude sample without projecting the state. A GPU kernel constructs a
fixed-size candidate with an FP64 log-domain exponential-race score, a uint64
global basis index, rank, validity, and error status. The score is
`log(-log(U)) - log(local_mass)`; comparison avoids direct clock division
underflow and overflow. Zero-mass ranks contribute inert candidates.

One `ncclAllGather` exchanges device candidates. A GPU metadata scan checks every
candidate's status and chooses the minimum `(score, rank, index)`. A global
zero-mass distribution or any relevant rank's invalid state or recoverable
preparation failure rejects the call before changing metadata, output bits, or
the public RNG cursor. Only the final selected candidate/status is copied to the
CPU for the synchronous API. There are no intermediate device-to-host transfers,
measurement-time MPI collectives, seed broadcasts, or result broadcasts.

Hierarchy construction, local sampling, candidate construction, NCCL exchange,
selection, and final readback are ordered on the simulator's CUDA stream. The
NCCL communicator and scratch device send, receive, and selected-result buffers
are simulator-owned and reused. MPI remains part of simulator initialization and
NCCL bootstrap. Setup uses the existing backend validation collectives, allocation
readiness check, and seed broadcast; it is outside the sampling protocol.
Cached calls use no sampling collective.

Completion polling checks both `cudaStreamQuery` and `ncclCommGetAsyncError`.
A CUDA or NCCL transport failure uses the existing communicator/job abort path;
it cannot safely be propagated by another candidate exchange on a broken stream.
Recoverable local preparation failures use a device sentinel and participate in
the same AllGather as successful ranks.

The backend uses the authoritative initialization seed snapshot and Philox
rank/domain streams from [MPI tiled race](tiled-race-measurement.md). Domain 0
selects the local amplitude; domain 1 generates the race clock on device. A
successful fresh call advances the public counter once; cached and rejected
calls do not advance it. Deferred unnormalized projection, requested-bit-only
metadata, logical mapping, and classical destinations are unchanged.

Numerical limits of the reused hierarchy remain: squared weights and local
totals must be representable as finite doubles. Log-domain keys do not repair
weight arithmetic overflow or underflow. The argmin is a serial GPU metadata
scan, so its work grows with GPU count; it is an experimental baseline.

## Validation status

CPU RNG/statistical and backend-selector tests passed in the implementation
environment. Source-copy integration harness generation also passed:

```sh
make -f Makefile.test test-measurement-race test-measurement-backend
```

GPU/MPI tests are added but **not run** here: there is no CUDA toolkit, MPI
compiler, or GPU runtime. Run on a configured system:

```sh
make test-measurement-race-device
make test-measurement-race-nccl-integration RANKS=1
make test-measurement-race-nccl-integration RANKS=2
make test-measurement-race-nccl-integration RANKS=4
```

Device tests cover full-width global indices, ties, error preservation, globally
inert candidates, extreme finite masses, host/device RNG and key agreement, and
failure sentinels. The shared public/private integration harness covers cross-rank
agreement, uneven probabilities, zero-mass ranks, single-supported-rank extreme
masses, coherence after partial measurement, mapping, destinations, cache behavior,
counter lifecycle, and collective rejection/retry.

## Evaluation protocol and limitations

No GPU latency results are available from this environment. Neither a speedup
nor GPU/distributed correctness has been established by hardware execution.
One collective may require multiple network rounds; the call count alone says
nothing about comparative latency. All supported ranks now perform local sampling,
and synchronization, GPU launches, and readback can dominate small workloads.

The paired benchmark accepts all four comparison backends:

```sh
make benchmark-measurement-backends RANKS=4 \
  BENCH_BACKENDS='tiled_race_nccl tiled_race tiled_cdf rank_sampled'
```

Use identical inputs, warmups, and repetition counts across available GPU counts
and node configurations. Save the CSV output, including cold-call and warmed
end-to-end latency, and record GPU, network, CUDA, NCCL, MPI, and buffer mode.
Benchmark barriers and timing aggregation MPI collectives are outside the API
sampling interval and must not be counted as sampling communication.

For the stage breakdown, build `make measurement-backend-profile` and run that
binary under Nsight Systems with CUDA, NCCL/MPI (as supported by the installed
profiler), and NVTX tracing. Use the following explicit ranges and correlated
GPU events:

| Range | Measurement |
| --- | --- |
| `measurement_api` | End-to-end synchronous API latency |
| `tiled_race_nccl.hierarchy_and_local_sample` | Local hierarchy and sampling kernels |
| `tiled_race_nccl.candidate_generation` | GPU candidate kernel |
| `tiled_race_nccl.allgather` | NCCL communication kernels |
| `tiled_race_nccl.argmin_selection` | Device selection kernel |
| `tiled_race_nccl.synchronization` | Host completion polling, including remaining GPU/network work |
| `tiled_race_nccl.final_result_transfer` | Final device-to-host copy and completion wait |

NVTX host enqueue durations are not GPU execution timings. Derive GPU durations
from correlated CUDA activities; use host ranges for synchronization/wait cost.
These measurements overlap (synchronization waits for earlier work), so do not
sum all columns as disjoint phases. Report first-call costs separately, use
maximum-rank end-to-end latency, and report stage distributions across ranks.
See [measurement performance](measurement-performance.md) for the shared paired
benchmark and profiling workflow. Hardware measurements remain pending.
