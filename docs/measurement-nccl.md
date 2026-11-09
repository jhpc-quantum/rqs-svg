# NCCL measurement communication

`rank_sampled` and `tiled_cdf` use one `ncclAllGather` and one
`ncclBroadcast` per successful fresh measurement, on the existing communicator
and CUDA stream. Cached requests require neither collective nor RNG advancement.
`grouped` and the `auto` dispatch policy are unchanged; auto-selected rank
sampling uses the new transport.

Persistent send/receive slices come from the simulator scratch arena and are
released with it. NCCL uses `ncclUint8` for compact records. Rank sampling uses host `double` chunk sums and rank CDF, transported as two
doubles per rank. This avoids host ABI dependence and matches NCCL `ncclDouble`.
Very large sums can lose precision or overflow sooner than with extended precision;
nonfinite totals are rejected. No new sampling kernels are introduced.

Tiled summaries flow directly from the GPU into all-gather. The gathered summaries
are read back once for host rank selection. Both backends broadcast successful
owner samples directly from device storage and read back the shared result once.
Rank sampling retains its chunk-weight readback and adds one upload of two
doubles. Tiled sampling removes the local summary readback; both remove
the owner-only sample readback and synchronization. These are source-level
transfer counts, not measured performance improvements. Small-message NCCL
latency may outweigh saved synchronization.

All ranks validate the same gathered summaries, use rank zero's authoritative
Philox draw, and commit metadata/counters only after validating the broadcast.
Recoverable local allocation/build failures and local counter exhaustion become
invalid summaries; recoverable owner launch failures become broadcast sentinels.
A CUDA/NCCL failure in the communication/readback boundary aborts the communicator
and MPI job, since a damaged CUDA context cannot reliably send a sentinel.
Transport hangs and hardware loss remain subject to NCCL/runtime timeout behavior.

## Validation and performance reproduction

CUDA compilation and distributed execution have not been verified in the editing
environment: `nvcc`, `mpicxx`, and GPU runtime tools are unavailable. On a configured
CUDA/MPI/NCCL host, run the existing private integration tests for each supported
rank count and metadata layout (one GPU per rank):

```sh
for metadata in constant global; do
  for ranks in 1 2 4 8; do
    for backend in rank_sampled tiled_cdf grouped auto; do
      QCS_MEASUREMENT_BACKEND=$backend make test-measurement-metadata \
        MEASUREMENT_METADATA=$metadata RANKS=$ranks
    done
  done
done
make test-measurement-tiled-cdf test-measurement-reduce
```

The private tests cover full/subset and cached measurements, mappings, coherence,
nonuniform correlated states, zero-mass ranks, nonfinite and globally zero norms,
rank-zero seed authority, counter exhaustion, and rank-local pre-collective
failure with retry. The nonuniform seeded test now runs for both NCCL backends.

For MPI/NCCL comparisons, build the parent revision and this revision in separate
source directories, using identical compiler flags and runtime allocations. Run
[the existing benchmark](measurement-performance.md) in each with separate output
directories, all four backends, and 1/2/4/8 ranks where supported. Include first-call
and warmed latency, single-node and multi-node runs. Use the profiling target
`make measurement-backend-profile` and Nsight Systems CUDA/NCCL tracing to compare
host waits and memcpy byte counts. No measured speedup or regression is claimed.

Local checks passed: host backend dispatch, measurement launch planning, work-buffer
allocation, SWAP launch planning, and Pauli string tests; private-source preparation
for both metadata layouts; and `git diff --check`. The benchmark build was attempted
and stopped at the missing `nvcc` executable. No GPU test or benchmark was run.
