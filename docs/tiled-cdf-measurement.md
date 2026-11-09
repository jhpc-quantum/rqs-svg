# Tiled CDF measurement backend

Set `QCS_MEASUREMENT_BACKEND=tiled_cdf` consistently on every MPI rank.
The default remains `grouped`. Explicit `rank_sampled` and the provisional
`auto` policy retain their previous behavior; auto does not select tiled CDF.
Unknown or inconsistent settings fail collectively during setup.

Full-register and subset measurements share a GPU sampler. Previously recorded
logical bits are translated through the current permutation into local conditions
and rank eligibility. Device tile masses and their CDF remain on the GPU. A
constant-size summary contains mass, status, and a Philox draw using the existing
seed/counter mapping. One allgather exchanges these summaries; rank zero's draw
selects a rank and its conditional variate. That rank samples a tile and amplitude
on the GPU and broadcasts the compact index/status. Host storage is O(ranks).
The device workspace persists between calls and is released with state storage.

Only fresh requested bits are committed, preserving deferred projection,
unnormalized complex amplitudes, and unrequested coherence across ranks. Cached
and empty requests do not scan, draw, or communicate. A successful fresh request
advances the lifetime counter once, including the final usable UINT64_MAX value.
Build failures travel in the rank summary; owner failures travel in the result
broadcast. Neither commits results or measurement metadata. Zero-mass ranks are
valid; invalid components or a globally zero norm fail on all ranks. CUDA context
loss and MPI transport failures are outside recoverable-error guarantees.

The successful fresh path uses two MPI measurement collectives, regardless of
request width. Rank-summary allocation and backend validation happen at setup.
The numerical interval rules inside a rank are those of the tiled primitives:
see [GPU primitive details](tiled-cdf-primitives.md).

## Regression coverage and execution

The private integration target explicitly enables tiled backend checks. Its
16-qubit state has at least two tiles per rank at eight ranks. Coverage includes
nontrivial mappings, reordered requests, cached results, differing per-rank seeds,
counters above 2^32, exhaustion, invalid norms from one rank, zero-mass ranks,
and full-register sampling from four correlated, nonuniform basis outcomes.
Shared public cases exercise classical destinations, subsets, follow-up gates,
coherence, saved unnormalized amplitudes, resets, and argument rejection.
Grouped diagnostic hooks and their RNG expectations remain confined to the
existing grouped harness; the tiled integration target does not invoke that harness.

On a CUDA/MPI/NCCL installation with one GPU per rank, run:

```sh
make test-measurement-tiled-cdf
for metadata in constant global; do
  for ranks in 1 2 4 8; do
    make test-measurement-tiled-integration MEASUREMENT_METADATA=$metadata RANKS=$ranks
    for backend in grouped rank_sampled auto; do
      QCS_MEASUREMENT_BACKEND=$backend make test-measurement-metadata \
        MEASUREMENT_METADATA=$metadata RANKS=$ranks
    done
  done
done
```

Compare semantic and statistical assertions, not seeded trajectories between
algorithms. Runtime collective counting, injected CUDA owner failures, and
performance profiling should also be checked on that installation.

## Validation performed for this integration

Passed locally:

- `make test-measurement-backend test-measurement-launch test-work-buffer`
  (C++17 with warnings treated as errors).
- Copied-source harness preparation for both constant and global metadata.
- `git diff --check`.

`make test-measurement-tiled-integration RANKS=1` reached the CUDA compile command
and failed because `nvcc` is unavailable. MPI compiler/runtime and NVIDIA runtime
tools are also unavailable here. CUDA compilation, GPU primitive execution, and
the entire 1/2/4/8-rank, two-placement runtime matrix remain unvalidated.
