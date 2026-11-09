# Global-only distributed measurement

`rank_sampled` and `tiled_cdf` omit owner amplitude sampling and the final
sample broadcast when every fresh target maps to a physical global bit.
Cached local targets do not prevent this optimization. The decision uses only
replicated request, cache, and mapping metadata, never rank activity or mass.
Rank mass validation and the authoritative rank-zero uniform remain unchanged;
metadata and the RNG counter are committed only after successful validation.
Requests needing any fresh local bit retain the existing sampling protocol.

The tiled backend still runs the identical metadata scan to preserve mass
rounding and validation, but does not write tile CDF entries. Its workspace
allocation is retained for reuse; a summary-only build cannot be sampled.

## Independent benchmark

Build `measurement-backend-benchmark` with the normal vendor configuration.
Run the following for each backend, using identical hardware/settings for the
parent commit and this commit (copy the benchmark driver with its new filter
into the baseline build):

```sh
QCS_MEASUREMENT_BACKEND=rank_sampled mpirun -n 4 \
  ./bin/measurement_backend_benchmark 24 5 30 global-only > rank-global.csv
QCS_MEASUREMENT_BACKEND=tiled_cdf mpirun -n 4 \
  ./bin/measurement_backend_benchmark 24 5 30 global-only > tiled-global.csv
```

The filter times only physical-global layouts, including partial caching, across
all existing state distributions and both mappings. Compare matching CSV rows;
all-cached calls are excluded. Timing remains maximum-rank API latency.

No timings or distributed runtime results were obtainable in the implementation
environment: CUDA, MPI, and GPU tooling are absent. Host backend and launch tests
pass. Harness generation also encounters an existing source-pattern mismatch
in `prepare_measurement_harness.py` for the reduction first-stage launch;
resolve that instrumentation mismatch before running the private MPI suite.
The new suite compares fast requests against requests requiring a fresh local
index with identical counters, and covers reversed mappings, cached local and
global targets, zero-mass ranks, invalid norms, and rank-local failures.
