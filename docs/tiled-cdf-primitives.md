# Tiled CDF CUDA primitives

`src/measurement_tiled_cdf.cuh` provides independent internal primitives for a
future `QCS_MEASUREMENT_BACKEND=tiled_cdf` backend. Dispatch and MPI are not
integrated; setting that selector is not supported yet. Existing backends are
unchanged.

## Execution and storage

Construct `qcs::tiled_cdf_detail::workspace` with the simulator stream, optionally
`reserve(local_count)`, then call `build(state, count, mask, value, active)`.
The returned device `rank_summary` contains the final tile CDF value and status
bits. `draw(conditional_uniform)` returns a device `sample_result` with a 64-bit
local index. These operations enqueue work without synchronization. Only compact
summaries/results need host transfers; production code never transfers tile
arrays. There is no RNG dependency.

One allocation contains two FP64 arrays of `ceil(count/4096)` elements followed
by a summary and result. Scan scratch is fixed shared memory, not a per-amplitude
device allocation. Capacity grows only when necessary, with checked size and
layout arithmetic. Allocation, replacement, and release use the owner stream.
Callers serialize operations, retain the state unchanged between build/draw,
consume results before reuse, and keep the stream alive through destruction.
`close()` reports release errors; destruction performs best-effort release.
Operations reject a different current CUDA device. The tile size is an internal
compile-time constant, currently 4096; scan sizes must remain multiples of 256
and fit device shared-memory limits when tuning it.

Each tile block loads adjacent amplitudes coalescently, computes FP64 squared
magnitudes with explicit round-to-nearest operations, and reduces them using a
fixed total-reduction tree. Masked/padded amplitudes contribute zero without
loading state; inactive ranks never dereference state, even if it is null.
A single block scans metadata in batches of 256 with a fixed carry order.
This avoids toolkit-dependent device-wide floating-point scan behavior and
needs no CUB temporary allocation. The metadata scan is linear but has limited
parallelism; its performance at very large tile counts remains to be measured.
Drawing binary-searches the tile CDF and rereads only the chosen tile, then
performs a parallel scan in amplitude-index order. No input writes occur.

## Numerical policy

Uniforms in [0,1] are accepted, including the upper endpoint for diagnostics.
Negative, nonfinite, or greater-than-one uniforms report `invalid_uniform`.
For a positive finite total, multiplication that rounds to the upper endpoint
is clamped to `nextafter(total,0)`. Selection uses strict prefix > threshold.
Given tile prefix `after`, preceding prefix `before`, and rank threshold `x`,
the child fraction is `(x-before)/(after-before)`. That fraction is scaled to
the selected tile's recomputed scan total, with the same endpoint clamp.
This explicitly handles differences between CDF interval width and tile mass.
The original tile mass must be positive, and the final integer minimum considers
only original positive amplitude weights whose prefixes exceed the threshold.
Zero, masked, padded, and out-of-range entries cannot win.

Status bits distinguish `zero_mass` (a valid empty local contribution) from
`invalid_weight`, `invalid_total`, `invalid_uniform`, and
`invalid_distribution`. Failure returns `UINT64_MAX`, never an amplitude index.
NaN/infinite effective weights, including overflow during squared magnitude,
set `invalid_weight`; such weights are replaced with zero only to complete the
metadata scan safely. Overflow of positive sums sets `invalid_total`. Any build
error prevents drawing, even if some other weights are valid. Masked and
inactive amplitudes do not participate in validation.

Underflow can erase tiny squared magnitudes. FP64 accumulation can erase tiny
intervals beside much larger weights, making them unreachable. There is no
promise of exact probabilities for arbitrary dynamic ranges. For a fixed
binary, device, configuration, and input, all floating point additions follow a
fixed tree and batch order; blocks write disjoint tile masses, and the only
sampling atomic is an exact integer minimum. Repeated build/draw calls are
therefore intended to return identical results. Cross-architecture, compiler,
or tile-size equivalence is not promised.

## Validation

Run `make test-measurement-tiled-cdf`. The focused CUDA executable uses an
independent long-double CPU hierarchy, dyadic exact-boundary fixtures and
non-dyadic mass checks. It covers partial and multiple tiles, multiple metadata
batches, zeros, endpoint and adjacent representable uniforms, masks, inactive
ranks, unnormalized/unequal weights, invalid weights/totals/uniforms, input
immutability, repeatability, capacity reuse, and checked 64-bit size boundaries.
It returns 77 when compiled but no CUDA device is available. Very large index
arithmetic is checked without allocating a correspondingly large state; actual
sampling above 2^32 requires a suitable large-memory GPU and is not exercised.

In the implementation environment `make test-measurement-tiled-cdf` stopped at
compilation because `nvcc` was not installed (make exit 2, compiler launch error
127). CUDA compilation, GPU correctness, repeatability, and performance remain
unverified and must be checked on the target CUDA system.

The distributed opt-in backend is described in [Tiled CDF measurement](tiled-cdf-measurement.md).

## Mass reduction optimization

The mass kernel prunes the Hillis-Steele scan to the ancestors of its final
4096th element. It combines adjacent subtrees, adding the right subtree to the
left with `__dadd_rn`, exactly as the original final scan value does. The first
seven levels use disjoint shared-memory endpoints; the last five use one warp's
shuffle instructions. There is no change in summation order or FP64 rounding.
The selected-tile scan and metadata scan remain unchanged.

Per tile, additions decrease from 45,057 to 4,095 and block barriers from 26 to
9, including loading and inter-tile reuse. Shared storage remains 32 KiB; only
surviving endpoints are read and written during reduction. These operation
counts are not measured speedups.

The focused CUDA test retains the original mass kernel and compares every tile
mass bit for bit and its invalid-weight status, including grid-stride reuse.
Fixtures cover nonuniform, sparse, zero, masked, inactive, overflow and widely
varying weights, plus small contributions near FP64 rounding boundaries. The
existing long-double mass tolerance of 1e-13 relative and sampling checks remain
unchanged.

For this optimization, host backend, launch and work-buffer tests passed, as did
100 host comparisons of the full scan and pruned tree. CUDA primitive, reducer
and integration tests and the complete API benchmark could not compile because
`nvcc` is unavailable; `nvidia-smi` is also unavailable. No kernel or API timings
were obtained. On a CUDA system, compare CUDA-event timings of `original_masses`
and `td::masses` from the focused test using identical warmed inputs and launch
geometry. Use `benchmark-measurement-backends BENCH_BACKENDS=tiled_cdf` on the
parent and optimized revisions for complete API measurements, with identical
rank counts, qubits, warmups and repetitions.

## Final sampling reduction

The final selection uses a per-thread minimum tile offset, warp shuffle minima,
and eight shared warp results reduced by the first warp. Thread 0 writes the
64-bit basis index and status. The sentinel is `tile_size`; an all-sentinel
result retains `UINT64_MAX` and `invalid_distribution`. No global selection
atomics or additional launches are needed. The floating-point prefix scan,
positive-weight guard, strict `prefix > threshold`, and `nextafter` endpoint
handling are unchanged. This change is independent of the mass reduction.

The focused test compares index and status with the frozen atomic sampling
kernel for every draw, including sparse, uniform, skewed, nonnormalized,
invalid and boundary distributions. Dense dyadic fixtures exercise warp and
thread-stride boundaries and endpoints. Host backend, launch and work-buffer
tests and 1,000 host reduction comparisons passed. CUDA compilation and the
requested two-rank integration run were blocked by missing `nvcc`; rank
agreement and GPU correctness remain unverified here. No timings are claimed.

On GPU hardware, run `make test-measurement-tiled-cdf` and
`make test-measurement-tiled-integration RANKS=2` (also with the target rank
count). The integration test checks MPI minimum/maximum agreement of measured
outcomes. Compare warmed CUDA-event timings of `test_reference::atomic_sample`
and `td::sample` with identical prebuilt metadata and launch geometry, covering
early, middle and endpoint draws on sparse and dense tiles. Compare complete
API latency using `benchmark-measurement-backends BENCH_BACKENDS=tiled_cdf`
on the parent and optimized revisions with identical hardware, rank counts,
qubits, warmups and repetitions.
