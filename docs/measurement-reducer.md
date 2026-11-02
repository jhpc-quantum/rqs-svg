# Internal measurement reducer (task 1)

The actual pre-migration production baseline is
`1ce55aff510934fd1502abf933e8c26a1db79133`. Retain this revision for task 3's
CUB comparison. Production measurement still uses its existing CUB path;
this change only adds an internal reducer, type compatibility assertions, and
a focused test executable. No public probability API or backend selector is added.

## Algorithm and attribution

`src/measurement_reduce.cuh` adapts the custom `warp_sum`/`block_sum`,
`shared_first_stage_body`, `second_stage_body`, and `reduction_resources` from
[naoto-aoki-fy/cuda-reduce](https://github.com/naoto-aoki-fy/cuda-reduce/blob/15bb3cd08e72a423e2be7d517003567f681b4f26/src/global_reduction.cu)
at commit `15bb3cd08e72a423e2be7d517003567f681b4f26`.
That pinned tree has no license file; the adapted source and README contain
no copyright/license notices. This attribution does not assert an upstream
license grant. No upstream CUB backend, runtime matrix, or dependency is imported.

Only custom block reduction, shared accumulators, and bin-major layout are
implemented. State amplitudes and accumulation use double precision, matching
`qcs::complex_t` and `qcs::float_t`. Histograms have `2^N` entries for `N=1..4`,
with raw squared norms compatible with `MeasureNorm<N>` element order.

## Indexing and layout contract

The descriptor lists ordered local target physical positions and their distinct
output positions. Remaining output bits come from `fixed_outcome`. Its mask/value
controls must contain **all** previously measured local qubits, including those
outside the current measurement group. Targets cannot overlap controls or each
other, and fixed bits cannot overlap variable output positions. Invalid counts,
shifts, masks, sizes, or products throw before launch.

For `P` fixed local controls the packed domain is `num_states_local >> P`.
Ascending insertion of control bits maps each packed index monotonically to
exactly one eligible physical index. Targets remain in the packed domain. Each
eligible amplitude is loaded once and contributes `re*re + im*im` to
`fixed_outcome | OR(((physical >> target[j]) & 1) << output_bit[j])`.
The `L=0` case accumulates only into `fixed_outcome`; other bins are overwritten
with zero. A zero state is valid here. Public measurement remains responsible
for rejecting zero total mass.

Each thread owns `sums[bin*256 + thread]` in dynamic shared memory. Every entry
is initialized on every call. Custom full-warp shuffles use separate static
warp-total scratch and barriers protect its reuse between bins. Stage one writes
`partials[bin*G + block]`; stage two reads with the same stride `G` and overwrites
every output. Both stages initially launch `G=8*SM_count` blocks of 256 threads,
independent of gate launch dimensions. Exact kernel attributes, static plus
dynamic shared limits, optional dynamic shared opt-in, and nonzero occupancy are
checked for each N on first use. Unsupported devices raise errors without a
fallback. For N=4 the dynamic request is 32 KiB plus compiled static scratch.

## Ownership and later integration

Construct `qcs::measurement_detail::workspace` on the owning CUDA device with
the caller's stream. The stream must outlive the workspace. The workspace is
noncopyable and calls must be serialized by its owner. `enqueue<N>(state, count,
descriptor, stream)` returns a device pointer to `2^N` doubles; it enqueues both
stages and does not synchronize or copy results to the host. The state is read-only.

The workspace caches per-N launch checks and owns a single growable allocation
for partials plus output. Growth allocates the replacement before queuing release
of old storage. Steady-state calls reuse capacity without allocation. Every used
element is overwritten; unused capacity is unspecified. Results remain valid
until the next invocation or release. Calls on a different device or stream are
rejected. `close()` queues release on the owning stream, temporarily selects the
owning device, restores the caller's device, and reports errors. Destruction
performs best-effort asynchronous release on that device/stream. Use explicit
`close()` before destroying the stream to obtain teardown error reporting.
For task 2, have `simulator_core` own this workspace and construct descriptors
from its current physical permutation and all recorded local measurements.

## Focused tests and execution status

`tests/measurement_reduce_test.cu` uses an independent physical-index CPU oracle
with compensated long-double sums. Every bin must be finite and satisfy
`abs(gpu - oracle) <= 1e-12 + 5e-12*abs(oracle)`; expected zero bins must be exactly
zero. The test also verifies the input state is unchanged. It covers every N/L
combination, small and grid-stride domains, control polarity and outside-group
controls, reversed/noncontiguous target order, nonuniform complex values,
unnormalized sparse/basis states, zeros, one eligible amplitude, descriptor/N/size
changes, poisoned buffers, capacity reuse, teardown/reinitialization, a nondefault
nonblocking stream, invalid descriptors, and bounded host mapping checks above
32 bits. Physical state counts are powers of two; a partial final grid stride
occurs when the device's `8*SM_count*256` stride does not divide the active count.

The analytic fixture has real amplitudes `1..8` and targets `[0,2]`: expected bins
are `[10,20,74,100]`. Reversing target order gives `[10,74,20,100]`.

Tests, builds, sanitizers, and validation were **not run**, as explicitly requested.
CUDA toolchain/device availability was not probed. No GPU correctness or build
success is claimed. Commands for a later authorized run, with the same vendor
include/compiler flags used for the library, are:

```sh
make sharedlibrary
make bin/measurement_reduce_test
make test-measurement-reduce
compute-sanitizer --tool memcheck ./bin/measurement_reduce_test
compute-sanitizer --tool racecheck ./bin/measurement_reduce_test
compute-sanitizer --tool synccheck ./bin/measurement_reduce_test
```

The focused executable needs CUDA and the existing `atlc` headers but does not
link MPI, NCCL, cuRAND, or libqcs. Source inspection and bounded host mapping
checks alone cannot establish GPU correctness. Production CUB integration and
performance comparisons remain tasks 2 and 3.
