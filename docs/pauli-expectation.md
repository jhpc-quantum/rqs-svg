# Pauli string expectation contract

The host-side input contract, standalone two-stage CUDA reducer, and CPU/GPU
test sources are implemented. The simulator shares its measurement workspace
with the Pauli reducer. The internal simulator-core operation integrates
projection, localization, and NCCL summation. Public C and C++ expectation
functions expose that operation.
The operator enum is available in [`qcs.h`](../include/qcs.h); internal helpers
are in [`pauli_string.hpp`](../src/pauli_string.hpp).

## Inputs and state semantics

A string consists of parallel arrays of logical qubit indices and
`qcs_pauli_operator` values (`QCS_PAULI_I`, `QCS_PAULI_X`, `QCS_PAULI_Y`,
`QCS_PAULI_Z`), a term count, and a finite real coefficient. Each qubit may
appear once, including identity entries. Array order does not affect the
operator. An empty string is identity; null arrays are allowed only when the
count is zero. Identity entries are validated and then removed from the
internal support list.

The result contract is `coefficient * <psi|P|psi>`, with no implicit
normalization. Identity therefore returns the coefficient times the squared
norm, including for unnormalized and projected states. The core operation
materializes all pending measurement projections before reduction while
preserving measurement outcomes and RNG state.

The simulator must exist and have allocated state storage. As with existing
gate operations, callers must initialize that storage using a state setter;
the simulator currently has no initialization flag. The allocation validator
checks pointers without dereferencing them. The future API must check the
simulator pointer before reading its core metadata.

## Dimensions, localization, and descriptor

Supported total qubit counts are 1 through 63. Local counts may be zero through
the total count. These bounds keep bit positions and the state dimension
representable with `uint64_t`, avoiding a shift by 64. The number of
non-identity terms must not exceed the local count. Thus a rank with zero
local qubits can evaluate only identity.

The validated input owns a copy of its non-identity terms. Descriptor
construction accepts a complete logical-to-physical permutation **after**
localization and rejects non-identity targets that remain global. Global
identity entries do not require localization. Helpers do not change the mapping
or perform communication.

`PauliStringDescriptor` contains the coefficient, X/Y flip mask, Y/Z phase
mask, and Y count modulo four. Its convention is
`P|i> = i^(number of Y) * (-1)^popcount(i & phase_mask) * |i ^ flip_mask>`.
Only local physical positions enter the descriptor masks. The coefficient
is retained for application once after the eventual cross-GPU sum.

For a nonzero flip mask, `pair_pivot` identifies its highest set bit.
`pair_first_index(k, pivot)` inserts a zero at that position. Given
`0 <= k < local_dim/2`, this enumerates each pair exactly once with
`i < (i ^ flip_mask)`. The index helper assumes valid inputs and has no
per-index validation branch. Diagonal strings must use the amplitude domain
instead of the pair domain.

## Host validation

Run from the repository root; CUDA, MPI, and NCCL are not needed:

```sh
make test-pauli-string
```

The tests reject invalid dimensions, counts, arrays, qubits, operators,
coefficients, mappings, allocation pointers, and oversized support. They also
check identity handling, mask bits above 32, and descriptor locality.

Pair coverage is exhaustive for every nonzero flip mask on one through nine
qubits, with additional wide-index checks. Numerical tests compare descriptor
pair arithmetic against an independent long-double CPU oracle that applies
explicit 2x2 matrices to the full state and takes the inner product. They
cover every Pauli string on one through four qubits using seeded random
complex amplitudes, reversed mappings, positive/negative/zero coefficients,
Y phases, Bell/GHZ states, and non-contiguous targets. The comparison uses
`abs(error) <= 1e-12 * (1 + abs(reference))`.

The CPU oracle's full-vector temporary storage is test-only; it is not a
proposal for GPU workspace allocation. These host tests do not establish GPU
accuracy, distributed behavior, or performance.

## Standalone CUDA reduction

[`pauli_reduce.cuh`](../src/pauli_reduce.cuh) reuses the measurement reducer's
256-thread block sum and amplitude type. Stage 1 uses grid-stride loops and
`min(4 * SM_count, ceil(work_items / 256))` blocks, where work items are all
amplitudes for diagonal strings and half the dimension for paired strings.
The paired specialization inserts a zero at the highest flip bit, reads two
amplitudes once, and accumulates twice the required real contribution. Y-phase
selection is uniform across a launch. There is no `i < j` test.

Stage 2 launches one 256-thread block and takes an explicit partial count.
Each thread can accumulate multiple partials before the common block sum.
The coefficient is deliberately excluded from the returned scalar for later
application after NCCL AllReduce.

`enqueue` borrows a caller-owned device buffer of at least `blocks + 1` doubles.
It writes partials at the start and returns the last buffer element as the
output slot. No allocation or stream synchronization occurs. State and scratch
must be disjoint, on the current device, and alive until stream completion.
Calls sharing scratch must be serialized on the same stream. The raw-pointer overload checks device/kernel resource limits at launch.
The simulator uses the checked workspace adapter described below.

On a configured CUDA machine, run:

```sh
make test-pauli-reduce
make test-measurement-reduce
compute-sanitizer --error-exitcode 1 --tool memcheck ./bin/pauli_reduce_test
compute-sanitizer --error-exitcode 1 --tool racecheck ./bin/pauli_reduce_test
compute-sanitizer --error-exitcode 1 --tool synccheck ./bin/pauli_reduce_test
```

The standalone CUDA test compares against the matrix oracle with
`abs(error) <= 1e-12 + 5e-12 * abs(reference)`. It covers exhaustive strings
through four qubits, random strings, signed Y eigenstates, Bell/GHZ states,
non-contiguous masks, low/middle/high pivots, dimensions down to one amplitude,
large domains, scratch poisoning/reuse/guards, invalid descriptors, and a
1025-partial Stage 2 case. Non-unit coefficients verify that the reducer leaves
coefficient application to its caller.

CUDA tools were absent from PATH during implementation, so the CUDA source and
test have not been compiled or executed. GPU numerical accuracy, sanitizer
results, and measurement regression results remain unverified.

## Shared measurement workspace

The simulator constructs a `pauli_detail::workspace` adapter immediately after
reserving the measurement slice and before carving out SWAP storage. The adapter
borrows that same slice through `measurement_detail::workspace::borrow_scalar`.
The measurement reservation already exceeds the Pauli requirement, so neither
the arena layout nor NCCL registration needs another region. The scalar output
is always the last double in the measurement slice; partials occupy its start.

The adapter caches the maximum `4 * SM_count` launch configuration and validates
kernel limits once at setup. Each call clamps that geometry to the current work
count and checks that the storage is allocated, has sufficient capacity, and
belongs to the current device and supplied stream. Borrowing never allocates or
grows storage. A closed workspace rejects borrowing. The standalone owning
measurement workspace may be reopened; the adapter obtains current storage on
each call instead of retaining a stale device pointer.

Measurement reduction, sampling, Pauli reduction, NCCL, and result consumption
must be queued serially on the owner stream. A result must be consumed on that
stream before the next workspace operation. Host calls sharing the workspace
must also be serialized, matching the existing simulator contract. No extra
stream synchronization or cross-stream sharing is introduced. Simulator teardown
joins existing work, destroys the adapter, then closes measurement storage before
releasing the backing work buffer.

The CUDA test source now includes alternating measurement reduction, device
sampling, and Pauli reduction on the same slice, without explicit synchronization
between operations. It checks poisoned storage, fixed output location,
pointer/capacity stability, adjacent SWAP guards, wrong streams, a wrong device
when available, capacity errors, repeated close, and standalone reopen/growth.
This coverage remains uncompiled and unexecuted here because `nvcc` is unavailable.

## Simulator-core operation

`simulator_core::expectation_pauli(qubits, operators, count, coefficient)` is an
internal synchronous operation returning `double`, exposed through the public
C/C++ entry points below. The operation requires allocated, explicitly initialized
state storage and matching arguments/call order on every participating rank.
Argument validation completes before pending projections or mappings are changed.

The path is:

1. Validate allocation, dimensions, input arrays, and non-identity support.
2. Materialize **all** pending projections using the current mapping, including
   measured qubits outside the requested string. This keeps the current raw
   branch amplitudes without renormalization.
3. Pass non-identity logical targets to the existing localization mechanism.
   Its communication and copy-back stream ordering are reused. Identity targets
   require no exchange. The updated logical/physical mapping is retained.
4. Build the descriptor from that updated mapping and enqueue both reduction
   stages using the shared measurement workspace.
5. Sum one device `double` per rank using in-place NCCL AllReduce on the owner
   stream. Ranks with zero local contribution also participate.
6. Copy one scalar to the host, synchronize that stream once, and multiply by
   the coefficient once.

Expectation does not use gate preparation/release bookkeeping, measurement
sampling, or classical-bit updates. Measured masks, recorded values/lists, and
measurement RNG state are preserved. Only the pending-projection mask is cleared
as its projections are applied. Subsequent gates and measurements use the
retained mapping. An exhausted measurement RNG does not prevent expectation.
No projection normalization, rank-specific early exit, intermediate host
synchronization, or additional communication path is introduced.

Run the private core harness on a configured CUDA/MPI/NCCL installation:

```sh
make test-pauli-core RANKS=1
make test-pauli-core RANKS=2
make test-pauli-core RANKS=4
```

`tests/pauli_core_test.cu` compares against the independent matrix oracle and
checks the physical slices against the expected logical state. It exercises
identity/reverse mappings, every single-qubit Pauli, mixed non-contiguous targets,
repeated localization, negative/zero coefficients, unnormalized states, pending
local/global projections inside/outside the request, rejection before mutation,
zero-contribution ranks, fixed workspace capacity, and subsequent gates/cached
and fresh measurements. It checks rank agreement and preservation of classical
bits, measured lists/masks, and the RNG cursor/exhaustion flag. Numerical
tolerances are `1e-12 + 5e-12 * abs(reference)` for expectations and `1e-12` for
logical amplitudes. These are test criteria, not measured results: the core
harness remains uncompiled and unexecuted here because `nvcc` is unavailable.

## Public API

Both entry points are declared in [`qcs.h`](../include/qcs.h):

```cpp
double qcs_simulator_expectation_pauli_cxx(
    qcs_simulator *sim, const bit_num_t *qubit_num_list,
    const qcs_pauli_operator *operators, bit_num_t qubit_num_count,
    double coefficient);

int qcs_simulator_expectation_pauli(
    qcs_simulator *sim, const bit_num_t *qubit_num_list,
    const qcs_pauli_operator *operators, bit_num_t qubit_num_count,
    double coefficient, double *result);
```

The C++ entry point throws on error. The C entry point uses the existing
exception callback and returns `1` on success or `0` on error. It requires a
non-null output pointer even for an empty string and writes the output only
on success. It validates that pointer before entering the core operation.

For example, after creating, allocating, and explicitly initializing `sim`,
compute `-0.5 * <X_0 Y_3>` from C:

```c
const bit_num_t qubits[] = {0, 3};
const qcs_pauli_operator operators[] = {QCS_PAULI_X, QCS_PAULI_Y};
double value;
if (!qcs_simulator_expectation_pauli(sim, qubits, operators, 2, -0.5, &value)) {
    /* The configured exception callback receives the error. */
    return 1;
}
```

The equivalent C++ call is:

```cpp
double value = qcs_simulator_expectation_pauli_cxx(sim, qubits, operators, 2, -0.5);
```

Pass null arrays with count zero for identity. Explicit identity operators are
also accepted. Null simulators, missing allocation, negative/excessive counts,
missing nonempty arrays, duplicate/out-of-range qubits, invalid operators,
non-finite coefficients, and support larger than the local qubit count are
rejected. Storage initialization remains a caller precondition because the
existing simulator does not track initialization separately from allocation.

These are synchronous collective calls. Every rank must call with the same
valid arrays, count, and coefficient in the same order. Input validation is
local; the API does not add a collective to detect mismatched arguments or
repair inconsistent rank participation. Calls sharing a simulator must be
serialized. The operation retains localization mapping changes, applies pending
projections, preserves measurement records/RNG/classical bits, and returns the
raw expectation including the coefficient without implicit normalization.

On a configured CUDA/MPI/NCCL machine:

```sh
make test-pauli-api RANKS=1
make test-pauli-api RANKS=2
make test-pauli-api RANKS=4
```

The integration executable links `tests/pauli_api_c.c` compiled as C11 with
`tests/pauli_api_test.cpp` and the shared library. Coverage includes both
interfaces, C status/callback behavior, unchanged output on rejected requests,
positive/negative/zero coefficients, empty/all-identity strings, signed Y
expectations, Bell correlations, and raw norms after measurement. The two
caller sources compile independently with strict warnings in this environment.
Linking and GPU execution remain unverified because `nvcc` is unavailable.
