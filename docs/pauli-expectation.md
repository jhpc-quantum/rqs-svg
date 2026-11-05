# Pauli string expectation contract

The host-side input contract and CPU tests are implemented. GPU reduction,
localization integration, and public expectation functions are future work.
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
norm, including for unnormalized and projected states. The future operation
must materialize all pending measurement projections before reduction while
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
