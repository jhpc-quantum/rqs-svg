# Pauli expectation validation

[API and implementation](pauli-expectation.md) · [Validation](pauli-validation.md) · [Performance](pauli-performance.md) · [Acceptance report](pauli-acceptance.md)

GPU source and test coverage are implemented but remain uncompiled and unrun
locally. Host checks pass; see the [acceptance report](pauli-acceptance.md) for
observed results and outstanding checks.

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

## Standalone GPU tests

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

## Workspace coverage

The CUDA test source now includes alternating measurement reduction, device
sampling, and Pauli reduction on the same slice, without explicit synchronization
between operations. It checks poisoned storage, fixed output location,
pointer/capacity stability, adjacent SWAP guards, wrong streams, a wrong device
when available, capacity errors, repeated close, and standalone reopen/growth.
This coverage remains uncompiled and unexecuted here because `nvcc` is unavailable.

## Core integration tests

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

## Public API tests

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

## Distributed regression matrix

The private core harness now alternates between the public C and C++ APIs.
It adds seeded random complex states and Pauli strings across five initial
permutations (identity, reverse, and three shuffled), maximum local support,
Y counts one through four, and fresh global X/Y/Z targets. Bell/GHZ fixtures,
local/global pending projections inside and outside requests, unnormalized
states, repeated mapping changes, and a reallocated one-local-qubit state
are compared with the independent CPU oracle. Logical amplitude comparisons
also reject non-finite values explicitly.

The existing per-call checks cover rank agreement, inverse mappings, state
preservation, workspace stability, classical bits, and measurement/RNG metadata.
No additional test hooks or communication are added to production code.

Use the runner from the repository root on a configured machine:

```sh
python3 scripts/validate_pauli.py --output /tmp/pauli-validation --sanitizers
# Preview commands without compiling or running any test:
python3 scripts/validate_pauli.py --output /tmp/pauli-plan --dry-run
# Smaller available hardware / compatible NCCL mode:
python3 scripts/validate_pauli.py --output /tmp/pauli-single --ranks 1 --buffer-modes 0
# Optional NVSHMEM installation (MPI-enabled, with suitable runtime setup):
python3 scripts/validate_pauli.py --output /tmp/pauli-nvshmem --nvshmem --make-arg NVSHMEM_HOME=/path/to/nvshmem
```

The default matrix builds NCCL modes 0/1/2 and constant/global metadata variants,
then runs on 1/2/4 ranks with one GPU per rank. Each configuration uses separate
binary/library paths and a forced rebuild. The core and private measurement
harnesses use the selected metadata variant; the shared-library API executable
and existing standalone regressions use the production metadata setting.
NVSHMEM selects mode 0 only. Mode compatibility and software prerequisites
remain as documented in [NCCL registration](nccl-buffer-registration.md) and
[advanced configuration](advanced-configuration.md).

The runner includes standalone Pauli/measurement reducers and initialization
on one rank, plus the distributed Pauli core, public API, SWAP, and both private
measurement regressions. `--sanitizers` adds memcheck, racecheck, and synccheck
for the standalone Pauli reducer and distributed core harness. Vendor flags can
be passed with repeated `--make-arg` assignments; `--mpiexec` accepts a launcher
and arguments for the target cluster. Existing `config.mk` is read as usual.
Output paths and matrix variables are reserved to keep configurations isolated.

Each command has a separate log and status in `report.json`, alongside source
revision and working-tree status. The runner stops at the first failed command
and exits nonzero. Dry runs have status `planned`, never `passed`; tests absent
from the report after a failure were not attempted. Retain the report and logs
when reporting numerical or sanitizer results. Timing is outside this runner's
scope.

During this change, host checks and matrix dry-run validation were performed.
The actual matrix build stopped at missing `nvcc`; CUDA, MPI launch, NCCL,
NVSHMEM, all multi-GPU runs, and sanitizer results remain unverified. Added
CUDA harness coverage has not been compiled or executed locally.
