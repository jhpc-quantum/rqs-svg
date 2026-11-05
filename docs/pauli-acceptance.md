# Pauli expectation implementation and acceptance report

[API and implementation](pauli-expectation.md) · [Validation](pauli-validation.md) · [Performance](pauli-performance.md)

## Overall status

The implementation and validation/benchmark tooling for one Pauli string are
present. Host checks pass. **GPU acceptance is pending:** CUDA source, kernels,
integration tests, and benchmarks have not been compiled or run locally because
`nvcc` is absent from PATH. No GPU numerical error, latency, speedup, regression
pass, or sanitizer result is claimed. The original completion criteria requiring
passing GPU tests and measured performance are not yet satisfied.

This report reviews the implementation through commit `28f58b4` and the final
documentation changes. Existing APIs retain their signatures; new expectation
entry points are additive. Full VQE, Hamiltonian batching, and whole-Hamiltonian
optimization remain outside scope.

## Added APIs

- `qcs_pauli_operator`: `QCS_PAULI_I`, `QCS_PAULI_X`, `QCS_PAULI_Y`, `QCS_PAULI_Z`.
- `qcs_simulator_expectation_pauli_cxx(sim, qubits, operators, count, coefficient)`:
  returns `double`, throws on error.
- `qcs_simulator_expectation_pauli(sim, qubits, operators, count, coefficient, result)`:
  returns 1/0 through the existing exception convention; writes only on success.

Both APIs return the raw `coefficient * <psi|P|psi>` without implicit
normalization. Calls are collective with matching valid arguments/order on all
ranks and serialized access to each simulator. Pending projections are applied;
measurement records, classical bits, and RNG state are preserved. Localization
mapping changes are retained for subsequent operations.

## Acceptance criteria review

“Source reviewed” records implementation inspection, not a successful GPU run.
Coverage listed below exists in source unless an observed pass is stated.

| Criterion | Implementation and evidence | Status |
| --- | --- | --- |
| GPU expectation for one string | Public API → core → Pauli workspace → two kernels → NCCL → scalar readback | Source reviewed; GPU build/run pending |
| Only `i < j` pairs, each once | Highest flip bit is cleared by zero insertion; host exhaustive masks on 1–9 qubits and wide-index checks | Host checks passed; GPU pending |
| No pair-selection branch | Production paired kernel directly enumerates `local_dim/2`; scan-and-branch code exists only in benchmark | Source reviewed |
| Correct twice-real contribution | One product component, Y-phase selection, parity sign, factor two; independent matrix CPU oracle | Host checks passed; GPU pending |
| Efficient diagonal path | Separate compile-time specialization reads each amplitude once | Source reviewed; timing pending |
| Global-bit SWAP and mapping | Existing localization/exchange reused; descriptor built after mapping update | Coverage added; runtime pending |
| Correct multi-GPU sum | All ranks, including zero contributors, call one-double NCCL AllReduce; coefficient applied afterward | Coverage added; 1/2/4-GPU runs pending |
| Grid-stride, block sums, two stages | 256 threads; capped 4×SM grid; explicit partial count; parallel Stage 2 | Source reviewed; GPU pending |
| No large extra temporary storage | Existing measurement slice is borrowed; checked fixed scalar output; no Pauli device allocation | Source reviewed; allocation trace pending |
| Stream ownership and repeated use | Device/stream/capacity checks; same-stream serialized reuse; adapter destroyed before backing storage | Coverage added; sanitizers pending |
| Deferred projections and subsequent operations | All pending projections materialized before SWAP; records/RNG retained | Coverage added; runtime pending |
| No gate/measurement regressions | Existing measurement, SWAP, and initialization tests are included in matrix runner | Host SWAP/work-buffer checks passed; GPU regressions pending |
| Accuracy tests added and passing | Independent long-double matrix oracle, public C/C++ tests, distributed fixtures | Host suite passed; GPU accuracy acceptance pending |
| Key performance characteristics verified | Stage/API timing harness, baseline, state-size/rank/layout sweep, raw CSV and percentiles | Reporting checks passed on synthetic data; GPU measurements pending |

## Observed validation

The final review reran the following checks successfully:

- `make test-pauli-string`, using a temporary executable path: input validation,
  descriptor mapping, pair coverage, phases, and independent CPU oracle.
- `make test-work-buffer` and `make test-swap-launch`, using temporary paths:
  host storage budget/alignment/exhaustion and SWAP launch planning.
- C11 and C++17 compilation of `tests/pauli_api_c.c` and
  `tests/pauli_api_test.cpp` with `-Wall -Wextra -Werror -pedantic`.
- Generation of both constant/global private measurement source variants.
- Regression and benchmark dry-run command plans, Makefile planning, and
  explicit failure-status reporting. These do not count as GPU test passes.
- Relative documentation links/anchors and `git diff --check`.

Earlier benchmark reporting checks used explicitly synthetic CSV values to check
percentiles, ratios, and rejection of missing/duplicate samples. They supplied
no hardware performance evidence. An earlier optional UBSan link attempt was
blocked by a missing UBSan runtime library; sanitizer acceptance remains open.

The final GPU build attempt stopped at `nvcc: No such file or directory` before
compiling the shared library. `nvcc`, `nvidia-smi`, `mpiexec`, and
`compute-sanitizer` were not found on PATH. This describes the available tools,
not a physical GPU inventory or the capabilities of another configured machine.

## Changed files across the implementation

| Area | Files |
| --- | --- |
| Public API and core | `include/qcs.h`, `src/qcs.cu` |
| Descriptor and CUDA reduction | `src/pauli_string.hpp`, `src/pauli_reduce.cuh` |
| Shared workspace | `src/measurement_reduce.cuh` |
| Build and private source copying | `Makefile`, `Makefile.test`, `scripts/prepare_measurement_harness.py` |
| CPU oracle and host tests | `tests/pauli_reference.hpp`, `tests/pauli_string_test.cpp` |
| GPU, core, and public tests | `tests/pauli_reduce_test.cu`, `tests/pauli_core_test.cu`, `tests/pauli_api_c.c`, `tests/pauli_api_test.cpp` |
| Distributed validation | `scripts/validate_pauli.py` |
| Benchmark | `tests/pauli_benchmark.cu`, `scripts/benchmark_pauli.py` |
| Documentation | `README.md`, `docs/usage.md`, `docs/measurement-reducer.md`, `docs/pauli-expectation.md`, `docs/pauli-validation.md`, `docs/pauli-performance.md`, this report |

## Remaining acceptance work

On a configured CUDA/MPI/NCCL machine, first run the
[regression matrix](pauli-validation.md#distributed-regression-matrix), including
available 1/2/4-GPU configurations and sanitizers. Test optional NVSHMEM separately
when supported. Preserve the logs, build flags, hardware placement, and exact
revision. Resolve any compilation or runtime failures before marking GPU
acceptance complete.

Then run the [performance sweep](pauli-performance.md), retain raw samples,
variability and regressions, and report diagonal/non-diagonal, stage, rank, and
SWAP-path behavior. Compare generated/scan baseline ratios; no memory-access
optimization is justified by measurements in this environment. A production
allocation/stream trace is still needed to confirm runtime allocation reuse
and ordering beyond source inspection.

The 1–63 total-qubit bound is an indexing/mask limit, not a practical allocation
promise. Non-identity support must fit local qubits; state initialization is a
caller precondition. Input validation is local and does not detect mismatched
rank arguments. Thread safety and concurrent use of one simulator are not added.
Large dimension arithmetic is covered on the host without allocating states of
those sizes. Multi-node and heterogeneous-GPU behavior remain unverified.
