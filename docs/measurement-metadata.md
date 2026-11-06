# Measurement gate metadata removal

`qcs_simulator_measure_many_to_clbits()` validates the request, orders targets
by physical location, calls `measure_qubit_groups()`, and restores caller order
for results and classical destinations. Each fresh group calls `measure_qubits<N>()`
and then `reduce_measure_norm<N>()` on participating ranks. Excluded ranks enqueue
zero weights; all ranks still participate in the existing NCCL reduction and
device sampling.

Measurement no longer calls `prepare_operating_gate()`. That function builds
`kernel_input_qnlist_struct` (target/control counts, positive-control and target
lists, measured-bit fields, and sorted operands), copies it to
`qcs_kernel_input_constant_addr` with `cudaMemcpyAsync`, and computes gate launch
dimensions. None of these outputs is read by the measurement reducer or sampler.
The reducer packs its independent descriptor and passes it by value to its
kernel; it derives its own launch configuration. Subsequent gates still call
`prepare_operating_gate()` before consuming its outputs.

This removes one explicit gate-metadata H2D copy per fresh measurement group
on ranks satisfying the prior global controls, along with buffer preparation,
operand sorting, and gate launch calculations. Cached groups and excluded ranks
already skipped that copy. Kernel argument delivery remains necessary.

The existing localization call remains: it populates the ordered physical target
list and clears exchange work lists. Measurement supplies only local unmeasured
targets, so this call does not exchange statevector data. Physical control checks
remain to map prior measurements, construct local control lists, and exclude
ranks with incompatible global controls. These lists supply the descriptor's
control mask/value; local target positions, outcome-bit positions, and fixed
cached/global outcomes also remain necessary. Validation, measured masks/value
lists, pending projections, RNG accounting, NCCL, readback, and classical-bit
updates are unchanged. No synchronization or communication was added.

## Validation for this change

- `git diff --check` passed.
- Strict C11 syntax checking of `examples/standalone/measure-only.c` passed.
- Private measurement harness source generation succeeded for constant and
  global gate-metadata modes.
- `make -k sharedlibrary test-measurement-reduce test-measurement-integration
  test-measurement-distributed` could not build because `nvcc` is absent from
  PATH. No GPU tests ran. Deterministic/uniform states, multiple groups,
  different mappings, and single-/multi-GPU runtime correctness remain unverified.
- `nvcc`, `nvidia-smi`, `mpiexec`, and `nsys` were absent from PATH. No transfer
  trace, measurement latency, speedup, or regression was measured.

See [measurement reducer validation and profiling](measurement-reducer.md) for
the existing regression commands. Run those on a configured CUDA/MPI machine,
including one, two, and four ranks and both metadata modes. Use
`examples/standalone/measure-only.c` for a representative end-to-end trace;
compare warmed measurements on identical hardware and configuration.
