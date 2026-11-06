# Usage and API

[README](../README.md) · [Building](building.md) · [Usage and API](usage.md) · [Advanced configuration](advanced-configuration.md)

All commands assume the repository root and a configured [build](building.md).

Use the [Pauli expectation API](pauli-expectation.md#public-api) to calculate
a single Pauli string expectation value on the GPU.

## Compiling a circuit

Use the helper script:

```sh
./compile_circuit.sh user_circuit.c -o user_circuit.so
```

Alternatively:

```sh
source ./env.bash
gcc -fPIC -shared user_circuit.c -o user_circuit.so
```

The loadable circuit examples live under [`examples/standalone/`](../examples/standalone/).

[`measure-only.c`](../examples/standalone/measure-only.c) measures all qubits
into their corresponding classical bits without applying any gates. With the
default zero initial state, every measured bit is zero. Set `NUM_QUBITS` to
the desired qubit count:

```sh
./compile_circuit.sh examples/standalone/measure-only.c -o measure-only.so
NUM_QUBITS=3 mpirun -np 1 ./bin/qcs --num-samples 10 measure-only.so
```

## Running

Use one MPI rank per GPU. Replace `NUM_GPUS` with the desired rank count.

You can execute the built simulator via:

```sh
mpirun -np NUM_GPUS ./bin/qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np NUM_GPUS ./bin/qcs --output-statevector data.bin user_circuit.so
```

## Initial state selection

Use `bin/qcs --initial-state flat --num-samples 10 circuit.so` to select
`zero` (the default), `flat`, `entangled`, `sequential`, `use_curand`, or
`load_statevector`. Explicit initialization after allocation and every subsequent
sample reset use the same initializer. Existing amplitudes and normalization
behavior are preserved.
`use_curand` reuses the existing seed on each reset, regenerating the same initial
vector while measurement sampling continues to advance its own RNG cursor.
`load_statevector` calls the existing `initialize_laod_statevector()` and rereads
`statevector_input.bin` from the working directory on every reset, retaining its
existing per-rank physical slice layout.

The original C function `qcs_simulator_allocate_memory(sim)` and its ABI remain
available and select zero, but allocation leaves state-vector contents
uninitialized and does not zero-fill them. Initialize explicitly before gates,
measurements, or saving the vector:

```c
qcs_simulator_allocate_memory(sim);
qcs_simulator_set_zero_state(sim);
```

C callers can select another method with
`qcs_simulator_allocate_memory_with_initial_state(sim, QCS_INITIAL_STATE_FLAT)`;
C++ callers can also use `qcs_simulator_allocate_memory(sim, QCS_INITIAL_STATE_FLAT)`.
The throwing C++ API has the corresponding two-argument
`qcs_simulator_allocate_memory_cxx` overload. These variants also only allocate
storage and remember the method; call `qcs_simulator_set_zero_state` (or
`qcs_simulator_set_zero_state_cxx` for the throwing API) to run it explicitly.
In particular, allocation with `load_statevector` does not read the input file.
Invalid enum values throw through
the C++ API and return failure through the C API's existing exception callback.
The legacy `qcs_simulator_set_zero_state` name now restores the selected method.
It discards measurement metadata; callers still reset the mapping and classical
bits separately. Explicit flat/sequential/entangled/random setters retain their
existing behavior and do not change the method remembered at allocation.

`make test-initial-state` builds and runs single-GPU regression coverage for all
six methods, deferred file loading, invalid enum rejection, explicit initialization
after legacy default allocation, and restoration after repeated measurements. Run it with one MPI rank on a CUDA/MPI/NCCL system.

## Using the shared library from C

The shared library exposes the same C API through [`include/qcs.h`](../include/qcs.h), plus
`qcs_simulator_create` and `qcs_simulator_destroy` helpers for C programs that
cannot allocate the opaque `qcs_simulator` type directly.

Build the shared library and then compile the C example:

```sh
make sharedlibrary
source ./env.bash
gcc examples/sharedlibrary/ghz.c -L./lib -lqcs -o ghz
mpirun -np 1 ./ghz --num-qubits 3
```

Code examples for linking against `libqcs.so` live under
[`examples/sharedlibrary/`](../examples/sharedlibrary/).

Python bindings are maintained separately at
[`rqs-svg-py`](https://github.com/naoto-aoki-fy/rqs-svg-py).

### Swapping pairs of qubits

`qcs_simulator_gate_swap` applies up to `qcs::gate::max_num_swaps` disjoint
SWAPs in one gate operation. Supply an even number of distinct targets, from 2
through twice that maximum:
entries 0 and 1 form one pair, entries 2 and 3 form the next, and so on.
Positive and negative controls condition all swaps together. The supported
maximum is set by the `qcs::gate::max_num_swaps` constexpr in `src/qcs.cu`;
other target counts are rejected.
The C++ API is `qcs_simulator_gate_swap_cxx`, which selects `qcs::gate::swap<N>()`
with a compile-time `N` based on the target count.
The numbered C APIs `qcs_simulator_gate_swap1` through `qcs_simulator_gate_swap4`
remain available and require exactly 2, 4, 6, and 8 targets, respectively.

```c
const bit_num_t targets[] = {0, 3, 1, 2};
/* SWAP(0, 3) and SWAP(1, 2) */
int ok = qcs_simulator_gate_swap(sim, targets, 4, NULL, 0, NULL, 0);
```

### Measuring multiple qubits

`qcs_simulator_measure_many` measures an ordered list of distinct logical
qubits in the Z basis and returns one correlated result per input element.
`qcs_simulator_measure_many_to_clbits` additionally writes each result to the
classical bit at the corresponding position in its destination list.  The two
counts must match and destinations must be distinct.  An empty request is a
successful no-op. Using the current logical-to-physical mapping, the `_to_clbits`
variant orders targets by descending local physical index, followed by ascending
global physical index, then forms measurement groups. Results and classical bit
destinations still correspond to the original input positions.

```c
const bit_num_t qubits[] = {5, 1, 3};
const bit_num_t clbits[] = {0, 2, 4};
bit_t results[3];

int ok = qcs_simulator_measure_many_to_clbits(
    sim, qubits, 3, clbits, 3, results);
```

All MPI ranks must invoke a measurement with the same ordered lists.  A later
measurement observes the already selected branch; prepare the state again for
an independent shot.

See [Measurement reducer](measurement-reducer.md) for grouped measurement,
GPU reduction and sampling, RNG guarantees, and validation commands.
