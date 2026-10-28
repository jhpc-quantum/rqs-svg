# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.h`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds a standalone `bin/qcs` executable and can also build `lib/libqcs.so` for programs that link against RQS-SVG directly.

## Building Simulator

The project requires NVIDIA's CUDA toolkit as well as NCCL and MPI.

Before building, set up the
[`atlc`](https://github.com/naoto-aoki-fy/atlc)
repository, which is required during the build process.

Define `CFLAGS_VENDOR`, `LDFLAGS_VENDOR`, and `GENCODE_FLAGS` in `config.mk`, which is automatically included by the Makefile:

```make
CFLAGS_VENDOR = -I/foo/bar
LDFLAGS_VENDOR = -L/foo/bar -lfoobar
GENCODE_FLAGS = -gencode=arch=compute_xx,code=sm_xx
```

These options can be obtained using the [`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) utility.

Then, `make` will build `bin/qcs`.

```sh
make
```

To build RQS-SVG as a shared library at `lib/libqcs.so`:

```sh
make sharedlibrary
```

If you will run programs that load or link against `lib/libqcs.so`, source the provided shell setup file from the repository root or from any other directory:

```sh
source ./env.bash
```

The setup file appends this repository's absolute `lib` directory path to `LD_LIBRARY_PATH`.

## Selecting the qubit-exchange method

NCCL remains the default. To enable the registered PUT algorithm (method 2 in
`buffer-swap/src/nvshmem_swap_bench.cu`), build with NVSHMEM support:

```sh
make clean
make USE_NVSHMEM=1
```

Add the NVSHMEM include and library directories to `CFLAGS_VENDOR` and
`LDFLAGS_VENDOR` in `config.mk`, alongside the existing CUDA/MPI/NCCL settings.
The build adds relocatable device code and links `nvshmem_host` and
`nvshmem_device`. Use an NVSHMEM version providing `nvshmemx_buffer_register`.
Its shared libraries must also be available on the runtime library path.
Rebuild after changing `USE_NVSHMEM`.

Select the method before creating the simulator, including when using the C or
Python library API. Set the same value on every MPI rank:

```sh
export QCS_SWAP_METHOD=2
export NVSHMEM_SYMMETRIC_SIZE=128M
mpirun -np 2 ./bin/qcs user_circuit.so
```

Unset `QCS_SWAP_METHOD`, or set it to `nccl` or `0`, to use NCCL. A build without
NVSHMEM rejects method `2` with an error. Values `1` and `3` from the benchmark
are not implemented in the simulator.

Method 2 registers the CUDA statevector outside the symmetric heap. The heap
holds staging buffers and signals: reserve at least 256 KiB plus 8 bytes per
block, with additional allocator headroom. There is one allocated block per SM
of the smallest GPU in the MPI job; the launch grid is capped by cooperative
launch capacity. `128M` is an example heap size; increase it if needed. Transfers
use chunks of up to 256 KiB, shortened at qubit-region boundaries.

This method requires a power-of-two number of ranks, distinct GPUs per local
rank, and Volta-or-newer GPUs supporting cooperative launch. All ranks must use
the same qubit count, mapping, and sequence of simulator operations. Transport
settings such as `NVSHMEM_IB_ENABLE_IBGDA` are left to the environment. XOR peer
rounds support exchanges involving multiple global qubits; a stream barrier
separates rounds before staging buffers are reused with a different peer.

CUDA compilation and NVSHMEM transport behavior require a target-system build
and execution.

## Compiling Circuit

Use the helper script:

```sh
./compile_circuit.sh user_circuit.c -o user_circuit.so
```

Alternatively:

```sh
source ./env.bash
gcc -fPIC -shared user_circuit.c -o user_circuit.so
```

The loadable circuit examples live under `examples/standalone/`.

## Using the Shared Library from C

The shared library exposes the same C API through `include/qcs.h`, plus
`qcs_simulator_create` and `qcs_simulator_destroy` helpers for C programs that
cannot allocate the opaque `qcs_simulator` type directly.

Build the shared library and then compile the C example:

```sh
make sharedlibrary
source ./env.bash
gcc examples/sharedlibrary/ghz.c -lqcs -o ghz
mpirun -np (NUM_GPUS) ./ghz
```

Code examples for linking against `libqcs.so` live under
`examples/sharedlibrary/`.

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
successful no-op.

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

## Running

You can execute the built simulator via:

```sh
source ./env.bash

mpirun -np (NUM_GPUS) ./bin/qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np (NUM_GPUS) ./bin/qcs --output-statevector data.bin user_circuit.so
```

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
