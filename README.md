# RQS-SVG

RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation using CUDA,
MPI, and NCCL. It provides common quantum gates, state preparation, and
measurement through a [C/C++ API](include/qcs.h), a standalone `bin/qcs`
executable, and a shared library, `lib/libqcs.so`.

## Build

You need an NVIDIA GPU, the CUDA toolkit, NCCL, MPI, C/C++ compilers, and the
[`atlc`](https://github.com/naoto-aoki-fy/atlc) headers.
Create `config.mk` with your installation's include paths, library flags
(including MPI), and GPU architecture flags:

```make
INCLUDE = -I./include -I/path/to/atlc/include
CFLAGS_VENDOR = -I/path/to/mpi/include -I/path/to/nccl/include
LDFLAGS_VENDOR = -L/path/to/mpi/lib -lmpi -L/path/to/nccl/lib
GENCODE_FLAGS = -gencode=arch=compute_xx,code=sm_xx
```

Replace the paths and `xx` with values for your system, then build:

```sh
make
```

See [Building and runtime setup](docs/building.md) for configuration help and
shared-library setup.

## Run an example

From the repository root, compile the supplied GHZ circuit and run it on one GPU:

```sh
./compile_circuit.sh examples/standalone/ghz_circuit.c -o ghz_circuit.so
NUM_QUBITS=3 mpirun -np 1 ./bin/qcs --num-samples 10 ghz_circuit.so
```

## Documentation

- [Building and runtime setup](docs/building.md): prerequisites, vendor flags, shared library, and library search paths.
- [Usage and API](docs/usage.md): circuit compilation, MPI execution, statevector output, initial states, C linking, SWAP, and measurement APIs.
- [Advanced configuration and implementation](docs/advanced-configuration.md): kernel constants, work buffers, NVSHMEM exchange, NCCL copy-back, and CUDA synchronization.
- [Measurement reducer](docs/measurement-reducer.md): grouped reduction, GPU sampling, RNG guarantees, regression tests, and profiling.
- [Explicit NCCL buffer registration](docs/nccl-buffer-registration.md): default buffer registration, alternative modes, and cluster validation.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
