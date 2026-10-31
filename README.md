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

### Storage for kernel constants

`use_global_memory_for_constants` near the top of `src/qcs.cu` defaults to
`false`, using CUDA constant memory. Set it to `true` and rebuild with
`make` to store the same data in device global memory. `make sharedlibrary`
rebuilds the library alone; restore `false` to return to constant memory.

To compare both modes on a CUDA/MPI system, build each setting with
`make clean && make target sharedlibrary`. For each build, compile and run the
same deterministic circuits using `./compile_circuit.sh`, then
`mpirun -np 1 ./bin/qcs --output-statevector MODE.bin circuit.so` (after
`source ./env.bash`). Compare the resulting state vectors and qubit mappings.
Include ordinary and controlled gates, and a circuit that prepares a known
measurement outcome, measures a qubit, then applies a gate to that measured
qubit. Use deterministic measurements so the two runs select the same branch.

### Optional NVSHMEM exchange

The `QCS_ENABLE_NVSHMEM` build definition selects the NVSHMEM exchange path.
It is absent in the normal build and is defined automatically when `NVSHMEM=1`
is passed to `make`. Build both the shared library and executable with
`make clean && make NVSHMEM=1 NVSHMEM_HOME=/path/to/nvshmem` to enable it.
Use the same option with `make sharedlibrary` for a library-only build. The
NVSHMEM installation must support MPI bootstrap and GPU-initiated PUT and
signals for the selected CUDA devices and transport. Configure MPI compiler
and link flags in `config.mk` as for the normal build; expose NVSHMEM's host
library at runtime (for example through `LD_LIBRARY_PATH`). Set
`NVSHMEM_SYMMETRIC_SIZE` large enough for a staging allocation of up to 4 MiB,
the signals, and NVSHMEM runtime overhead on each GPU. All MPI ranks must
participate in the same simulator operations in the same order, with one GPU
per rank and NVSHMEM PE numbering equal to MPI rank numbering. Unified memory
is unsupported in this mode. The state vector uses registered `cudaMalloc`
storage; staging and signals use the symmetric heap. The staging capacity is
the smallest available capacity across ranks, capped at 4 MiB.

The registered PUT path uses one staging slot. It transfers each chunk before
waiting for its peer's ready signal, copies the received data into the state
vector, then exchanges consumed signals before either peer reuses the slot.
The monotonically increasing signals persist across exchanges. Kernels use
NVSHMEM collective launch and execute on the simulator stream. The
`exchange_pipeline_copy_back` setting affects only the NCCL path.

For validation, build a baseline with the selector `false` for each value of
`exchange_pipeline_copy_back`, and an NVSHMEM build with the selector `true`.
Run the same deterministic circuits using the commands below at two and four
ranks, saving each output under a different filename. Compare the binary state
vectors and reported qubit mappings. Include no-exchange, self-peer,
single- and multiple-qubit exchanges, repeated exchanges followed by gates,
and multiple peers. Temporarily set the 262144-element staging cap to 3 in a
validation build to force multiple chunks and a partial final chunk; restore it
and the selector before normal use. Delay one rank in a test harness to exercise
uneven progress. These checks require a multi-GPU MPI/NVSHMEM installation.

### Exchange copy-back pipeline

`exchange_pipeline_copy_back` near the top of `src/qcs.cu` defaults to
`true`. It splits the existing swap buffer into two receive slots and overlaps
copy-back on a separate stream with subsequent NCCL transfers. If the buffer
contains fewer than two complex elements, the exchange uses the original
single-stream path. Set the constant to `false` and rebuild to use that path
for every exchange; no pipeline streams or events are created in that build.

To validate on a CUDA/NCCL multi-GPU system, build both settings with `make`,
then run the same deterministic, unmeasured circuits with `mpirun -np 2` and,
where available, `mpirun -np 4`, using `--output-statevector` to save their
results. Compare the resulting state vectors and logical-to-physical mappings
against the disabled build. Exercise circuits with no exchange, consecutive
exchanges followed by gates, and exchanges spanning multiple local areas and
peers. To exercise slot reuse and the one-element fallback, temporarily limit
the swap-buffer allocation in a validation build to suitable sizes; include at
least three chunks for the reuse case. Restore the allocation before normal
use. Compare exchange timings under identical circuit, rank, and buffer-size
conditions if measuring performance.

### CUDA synchronization settings

The compile-time settings near the top of `src/qcs.cu` select stream and event
synchronization independently:

- `stream_sync_flags` accepts `cudaDeviceScheduleSpin`,
  `cudaDeviceScheduleYield`, `cudaDeviceScheduleAuto`, or
  `cudaDeviceScheduleBlockingSync` (the default).
  It controls the CUDA device scheduling policy used while synchronizing streams.
- `event_sync` accepts `event_sync_mode::default_sync`, `blocking_sync` (the
  default), or `busy_loop`. Blocking mode creates events with
  `cudaEventBlockingSync`; busy-loop mode polls with `cudaEventQuery`.
- `event_busy_loop_yield` controls whether the busy loop yields the host thread
  between incomplete queries. Its default is `true` and it has no effect in
  the other event modes.

Edit these constants and run `make` to rebuild the simulator, or run
`make sharedlibrary` to rebuild the shared library alone.

If you will run programs that load or link against `lib/libqcs.so`, source the provided shell setup file from the repository root or from any other directory:

```sh
source ./env.bash
```

The setup file appends this repository's absolute `lib` directory path to `LD_LIBRARY_PATH`.

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

Measurements use groups of up to eight qubits, with any remainder measured
together as a smaller group. Each group containing unmeasured qubits computes
joint probabilities with one CUB reduction per participating rank, then one MPI
probability sum and one result broadcast. The first target is the
least significant bit of the internal joint outcome. Already measured qubits
retain their values, and projection remains deferred until needed by a gate or
statevector output. Global targets use rank bits without exchanging the
statevector.

The internal `measure_qubits<N>` and `IndirectLoad<N, L>` templates support
widths 1–4, where `L` counts unmeasured local targets. `measure_group_size` in
`src/qcs.cu` selects the production grouping width; the public C API is unchanged.
The device result buffer grows to `sizeof(MeasureNorm<N>)` as needed, and CUB
scratch requirements are queried for each reduction configuration. Both buffers
are reused. Wider groups increase reduction storage exponentially and are not
necessarily faster. Joint sampling preserves probability distributions, but
consumes fewer random draws, so seeded results can differ from sequential
measurement.

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
