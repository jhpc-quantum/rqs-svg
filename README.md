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
