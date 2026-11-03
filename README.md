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

By default, `bin/qcs` embeds the absolute directory containing `libqcs.so`
in its runtime library search path using `-Wl,-rpath`. No `LD_LIBRARY_PATH`
setting is needed to locate `libqcs.so`. Other dependencies, such as CUDA,
NCCL, and MPI, must still be available to the runtime loader. Rebuild the
executable if you move the repository.

To disable the embedded search path, rebuild with:

```sh
make clean
make QCS_RPATH=0
```

With this option, use `source ./env.bash` or configure the runtime library
search path yourself. Rebuild the executable when changing `QCS_RPATH`.

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

### Device work buffer

After allocating the state-vector storage, the simulator queries the
remaining GPU memory and allocates `floor(free_bytes * 95 / 100)` bytes as one
work buffer. Measurement partials, output bins and sampler scratch use a fixed
slice sized for every supported group width (1 through 4). The NCCL swap buffer
uses another slice, capped at half the local state vector and reduced to a
fitting power of two when necessary. Slices are aligned to 256 bytes and never
freed individually. Insufficient capacity raises an error before use.

NVSHMEM requires symmetric addresses for staging and signals. The same work
buffer owner therefore also holds a separate symmetric backing allocation and
carves staging and signal slices from it. This allocation precedes the 95%
CUDA memory query; its size remains determined by the exchange grid. The
unallocated 5% of ordinary free memory remains available to runtime libraries.

Run `make test-work-buffer` for host tests of budget arithmetic, alignment and
capacity limits, and `make test-measurement-reduce` on CUDA hardware for shared
buffer isolation and measurement correctness.

### Optional NVSHMEM exchange

NVSHMEM is disabled by default. Build with
`make clean && make NVSHMEM=1 NVSHMEM_HOME=/path/to/nvshmem` to enable it.
This defines `QCS_ENABLE_NVSHMEM`, which automatically sets
`exchange_nvshmem_registered_put` in `src/qcs.cu` to `true` and selects
NVSHMEM registered-PUT exchange. When the macro is not defined, the constant
is `false` and the simulator uses NCCL exchange with its
`exchange_pipeline_copy_back` setting. Clean and rebuild when switching
between NCCL and NVSHMEM builds.

The installation must support MPI bootstrap, registered GPU buffers, device PUT
and signals, and collective launches. Configure MPI compiler and link flags in
`config.mk`, expose the NVSHMEM host library at runtime, and provide sufficient
`NVSHMEM_SYMMETRIC_SIZE` for staging, signals, and runtime overhead. All ranks
must execute the same simulator operations in the same order, with one GPU per
rank and PE numbering equal to MPI rank numbering. Unified memory is not
supported by this path. The state vector is registered once; staging and
signals are symmetric allocations reused until cleanup.

Each block has 32 threads, a 512 KiB staging slot (32,768 complex values),
and independent ready and consumed signals. The grid contains exactly 16
blocks per SM. All PEs must have the same SM count and support the requested
grid under both CUDA hardware and NVSHMEM collective-launch limits; otherwise
initialization fails with a diagnostic. Rank 0 prints the SM count, grid,
threads, chunk size, and staging allocation. Set `NVSHMEM_SYMMETRIC_SIZE` on
every rank to at least `SM count * 16 * (512 KiB + 16 bytes)` plus NVSHMEM
runtime and allocation overhead. For example, a GPU with 80 SMs needs 640 MiB
of staging and 20 KiB of signals, plus overhead. Ensure the GPU has enough
memory for the symmetric heap, state vector, and other allocations. NVSHMEM
may reserve the heap during initialization; free GPU memory after initialization
does not indicate remaining space within it. One capacity query and one MPI
all-reduce establish this layout at allocation time. Each exchange uses one collective kernel launch per nonzero
XOR peer phase (`2^k - 1` launches for `k` exchanged global qubits), no MPI
collectives, and one stream-ordered NVSHMEM all-PE barrier per phase. Each
kernel distributes all regions and chunks for its peer across blocks and
performs successive rounds on the GPU. The barrier prevents a new peer's
signal from satisfying an earlier peer's wait; it can limit performance when
many peers participate. The kernel follows the reference method 2 PUT, ready
wait, copy, consumed notification, quiet and wait sequence. Generations
increase across phases and exchanges, with overflow rejected. All work is
ordered on the simulator stream before subsequent gates. An additional
stream-ordered barrier after signal initialization prevents peers from
signaling counters that have not yet been cleared.

For validation, build NCCL baselines for both `exchange_pipeline_copy_back`
settings with `NVSHMEM=0` and an NVSHMEM build with `NVSHMEM=1`. Run identical
deterministic circuits at two and four ranks (and eight if available), saving
and comparing state vectors and qubit mappings. Include single and multiple
qubit exchanges, multiple peers, non-contiguous regions, repeated exchanges,
and subsequent gates. Exercise multiple rounds and partial chunks with suitable
state sizes, or temporarily reduce `exchange_chunk_bytes` in a validation
build and restore it afterwards. Delay one rank in a harness to exercise uneven
progress. Measure exchange latency and bandwidth under identical workloads;
these tests require a multi-GPU MPI/NVSHMEM installation.

### Exchange copy-back pipeline

`exchange_pipeline_copy_back` near the top of `src/qcs.cu` defaults to
`true` and applies only to NCCL exchange (when `QCS_ENABLE_NVSHMEM` is not
defined). It splits the existing swap buffer into two receive slots and overlaps
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

For programs without an embedded runtime search path that load or link against
`lib/libqcs.so`, source the provided shell setup file:

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

Measurements use groups of up to four qubits (the current maximum and default),
with any remainder measured together as a smaller group. Each group containing
unmeasured qubits computes joint unnormalized squared norms in double precision
with a custom two-stage CUDA reduction. The first stage visits only states
matching all previously measured local controls, accumulates bins in shared
memory, and writes bin-major partial sums. The second stage reduces those
partials into the outcome bins. The first target is the least significant bit
of the internal joint outcome; unsorted targets and logical-to-physical mappings
retain their caller-order output positions.

Each fresh group sums corresponding outcome bins with an in-place NCCL
AllReduce. Inactive ranks contribute zeros and still reduce and sample. On each
GPU, a fixed CUB warp scan forms cumulative weights and a local Philox draw
selects the first positive bin with cumulative weight strictly above `u*total`.
Weights need not be normalized. Invalid components or totals fail through the
existing C error return (or C++ exception). Only an eight-byte outcome/status
returns to the CPU; probability arrays stay on device.

The RNG maps the common setup seed and a simulator-lifetime 64-bit group counter
to a uniform in `[0,1)`. Each successfully sampled group advances once on every
rank, including deterministic distributions and ranks contributing zero. Empty
and entirely cached groups do not advance. State preparation, measurement resets
and between-shot resets preserve the counter; exhaustion throws without wrapping.
The setup seed broadcast remains; sampling broadcasts no state, draw or outcome.

Equal seed/counter inputs yield equal raw draws. Agreement within a distributed
run requires identical NCCL bins and compatible compiled scan arithmetic on all
GPUs. Exact outcome sequences across GPU counts, decompositions, architectures
or toolkit versions are not guaranteed: floating-point reduction can change.
The legacy CPU RNG sequence is not preserved.

Global-only targets use rank bits without exchanging amplitudes or changing the
mapping. Already measured targets retain recorded values, and entirely cached
groups need no collective or draw. Projection remains deferred until a gate or
statevector output requires it. `measure_group_size` selects widths 1–4, with a
maximum and default of four. The simulator workspace reuses partial/output/result
storage after warm-up on the owning device and stream.

See [measurement documentation](docs/measurement-reducer.md) for RNG mapping,
numerical boundaries, regression coverage, multi-node launch instructions,
profiling, and the execution report. On a configured CUDA/MPI machine:

```sh
make target sharedlibrary
make test-measurement-reduce
make test-measurement-integration
make test-measurement-distributed RANKS=1
make test-measurement-distributed RANKS=2
make test-measurement-distributed RANKS=4
mpiexec -n 2 ./bin/measurement_integration_test
```

These tests have not been compiled or run for this change; tests and validation
were explicitly skipped.

## Running

You can execute the built simulator via:

```sh
mpirun -np (NUM_GPUS) ./bin/qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np (NUM_GPUS) ./bin/qcs --output-statevector data.bin user_circuit.so
```

### Initial state selection

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


Optional NCCL exchange experiments: [explicit buffer registration](docs/nccl-buffer-registration.md) and [CUDA Graph execution/registration](docs/nccl-cuda-graphs.md). Both are disabled by default.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
