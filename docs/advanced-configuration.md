# Advanced configuration and implementation

[README](../README.md) · [Building](building.md) · [Usage and API](usage.md) · [Advanced configuration](advanced-configuration.md)

Compile-time settings require rebuilding. See [Building](building.md) for vendor flags
and runtime library setup; all shell commands assume the repository root.

## NCCL environment defaults

Before any NCCL initialization, the simulator sets these defaults on every MPI
rank when the corresponding environment variable is unset or empty:

- `NCCL_NCHANNELS_PER_NET_PEER=8`
- `NCCL_P2P_NET_CHUNKSIZE=262144`

These are configurable performance-tuning defaults; they are not universally
optimal. Non-empty values are preserved. To override them without rebuilding,
export your preferred values before launching, for example:

```bash
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_P2P_NET_CHUNKSIZE=131072
mpirun -np 2 ./bin/qcs circuit.so
```

Ensure your MPI launcher propagates these variables to every rank. A failure to
set any default is checked collectively, so all ranks stop before NCCL
initialization. The mode-specific `NCCL_PXN_DISABLE` handling described below
uses the same collective check.

## Storage for kernel constants

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

## Device work buffer

See also [measurement workspace ownership](measurement-reducer.md#algorithm-and-ownership).

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

## Optional NVSHMEM exchange

NVSHMEM is disabled by default. Build with
`make clean && make NVSHMEM=1 NCCL_BUFFER_MODE=0 NVSHMEM_HOME=/path/to/nvshmem` to enable it.
This defines `QCS_ENABLE_NVSHMEM`, which automatically sets
`exchange_nvshmem_registered_put` in `src/qcs.cu` to `true` and selects
NVSHMEM registered-PUT exchange. When the macro is not defined, the constant
is `false` and the simulator uses NCCL exchange with its
`exchange_pipeline_copy_back` setting. Clean and rebuild when switching
between NCCL and NVSHMEM builds.

When NVSHMEM is enabled, initialization defaults `NVSHMEM_IB_ENABLE_IBGDA`
to `1` if it is unset or empty. Non-empty user values are preserved.

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

## Exchange copy-back pipeline

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

## CUDA synchronization settings

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

## Explicit NCCL buffer registration

The default NCCL exchange uses CUDA streams with pipelined copy-back.
[Explicit buffer registration](nccl-buffer-registration.md) is enabled by default
(`NCCL_BUFFER_MODE=2`), using `ncclMemAlloc` and `ncclCommRegister`. It requires
NCCL headers and runtime >= 2.31.2, NCCL exchange (`NVSHMEM=0`), and device memory.
Before communicator initialization, mode 2 sets `NCCL_PXN_DISABLE=1` when unset
or empty. Non-empty values are never overwritten and must equal `1` to pass
collective validation. Modes 0 and 1 leave the environment variable unchanged.
Use `make -B NCCL_BUFFER_MODE=0` for CUDA allocation or
`make -B NCCL_BUFFER_MODE=1` for NCCL allocation without registration. Force a
rebuild when switching modes; changing the Make variable alone does not rebuild
existing binaries. See the guide for allocation lifecycle and the cluster harness.
