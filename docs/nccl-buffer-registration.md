# Explicit NCCL buffer registration

For the optional graph modes and their separate registration controls, see
[CUDA Graph experiments](nccl-cuda-graphs.md).

Build-time `NCCL_BUFFER_MODE` selects the experiment (default `0`):

| Mode | State and scratch allocations | Explicit registration |
| --- | --- | --- |
| 0 | Original CUDA allocators | Disabled |
| 1 | `ncclMemAlloc` / `ncclMemFree` | Disabled |
| 2 | Same as mode 1 | `ncclCommRegister` / `ncclCommDeregister` |

Modes 1 and 2 require NCCL headers and runtime >= 2.31.2, NCCL exchange
(`NVSHMEM=0`), and device memory rather than managed memory. Mode 2 additionally
requires `NCCL_PXN_DISABLE=1` on every rank. Unsupported or inconsistent settings
are rejected collectively. Allocation is a collective application operation:
every rank must enter it in the same order. An existing allocation must be freed
before reallocation; rejected repeated allocation preserves the existing state.
Changing a Make variable does not invalidate an existing binary: force rebuilds
or use distinct output paths as below.

The entire state vector backs all send offsets. Both receive slots are slices of
the shared scratch allocation, so its owner uses the compatible allocator too.
Registration covers the entire scratch allocation, including measurement
AllReduce inputs/outputs, rather than separately registering overlapping slices.
The scratch budget remains 95% of free memory sampled after the committed state
allocation. VMM allocation rounding and registration metadata consume additional
physical resources; an allocation failure is reported rather than changing the
budget or falling back to a different allocator. Modes 1/2 select the minimum
swap capacity across ranks to keep chunk boundaries consistent.

Each region is registered once per allocation lifetime on the core communicator.
The exchange loop, grouping, two-slot copy pipeline, and permutation update are
unchanged. Teardown synchronizes the communication and copy streams, deregisters
both regions while the communicator exists, then frees memory using its matching
allocator. Allocation/registration errors are collected across ranks; successful
partial allocations/handles are rolled back on every rank before an error is
returned. `free_memory` supports subsequent allocations on the same communicator.
Communicator destruction precedes MPI finalization. Fatal CUDA/NCCL teardown
errors are reported; allocations whose deregistration fails are retained rather
than freed while registered, and the caller should terminate the failed job.

## Build and run on the cluster

Use the site's existing `config.mk` for CUDA architecture flags, MPI include/link
flags, and NCCL paths. Ensure both compilation and runtime resolve NCCL 2.31.2.
For a production binary, for example:

```sh
make -B NVSHMEM=0 NCCL_BUFFER_MODE=2 sharedlibrary target
```

Build all three standalone harnesses with identical compiler/vendor flags:

```sh
for mode in 0 1 2; do
  make -B nccl-buffer-harness NVSHMEM=0 NCCL_BUFFER_MODE="$mode"
done
```

Use the scheduler allocation and site MPI options for **one rank per GPU, four
ranks per GB200 node**; keep CPU binding, rank placement, plugin, and NIC selection
identical. For Open MPI, an example inside an allocation is:

```sh
export NCCL_PXN_DISABLE=1
export NCCL_NCHANNELS_PER_NET_PEER=8  # replace with the site's tuned value
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,REG,NET
export NCCL_DEBUG_FILE='/tmp/nccl.%h.%p.log'
for mode in 0 1 2; do
  mpiexec -n 64 --map-by ppr:4:node --bind-to core \
    ./bin/nccl-buffer-"$mode"-graph-0 10 6 30
done
```

Confirm the site's MPI launcher exports these variables to every node. Preserve
logs separately for each mode/job. Keep PXN, channel counts, cap, local bits,
repetitions, topology, affinity, plugin version, and all workload settings equal
between modes 1/2. Record the actual printed cap; a memory-constrained run may use
less than requested. Also compare production workloads with identical circuit,
qubit count, initial state, and application completion synchronization. The
harness measures a fixed repeated exchange, not end-to-end circuit performance.

The harness uses a nonuniform complex state and a CPU amplitude oracle, checks
physical exchanges against a bit-permutation reference, then checks Hadamard
gates on local/global logical qubits against a CPU reference using the current
physical-to-logical mapping. It performs three allocate/check/benchmark/free
cycles per process. It warms up five exchanges, then times completed exchanges
including both streams' copy-back and reports maximum elapsed time across ranks.
`initialization_s` includes the complete allocation transaction; `allocation_s`
and `registration_s` separately report maximum local phase times (their maxima
need not occur on the same rank and need not sum to total initialization).
Host fixture construction and amplitude comparisons are outside steady-state
timing. This deliberately small harness keeps the full oracle on each rank.

Boundary and lifecycle commands (run for all modes and at 2, 4, and 64 ranks):

```sh
for mode in 0 1 2; do
  for cap in 1 2 6 512; do
    mpiexec -n 4 --map-by ppr:4:node --bind-to core \
      ./bin/nccl-buffer-"$mode"-graph-0 10 "$cap" 30
  done
done
```

Cap 1 exercises the single-slot path; 2 exercises the minimum pipeline; 6 gives
three elements per slot and a short final chunk for a 512-element region; 512
exercises an exact division. Low-bit exchanges also exercise one-element regions.
Run fresh MPI jobs repeatedly for complete setup/run/dispose cycles: MPI cannot
be initialized again after finalization within the same process. On the cluster,
also check rejection of mode 2 without PXN disabled, mixed mode binaries via the
MPI launcher's MPMD syntax, managed-memory requests, and allocation pressure on
one rank. Inject a registration failure on the second registration to exercise
rollback of the first handle, then retry allocation on the surviving communicator.
These failure cases require site-specific instrumentation and are not automated
by the harness.

## Verify actual NET use

The [NCCL 2.31.2 registration documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/usage/bufferreg.html)
explains allocator requirements and the PXN restriction; the
[communicator API reference](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/api/comms.html)
describes registration/deregistration and allocation/free APIs.

Inspect `REG` and `NET` output on every participating rank. Correlate registered
base addresses and lengths with state/scratch allocations and the NET transport
selected for the send/receive connection. Verify the SPCX plugin advertises and
uses registration of the application buffers, with the correct device/region,
for inter-node transfers. Compare mode 1 versus mode 2 logs and plugin counters.
A successful registration API return only records acceptance; an existing
`GDRDMA` connection message only describes transport capability. Neither proves
that these exchange payloads bypass internal staging. INFO messages and exact
wording depend on NCCL/plugin build. If they do not identify the buffers used
by the transfer, repeat a short diagnostic run with `NCCL_DEBUG=TRACE` and
`NCCL_DEBUG_SUBSYS=REG,NET`, enable the site's SPCX registration/transfer tracing,
or instrument the plugin's registration and send/receive buffer paths. Correlate
actual transfer pointers/offsets and memory handles with the registered regions.
Report registered NET use as unconfirmed if that evidence is unavailable. Disable
diagnostic logging for timing runs, identically across modes.

## Execution status

No tests, builds, validation commands, or benchmarks were run for this change,
as requested. All cluster checks and all performance measurements remain pending.
The provided harness is uncompiled and unrun. No performance gain is claimed.
