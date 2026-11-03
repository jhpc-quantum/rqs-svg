# NCCL exchange CUDA Graph experiments

`NCCL_GRAPH_MODE` is a build-time opt-in. The default is `0` (the existing
stream execution). Buffer allocation/explicit registration remains independently
selected by `NCCL_BUFFER_MODE`. Force a rebuild when changing either setting.

| Buffer mode | Graph mode | Experiment | GRAPH_REGISTER | LOCAL_REGISTER |
| --- | --- | --- | --- | --- |
| 0 | 0 | Original allocator and stream execution | 0 | 0 |
| 1 | 0 | Compatible allocation, unregistered stream control | 0 | 0 |
| 2 | 0 | Stream execution with explicit registration | 0 | 1 |
| 1 | 1 | Graph execution, registration disabled | 0 | 0 |
| 2 | 1 | Graph execution with explicit registration | 0 | 1 |
| 1 | 2 | Graph execution requesting graph-managed registration | 1 | 0 |

The last two columns name `NCCL_` environment variables. Export them before
process startup. Graph modes require their exact values, `NCCL_PXN_DISABLE=1`,
`NCCL_RUNTIME_CONNECT=0`, `NCCL_GRAPH_STREAM_ORDERING=1`, and
`NCCL_GRAPH_MIXING_SUPPORT=1` on every rank. These requirements keep connection
setup outside capture and retain NCCL ordering for mixed captured exchanges and
uncaptured measurement collectives. Managed memory and NVSHMEM are rejected.
Graph mode 1 requires NCCL headers/runtime >= 2.31.2; mode 2 is restricted to the
audited 2.31.2 headers/runtime. Graph mode 2 requires buffer mode 1, so the
application never calls `ncclCommRegister` for either participating allocation.
Mode 1 with buffer mode 0 is also available but is not an allocator-matched control.

Graph execution success is **not** proof of registered NET payloads. Graph mode
2 requests the supported upstream path; plugin or transport limitations can still
cause NCCL to use staging. It does not report registration as confirmed.

## Version-specific registration audit

The [NCCL 2.31.2 environment documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/env.html#nccl-graph-register)
still describes `NCCL_GRAPH_REGISTER` in terms of CollNet. However, the upstream
[`v2.31.2-1` P2P registration implementation](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/register/sendrecv_reg.cc)
contains `ncclRegisterP2pNetBuffer`: persistent (captured) plans with graph
registration enabled call `ncclNetGraphRegisterBuffer`. Local registration is a
separate fallback. NET device type `NCCL_NET_DEVICE_UNPACK` excludes both paths.

The [P2P planner](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/enqueue.cc)
(`addP2pToPlan`) attempts NET user buffers only for positive lengths, same-process
proxies, Simple protocol, eligible user-buffer tasks, and no PXN. It checks
`NCCL_DIRECT_NIC` and registration success for each selected channel. Small
messages selecting LL will therefore not exercise registered NET payloads.
For a registration-focused diagnostic/comparison, set
`NCCL_P2P_LL_THRESHOLD=0` identically in **all** modes. Keep a separate default
protocol comparison if that reflects the production workload.

The [NET transport](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/net.cc)
(`ncclNetGraphRegisterBuffer`, `netRegisterBuffer`, `sendProxyRegBuffer`, and
`recvProxyRegBuffer`) queries the allocation range, creates a graph registration,
and registers it through the connection's proxy/plugin. Failed registration can
leave the user-buffer flag unset rather than fail capture. Multiple physical
segments need the NCCL multi-segment gate and DMA-BUF support; the non-DMA-BUF
`regMr` fallback rejects multiple segments. The plugin must support the selected
CUDA memory registration (`regMr` or `regMrDmaBuf`), return a usable memory handle,
and use that handle with the application payload in send/receive operations.
Graph plan cleanup releases graph registrations after graph references are gone.
These are source-level findings, not measurements of this installation.

No SPCX source, installed plugin binary, or cluster diagnostics were supplied.
Its exact device type, CUDA/DMA-BUF registration support, vendor patches, and
actual payload path remain unknown. Record the loaded NCCL/plugin build IDs and
compare vendor source with the linked upstream tag. A local policy check cannot
establish plugin support through NCCL's public communicator API.

Use `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,REG,NET`, and a unique
`NCCL_DEBUG_FILE` per process/job. Correlate `NET register userbuff` and
`NET reuse buffer` addresses/lengths/handles with state and scratch allocations,
connections, channels, and inter-node operations. Log wording follows the cited
upstream source and may differ in vendor builds. If needed, instrument:

1. `ncclRegisterP2pNetBuffer` with persistent/graph/local parameter values and
   returned registration flag, pointer, length, and connection.
2. `addP2pToPlan` with protocol, PXN, direct-NIC, and final `netRegistered` per direction.
3. SPCX `regMr`/`regMrDmaBuf` and send/receive callbacks with payload pointers,
   memory handles, and selected NIC/connection. Correlate the actual payload
   range with the application allocation, rather than an internal staging buffer.

Collect this evidence for both send and receive, on every inter-node channel.
With graph mode 1/buffer mode 1, graph/local registration are both disabled;
NCCL still registers its internal transport buffers, which is expected. With
buffer mode 2/graph mode 1, graph registration is disabled so it cannot mask
explicit registration. With graph mode 2, local registration is disabled so it
cannot mask graph registration. An API success, graph capture, GDRDMA connection
message, or `NCCL_GRAPH_REGISTER=1` alone is insufficient evidence. If the plugin
excludes registration, report that limitation and call the result graph execution
with unconfirmed/unused graph registration. Never label it verified zero-copy.

## Capture, reuse, mapping, and teardown

The boundary is one complete `exchange_qubits_enqueue` invocation: all ordered
NCCL send/receive groups plus all device copy-back operations. Allocation,
registration, connection initialization, mapping determination, host mapping
updates, and CUDA host waits are outside capture. The communication stream is
the origin. A recorded transfer event forks the copy stream, slot-reuse waits
preserve overlap without overwriting a live receive slot, and the final copy-done
wait joins the copy stream back to the origin before ending capture. A single
slot uses the origin stream throughout. See the
[CUDA capture dependency rules](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

One cache entry is retained per simulator, with a conservative upper bound of
32,768 scheduled chunks (including self areas) per capture. Larger schedules
fail with a diagnostic; use larger slots or graph mode 0. This bounds application
capture work, though NCCL's internal node count depends on its configuration.
The key includes communicator, state/scratch/receive addresses and sizes, origin
and copy streams, rank/count, local state size, ordered local/global swap lists,
slot capacity, pipeline setting, and registration/execution modes. These derive
all peers, operation ordering, offsets, lengths, datatypes, and dependencies.
Communicator configuration is immutable for its lifetime; changing environment
settings after startup is unsupported. Events live for the whole simulator lifetime.

All ranks enter exchanges in the same host order, including zero-swap calls.
Each graph exchange coordinates counts and cache hits through MPI. If any rank
misses, all ranks retire their old graph and capture again. The rebuild checks
identical physical schedules and slot capacity, drains previous work before
capture, and completes instantiation on every rank before the first launch.
Steady replay includes the MPI coordination overhead; this is deliberately a
conservative implementation, so do not interpret its timing as bare graph-launch
latency. It is not intended for divergent rank-local exchange invocation.
Capture, instantiation, schedule, or launch failures terminate the MPI job with
rank diagnostics; there is no rank-local fallback that could strand peers in
NCCL. A driver/plugin hang inside NCCL still requires the site's job timeout.

Capture does not execute the exchange; the first graph launch does. Neither
capture nor replay mutates host permutation tables. `localize_target_qubits`
updates those tables exactly once after either enqueue or graph launch.
The captured schedule uses physical qubits, so resetting a logical mapping does
not by itself invalidate the device graph. A different resulting physical
schedule does invalidate it. Resetting mapping is bookkeeping, not a physical
state permutation: a harness must reload the corresponding physical state too.

On invalidation and free, outstanding origin/copy work is joined, the executable
is destroyed, then the source graph is destroyed. Only then are explicit handles
deregistered, state/scratch freed, and finally the communicator destroyed.
Every allocation lifetime starts with an empty cache, even if pointers are reused.
The same ordering applies to repeated allocate/free cycles. Graph setup failures
are fail-stop; allocation/explicit-registration failures before graph capture
retain the existing collective rollback behavior.

## Reproducible cluster checks and timings (not executed)

Build six separate harness binaries with the same site's `config.mk`:

```sh
for pair in '0 0' '1 0' '2 0' '1 1' '2 1' '1 2'; do
  set -- $pair
  make -B nccl-buffer-harness NVSHMEM=0 NCCL_BUFFER_MODE="$1" NCCL_GRAPH_MODE="$2"
done
```

For example, inside an allocation with one MPI rank/GPU, four GB200 GPUs/node:

```sh
export NCCL_PXN_DISABLE=1 NCCL_RUNTIME_CONNECT=0
export NCCL_GRAPH_STREAM_ORDERING=1 NCCL_GRAPH_MIXING_SUPPORT=1
export NCCL_NCHANNELS_PER_NET_PEER=8  # use the site's tuned value for every mode
export NCCL_P2P_LL_THRESHOLD=0       # registration-focused comparison
for pair in '0 0' '1 0' '2 0' '1 1' '2 1' '1 2'; do
  set -- $pair
  export NCCL_GRAPH_REGISTER=0 NCCL_LOCAL_REGISTER=0
  if [ "$1" = 2 ]; then export NCCL_LOCAL_REGISTER=1; fi
  if [ "$2" = 2 ]; then export NCCL_GRAPH_REGISTER=1; fi
  mpiexec -n 64 --map-by ppr:4:node --bind-to core \
    ./bin/nccl-buffer-"$1"-graph-"$2" 10 512 30
done
```

Ensure MPI exports settings to all nodes. Keep allocator, PXN, channel count,
application cap, NCCL network chunk sizes, protocol settings, rank placement,
plugin/NIC, and workload identical across the matched controls. Preserve metadata
and diagnostic logs per mode/job, then turn debug logging off identically for
timing runs. Production builds use the same two Make variables with
`make -B sharedlibrary target`. Keep the default protocol tests as a separate series.

The harness compares nonuniform complex amplitudes to a CPU oracle, exercises
one-element and tail chunks, changes global peers and local offsets, replays each
schedule seven times after its first exchange, changes slot capacity, resets
mapping/reloads the physical state, and runs local/global Hadamards twice.
Three allocation/check/timing/free cycles exercise pointer and registration
lifetimes. Run at 2, 4, and 64 ranks, caps 1/2/6/512, and repeat fresh MPI jobs to
exercise communicator lifecycle. To confirm inter-node registration, the selected
global peer must cross nodes; a two-rank single-node run cannot establish that.
Also exercise mismatched graph modes, missing environment controls, managed
memory, an oversized capture, and injected capture/instantiation failures on one
rank under a scheduler timeout. These failure injections need site instrumentation.

Output separates `initialization_s`, `allocation_s`, `registration_s`,
`graph_setup_s`, `capture_s`, `instantiate_s`, `first_launch_s`, and `exchange_s`. Graph setup
numbers describe the last rebuilt schedule, which is the fixed timed schedule
after warm-up. Graph setup covers schedule checks, cache retirement, and stream
drains. Capture includes host enqueue/planning; instantiation excludes
capture. First launch is measured separately through origin-stream completion
(including joined copy-back); the first-launch synchronization is enabled only
by the harness. `exchange_s` is completed steady work per iteration after five
warm-ups, including MPI cache coordination and copy-back. Every reported duration
uses the maximum across ranks. Graph build/replay counters are cumulative local
counts; non-graph graph-specific durations/counters are zero. Fixture generation,
CPU checking, invalidation, and initial drain waits are outside replay timing.
These small workloads are correctness fixtures, not a substitute for matched
production circuit measurements.

No compilation, tests, validation commands, sanitizer runs, cluster checks, or
benchmarks were run for this change, as requested. The new harness and graph
implementation are uncompiled and unrun. Registration selection on SPCX,
numerical correctness, failure handling, and performance remain unverified.
