# NCCL XOR exchange scheduling

The NCCL path of `exchange_qubits()` processes ascending nonzero submasks of
its exchanged rank-bit mask. Each phase pairs rank `r` with `r ^ delta`.
`src/exchange_schedule.hpp` supplies the host enumeration without allocating
mapping storage. NVSHMEM and the public API are unchanged.

For pair `(b[i], g[i])`, where `g[i]` is the global physical position minus the
number of local qubits, a rank's outgoing local region fixes bit `b[i]` to
its peer's bit `g[i]`. An ascending ordinal fills all other local bits above
the minimum exchanged position. Lower bits form the existing contiguous area.
At the peer, the same ordinal fixes exchanged local bits to the original rank's
global bits. This is exactly the simultaneous physical bit permutation.

The unsigned successor `(delta - global_mask) & global_mask`, starting at zero,
enumerates ascending submasks and terminates on zero. It avoids scanning gaps in
noncontiguous masks. Valid mappings contain distinct in-range local positions
and distinct in-range global positions; mappings and dimensions agree across
ranks. Pair ordering may be arbitrary but must describe the same mapping.

All non-self regions occur exactly once. For k pairs and n local qubits, each
rank sends and receives `(2^k - 1) * 2^(n-k)` complex amplitudes. Chunk sizes,
NCCL group counts, send/receive order, staging buffers, registration, and CUDA
stream/event dependencies are preserved. Slot lifetime spans all regions and
phases. No phase barrier is added. One-pair communicated-region order is unchanged.

Communicating ranks must use compatible chunk boundaries. Buffer modes 1/2
normalize staging capacity during allocation. Mode 0 retains its existing local
capacity policy: unequal effective capacities can cause message mismatches.
This change does not negotiate capacities or repair that existing limitation.

## Correctness validation

Run host coverage with:

```sh
make test-exchange-schedule
```

The test shares the production enumeration and independently computes bit
transpositions. It exhaustively covers 7,140 rank/mapping cases with 1–5 local
bits and 1–3 rank bits, including zero pairs and all ordered distinct pairings.
It checks old/new region coverage, uniqueness, reciprocal destinations, single-pair
order, nonuniform complex amplitudes, finite results, volume, chunk tails and group
counts with pipelining modeled both ways. Sparse high-bit and unsigned-wraparound
checks exercise the mask successor. This does not execute CUDA events or NCCL.

On a configured CUDA/NCCL/MPI installation, run both production-pipeline and private
non-pipelined harness variants:

```sh
make test-exchange RANKS=1
make test-exchange RANKS=2
make test-exchange RANKS=4
make test-exchange RANKS=8
make test-exchange RANKS=16
# Repeat in clean/rebuilt binaries for NCCL_BUFFER_MODE=0, 1 and 2.
```

The private variant changes only the pipeline constant in a generated source
copy. The GPU harness uses eight local bits, deterministic nonuniform complex
amplitudes, an independent transposition oracle, physical and logical comparisons,
permutation metadata checks, and explicit nonfinite rejection. It covers zero
through three pairs where rank count permits, contiguous and sparse mappings,
reordered pairs, capacities 1/2/6/10, repeated slot reuse, partial chunks, inverse
exchanges, consecutive exchanges, and an immediate consuming kernel on the owner
stream. Capacity overrides are identical across ranks, including mode 0. The
readback synchronizes only the owner stream to exercise the final copy-back join.
Use a launcher timeout when testing distributed liveness, and include cross-node
runs. These tests do not benchmark performance.

Validation in the implementation environment: host scheduling, SWAP launch, and
work-buffer tests pass. Production and GPU harness builds stop because `nvcc` is
unavailable; MPI launchers are also unavailable. Neither GPU correctness nor NCCL
deadlock freedom has been experimentally validated here.

Additional environment limitations: AddressSanitizer and UndefinedBehaviorSanitizer
host builds cannot link because their runtime libraries are missing. The existing
measurement-harness preparation script fails its expected first-stage launch
pattern check; the same failure was reproduced from an unmodified HEAD archive.
Its header-copy list now includes the scheduling helper, but the unrelated
instrumentation-pattern failure is left unchanged. The new non-pipelined exchange
source generator was executed successfully, and a whitespace-independent source
comparison confirmed that the NCCL transfer/copy-back body is unchanged.
