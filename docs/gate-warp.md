# Experimental local warp/register gate backend

Select `QCS_GATE_BACKEND=warp` before applying gates. The default (including
unrecognized values) is the original kernel. No performance improvement is
claimed: this environment has neither nvcc nor a CUDA GPU.

## Mapping and synchronization

`gate_warp_mapping.hpp` excludes positive and negative local controls from the
free dimensions, assigns the lowest five remaining physical positions to lane
bits, assigns higher target positions to registers, and assigns other positions
to warp IDs. Positive controls are inserted as ones; negative controls stay zero.
The three disjoint sets plus fixed controls partition the local index bits.
Thus each eligible amplitude has exactly one (warp, lane, register) owner.
A selected space with fewer than five free bits falls back entirely.

Target-to-lane and target-to-register tables retain the caller's target order.
Each lane gathers a full input vector with target zero as the basis LSB. For
column c it selects that column's register uniformly, then shuffles using the
lane target bits XOR the column target bits. Non-target lane bits never change.
Every live warp executes the same sequence of full-mask shuffles. Tail exits
are whole-warp exits. FP64 real and imaginary components are shuffled separately;
these are not single 32-bit hardware instructions. All owned inputs are loaded
before any writes, and a warp barrier separates calculation from stores.
No input or output group crosses a warp.

The adapter calls the existing specialized `GateType::apply()` on private
inputs/outputs. It initializes outputs because some structured gates leave
identity rows untouched. Only owned output rows are committed. This preserves
specialized arithmetic, but repeats arithmetic across lanes and uses dynamically
indexed small arrays. Register placement and spills must be measured, not assumed.
This is a correctness-oriented initial adapter, not a tuned Hopper implementation.

## Dispatch and limits

One through three targets are supported, with up to eight amplitudes owned per
thread. The existing rank-level control decision, preparation (including any
nonlocal-to-local target exchanges), stream, and completion path remain intact.
The new kernel only accesses the resulting local state. Four-target gates,
partial-warp spaces, measured input constraints, invalid mapping metadata, and
excessive grids use the original kernel. Diagonal and identity gates use the
original communication-free arithmetic, without shuffle exchanges. The API does
not gain a dense gate entry point; the private harness tests the generic adapter
with dense matrices. MPI/NCCL and measurement behavior are unchanged.

## Reproduce validation and timing

```
make test-gate-warp-mapping
make gate-warp-harness
./bin/gate_warp_harness 12 > gate-12.csv
./bin/gate_warp_harness 20 > gate-20.csv
./bin/gate_warp_harness 26 > gate-26.csv
compute-sanitizer --tool memcheck ./bin/gate_warp_harness 12
compute-sanitizer --tool racecheck ./bin/gate_warp_harness 12
compute-sanitizer --tool synccheck ./bin/gate_warp_harness 12
```

Configure the normal MPI/vendor include/link flags first (see [building](building.md)).
For GH200 set `GENCODE_FLAGS='-gencode arch=compute_90,code=sm_90'`.
The harness compares production original and warp kernels on identical inputs,
checks inactive controls, and compares dense gates against an independent CPU
matrix multiplication. It includes the five requested layouts, target order
reversal, mixed controls, one-target bit sweeps, two/three-target dense gates,
Hadamard, X, Z, and a small-space fallback. Dense matrices are nonsymmetric
complex unitaries with independent row/column phases. The host test adds seeded
random mapping cases; the CUDA harness also includes 20 seeded random three-target layouts. Full
simulator dispatch/MPI integration coverage remains future validation work.

Timing excludes allocation, metadata transfers, initialization and result copies.
Five warmups precede 31 event-timed launches per case/backend. CSV reports median,
p10 and p90 milliseconds. Repeated gates act in place; data are reset before each
backend. Event timing includes GPU idle gaps if host submission cannot keep up;
use larger statevectors and inspect launch overhead before drawing conclusions.
The same binary tests correctness before timing. Z is launched directly in the
harness to expose adapter overhead, although production diagonal dispatch falls
back. CSV reports first target and control counts; consult the deterministic
case sequence in the source for full layouts.

```
make -B gate-warp-harness NVCC='nvcc --forward-unknown-to-host-compiler' CFLAGS_VENDOR='YOUR_VENDOR_FLAGS -Xptxas=-v'
ncu --set full --kernel-name-base demangled ./bin/gate_warp_harness 20
nsys profile --trace=cuda -o gate-warp ./bin/gate_warp_harness 20
```

Inspect duration, DRAM throughput, L1/L2 transactions, registers/thread, local
loads/stores, occupancy, shuffle instructions and stall reasons. Metric names
vary by Nsight version; select supported metrics with `ncu --query-metrics`.
Keep compiler, GPU, clocks, state size and layout fixed for comparisons.

## Confirmed and outstanding

Confirmed here: host mapping test and eight existing host test targets pass;
`git diff --check` passes. `make gate-warp-harness` fails because nvcc is absent.
No CUDA compilation, device regression, sanitizer, generated-resource inspection,
or GPU timing was possible. This backend therefore remains opt-in and unvalidated
on device. No speedup or device correctness claim is made.

Comparisons B (simplified original ownership) and C (uncompressed control masking)
are not implemented. The current benchmark compares A (original) and D
(control-compressed hybrid). Future work should first validate on CUDA, then
measure spills and redundant apply arithmetic before adding output-row adapters,
diagonal specialization, coefficient caching, or additional indexing variants.

Additional local checks: `python3 -m unittest discover -s tests -p 'test_*.py'`
passes all six tests. An attempted host ASan/UBSan build using
`g++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined tests/gate_warp_mapping_test.cpp -o /tmp/gate_warp_mapping_sanitized`
cannot link because the installed compiler's `libasan.so.6.0.0` is missing.
Existing host checks were run with:

```
make test-work-buffer test-swap-launch test-pauli-string test-measurement-launch test-measurement-backend test-measurement-race test-exchange-schedule
```

All seven targets pass (eight host targets including the new mapping test).
