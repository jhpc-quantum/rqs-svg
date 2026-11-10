# Hadamard launch experiment

Select a launch mode at runtime:

```sh
QCS_H_LAUNCH_MODE=baseline ./benchmark
QCS_H_LAUNCH_MODE=A ./benchmark
QCS_H_LAUNCH_MODE=B ./benchmark
QCS_H_LAUNCH_MODE=grid-stride ./benchmark
```

The unset/empty default is baseline. Baseline uses the original generic kernel
with up to 512 threads per block; A and B use the same full-grid kernel with
up to 256 and 1024 threads, respectively. Small full-grid launches reduce the
block size to the logical work count. This checkout did not previously contain
the launch-mode selector; the original default launch is retained.

The experiment applies only to single-rank, local, uncontrolled Hadamard gates.
Controlled Hadamard gates, distributed execution, and other gates retain their
original launch behavior. Invalid mode values are rejected on eligible gates.

Grid-stride uses the generic gate's existing indexing, amplitude loads/stores,
and arithmetic through a shared device function. The work count is
`1ULL << (num_qubits_local - num_operand_qubits)`, using prepared operands.
The launch uses the lesser of the blocks needed for this work and the actual
CUDA device SM count times the compile-time multiplier. Threads loop over
64-bit logical ordinals with a stride of `blockDim.x * gridDim.x`. The loop
bounds cover partial final iterations and launches larger than the work count.
Launches use the existing stream and do not synchronize it.

Edit these constants in `src/hadamard_launch.hpp`, then rebuild:

```cpp
constexpr int kHadamardGridStrideBlockSize = 256;
constexpr int kHadamardGridStrideBlocksPerSM = 8;
```

No runtime geometry overrides are provided. Each grid-stride launch reports its
mode, constants, actual block count, GPU SM count, and logical work count to
stderr. Redirect stderr when collecting timings; account for host diagnostics
when interpreting end-to-end measurements.

## Validation

```sh
make test-hadamard-launch          # host geometry and exact-once coverage
make test-hadamard-grid-stride     # requires CUDA, MPI, NCCL, atlc, one GPU/rank
```

The GPU regression compares every amplitude against baseline for 1, 2, 5, 10,
and 20 qubits, low/high targets, H, X, and controlled H. The largest case
requires multiple iterations on GPUs with fewer than 256 SMs at the default
geometry. Host tests explicitly force multiple iterations and uneven tails
using a simulated two-SM geometry, and check counts above 32 bits.

For both 256 and 512, edit the block-size constant, rebuild both tests, and run
them. Restore 256 afterward. The library Makefile tracks the constants header.
On systems with 256 or more SMs, increase the GPU test's largest qubit count
to force multiple iterations (subject to available statevector memory).

In the implementation environment, host tests passed with separate builds for
256 and 512 threads. CUDA compilation, GPU amplitude comparisons for either
block size, and performance measurements were not executed: neither nvcc nor
nvidia-smi was available. GPU correctness and speedup remain unverified.
