#pragma once
#include "gate_warp_mapping.hpp"

namespace qcs { namespace gate_warp {
__device__ inline uint64_t deposit(uint64_t value, const int* positions, int count)
{
    uint64_t result = 0;
    for (int b = 0; b < count; ++b) result |= ((value >> b) & 1) << positions[b];
    return result;
}
// Reuse specialized gate arithmetic. Each lane gathers one complete gate input,
// computes into private storage, and commits only the rows it owns.
template<class Gate, int R>
__global__ void execute(Gate gate, mapping m, qcs::complex_t* state)
{
    constexpr int N = 1 << Gate::num_target_qubits;
    constexpr int NR = 1 << R;
    uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t warp = thread >> 5;
    if (warp >= m.warps) return; // Uniform across the entire warp.
    unsigned lane = threadIdx.x & 31;
    uint64_t base = m.positive | deposit(lane, m.lane, 5)
                    | deposit(warp, m.remaining, m.remaining_count);
    qcs::complex_t owned[NR], input[N], output[N];
    const qcs::complex_t* in[N];
    qcs::complex_t* out[N];
    #pragma unroll
    for (int r = 0; r < NR; ++r)
        owned[r] = state[base | deposit(r, m.registers, R)];
    #pragma unroll
    for (int c = 0; c < N; ++c) {
        unsigned mask = 0;
        int reg = 0;
        #pragma unroll
        for (int j = 0; j < Gate::num_target_qubits; ++j) {
            if (m.target_lane[j] >= 0)
                mask |= (((lane >> m.target_lane[j]) ^ (c >> j)) & 1) << m.target_lane[j];
            else reg |= ((c >> j) & 1) << m.target_register[j];
        }
        // Select before shuffling: every source lane selects the same register.
        auto value = owned[reg];
        input[c] = qcs::complex_t(__shfl_xor_sync(0xffffffff, value.real(), mask),
                                  __shfl_xor_sync(0xffffffff, value.imag(), mask));
        output[c] = input[c]; // apply() can leave identity rows untouched.
        in[c] = &input[c]; out[c] = &output[c];
    }
    gate.apply(in, out);
    // All original global inputs have been loaded before any lane writes.
    __syncwarp();
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        int row = 0;
        #pragma unroll
        for (int j = 0; j < Gate::num_target_qubits; ++j)
            row |= ((m.target_lane[j] >= 0 ? (lane >> m.target_lane[j])
                                         : (r >> m.target_register[j])) & 1) << j;
        state[base | deposit(r, m.registers, R)] = output[row];
    }
}
}}
