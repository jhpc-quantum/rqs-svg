#pragma once

#include "pauli_string.hpp"
#include "measurement_reduce.cuh"

namespace qcs::pauli_detail
{
using state_type = measurement_detail::state_type;
constexpr unsigned block_size = 256;
constexpr uint64_t blocks_per_sm = 4;
static_assert(block_size == measurement_detail::block_size,
              "shared block reduction requires matching block sizes");

// Return the number of Stage 1 work items. In particular, reject inconsistent
// Y phases before selecting a diagonal path or evaluating __clzll(0).
inline uint64_t validate_descriptor(uint64_t local_dim, const PauliStringDescriptor& d)
{
    if (!local_dim || (local_dim & (local_dim - 1)))
        throw std::runtime_error("Pauli reducer: local dimension must be a power of two");
    if ((d.flip_mask | d.phase_mask) & ~(local_dim - 1))
        throw std::runtime_error("Pauli reducer: mask outside local state");
    unsigned ys = 0;
    for (uint64_t bits = d.flip_mask & d.phase_mask; bits; bits &= bits - 1) ++ys;
    if (d.y_count_mod4 != ys % 4 || !std::isfinite(d.coefficient))
        throw std::runtime_error("Pauli reducer: invalid phase or coefficient");
    return d.flip_mask ? local_dim / 2 : local_dim;
}

template<bool Paired>
__global__ void first_stage(const state_type* state, uint64_t local_dim,
                            PauliStringDescriptor d, double* partials)
{
    __shared__ double scratch[block_size / 32];
    uint64_t lower = 0;
    if constexpr (Paired)
        lower = (UINT64_C(1) << (63 - __clzll(d.flip_mask))) - 1;
    const uint64_t count = Paired ? local_dim / 2 : local_dim;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    double sum = 0;
    for (uint64_t k = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; k < count;)
    {
        uint64_t i = k;
        if constexpr (Paired) i = (k & lower) | ((k & ~lower) << 1);
        const state_type a = state[i];
        double value;
        if constexpr (Paired)
        {
            const state_type b = state[i ^ d.flip_mask];
            // conj(b)*a: only the component needed by i^nY is computed.
            // The phase selection is uniform across the entire launch.
            if (d.y_count_mod4 & 1)
                value = b.imag() * a.real() - b.real() * a.imag();
            else
                value = b.real() * a.real() + b.imag() * a.imag();
            if (d.y_count_mod4 & 2) value = -value;
            value *= 2;
        }
        else value = a.real() * a.real() + a.imag() * a.imag();
        sum += (__popcll(i & d.phase_mask) & 1) ? -value : value;
        // Avoid wraparound for the largest representable local dimensions.
        if (count - k <= stride) break;
        k += stride;
    }
    double total = measurement_detail::block_sum(sum, scratch);
    if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

__global__ void second_stage(const double* partials, unsigned partial_count, double* output)
{
    __shared__ double scratch[block_size / 32];
    double sum = 0;
    for (uint64_t k = threadIdx.x; k < partial_count; k += blockDim.x) sum += partials[k];
    double total = measurement_detail::block_sum(sum, scratch);
    if (threadIdx.x == 0) *output = total;
}

struct launch_config
{
    unsigned blocks;
};

inline launch_config resources(int device, uint64_t work_items)
{
    cudaDeviceProp prop{};
    ATLC_CHECK_CUDA(cudaGetDeviceProperties, &prop, device);
    uint64_t blocks = measurement_detail::checked_product(blocks_per_sm,
        prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 0);
    uint64_t needed = work_items / block_size + (work_items % block_size != 0);
    blocks = std::min(blocks, needed);
    if (!blocks || blocks > static_cast<uint64_t>(prop.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned>::max() || prop.warpSize != 32 ||
        block_size > static_cast<unsigned>(prop.maxThreadsPerBlock) ||
        block_size > static_cast<unsigned>(prop.maxThreadsDim[0]))
        throw std::runtime_error("Pauli reducer: unsupported launch geometry");
    for (const void* kernel : {reinterpret_cast<const void*>(first_stage<false>),
                               reinterpret_cast<const void*>(first_stage<true>),
                               reinterpret_cast<const void*>(second_stage)})
    {
        cudaFuncAttributes attr{};
        ATLC_CHECK_CUDA(cudaFuncGetAttributes, &attr, kernel);
        int residency = 0;
        ATLC_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor, &residency, kernel, block_size, 0);
        if (attr.maxThreadsPerBlock < static_cast<int>(block_size) ||
            attr.sharedSizeBytes > prop.sharedMemPerBlock || !residency)
            throw std::runtime_error("Pauli reducer: kernel resource limits exceeded");
    }
    return {static_cast<unsigned>(blocks)};
}

// Internal launch after validation and workspace/geometry checks.
inline void launch(const state_type* state, uint64_t local_dim,
                   const PauliStringDescriptor& d, unsigned blocks,
                   double* partials, double* output, cudaStream_t stream)
{
    if (d.flip_mask)
        first_stage<true><<<blocks, block_size, 0, stream>>>(state, local_dim, d, partials);
    else
        first_stage<false><<<blocks, block_size, 0, stream>>>(state, local_dim, d, partials);
    ATLC_CHECK_CUDA(cudaGetLastError);
    second_stage<<<1, block_size, 0, stream>>>(partials, blocks, output);
    ATLC_CHECK_CUDA(cudaGetLastError);
}

// Non-owning adapter over the measurement slice. The slice and its stream must
// outlive this adapter. No device pointer is cached, so close/growth of the
// standalone measurement workspace is checked on every borrow. Serialized host
// calls are required, as for measurement; all device uses share the owner stream.
class workspace
{
    measurement_detail::workspace& storage_;
    unsigned max_blocks_;

public:
    workspace(measurement_detail::workspace& storage, cudaStream_t stream)
        : storage_(storage), max_blocks_(0)
    {
        storage_.borrow_scalar(1, stream); // Check owner and existing allocation.
        max_blocks_ = resources(storage_.device(), UINT64_MAX).blocks;
        storage_.borrow_scalar(max_blocks_, stream); // No allocation or growth.
    }
    workspace(const workspace&) = delete;
    workspace& operator=(const workspace&) = delete;

    double* enqueue(const state_type* state, uint64_t local_dim,
                    const PauliStringDescriptor& d, cudaStream_t stream)
    {
        if (!state) throw std::runtime_error("Pauli reducer: missing state");
        uint64_t count = validate_descriptor(local_dim, d);
        unsigned blocks = static_cast<unsigned>(std::min<uint64_t>(max_blocks_,
            count / block_size + (count % block_size != 0)));
        auto borrowed = storage_.borrow_scalar(blocks, stream);
        launch(state, local_dim, d, blocks, borrowed.partials, borrowed.output, stream);
        return borrowed.output;
    }
};

// Borrowed storage only: caller provides capacity >= resources(...).blocks+1
// doubles on the current device. Storage/state must not overlap and must remain
// alive until the stream completes. Calls sharing storage must be serialized on
// that stream. Every used element is overwritten; no allocation or sync occurs.
// Return a fixed output slot, valid until reuse. The result excludes coefficient;
// the simulator applies it once after NCCL AllReduce.
inline double* enqueue(const state_type* state, uint64_t local_dim,
                       const PauliStringDescriptor& d, double* workspace,
                       uint64_t capacity, cudaStream_t stream)
{
    if (!state || !workspace) throw std::runtime_error("Pauli reducer: missing state/workspace");
    uint64_t work_items = validate_descriptor(local_dim, d);
    int device;
    ATLC_CHECK_CUDA(cudaGetDevice, &device);
    auto config = resources(device, work_items);
    if (capacity <= config.blocks)
        throw std::runtime_error("Pauli reducer: workspace capacity exceeded");
    double* output = workspace + capacity - 1;
    launch(state, local_dim, d, config.blocks, workspace, output, stream);
    return output;
}
} // namespace qcs::pauli_detail
