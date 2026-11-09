#pragma once

// Internal adaptation of naoto-aoki-fy/cuda-reduce, commit
// 15bb3cd08e72a423e2be7d517003567f681b4f26, src/global_reduction.cu:
// warp_sum, custom block_sum, shared_first_stage_body, second_stage_body,
// and reduction_resources. Upstream supplies no license file or source notices
// at that revision. Only custom + shared accumulators + bin-major is used here.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include <cuda/std/complex>
#include <atlc/check_cuda.hpp>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "work_buffer.hpp"
#include "measurement_launch.hpp"

namespace qcs::measurement_detail
{
using real_type = double;
using state_type = cuda::std::complex<real_type>;
constexpr unsigned block_size = measurement_block_size;
constexpr uint64_t blocks_per_sm = 8;
static_assert(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0,
              "measurement reduction requires complete warps");

inline uint64_t checked_product(uint64_t a, uint64_t b)
{
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::runtime_error("measurement reducer: size product overflow");
    return a * b;
}

inline size_t allocation_bytes(uint64_t elements, size_t element_size)
{
    uint64_t bytes = checked_product(elements, element_size);
    if (bytes > std::numeric_limits<size_t>::max())
        throw std::runtime_error("measurement reducer: allocation exceeds size_t");
    return static_cast<size_t>(bytes);
}

template<int N>
struct descriptor
{
    static_assert(N >= 1 && N <= 4);
    int local_targets = 0;
    std::array<int, N> local_target_physical{};
    std::array<int, N> local_outcome_bits{};
    uint64_t fixed_outcome = 0;
    // Include ALL previously measured local controls, even outside this group.
    uint64_t control_mask = 0;
    uint64_t control_value = 0;
};

// Kernel argument passed by value; no asynchronous metadata-copy lifetime.
template<int N>
struct packed_descriptor
{
    int local_targets = 0;
    int target_physical[N]{};
    int outcome_bits[N]{};
    uint64_t fixed_outcome = 0;
    uint64_t control_value = 0;
    unsigned control_count = 0;
    unsigned control_positions[63]{};
    uint64_t active_count = 0;

    __host__ __device__ uint64_t physical_index(uint64_t packed) const
    {
        // Ascending insertions preserve the order of the non-control bits.
        for (unsigned j = 0; j < control_count; ++j)
        {
            unsigned p = control_positions[j];
            uint64_t low = (UINT64_C(1) << p) - 1;
            packed = (packed & low) | ((packed & ~low) << 1)
                   | (control_value & (UINT64_C(1) << p));
        }
        return packed;
    }
};

template<int N>
packed_descriptor<N> pack(const descriptor<N>& d, uint64_t num_states_local)
{
    if (!num_states_local || (num_states_local & (num_states_local - 1)))
        throw std::runtime_error("measurement reducer: state count must be a power of two");
    // Also bounds every valid physical index and prevents unaddressable states.
    allocation_bytes(num_states_local, sizeof(state_type));
    unsigned qubits = 0;
    for (uint64_t v = num_states_local; v > 1; v >>= 1) ++qubits;
    if (d.local_targets < 0 || d.local_targets > N ||
        d.fixed_outcome >= (UINT64_C(1) << N) ||
        (d.control_mask & ~(num_states_local - 1)) ||
        (d.control_value & ~d.control_mask))
        throw std::runtime_error("measurement reducer: invalid descriptor bounds");
    packed_descriptor<N> result{};
    result.local_targets = d.local_targets;
    result.fixed_outcome = d.fixed_outcome;
    result.control_value = d.control_value;
    uint64_t targets = 0, outputs = 0;
    for (int j = 0; j < d.local_targets; ++j)
    {
        int p = d.local_target_physical[j], o = d.local_outcome_bits[j];
        if (p < 0 || static_cast<unsigned>(p) >= qubits || o < 0 || o >= N)
            throw std::runtime_error("measurement reducer: target/output shift out of range");
        uint64_t target = UINT64_C(1) << p, output = UINT64_C(1) << o;
        if ((target & (targets | d.control_mask)) || (output & outputs) ||
            (output & d.fixed_outcome))
            throw std::runtime_error("measurement reducer: overlapping targets, controls or output bits");
        targets |= target;
        outputs |= output;
        result.target_physical[j] = p;
        result.outcome_bits[j] = o;
    }
    for (unsigned p = 0; p < qubits; ++p)
        if (d.control_mask & (UINT64_C(1) << p))
            result.control_positions[result.control_count++] = p;
    result.active_count = num_states_local >> result.control_count;
    return result;
}

__device__ inline double warp_sum(double value)
{
    for (unsigned delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

__device__ inline double block_sum(double value, double* scratch)
{
    unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    value = warp_sum(value);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    double total = 0;
    if (warp == 0)
    {
        total = lane < block_size / 32 ? scratch[lane] : 0.0;
        total = warp_sum(total);
    }
    __syncthreads(); // Finish all readers before the next bin reuses scratch.
    return total; // Only thread zero consumes this result.
}

// Compact=false retains the full-bin reference path for exact regression checks.
template<int N, bool Compact = false>
__global__ void first_stage(const state_type* state, packed_descriptor<N> d, double* partials)
{
    extern __shared__ double sums[];
    __shared__ double scratch[block_size / 32];
    const uint64_t bins = UINT64_C(1) << (Compact ? d.local_targets : N);
    const uint64_t t = threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * block_size;
    for (uint64_t bin = 0; bin < bins; ++bin)
        sums[bin * block_size + t] = 0;
    for (uint64_t packed = uint64_t(blockIdx.x) * block_size + t;
         packed < d.active_count;)
    {
        const uint64_t i = d.physical_index(packed);
        const state_type amplitude = state[i];
        uint64_t outcome = Compact ? 0 : d.fixed_outcome;
        for (int j = 0; j < d.local_targets; ++j)
            outcome |= ((i >> d.target_physical[j]) & UINT64_C(1)) << (Compact ? j : d.outcome_bits[j]);
        double re = amplitude.real(), im = amplitude.imag();
        sums[outcome * block_size + t] += re * re + im * im;
        if (d.active_count - packed <= stride) break;
        packed += stride;
    }
    // With one block, write the full external layout directly. Only thread
    // zero writes output, so zero initialization cannot race with bin stores.
    if constexpr (Compact)
        if (gridDim.x == 1 && t == 0)
            for (uint64_t bin = 0; bin < (UINT64_C(1) << N); ++bin)
                partials[bin] = 0.0;
    for (uint64_t bin = 0; bin < bins; ++bin)
    {
        uint64_t output_bin = bin;
        if constexpr (Compact)
            if (gridDim.x == 1)
            {
                output_bin = d.fixed_outcome;
                for (int j = 0; j < d.local_targets; ++j)
                    output_bin |= ((bin >> j) & 1) << d.outcome_bits[j];
            }
        double total = block_sum(sums[bin * block_size + t], scratch);
        if (t == 0) partials[output_bin * uint64_t(gridDim.x) + blockIdx.x] = total;
    }
}

template<int N, bool Compact = false>
__global__ void second_stage(const double* partials, double* output, uint64_t partial_blocks,
                             packed_descriptor<N> d = {})
{
    __shared__ double scratch[block_size / 32];
    const uint64_t blocks = partial_blocks; // First-stage bin-major stride.
    for (uint64_t bin = blockIdx.x; bin < (UINT64_C(1) << N); bin += gridDim.x)
    {
        uint64_t local_bin = bin;
        if constexpr (Compact)
        {
            uint64_t variable_mask = 0;
            local_bin = 0;
            for (int j = 0; j < d.local_targets; ++j)
            {
                variable_mask |= UINT64_C(1) << d.outcome_bits[j];
                local_bin |= ((bin >> d.outcome_bits[j]) & 1) << j;
            }
            // Uniform across the block: no thread skips a shared-memory barrier.
            if ((bin & ~variable_mask) != d.fixed_outcome)
            {
                if (threadIdx.x == 0) output[bin] = 0.0;
                continue;
            }
        }
        double value = 0;
        for (uint64_t block = threadIdx.x; block < blocks; block += block_size)
            value += partials[local_bin * blocks + block];
        double total = block_sum(value, scratch);
        if (threadIdx.x == 0) output[bin] = total;
    }
}

// The sampler always uses one complete warp, independently of reducer geometry.
// Keep original weights intact: eligibility is based on weights, not prefixes.
enum class sample_status : uint32_t
{
    success = 0,
    invalid_weight = 1,
    invalid_total = 2,
};

struct sample_result
{
    uint32_t outcome;
    sample_status status;
};
static_assert(sizeof(sample_result) == sizeof(double));

struct rounded_add
{
    __device__ double operator()(double a, double b) const
    {
        return __dadd_rn(a, b);
    }
};

// Internal diagnostics are compiled out of the production specialization.
struct sample_diagnostics
{
    double cumulative[16];
    uint4 raw;
    double uniform;
};

__device__ inline uint4 measurement_random(uint64_t seed, uint64_t counter)
{
    curandStatePhilox4_32_10_t rng;
    curand_init(static_cast<unsigned long long>(seed),
                static_cast<unsigned long long>(counter), 0ULL, &rng);
    return curand4(&rng);
}

template<int N, bool Inspect = false>
__global__ void sample_outcome(const double* weights, uint64_t seed,
                               uint64_t counter, sample_result* result,
                               sample_diagnostics* diagnostics = nullptr,
                               double controlled_uniform = -1.0)
{
    static_assert(N >= 1 && N <= 4);
    constexpr unsigned bins = 1u << N;
    using scan = cub::BlockScan<double, 32, cub::BLOCK_SCAN_RAKING>;
    __shared__ typename scan::TempStorage scratch;
    __shared__ double cumulative[32];
    unsigned lane = threadIdx.x;
    double weight = lane < bins ? weights[lane] : 0.0;
    double prefix;
    scan(scratch).InclusiveScan(weight, prefix, rounded_add{});
    cumulative[lane] = prefix;
    __syncthreads();
    if constexpr (Inspect)
        if (lane < bins) diagnostics->cumulative[lane] = prefix;
    if (lane != 0) return;
    if constexpr (Inspect)
    {
        diagnostics->raw = measurement_random(seed, counter);
        uint64_t m = (uint64_t(diagnostics->raw.x) << 21) | (uint64_t(diagnostics->raw.y) >> 11);
        diagnostics->uniform = static_cast<double>(m) * 0x1.0p-53;
    }

    *result = {0, sample_status::invalid_weight};
    for (unsigned i = 0; i < bins; ++i)
        if (!isfinite(weights[i]) || !(weights[i] >= 0.0)) return;
    double total = cumulative[bins - 1];
    result->status = sample_status::invalid_total;
    if (!isfinite(total) || !(total > 0.0)) return;

    // No rank, geometry or probability data enters this mapping. One Philox
    // subsequence per logical measurement; fixed offset and fixed raw outputs.
    uint4 bits = measurement_random(seed, counter);
    uint64_t mantissa = (uint64_t(bits.x) << 21) | (uint64_t(bits.y) >> 11);
    double u = static_cast<double>(mantissa) * 0x1.0p-53; // Exactly [0,1).
    if constexpr (Inspect)
        if (controlled_uniform >= 0.0) u = controlled_uniform;
    double threshold = __dmul_rn(u, total);
    if (!(threshold < total)) threshold = nextafter(total, 0.0);

    unsigned last_positive = 0;
    for (unsigned i = 0; i < bins; ++i)
    {
        if (!(weights[i] > 0.0)) continue;
        last_positive = i;
        if (cumulative[i] > threshold)
        {
            *result = {i, sample_status::success};
            return;
        }
    }
    // Endpoint rounding must never choose a zero-weight trailing bin.
    *result = {last_positive, sample_status::success};
}

struct launch_config
{
    unsigned blocks = 0;
    size_t dynamic_shared = 0;
};

template<int N>
launch_config resources(int device)
{
    cudaDeviceProp prop{};
    ATLC_CHECK_CUDA(cudaGetDeviceProperties, &prop, device);
    uint64_t blocks = checked_product(blocks_per_sm,
                                     prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 0);
    if (!blocks || blocks > static_cast<uint64_t>(prop.maxGridSize[0]) ||
        blocks > std::numeric_limits<unsigned>::max() || prop.warpSize != 32 ||
        block_size > static_cast<unsigned>(prop.maxThreadsPerBlock) ||
        block_size > static_cast<unsigned>(prop.maxThreadsDim[0]))
        throw std::runtime_error("measurement reducer: unsupported grid/block/warp limits");
    const void* first = reinterpret_cast<const void*>(first_stage<N, true>);
    const void* second = reinterpret_cast<const void*>(second_stage<N, true>);
    cudaFuncAttributes fa{}, sa{};
    ATLC_CHECK_CUDA(cudaFuncGetAttributes, &fa, first);
    ATLC_CHECK_CUDA(cudaFuncGetAttributes, &sa, second);
    size_t dynamic = allocation_bytes(checked_product(block_size, UINT64_C(1) << N), sizeof(double));
    // Static scratch is measured from the exact compiled kernels, not estimated.
    uint64_t total = fa.sharedSizeBytes;
    if (total > std::numeric_limits<uint64_t>::max() - dynamic)
        throw std::runtime_error("measurement reducer: shared-memory size overflow");
    total += dynamic;
    if (total > prop.sharedMemPerBlock && total <= prop.sharedMemPerBlockOptin &&
        dynamic <= static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        ATLC_CHECK_CUDA(cudaFuncSetAttribute, first,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic));
        ATLC_CHECK_CUDA(cudaFuncGetAttributes, &fa, first);
    }
    if (total > (prop.sharedMemPerBlockOptin > prop.sharedMemPerBlock
                     ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock) ||
        fa.maxDynamicSharedSizeBytes < 0 || dynamic > static_cast<size_t>(fa.maxDynamicSharedSizeBytes) ||
        sa.sharedSizeBytes > prop.sharedMemPerBlock ||
        fa.maxThreadsPerBlock < static_cast<int>(block_size) ||
        sa.maxThreadsPerBlock < static_cast<int>(block_size))
        throw std::runtime_error("measurement reducer: kernel shared-memory/thread limits exceeded (dynamic="
                                 + std::to_string(dynamic) + ", static=" + std::to_string(fa.sharedSizeBytes) + ")");
    int first_residency = 0, second_residency = 0;
    ATLC_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor, &first_residency, first, block_size, dynamic);
    ATLC_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor, &second_residency, second, block_size, 0);
    if (!first_residency || !second_residency)
        throw std::runtime_error("measurement reducer: zero kernel residency");
    return {static_cast<unsigned>(blocks), dynamic};
}

// Single owner, bound to the construction device and caller-owned stream.
// Keep that stream alive through close/destruction. Calls must be serialized by
// the owner. Result storage is valid until the next reserve, enqueue,
// enqueue_zero, borrow_scalar or close, and may be consumed in place on the owner stream.
class workspace
{
    int device_ = -1;
    cudaStream_t stream_;
    double* data_ = nullptr;
    uint64_t capacity_ = 0;
    bool owns_data_ = true;
    std::array<launch_config, 4> configs_{};

    void require_owner(cudaStream_t stream) const
    {
        int device;
        ATLC_CHECK_CUDA(cudaGetDevice, &device);
        if (device != device_ || stream != stream_)
            throw std::runtime_error("measurement reducer: workspace device/stream mismatch");
    }

public:
    struct scalar_storage
    {
        double* partials;
        double* output;
    };

    // Share the existing registered slice with scalar reductions on the owner
    // stream. Borrowing never grows storage. All earlier consumers must already
    // be queued on that stream; the next workspace operation invalidates results.
    scalar_storage borrow_scalar(uint64_t partial_count, cudaStream_t stream)
    {
        require_owner(stream);
        if (!data_ || !partial_count || partial_count >= capacity_)
            throw std::runtime_error("measurement reducer: scalar workspace capacity exceeded or closed");
        return {data_, data_ + capacity_ - 1};
    }

    explicit workspace(cudaStream_t stream) : stream_(stream)
    {
        ATLC_CHECK_CUDA(cudaGetDevice, &device_);
    }
    workspace(cudaStream_t stream, qcs::buffer_arena& arena) : workspace(stream)
    {
        // Reserve the largest supported reduction before swap storage is cut
        // out, so later measurements never allocate or compete with exchange.
        configs_ = {resources<1>(device_), resources<2>(device_),
                    resources<3>(device_), resources<4>(device_)};
        for (unsigned n = 1; n <= configs_.size(); ++n)
            capacity_ = std::max(capacity_, checked_product(UINT64_C(1) << n,
                                                          uint64_t(configs_[n - 1].blocks) + 1));
        data_ = static_cast<double*>(arena.take(allocation_bytes(capacity_, sizeof(double))));
        owns_data_ = false;
    }
    workspace(const workspace&) = delete;
    workspace& operator=(const workspace&) = delete;
    ~workspace() noexcept
    {
        // close() is the explicit, error-reporting teardown. Destruction queues
        // release on the owner stream without throwing or synchronizing.
        if (!data_ || !owns_data_) return;
        int previous;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device_ && cudaSetDevice(device_) != cudaSuccess) return;
        cudaFreeAsync(data_, stream_);
        if (previous != device_) cudaSetDevice(previous);
    }
    void close()
    {
        if (!data_) return;
        if (!owns_data_) { data_ = nullptr; capacity_ = 0; return; }
        int previous;
        ATLC_CHECK_CUDA(cudaGetDevice, &previous);
        if (previous != device_) ATLC_CHECK_CUDA(cudaSetDevice, device_);
        cudaError_t status = cudaFreeAsync(data_, stream_);
        if (status == cudaSuccess) { data_ = nullptr; capacity_ = 0; }
        cudaError_t restored = previous != device_ ? cudaSetDevice(previous) : cudaSuccess;
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("measurement reducer: release: ") + cudaGetErrorString(status));
        if (restored != cudaSuccess)
            throw std::runtime_error(std::string("measurement reducer: restore device: ") + cudaGetErrorString(restored));
    }

    template<int N>
    launch_config reserve(cudaStream_t stream)
    {
        static_assert(N >= 1 && N <= 4);
        require_owner(stream);
        auto& config = configs_[N - 1];
        if (!config.blocks) config = resources<N>(device_);
        uint64_t required = checked_product(UINT64_C(1) << N, uint64_t(config.blocks) + 1);
        if (required > capacity_)
        {
            if (!owns_data_)
                throw std::runtime_error("measurement reducer: external workspace capacity exceeded");
            double* next = nullptr;
            ATLC_CHECK_CUDA(cudaMallocAsync, &next, allocation_bytes(required, sizeof(double)), stream_);
            if (data_)
            {
                cudaError_t status = cudaFreeAsync(data_, stream_);
                if (status != cudaSuccess)
                {
                    cudaFreeAsync(next, stream_);
                    throw std::runtime_error(std::string("measurement reducer: grow release: ") + cudaGetErrorString(status));
                }
            }
            data_ = next;
            capacity_ = required;
        }
        return config;
    }

    template<int N>
    double* enqueue_zero(cudaStream_t stream)
    {
        auto config = reserve<N>(stream);
        uint64_t partial_elements = checked_product(UINT64_C(1) << N, config.blocks);
        double* output = data_ + partial_elements;
        ATLC_CHECK_CUDA(cudaMemsetAsync, output, 0,
                        allocation_bytes(UINT64_C(1) << N, sizeof(double)), stream);
        return output;
    }

    template<int N>
    double* enqueue(const state_type* state, uint64_t num_states_local,
                    const descriptor<N>& d, cudaStream_t stream)
    {
        if (!state) throw std::runtime_error("measurement reducer: null state");
        auto packed = pack(d, num_states_local);
        auto config = reserve<N>(stream);
        uint64_t partial_elements = checked_product(UINT64_C(1) << N, config.blocks);
        double* output = data_ + partial_elements;
        // Capacity/output offset stay at the maximum; only live partials are packed.
        const unsigned blocks = measurement_blocks(packed.active_count, config.blocks);
        // Only fresh local outcomes need accumulators. Keep traversal, launch
        // geometry and reduction trees unchanged, including the full-local case.
        const size_t shared = allocation_bytes(
            checked_product(block_size, UINT64_C(1) << packed.local_targets), sizeof(double));
        first_stage<N, true><<<blocks, block_size, shared, stream>>>(
            state, packed, blocks == 1 ? output : data_);
        ATLC_CHECK_CUDA(cudaGetLastError);
        if (blocks != 1)
        {
            second_stage<N, true><<<(1u << N), block_size, 0, stream>>>(data_, output, blocks, packed);
            ATLC_CHECK_CUDA(cudaGetLastError);
        }
        return output;
    }

    template<int N>
    sample_result* enqueue_sample(const double* weights, uint64_t seed,
                                  uint64_t counter, cudaStream_t stream)
    {
        static_assert(N >= 1 && N <= 4);
        require_owner(stream);
        if (!data_ || !weights)
            throw std::runtime_error("measurement sampler: missing weights/workspace");
        // Reduction partials are dead after NCCL. Reuse their aligned storage
        // for the compact result, disjoint from the live output weights.
        auto* result = reinterpret_cast<sample_result*>(data_);
        sample_outcome<N><<<1, 32, 0, stream>>>(weights, seed, counter, result);
        ATLC_CHECK_CUDA(cudaGetLastError);
        return result;
    }

    // Internal diagnostics used by the focused poison/reuse test.
    double* data() const { return data_; }
    uint64_t capacity() const { return capacity_; }
    int device() const { return device_; }
};
} // namespace qcs::measurement_detail
