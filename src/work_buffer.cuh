#pragma once

#include "work_buffer.hpp"
#include <cuda_runtime.h>
#include <atlc/check_cuda.hpp>
#ifdef QCS_ENABLE_NVSHMEM
#include <nvshmem.h>
#endif

namespace qcs
{
// Owns scratch storage. Ordinary slices share 95% of the free CUDA memory
// sampled after state allocation. NVSHMEM destinations require a separate
// symmetric backing allocation, owned here and carved with the same interface.
class work_buffer
{
    void* data_ = nullptr;
    cudaStream_t stream_{};
#ifdef QCS_ENABLE_NVSHMEM
    void* symmetric_data_ = nullptr;
#endif

public:
    buffer_arena device;
    buffer_arena symmetric;
    work_buffer() = default;
    work_buffer(const work_buffer&) = delete;
    work_buffer& operator=(const work_buffer&) = delete;

    void allocate_remaining(cudaStream_t stream)
    {
        if (data_) throw std::runtime_error("work buffer: already allocated");
        stream_ = stream;
        // Account for the committed state allocation, including async pools.
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        size_t free_bytes = 0, total_bytes = 0;
        ATLC_CHECK_CUDA(cudaMemGetInfo, &free_bytes, &total_bytes);
        size_t bytes = buffer_arena::budget(free_bytes);
        if (!bytes) throw std::runtime_error("work buffer: no free device memory");
        ATLC_CHECK_CUDA(cudaMalloc, &data_, bytes);
        device = buffer_arena(data_, bytes);
    }

#ifdef QCS_ENABLE_NVSHMEM
    void allocate_symmetric(size_t bytes)
    {
        if (symmetric_data_) throw std::runtime_error("work buffer: symmetric storage already allocated");
        symmetric_data_ = nvshmem_align(256, bytes);
        if (!symmetric_data_)
            throw std::runtime_error("work buffer: NVSHMEM allocation failed; increase NVSHMEM_SYMMETRIC_SIZE");
        symmetric = buffer_arena(symmetric_data_, bytes);
    }
#endif

    // The caller joins exchange streams and PEs before releasing any slice.
    void close()
    {
#ifdef QCS_ENABLE_NVSHMEM
        if (symmetric_data_) { nvshmem_free(symmetric_data_); symmetric_data_ = nullptr; }
#endif
        symmetric = buffer_arena();
        if (data_)
        {
            ATLC_CHECK_CUDA(cudaFreeAsync, data_, stream_);
            data_ = nullptr;
        }
        device = buffer_arena();
    }
};
} // namespace qcs
