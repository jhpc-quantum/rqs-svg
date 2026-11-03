#pragma once
#include <nccl.h>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifndef QCS_NCCL_BUFFER_MODE
#define QCS_NCCL_BUFFER_MODE 0
#endif

namespace qcs {
// 0: original CUDA allocation; 1: NCCL allocation; 2: allocation + registration.
constexpr int nccl_buffer_mode = QCS_NCCL_BUFFER_MODE;
static_assert(nccl_buffer_mode >= 0 && nccl_buffer_mode <= 2, "invalid NCCL buffer mode");
inline void check_buffer_nccl(ncclResult_t result, const char* operation)
{
    if (result != ncclSuccess)
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
}
// Keep baseline builds usable with old NCCL headers and libraries.
inline void* allocate_nccl_buffer(size_t bytes)
{
#if NCCL_VERSION_CODE >= 23102
    void* ptr = nullptr;
    check_buffer_nccl(ncclMemAlloc(&ptr, bytes), "ncclMemAlloc");
    return ptr;
#else
    throw std::runtime_error("NCCL buffer modes 1/2 require NCCL >= 2.31.2 headers and library");
#endif
}
inline void free_nccl_buffer(void* ptr)
{
#if NCCL_VERSION_CODE >= 23102
    check_buffer_nccl(ncclMemFree(ptr), "ncclMemFree");
#else
    throw std::runtime_error("NCCL buffer allocation unavailable");
#endif
}
inline void register_nccl_buffer(ncclComm_t comm, void* ptr, size_t bytes, void** handle)
{
#if NCCL_VERSION_CODE >= 23102
    check_buffer_nccl(ncclCommRegister(comm, ptr, bytes, handle), "ncclCommRegister");
#else
    throw std::runtime_error("NCCL buffer registration unavailable");
#endif
}
inline void deregister_nccl_buffer(ncclComm_t comm, void*& handle)
{
    if (!handle) return;
#if NCCL_VERSION_CODE >= 23102
    check_buffer_nccl(ncclCommDeregister(comm, handle), "ncclCommDeregister");
    handle = nullptr;
#else
    throw std::runtime_error("NCCL buffer deregistration unavailable");
#endif
}
} // namespace qcs
