#pragma once
#include <cuda_runtime.h>
#include <atlc/check_cuda.hpp>
#include <functional>
namespace measurement_probe {
inline cudaEvent_t start, stop;
inline bool enabled = false, recorded = false;
inline std::function<void(int, const int*, const double*, const double*)> observer;
inline void begin(cudaStream_t stream) {
    if (enabled) { ATLC_CHECK_CUDA(cudaEventRecord, start, stream); recorded = true; }
}
inline void end(cudaStream_t stream) {
    if (enabled) ATLC_CHECK_CUDA(cudaEventRecord, stop, stream);
}
}
