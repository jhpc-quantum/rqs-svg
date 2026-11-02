#pragma once
#include <cuda_runtime.h>
#include <atlc/check_cuda.hpp>
#include <functional>
namespace measurement_probe {
inline cudaEvent_t start, stop;
inline bool enabled = false, recorded = false;
inline std::function<void(int, const int*, const double*, const double*)> observer;
inline std::function<void(unsigned, unsigned)> result_observer;
inline double local[16], global[16];
inline void capture_local(int n, const double* device, cudaStream_t stream) {
    ATLC_CHECK_CUDA(cudaMemcpyAsync, local, device, (1 << n) * sizeof(double), cudaMemcpyDeviceToHost, stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
}
inline void observe(int n, const int* targets, const double* device, cudaStream_t stream) {
    ATLC_CHECK_CUDA(cudaMemcpyAsync, global, device, (1 << n) * sizeof(double), cudaMemcpyDeviceToHost, stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    observer(n, targets, local, global);
}
inline void begin(cudaStream_t stream) {
    if (enabled) { ATLC_CHECK_CUDA(cudaEventRecord, start, stream); recorded = true; }
}
inline void end(cudaStream_t stream) {
    if (enabled) ATLC_CHECK_CUDA(cudaEventRecord, stop, stream);
}
}
