#pragma once
#include "measurement_profile.hpp"
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qcs::tiled_cdf_detail {
using state_type = cuda::std::complex<double>;
constexpr unsigned threads = 256;
constexpr unsigned tile_size = 4096;
enum status : unsigned { success = 0, invalid_weight = 1, invalid_total = 2,
    zero_mass = 4, invalid_uniform = 8, invalid_distribution = 16 };
struct rank_summary { double mass; unsigned status; double uniform; };
struct sample_result { uint64_t index; unsigned status; };
inline void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__host__ __device__ inline uint64_t tiles_for(uint64_t count) {
    return count / tile_size + (count % tile_size != 0);
}
inline size_t bytes_for(uint64_t count, size_t width) {
    if (count > std::numeric_limits<size_t>::max() / width)
        throw std::overflow_error("tiled CDF workspace size overflow");
    return size_t(count) * width;
}
// Fixed Hillis-Steele tree. Each round snapshots all operands before writes.
// No inter-block floating point atomics or implementation-dependent CUB scan.
template<unsigned N>
__device__ void scan(double* values) {
    double next[N / threads];
    for (unsigned offset = 1; offset < N; offset *= 2) {
        for (unsigned k = 0; k < N / threads; ++k) {
            unsigned i = threadIdx.x + k * threads;
            next[k] = i >= offset ? __dadd_rn(values[i], values[i-offset]) : values[i];
        }
        __syncthreads();
        for (unsigned k = 0; k < N / threads; ++k)
            values[threadIdx.x + k * threads] = next[k];
        __syncthreads();
    }
}
__device__ double weight(const state_type* state, uint64_t i, uint64_t count,
                         uint64_t mask, uint64_t value, bool active) {
    if (!active || i >= count || (i & mask) != value) return 0;
    auto a = state[i];
    return __dadd_rn(__dmul_rn(a.real(), a.real()), __dmul_rn(a.imag(), a.imag()));
}
__global__ void masses(const state_type* state, uint64_t count, uint64_t mask,
                       uint64_t value, bool active, double* mass, rank_summary* summary) {
    __shared__ double sums[tile_size];
    for (uint64_t t = blockIdx.x; t < tiles_for(count); t += gridDim.x) {
        for (unsigned j = threadIdx.x; j < tile_size; j += threads) {
            // count is checked against addressable state storage on the host.
            double w = weight(state, t * tile_size + j, count, mask, value, active);
            if (!isfinite(w) || w < 0) {
                atomicOr(&summary->status, unsigned(invalid_weight)); w = 0;
            }
            sums[j] = w;
        }
        __syncthreads();
        // Keep only the ancestors of the old scan's final element. Each
        // subtree adds its right half to its left half, with the same rounding.
        for (unsigned stride = 2; stride <= tile_size / 32; stride *= 2) {
            for (unsigned i = threadIdx.x; i < tile_size / stride; i += threads) {
                unsigned end = (i + 1) * stride - 1;
                sums[end] = __dadd_rn(sums[end], sums[end - stride / 2]);
            }
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            unsigned lane = threadIdx.x;
            double total = sums[(lane + 1) * (tile_size / 32) - 1];
            for (unsigned offset = 1; offset < 32; offset *= 2) {
                double left = __shfl_up_sync(0xffffffff, total, offset);
                if ((lane & (2 * offset - 1)) == 2 * offset - 1)
                    total = __dadd_rn(total, left);
            }
            if (lane == 31) mass[t] = total;
        }
        __syncthreads();
    }
}
// Bounded shared-memory scan of metadata batches, with a fixed carry order.
__global__ void cdf_scan(const double* masses, double* cdf, uint64_t tiles,
                         rank_summary* summary) {
    __shared__ double batch[threads];
    __shared__ double carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (uint64_t begin = 0; begin < tiles; begin += threads) {
        uint64_t i = begin + threadIdx.x;
        batch[threadIdx.x] = i < tiles ? masses[i] : 0;
        __syncthreads();
        scan<threads>(batch);
        if (cdf && i < tiles) cdf[i] = __dadd_rn(carry, batch[threadIdx.x]);
        __syncthreads();
        if (threadIdx.x == 0) carry = __dadd_rn(carry, batch[threads-1]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        summary->mass = carry;
        if (!isfinite(carry) || carry < 0) summary->status |= invalid_total;
        else if (carry == 0) summary->status |= zero_mass;
    }
}
__device__ double threshold(double fraction, double total) {
    double x = __dmul_rn(fraction, total);
    return x < total ? x : nextafter(total, 0.0);
}
__global__ void sample(const state_type* state, uint64_t count, uint64_t mask,
                       uint64_t value, bool active, const double* masses,
                       const double* cdf, const rank_summary* summary,
                       double uniform, sample_result* result) {
    __shared__ double sums[tile_size];
    __shared__ uint64_t tile;
    __shared__ double fraction;
    __shared__ unsigned failure;
    __shared__ unsigned warp_minima[threads / 32];
    double original[tile_size / threads];
    if (threadIdx.x == 0) {
        failure = summary->status;
        if (!isfinite(uniform) || uniform < 0 || uniform > 1) failure |= invalid_uniform;
        *result = {UINT64_MAX, failure};
        tile = UINT64_MAX;
        if (!failure) {
            double x = threshold(uniform, summary->mass);
            uint64_t lo = 0, hi = tiles_for(count);
            while (lo < hi) {
                uint64_t mid = lo + (hi-lo)/2;
                if (cdf[mid] > x) hi = mid; else lo = mid+1;
            }
            if (lo == tiles_for(count) || !(masses[lo] > 0)) failure = invalid_distribution;
            else {
                double before = lo ? cdf[lo-1] : 0;
                double width = cdf[lo] - before;
                fraction = (x-before)/width;
                tile = lo;
            }
            result->status = failure;
        }
    }
    __syncthreads();
    if (failure) return;
    for (unsigned k = 0; k < tile_size / threads; ++k) {
        unsigned j = threadIdx.x + k * threads;
        original[k] = weight(state, tile*tile_size+j, count, mask, value, active);
        sums[j] = original[k];
    }
    __syncthreads();
    scan<tile_size>(sums);
    double total = sums[tile_size-1];
    if (!isfinite(total) || !(total > 0)) {
        if (threadIdx.x == 0) result->status = invalid_distribution;
        return;
    }
    double x = threshold(fraction, total);
    // Reduce tile-relative indices; tile_size is also the no-candidate sentinel.
    unsigned candidate = tile_size;
    for (unsigned k = 0; k < tile_size / threads; ++k) {
        unsigned j = threadIdx.x + k * threads;
        if (original[k] > 0 && sums[j] > x)
            candidate = min(candidate, j);
    }
    for (unsigned offset = 16; offset; offset /= 2)
        candidate = min(candidate, __shfl_down_sync(0xffffffff, candidate, offset));
    unsigned lane = threadIdx.x % 32;
    if (lane == 0) warp_minima[threadIdx.x / 32] = candidate;
    __syncthreads();
    if (threadIdx.x < 32) {
        candidate = lane < threads / 32 ? warp_minima[lane] : tile_size;
        for (unsigned offset = 16; offset; offset /= 2)
            candidate = min(candidate, __shfl_down_sync(0xffffffff, candidate, offset));
        if (lane == 0)
            *result = candidate == tile_size
                ? sample_result{UINT64_MAX, invalid_distribution}
                : sample_result{tile * tile_size + candidate, success};
    }
}

// Single owner; stream must outlive workspace. Results remain device-resident,
// valid until the next operation. State must stay unchanged between build/sample.
class workspace {
    cudaStream_t stream_;
    int device_;
    void* storage_ = nullptr;
    uint64_t capacity_ = 0;
    uint64_t count_ = 0, mask_ = 0, value_ = 0;
    bool active_ = false, built_ = false;
    const state_type* state_ = nullptr;
    double* masses_ = nullptr;
    double* cdf_ = nullptr;
    rank_summary* summary_ = nullptr;
    sample_result* result_ = nullptr;
    void owner() const {
        int device; check(cudaGetDevice(&device));
        if (device != device_) throw std::runtime_error("tiled CDF workspace device mismatch");
    }
public:
    explicit workspace(cudaStream_t stream) : stream_(stream) { check(cudaGetDevice(&device_)); }
    workspace(const workspace&) = delete;
    workspace& operator=(const workspace&) = delete;
    ~workspace() noexcept {
        int previous;
        if (storage_ && cudaGetDevice(&previous) == cudaSuccess) {
            if (cudaSetDevice(device_) == cudaSuccess) cudaFreeAsync(storage_, stream_);
            cudaSetDevice(previous);
        }
    }
    void close() {
        owner();
        if (storage_) check(cudaFreeAsync(storage_, stream_));
        storage_ = nullptr; capacity_ = 0; built_ = false;
    }
    void reserve(uint64_t count) {
        owner();
        bytes_for(count, sizeof(state_type));
        uint64_t tiles = std::max<uint64_t>(tiles_for(count), 1);
        if (tiles <= capacity_) return;
        size_t arrays = bytes_for(tiles, 2 * sizeof(double));
        if (arrays > SIZE_MAX - sizeof(rank_summary) - sizeof(sample_result))
            throw std::overflow_error("tiled CDF workspace layout overflow");
        void* next = nullptr;
        check(cudaMallocAsync(&next, arrays + sizeof(rank_summary) + sizeof(sample_result), stream_));
        if (storage_) {
            auto error = cudaFreeAsync(storage_, stream_);
            if (error != cudaSuccess) { cudaFreeAsync(next, stream_); check(error); }
        }
        storage_ = next; capacity_ = tiles; built_ = false;
        masses_ = static_cast<double*>(next); cdf_ = masses_ + tiles;
        summary_ = reinterpret_cast<rank_summary*>(cdf_ + tiles);
        result_ = reinterpret_cast<sample_result*>(summary_ + 1);
    }
    const rank_summary* build(const state_type* state, uint64_t count,
                             uint64_t mask = 0, uint64_t value = 0, bool active = true, bool build_cdf = true) {
        owner();
        if ((active && count && !state) || (value & ~mask))
            throw std::invalid_argument("tiled CDF invalid state or conditioning");
        QCS_MEASURE_RANGE("tiled_cdf.workspace_reserve");
        reserve(count);
        state_ = state; count_ = count; mask_ = mask; value_ = value; active_ = active;
        built_ = false;
        check(cudaMemsetAsync(summary_, 0, sizeof(rank_summary), stream_));
        uint64_t tiles = tiles_for(count);
        QCS_MEASURE_STAGE("tiled_cdf.tile_reduction");
        if (tiles) {
            masses<<<unsigned(std::min<uint64_t>(tiles, 65535)), threads, 0, stream_>>>(
                state, count, mask, value, active, masses_, summary_);
            check(cudaGetLastError());
        }
        QCS_MEASURE_STAGE("tiled_cdf.device_cdf_construction");
        // Retain the identical scan arithmetic and validation for rank mass;
        // omit all tile-prefix stores when no local sample can be requested.
        cdf_scan<<<1, threads, 0, stream_>>>(masses_, build_cdf ? cdf_ : nullptr, tiles, summary_);
        check(cudaGetLastError()); built_ = build_cdf;
        return summary_;
    }
    const sample_result* draw(double uniform) {
        owner();
        if (!built_) throw std::runtime_error("tiled CDF build required before draw");
        sample<<<1, threads, 0, stream_>>>(state_, count_, mask_, value_, active_,
                                         masses_, cdf_, summary_, uniform, result_);
        check(cudaGetLastError()); return result_;
    }
    uint64_t capacity() const { return capacity_; }
};
} // namespace qcs::tiled_cdf_detail
