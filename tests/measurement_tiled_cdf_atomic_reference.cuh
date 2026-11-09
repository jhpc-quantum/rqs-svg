#pragma once
// Frozen pre-reduction sampling kernel for exact index/status comparisons.
#include "../src/measurement_tiled_cdf.cuh"
namespace qcs::tiled_cdf_detail::test_reference {
__global__ void atomic_sample(const state_type* state, uint64_t count, uint64_t mask,
                       uint64_t value, bool active, const double* masses,
                       const double* cdf, const rank_summary* summary,
                       double uniform, sample_result* result) {
    __shared__ double sums[tile_size];
    __shared__ uint64_t tile;
    __shared__ double fraction;
    __shared__ unsigned failure;
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
    // Atomic integer min selects the first positive entry whose prefix exceeds x.
    for (unsigned k = 0; k < tile_size / threads; ++k) {
        unsigned j = threadIdx.x + k * threads;
        if (original[k] > 0 && sums[j] > x)
            atomicMin(reinterpret_cast<unsigned long long*>(&result->index),
                      static_cast<unsigned long long>(tile*tile_size+j));
    }
    __syncthreads();
    if (threadIdx.x == 0 && result->index == UINT64_MAX) result->status = invalid_distribution;
}

} // namespace qcs::tiled_cdf_detail::test_reference
