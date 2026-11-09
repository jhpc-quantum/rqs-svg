#include "../src/measurement_tiled_cdf.cuh"
#include "measurement_tiled_cdf_atomic_reference.cuh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
namespace td = qcs::tiled_cdf_detail;
using State = td::state_type;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
// Independent serial long-double hierarchy. Exact-boundary fixtures use dyadic
// weights, so reference boundaries are exactly representable on both devices.
uint64_t reference(const std::vector<State>& a, uint64_t mask, uint64_t value, double u) {
    std::vector<long double> masses(td::tiles_for(a.size()), 0);
    for (size_t i = 0; i < a.size(); ++i)
        if ((i & mask) == value) {
            long double re = a[i].real(), im = a[i].imag();
            masses[i/td::tile_size] += re*re + im*im;
        }
    long double total = 0;
    for (auto m : masses) total += m;
    long double x = u == 1 ? std::nextafter(total, 0.L) : total*u;
    size_t tile = 0;
    for (; tile < masses.size(); ++tile) {
        if (x < masses[tile]) break;
        x -= masses[tile];
    }
    for (size_t i = tile*td::tile_size; i < a.size(); ++i) {
        if ((i & mask) != value) continue;
        long double re = a[i].real(), im = a[i].imag(), w = re*re + im*im;
        if (w > 0 && x < w) return i;
        x -= w;
    }
    throw std::runtime_error("reference failed");
}
// Frozen original mass kernel: compare the full scan's last value bit for bit.
__global__ void original_masses(const State* state, uint64_t count, uint64_t mask,
                                uint64_t value, bool active, double* mass,
                                td::rank_summary* summary) {
    __shared__ double sums[td::tile_size];
    for (uint64_t t = blockIdx.x; t < td::tiles_for(count); t += gridDim.x) {
        for (unsigned j = threadIdx.x; j < td::tile_size; j += td::threads) {
            double w = td::weight(state, t * td::tile_size + j, count, mask, value, active);
            if (!isfinite(w) || w < 0) {
                atomicOr(&summary->status, unsigned(td::invalid_weight)); w = 0;
            }
            sums[j] = w;
        }
        __syncthreads();
        td::scan<td::tile_size>(sums);
        if (threadIdx.x == 0) mass[t] = sums[td::tile_size-1];
        __syncthreads();
    }
}
void compare_masses(const State* state, uint64_t count, uint64_t mask,
                    uint64_t value, bool active, cudaStream_t stream) {
    if (!count) return;
    size_t tiles = td::tiles_for(count);
    double *old_mass, *new_mass;
    td::rank_summary *old_status, *new_status;
    td::check(cudaMallocAsync(&old_mass, tiles*sizeof(double), stream));
    td::check(cudaMallocAsync(&new_mass, tiles*sizeof(double), stream));
    td::check(cudaMallocAsync(&old_status, sizeof(*old_status), stream));
    td::check(cudaMallocAsync(&new_status, sizeof(*new_status), stream));
    td::check(cudaMemsetAsync(old_status, 0, sizeof(*old_status), stream));
    td::check(cudaMemsetAsync(new_status, 0, sizeof(*new_status), stream));
    // One block also exercises shared-memory reuse across consecutive tiles.
    original_masses<<<1, td::threads, 0, stream>>>(state,count,mask,value,active,old_mass,old_status);
    td::masses<<<1, td::threads, 0, stream>>>(state,count,mask,value,active,new_mass,new_status);
    td::check(cudaGetLastError());
    std::vector<double> before(tiles), after(tiles);
    td::rank_summary a{}, b{};
    td::check(cudaMemcpyAsync(before.data(), old_mass, tiles*sizeof(double), cudaMemcpyDeviceToHost, stream));
    td::check(cudaMemcpyAsync(after.data(), new_mass, tiles*sizeof(double), cudaMemcpyDeviceToHost, stream));
    td::check(cudaMemcpyAsync(&a, old_status, sizeof(a), cudaMemcpyDeviceToHost, stream));
    td::check(cudaMemcpyAsync(&b, new_status, sizeof(b), cudaMemcpyDeviceToHost, stream));
    td::check(cudaStreamSynchronize(stream));
    require(std::memcmp(before.data(), after.data(), tiles*sizeof(double)) == 0, "mass rounding changed");
    require(a.status == b.status, "mass status changed");
    td::check(cudaFreeAsync(old_mass, stream));
    td::check(cudaFreeAsync(new_mass, stream));
    td::check(cudaFreeAsync(old_status, stream));
    td::check(cudaFreeAsync(new_status, stream));
}
int main() {
    try {
        require(td::tiles_for(UINT64_MAX) == UINT64_MAX/4096 + 1, "64-bit tile count");
        require(td::tiles_for((UINT64_C(1)<<32)+1) == (UINT64_C(1)<<20)+1, "32-bit count truncation");
        bool rejected = false;
        try { td::bytes_for(UINT64_MAX, sizeof(State)); }
        catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "size overflow accepted");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            std::cerr << "SKIP: CUDA device unavailable (host size checks passed)\n"; return 77;
        }
        cudaStream_t stream; td::check(cudaStreamCreate(&stream));
        {
            td::workspace ws(stream);
            auto run = [&](const std::vector<State>& a, uint64_t mask, uint64_t value,
                           bool active, unsigned expected_status, const std::vector<double>& draws,
                           uint64_t expected_index = UINT64_MAX) {
                State* device = nullptr;
                if (!a.empty()) {
                    td::check(cudaMallocAsync(&device, a.size()*sizeof(State), stream));
                    td::check(cudaMemcpyAsync(device, a.data(), a.size()*sizeof(State), cudaMemcpyHostToDevice, stream));
                }
                compare_masses(device, a.size(), mask, value, active, stream);
                const auto* summary_device = ws.build(device, a.size(), mask, value, active);
                td::rank_summary summary{};
                td::check(cudaMemcpyAsync(&summary, summary_device, sizeof(summary), cudaMemcpyDeviceToHost, stream));
                td::check(cudaStreamSynchronize(stream));
                require((summary.status & expected_status) == expected_status, "summary missing status");
                if (!expected_status) {
                    require(summary.status == 0, "unexpected summary status");
                    long double total = 0;
                    for (size_t i=0; i<a.size(); ++i) if ((i & mask) == value) {
                        long double re=a[i].real(), im=a[i].imag(); total += re*re+im*im;
                    }
                    require(std::abs((long double)summary.mass-total) <= total*1e-13L, "rank mass differs from CPU");
                }
                size_t tiles = std::max<size_t>(td::tiles_for(a.size()), 1);
                double *masses, *cdf;
                td::rank_summary* baseline_summary;
                td::sample_result* baseline_result;
                td::check(cudaMallocAsync(&masses, tiles*sizeof(double), stream));
                td::check(cudaMallocAsync(&cdf, tiles*sizeof(double), stream));
                td::check(cudaMallocAsync(&baseline_summary, sizeof(summary), stream));
                td::check(cudaMallocAsync(&baseline_result, sizeof(td::sample_result), stream));
                td::check(cudaMemsetAsync(baseline_summary, 0, sizeof(summary), stream));
                if (!a.empty())
                    td::masses<<<1, td::threads, 0, stream>>>(device, a.size(), mask, value, active, masses, baseline_summary);
                td::cdf_scan<<<1, td::threads, 0, stream>>>(masses, cdf, td::tiles_for(a.size()), baseline_summary);
                td::check(cudaGetLastError());
                auto capacity = ws.capacity();
                for (double u : draws) {
                    td::sample_result first{};
                    for (int repetition=0; repetition<3; ++repetition) {
                        // Rebuild verifies repeatability of the entire hierarchy.
                        ws.build(device, a.size(), mask, value, active);
                        auto* result_device = ws.draw(u);
                        td::test_reference::atomic_sample<<<1, td::threads, 0, stream>>>(
                            device, a.size(), mask, value, active, masses, cdf,
                            baseline_summary, u, baseline_result);
                        td::check(cudaGetLastError());
                        td::sample_result baseline{};
                        td::check(cudaMemcpyAsync(&baseline, baseline_result, sizeof(baseline), cudaMemcpyDeviceToHost, stream));
                        td::sample_result result{};
                        td::check(cudaMemcpyAsync(&result, result_device, sizeof(result), cudaMemcpyDeviceToHost, stream));
                        td::check(cudaStreamSynchronize(stream));
                        require(result.index == baseline.index && result.status == baseline.status,
                                "sample differs from atomic baseline");
                        if (repetition) require(result.index == first.index && result.status == first.status, "not repeatable");
                        first = result;
                        if (expected_status) require(result.status != 0 && result.index == UINT64_MAX, "invalid distribution sampled");
                        else if (!std::isfinite(u) || u<0 || u>1)
                            require(result.status == td::invalid_uniform, "invalid uniform accepted");
                        else require(result.status == 0 && result.index == (expected_index == UINT64_MAX ? reference(a, mask, value, u) : expected_index), "sample differs from CPU");
                    }
                }
                require(capacity == ws.capacity(), "workspace grew after reserve");
                td::check(cudaFreeAsync(masses, stream));
                td::check(cudaFreeAsync(cdf, stream));
                td::check(cudaFreeAsync(baseline_summary, stream));
                td::check(cudaFreeAsync(baseline_result, stream));
                if (device) {
                    std::vector<State> after(a.size());
                    td::check(cudaMemcpyAsync(after.data(), device, a.size()*sizeof(State), cudaMemcpyDeviceToHost, stream));
                    td::check(cudaStreamSynchronize(stream));
                    require(std::memcmp(after.data(), a.data(), a.size()*sizeof(State)) == 0, "input modified");
                    td::check(cudaFreeAsync(device, stream));
                }
            };
            std::vector<double> thresholds{0, std::nextafter(0.,1.),
                std::nextafter(.25,0.), .25, std::nextafter(.25,1.),
                std::nextafter(.5,0.), .5, std::nextafter(.5,1.), std::nextafter(1.,0.), 1};
            for (size_t n : {size_t(17), size_t(4096), size_t(8192), size_t(8209)}) {
                std::vector<State> a(n, State(0,0));
                a[1]=State(1,0); a[n/2]=State(1,0); a[n-2]=State(0,1);
                a[n-1]=State(1,0);
                run(a,0,0,true,0,thresholds);
                run(a,1,1,true,0,{0,.2,.6,1});
                run(a,0,0,false,td::zero_mass,{0,1});
            }
            // Dense contenders and winners on both sides of warp/stride boundaries.
            std::vector<double> dense_draws{0, 1};
            for (unsigned i : {31u, 32u, 63u, 64u, 255u, 256u, 4095u}) {
                double u = double(i) / td::tile_size;
                dense_draws.push_back(std::nextafter(u, 0.));
                dense_draws.push_back(u);
                dense_draws.push_back(std::nextafter(u, 1.));
            }
            run(std::vector<State>(td::tile_size, State(2,0)),0,0,true,0,dense_draws);
            // More than one metadata scan batch, including a partial batch.
            std::vector<State> large(257*td::tile_size+3, State(0,0));
            large[1]=State(1,0); large.back()=State(1,0);
            run(large,0,0,true,0,thresholds);
            std::vector<State> unequal(5000, State(0,0));
            unequal[0]=State(1e-150,0); unequal[4000]=State(1e100,0); unequal[4999]=State(1e-150,0);
            run(unequal,0,0,true,0,{0,.5,1});
            // Tile mass is 3 but its CDF interval width is 4. Endpoint clamping
            // gives residual 2/4, then threshold 1.5 in the child total of 3.
            std::vector<State> residual(4099, State(0,0));
            residual[0]=State(1e8,0);
            residual[4096]=residual[4097]=residual[4098]=State(1,0);
            run(residual,0,0,true,0,{1},4097);
            std::vector<State> rank_overflow(8192, State(0,0));
            rank_overflow[0]=rank_overflow[4096]=State(1e154,0);
            run(rank_overflow,0,0,true,td::invalid_total,{0,1});
            std::vector<State> rounded(8209, State(.1,.2));
            run(rounded,0,0,true,0,{0,.123,.789,1});
            std::vector<State> nonuniform(8209);
            for (size_t i = 0; i < nonuniform.size(); ++i)
                nonuniform[i] = State(double(1 + i % 31) / 37, double(i % 7) / 19);
            run(nonuniform,0,0,true,0,{0,.123,.789,1});
            run(nonuniform,7,3,true,0,{0,.123,.789,1});
            // Small contributions around an FP64 rounding boundary at every level.
            std::vector<State> difficult(8209);
            for (size_t i = 0; i < difficult.size(); ++i)
                difficult[i] = State(i % 127 == 0 ? 1. : std::ldexp(1., -27), 0);
            run(difficult,0,0,true,0,{0,.123,.789});
            run({},0,0,true,td::zero_mass,{0,1});
            run(std::vector<State>(17,State(0,0)),0,0,true,td::zero_mass,{0,1});
            for (double bad : std::vector<double>{NAN, INFINITY, 1e308}) {
                std::vector<State> a(17,State(0,0)); a[0]=State(bad,0); a[1]=State(1,0);
                run(a,0,0,true,td::invalid_weight,{0,1});
                run(a,1,1,true,0,{0,1}); // invalid masked amplitude ignored
                run(a,0,0,false,td::zero_mass,{0,1});
            }
            run(std::vector<State>(4096,State(1e153,0)),0,0,true,td::invalid_total,{0,1});
            run({State(1,0)},0,0,true,0,{-1,2,NAN,INFINITY});
            // No state pointer is needed by an inactive rank.
            ws.build(nullptr,4096,0,0,false);
            ws.close(); td::check(cudaStreamSynchronize(stream));
        }
        td::check(cudaStreamDestroy(stream));
        std::cout << "tiled CDF tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
