#include "../src/measurement_backend.hpp"
#include <stdexcept>
#include <initializer_list>

int main()
{
    using namespace qcs::measurement_detail;
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("backend dispatch regression"); };
    check(backend_mode(nullptr) == 0 && backend_mode("grouped") == 0);
    check(backend_mode("rank_sampled") == 1 && backend_mode("auto") == 2);
    check(backend_mode("tiled_cdf") == 3);
    check(backend_mode("tiled_race") == 4);
    check(backend_mode("tiled_race_nccl") == 5);
    check(backend_mode("") == -1 && backend_mode("unknown") == -1);
    backend_workload w;
    w.local_states = UINT64_C(1) << 26;
    w.ranks = 32;
    w.fresh_local = 16;
    check(prefer_rank_sampling(w));
    w.fresh_local = 4;
    check(!prefer_rank_sampling(w));
    w.fresh_local = 8;
    w.fresh_global = 7;
    check(!prefer_rank_sampling(w));
    w.fresh_global = 8;
    check(prefer_rank_sampling(w));
    for (int ranks : {64, 128}) {
        w.ranks = ranks;
        check(!prefer_rank_sampling(w));
    }
    w.ranks = 32;
    w.measured_global = 1;
    check(!prefer_rank_sampling(w));
    w.measured_global = 0;
    w.measured_local = 7;
    check(!prefer_rank_sampling(w));
    w.measured_local = 6;
    check(prefer_rank_sampling(w));
    w.fresh_local = w.fresh_global = 0;
    check(!prefer_rank_sampling(w));
    // Exact legacy boundaries: these are compatibility checks, not fitted data.
    for (int ranks : {1, 2, 4, 8, 16, 32, 64}) {
        for (int local : {7, 8, 15, 16}) {
            for (int global : {0, 1, 8}) {
                for (int exponent : {19, 20, 21}) {
                    backend_workload b;
                    b.ranks = ranks;
                    b.fresh_local = local;
                    b.fresh_global = global;
                    b.local_states = UINT64_C(1) << exponent;
                    const bool expected = ranks <= 32 && local >= 8 &&
                        local + global >= 16 && exponent >= 20;
                    check((select_auto_backend(b) == automatic_backend::rank_sampled) == expected);
                }
            }
        }
    }
    const uint64_t request = (UINT64_C(1) << 24) - 1;
    for (uint64_t measured : {UINT64_C(3), UINT64_C(3) << 22,
                              (UINT64_C(1) << 0) | (UINT64_C(1) << 23)}) {
        auto b = workload_from_masks(request, measured, 26, 4, UINT64_C(1) << 26);
        check(b.fresh_local == 22 && b.fresh_global == 0 && b.measured_local == 2);
        check(select_auto_backend(b) == automatic_backend::rank_sampled);
    }
    auto mixed = workload_from_masks(request, UINT64_C(1) << 23, 20, 16, UINT64_C(1) << 20);
    check(mixed.fresh_local == 20 && mixed.fresh_global == 3 && mixed.measured_global == 1);
    check(select_auto_backend(mixed) == automatic_backend::grouped);
    for (uint64_t targets : {UINT64_C(0), request, UINT64_C(1) << 63}) {
        auto cached = workload_from_masks(targets, targets, 26, 4, UINT64_C(1) << 26);
        check(cached.fresh_local == 0 && cached.fresh_global == 0);
        check(select_auto_backend(cached) == automatic_backend::grouped);
    }
    auto global_only = workload_from_masks(UINT64_C(15) << 26, 0, 26, 16, UINT64_C(1) << 26);
    check(global_only.fresh_local == 0 && global_only.fresh_global == 4);
    check(select_auto_backend(global_only) == automatic_backend::grouped);
    for (int invalid : {-1, 64})
        check(select_auto_backend(workload_from_masks(request, 0, invalid, 4,
              UINT64_C(1) << 26)) == automatic_backend::grouped);
    for (int invalid : {-1, 65, 2147483647}) {
        w.fresh_local = invalid;
        check(select_auto_backend(w) == automatic_backend::grouped);
    }
}
