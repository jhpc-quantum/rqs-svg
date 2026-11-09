#pragma once
#include <cstdint>
#include <cstring>

namespace qcs::measurement_detail
{
// Stable environment selection codes; explicit selections bypass auto policy.
inline int backend_mode(const char* name)
{
    if (!name || std::strcmp(name, "grouped") == 0) return 0;
    if (std::strcmp(name, "rank_sampled") == 0) return 1;
    if (std::strcmp(name, "auto") == 0) return 2;
    if (std::strcmp(name, "tiled_cdf") == 0) return 3;
    if (std::strcmp(name, "tiled_race") == 0) return 4;
    if (std::strcmp(name, "tiled_race_nccl") == 0) return 5;
    return -1;
}

struct backend_workload
{
    int fresh_local = 0, fresh_global = 0;
    int measured_local = 0, measured_global = 0;
    int ranks = 1;
    uint64_t local_states = 0;
};

// Build counts from physical masks after applying the replicated mapping. Keeping
// this conversion on the host makes cached targets and bit placement testable.
// No sampled values, active-rank flags or device data enter this interface.
inline backend_workload workload_from_masks(uint64_t requested, uint64_t measured,
                                            int local_qubits, int ranks,
                                            uint64_t local_states)
{
    backend_workload w{};
    w.ranks = ranks;
    w.local_states = local_states;
    if (local_qubits < 0 || local_qubits >= 64) {
        w.ranks = 0; // Invalid metadata takes the conservative fallback.
        return w;
    }
    const uint64_t fresh = requested & ~measured;
    for (int p = 0; p < 64; ++p) {
        const uint64_t bit = UINT64_C(1) << p;
        if (fresh & bit) {
            if (p < local_qubits) ++w.fresh_local;
            else ++w.fresh_global;
        }
        if (measured & bit) {
            if (p < local_qubits) ++w.measured_local;
            else ++w.measured_global;
        }
    }
    return w;
}

// Provisional opt-in policy, not a fitted performance model. Favor fewer full
// scans only for large fresh requests and substantial effective local work.
inline bool prefer_rank_sampling(const backend_workload& w)
{
    if (w.ranks < 1 || w.ranks > 32 || w.measured_global != 0 ||
        w.measured_local < 0 || w.measured_local >= 64 ||
        w.fresh_local < 0 || w.fresh_local > 64 ||
        w.fresh_global < 0 || w.fresh_global > 64) return false;
    const uint64_t effective_states = w.local_states >> w.measured_local;
    return w.fresh_local >= 8 && w.fresh_local + w.fresh_global >= 16 &&
           effective_states >= (UINT64_C(1) << 20);
}
// Tiled CDF is deliberately ineligible until hardware correctness and crossover
// evidence exists. Keep the policy result separate from environment mode codes.
enum class automatic_backend { grouped, rank_sampled };

inline automatic_backend select_auto_backend(const backend_workload& w)
{
    return prefer_rank_sampling(w) ? automatic_backend::rank_sampled
                                   : automatic_backend::grouped;
}
} // namespace qcs::measurement_detail
