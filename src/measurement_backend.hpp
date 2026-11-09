#pragma once
#include <cstdint>
#include <cstring>

namespace qcs::measurement_detail
{
// Stable internal selection codes; auto continues to choose only legacy paths.
inline int backend_mode(const char* name)
{
    if (!name || std::strcmp(name, "grouped") == 0) return 0;
    if (std::strcmp(name, "rank_sampled") == 0) return 1;
    if (std::strcmp(name, "auto") == 0) return 2;
    if (std::strcmp(name, "tiled_cdf") == 0) return 3;
    return -1;
}

struct backend_workload
{
    int fresh_local = 0, fresh_global = 0;
    int measured_local = 0, measured_global = 0;
    int ranks = 1;
    uint64_t local_states = 0;
};

// Provisional opt-in policy, not a fitted performance model. Favor fewer full
// scans only for large fresh requests and substantial effective local work.
inline bool prefer_rank_sampling(const backend_workload& w)
{
    if (w.ranks < 1 || w.ranks > 32 || w.measured_global != 0 ||
        w.measured_local < 0 || w.measured_local >= 64) return false;
    const uint64_t effective_states = w.local_states >> w.measured_local;
    return w.fresh_local >= 8 && w.fresh_local + w.fresh_global >= 16 &&
           effective_states >= (UINT64_C(1) << 20);
}
} // namespace qcs::measurement_detail
