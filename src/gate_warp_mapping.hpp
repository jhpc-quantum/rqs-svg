#pragma once
#include <cstdint>
#include <vector>

namespace qcs { namespace gate_warp {
// All fields are kernel arguments, independent of mutable constant metadata.
struct mapping {
    uint64_t positive = 0, negative = 0, warps = 0;
    int lane[5] = {}, remaining[63] = {}, registers[3] = {};
    int target_lane[3] = {-1, -1, -1}, target_register[3] = {-1, -1, -1};
    int remaining_count = 0, register_count = 0;
};
inline bool build(int n, const std::vector<int>& targets,
                  const std::vector<int>& negative, const std::vector<int>& positive,
                  mapping& m)
{
    m = mapping{};
    if (n < 5 || n > 62 || targets.empty() || targets.size() > 3) return false;
    uint64_t used = 0, target_mask = 0;
    auto insert = [&](int bit, uint64_t& mask) {
        if (bit < 0 || bit >= n || (used & (uint64_t(1) << bit))) return false;
        used |= uint64_t(1) << bit;
        mask |= uint64_t(1) << bit;
        return true;
    };
    for (int p : positive) if (!insert(p, m.positive)) return false;
    for (int p : negative) if (!insert(p, m.negative)) return false;
    for (int p : targets) if (!insert(p, target_mask)) return false;
    int lanes = 0;
    for (int p = 0; p < n; ++p) {
        if (((m.positive | m.negative) >> p) & 1) continue;
        if (lanes < 5) m.lane[lanes++] = p;
        else if ((target_mask >> p) & 1) m.registers[m.register_count++] = p;
        else m.remaining[m.remaining_count++] = p;
    }
    if (lanes != 5) return false; // No partial warp collectives.
    for (unsigned j = 0; j < targets.size(); ++j) {
        for (int b = 0; b < 5; ++b) if (targets[j] == m.lane[b]) m.target_lane[j] = b;
        for (int b = 0; b < m.register_count; ++b)
            if (targets[j] == m.registers[b]) m.target_register[j] = b;
    }
    m.warps = uint64_t(1) << m.remaining_count;
    return true;
}
}}
