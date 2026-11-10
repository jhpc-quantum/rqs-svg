#ifndef QCS_HADAMARD_LAUNCH_HPP
#define QCS_HADAMARD_LAUNCH_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace qcs::hadamard_detail
{
    // Edit these constants and rebuild to change grid-stride geometry.
    constexpr int kHadamardGridStrideBlockSize = 256;
    constexpr int kHadamardGridStrideBlocksPerSM = 8;
    static_assert(kHadamardGridStrideBlockSize > 0 &&
                  kHadamardGridStrideBlockSize <= 1024);
    static_assert(kHadamardGridStrideBlocksPerSM > 0);

    enum class mode { baseline, a, b, grid_stride };

    inline mode selected_mode()
    {
        const char *value = std::getenv("QCS_H_LAUNCH_MODE");
        if (!value || !*value || !std::strcmp(value, "baseline")) return mode::baseline;
        if (!std::strcmp(value, "A")) return mode::a;
        if (!std::strcmp(value, "B")) return mode::b;
        if (!std::strcmp(value, "grid-stride")) return mode::grid_stride;
        throw std::runtime_error("QCS_H_LAUNCH_MODE must be baseline, A, B, or grid-stride");
    }

    struct geometry { uint64_t blocks; uint64_t threads; };

    inline geometry launch(mode selection, uint64_t work, int sm_count)
    {
        uint64_t threads = selection == mode::a ? 256 : selection == mode::b ? 1024 : 512;
        if (selection == mode::grid_stride)
        {
            if (sm_count <= 0) throw std::runtime_error("invalid GPU SM count");
            threads = kHadamardGridStrideBlockSize;
            return {std::min(work / threads + (work % threads != 0),
                             uint64_t(sm_count) * kHadamardGridStrideBlocksPerSM), threads};
        }
        threads = std::min(threads, work);
        return {threads ? work / threads + (work % threads != 0) : 0, threads};
    }
}
#endif
