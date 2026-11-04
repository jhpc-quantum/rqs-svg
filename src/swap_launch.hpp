#ifndef QCS_SWAP_LAUNCH_HPP
#define QCS_SWAP_LAUNCH_HPP

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace qcs::swap_detail
{
    struct launch_chunk
    {
        uint64_t offset;
        uint64_t count;
        unsigned int blocks;
        unsigned int threads;
    };

    // Host-only planning, shared by production and tests. Limits are queried once
    // for the simulator's device/kernel, rather than on each SWAP operation.
    struct launch_limits
    {
        unsigned int threads;
        unsigned int blocks;

        launch_limits(int requested_threads, int device_threads, int dimension_threads,
                      int kernel_threads, int grid_blocks)
        {
            int const supported = std::min({requested_threads, device_threads,
                                            dimension_threads, kernel_threads});
            if (supported <= 0 || grid_blocks <= 0)
                throw std::runtime_error("invalid CUDA SWAP launch limits");
            threads = static_cast<unsigned int>(supported);
            blocks = static_cast<unsigned int>(grid_blocks);
        }

        launch_chunk next(uint64_t total, uint64_t offset) const
        {
            if (offset >= total)
                throw std::runtime_error("CUDA SWAP launch offset outside work range");
            uint64_t const block = std::min<uint64_t>(threads, total);
            // Both factors come from positive CUDA int limits; widen before
            // multiplying. Subtraction and quotient/remainder avoid overflow
            // even when total is UINT64_MAX.
            uint64_t const capacity = uint64_t(blocks) * block;
            uint64_t const count = std::min(total - offset, capacity);
            uint64_t const grid = count / block + (count % block != 0);
            return {offset, count, static_cast<unsigned int>(grid),
                    static_cast<unsigned int>(block)};
        }
    };
}
#endif
