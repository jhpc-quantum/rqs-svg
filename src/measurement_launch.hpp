#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace qcs::measurement_detail
{
constexpr unsigned measurement_block_size = 256;

// One block per tile, capped at the existing throughput configuration. Division
// before rounding avoids overflow even for the largest uint64_t workload.
inline unsigned measurement_blocks(uint64_t active_count, unsigned maximum)
{
    if (!active_count || !maximum)
        throw std::runtime_error("measurement reducer: empty launch geometry");
    const uint64_t tiles = active_count / measurement_block_size
                         + (active_count % measurement_block_size != 0);
    return static_cast<unsigned>(std::min(tiles, uint64_t(maximum)));
}
}
