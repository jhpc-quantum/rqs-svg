#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace qcs::exchange_detail
{
    // Valid, distinct physical local/global pairs; dimensions and pair lists
    // must agree across ranks. No ownership or allocation of mapping storage.
    struct schedule
    {
        const std::vector<int>& local;
        const std::vector<int>& global;
        int local_bits, low, area_bits;
        uint64_t global_mask = 0, fixed_mask = 0, regions;

        schedule(int n, const std::vector<int>& b, const std::vector<int>& g)
            : local(b), global(g), local_bits(n),
              low(b.empty() ? n : *std::min_element(b.begin(), b.end())),
              area_bits(n - low), regions(UINT64_C(1) << (area_bits - static_cast<int>(b.size())))
        {
            for (size_t i = 0; i < b.size(); ++i)
            {
                global_mask |= UINT64_C(1) << (g[i] - n);
                fixed_mask |= UINT64_C(1) << (b[i] - low);
            }
        }

        // Ascending submasks without scanning gaps in sparse global masks.
        // Unsigned wraparound is intentional; zero terminates the sequence.
        uint64_t next(uint64_t delta) const { return (delta - global_mask) & global_mask; }

        uint64_t fixed_value(uint64_t peer) const
        {
            uint64_t value = 0;
            for (size_t i = 0; i < local.size(); ++i)
                value |= ((peer >> (global[i] - local_bits)) & 1) << (local[i] - low);
            return value;
        }

        uint64_t area(uint64_t fixed, uint64_t ordinal) const
        {
            for (int bit = 0; bit < area_bits; ++bit)
            {
                if (!(fixed_mask & (UINT64_C(1) << bit)))
                {
                    fixed |= (ordinal & 1) << bit;
                    ordinal >>= 1;
                }
            }
            return fixed;
        }
    };
}
