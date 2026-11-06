#include "../src/measurement_launch.hpp"
#include <cassert>
#include <limits>
#include <iostream>

int main()
{
    using namespace qcs::measurement_detail;
    for (unsigned maximum : {1u, 8u, 1024u})
    {
        for (uint64_t active = 1; active <= uint64_t(maximum) * 256 + 1; ++active)
        {
            unsigned blocks = measurement_blocks(active, maximum);
            assert(blocks >= 1 && blocks <= maximum);
            assert(blocks == std::min((active + 255) / 256, uint64_t(maximum)));
            assert(uint64_t(blocks - 1) * 256 < active);
        }
        assert(measurement_blocks(std::numeric_limits<uint64_t>::max(), maximum) == maximum);
    }
    for (auto invalid : {std::pair<uint64_t, unsigned>{0, 8}, {1, 0}})
    {
        bool rejected = false;
        try { measurement_blocks(invalid.first, invalid.second); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "measurement launch geometry passed\n";
}
