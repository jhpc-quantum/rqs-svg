#include "../src/swap_launch.hpp"
#include <climits>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

static void require(bool ok)
{
    if (!ok) throw std::runtime_error("SWAP launch planning check failed");
}

static void check_chunk(const qcs::swap_detail::launch_limits& limits,
                        uint64_t total, uint64_t offset)
{
    auto const chunk = limits.next(total, offset);
    require(chunk.offset == offset && chunk.count > 0 && chunk.count <= total - offset);
    require(chunk.threads > 0 && chunk.threads <= limits.threads);
    require(chunk.blocks > 0 && chunk.blocks <= limits.blocks);
    uint64_t const launched = uint64_t(chunk.blocks) * chunk.threads;
    require(launched >= chunk.count && launched - chunk.count < chunk.threads);
    require(chunk.count == std::min(total - offset, uint64_t(limits.blocks) * chunk.threads));
}

int main()
{
    using qcs::swap_detail::launch_limits;
    // Exhaustively simulate the device guard and ordinal offset on small grids.
    // Every ordinal must have exactly one owning thread, including final tails.
    for (int threads : {1, 3, 37, 512})
    for (int blocks : {1, 2, 7})
    for (uint64_t total = 1; total <= 2049; ++total)
    {
        launch_limits limits(threads, 1024, 1024, 1024, blocks);
        std::vector<unsigned char> visits(total, 0);
        for (uint64_t offset = 0; offset < total; )
        {
            check_chunk(limits, total, offset);
            auto const chunk = limits.next(total, offset);
            for (uint64_t thread = 0; thread < uint64_t(chunk.blocks) * chunk.threads; ++thread)
                if (thread < chunk.count) require(++visits[chunk.offset + thread] == 1);
            offset += chunk.count;
        }
        for (auto count : visits) require(count == 1);
    }
    launch_limits actual(512, 1024, 1024, 1024, INT_MAX);
    uint64_t const old_capacity = UINT64_C(65535) * 512;
    for (uint64_t total : {old_capacity - 1, old_capacity, old_capacity + 1,
                           UINT64_C(1) << 26})
    {
        check_chunk(actual, total, 0);
        require(actual.next(total, 0).count == total);
    }
    require(actual.next(old_capacity + 1, 0).blocks == 65536);
    for (int grids : {3, 65535, INT_MAX})
    {
        launch_limits limits(512, 1024, 1024, 1024, grids);
        uint64_t const capacity = uint64_t(grids) * 512;
        for (uint64_t total : {capacity - 1, capacity, capacity + 1, 2 * capacity + 7})
        {
            uint64_t offset = 0;
            unsigned int launches = 0;
            while (offset < total)
            {
                check_chunk(limits, total, offset);
                offset += limits.next(total, offset).count;
                ++launches;
            }
            require(offset == total && launches <= 3);
        }
        uint64_t const maximum = std::numeric_limits<uint64_t>::max();
        for (uint64_t offset : {UINT64_C(0), maximum - capacity - 1, maximum - capacity,
                                maximum - 511, maximum - 1})
            check_chunk(limits, maximum, offset);
    }
    require(launch_limits(512, 256, 128, 64, 7).threads == 64);
    require(launch_limits(512, 32, 128, 64, 7).threads == 32);
    require(launch_limits(512, 256, 16, 64, 7).threads == 16);
    for (int which = 0; which < 5; ++which)
    {
        int args[] = {512, 1024, 1024, 1024, INT_MAX};
        args[which] = 0;
        bool rejected = false;
        try { launch_limits invalid(args[0], args[1], args[2], args[3], args[4]); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected);
    }
    for (auto pair : {std::pair<uint64_t, uint64_t>{0, 0}, {3, 3}, {3, 4}})
    {
        bool rejected = false;
        try { actual.next(pair.first, pair.second); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected);
    }
    std::cout << "SWAP launch planning passed (coverage, tails, limits, 65535 boundary, uint64 overflow)\n";
}
