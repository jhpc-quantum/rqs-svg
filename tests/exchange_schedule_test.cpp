#include "../src/exchange_schedule.hpp"
#include <complex>
#include <iostream>
#include <stdexcept>
#include <utility>

using indices = std::vector<int>;
using message = std::pair<uint64_t, uint64_t>;
static void require(bool ok) { if (!ok) throw std::runtime_error("exchange schedule mismatch"); }
static uint64_t swapped(uint64_t index, const indices& b, const indices& g)
{
    // Independent oracle: sequential physical bit transpositions.
    for (size_t i = 0; i < b.size(); ++i)
        if (((index >> b[i]) ^ (index >> g[i])) & 1)
            index ^= (UINT64_C(1) << b[i]) | (UINT64_C(1) << g[i]);
    return index;
}
static uint64_t cases = 0;
static void check(int n, int h, const indices& b, const indices& g)
{
    qcs::exchange_detail::schedule s(n, b, g);
    uint64_t A = UINT64_C(1) << s.low, size = UINT64_C(1) << n;
    using value = std::complex<double>;
    std::vector<value> initial(size << h), result(size << h);
    for (uint64_t i = 0; i < initial.size(); ++i)
        initial[i] = value(double(i + 1) / initial.size(), double(int(i % 17) - 8) / 17);
    result = initial;
    for (uint64_t r = 0; r < (UINT64_C(1) << h); ++r)
    {
        std::vector<message> old, current;
        for (uint64_t a = 0; a < size / A; ++a)
        {
            uint64_t peer = r;
            for (size_t i = 0; i < b.size(); ++i)
                peer = (peer & ~(UINT64_C(1) << (g[i] - n))) |
                    (((a >> (b[i] - s.low)) & 1) << (g[i] - n));
            if (peer != r) old.emplace_back(peer, a);
        }
        uint64_t previous_delta = 0;
        for (uint64_t d = s.next(0); d; d = s.next(d))
        {
            require(d > previous_delta); previous_delta = d;
            uint64_t peer = r ^ d;
            require((peer ^ d) == r);
            uint64_t previous_area = 0;
            for (uint64_t z = 0; z < s.regions; ++z)
            {
                uint64_t a = s.area(s.fixed_value(peer), z);
                uint64_t remote = s.area(s.fixed_value(r), z);
                require(!z || a > previous_area); previous_area = a;
                current.emplace_back(peer, a);
                for (uint64_t x = 0; x < A; ++x)
                {
                    result[r * size + a * A + x] = initial[peer * size + remote * A + x];
                    require(swapped(r * size + a * A + x, b, g) ==
                            peer * size + remote * A + x);
                }
            }
        }
        if (b.size() == 1) require(old == current); // Existing single-pair order.
        auto ordered = current;
        std::sort(old.begin(), old.end()); std::sort(ordered.begin(), ordered.end());
        require(old == ordered);
        require(std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end());
        require(current.size() * A == (b.empty() ? 0 : size - (size >> b.size())));
        // Model the unchanged per-region chunks for both pipeline settings,
        // including odd staging capacities and partial final chunks.
        for (bool pipeline : {false, true})
        for (uint64_t capacity : {1, 2, 6, 10})
        {
            uint64_t chunk = std::min(A, pipeline && capacity >= 2 ? capacity / 2 : capacity);
            uint64_t groups = 0, expected_groups = old.size() * ((A - 1) / chunk + 1);
            for (auto [peer, a] : current)
            for (uint64_t pos = 0; pos < A; pos += chunk)
            {
                uint64_t len = std::min(chunk, A - pos);
                uint64_t destination = swapped(r * size + a * A + pos, b, g);
                require(destination / size == peer);
                require(swapped(destination + len - 1, b, g) == r * size + a * A + pos + len - 1);
                ++groups;
            }
            require(groups == expected_groups);
        }
        ++cases;
    }
    for (uint64_t i = 0; i < result.size(); ++i)
    {
        require(std::isfinite(result[i].real()) && std::isfinite(result[i].imag()));
        require(result[i] == initial[swapped(i, b, g)]);
    }
}
static void enumerate(int n, int h, int k, indices& b, indices& g)
{
    if (int(b.size()) == k) { check(n, h, b, g); return; }
    for (int x = 0; x < n; ++x)
    for (int y = n; y < n + h; ++y)
        if (std::find(b.begin(), b.end(), x) == b.end() &&
            std::find(g.begin(), g.end(), y) == g.end())
        {
            b.push_back(x); g.push_back(y);
            enumerate(n, h, k, b, g);
            b.pop_back(); g.pop_back();
        }
}
int main()
{
    for (int n = 1; n <= 5; ++n)
    for (int h = 1; h <= 3; ++h)
    for (int k = 0; k <= std::min(n, h); ++k)
    {
        indices b, g; enumerate(n, h, k, b, g);
    }
    // Sparse high bits and the unsigned successor's wraparound boundary.
    indices b{1, 4}, g{6, 63};
    qcs::exchange_detail::schedule high(5, b, g);
    require(high.next(0) == 2);
    require(high.next(2) == (UINT64_C(1) << 58));
    require(high.next(high.global_mask) == 0);
    indices top_b{0}, top_g{64};
    qcs::exchange_detail::schedule top(1, top_b, top_g);
    require(top.next(0) == (UINT64_C(1) << 63));
    require(top.next(top.global_mask) == 0);
    std::cout << "PASS: " << cases << " rank/mapping cases, regions, destinations, chunks and group counts\n";
}
