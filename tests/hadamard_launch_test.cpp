#include "../src/hadamard_launch.hpp"
#include <cassert>
#include <vector>

int main()
{
    using namespace qcs::hadamard_detail;
    for (const char *name : {"baseline", "A", "B", "grid-stride"})
    {
        setenv("QCS_H_LAUNCH_MODE", name, 1);
        for (uint64_t work : {UINT64_C(1), UINT64_C(7), UINT64_C(256),
                              UINT64_C(513), UINT64_C(8193), UINT64_C(131073)})
        {
            auto g = launch(selected_mode(), work, 2);
            std::vector<int> visits(work);
            uint64_t stride = g.blocks * g.threads;
            if (selected_mode() == mode::grid_stride)
            {
                assert(g.threads == kHadamardGridStrideBlockSize);
                assert(g.blocks <= 2 * kHadamardGridStrideBlocksPerSM);
                for (uint64_t thread = 0; thread < stride; ++thread)
                    for (uint64_t i = thread; i < work; i += stride) ++visits[i];
                for (int count : visits) assert(count == 1);
            }
            assert(stride > 0);
        }
    }
    auto large = launch(mode::grid_stride, UINT64_C(1) << 40, 80);
    assert(large.blocks == 80 * kHadamardGridStrideBlocksPerSM);
    setenv("QCS_H_LAUNCH_MODE", "invalid", 1);
    bool rejected = false;
    try { selected_mode(); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
