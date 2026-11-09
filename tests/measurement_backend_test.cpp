#include "../src/measurement_backend.hpp"
#include <stdexcept>
#include <initializer_list>

int main()
{
    using namespace qcs::measurement_detail;
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("backend dispatch regression"); };
    check(backend_mode(nullptr) == 0 && backend_mode("grouped") == 0);
    check(backend_mode("rank_sampled") == 1 && backend_mode("auto") == 2);
    check(backend_mode("tiled_cdf") == 3);
    check(backend_mode("tiled_race") == 4);
    check(backend_mode("") == -1 && backend_mode("unknown") == -1);
    backend_workload w;
    w.local_states = UINT64_C(1) << 26;
    w.ranks = 32;
    w.fresh_local = 16;
    check(prefer_rank_sampling(w));
    w.fresh_local = 4;
    check(!prefer_rank_sampling(w));
    w.fresh_local = 8;
    w.fresh_global = 7;
    check(!prefer_rank_sampling(w));
    w.fresh_global = 8;
    check(prefer_rank_sampling(w));
    for (int ranks : {64, 128}) {
        w.ranks = ranks;
        check(!prefer_rank_sampling(w));
    }
    w.ranks = 32;
    w.measured_global = 1;
    check(!prefer_rank_sampling(w));
    w.measured_global = 0;
    w.measured_local = 7;
    check(!prefer_rank_sampling(w));
    w.measured_local = 6;
    check(prefer_rank_sampling(w));
    w.fresh_local = w.fresh_global = 0;
    check(!prefer_rank_sampling(w));
}
