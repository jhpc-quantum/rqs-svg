#include "../src/measurement_race.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <set>
namespace r = qcs::race_detail;
static void check(bool ok) { if (!ok) std::abort(); }
int main() {
    // Published Random123 Philox4x32-10 all-zero known answer.
    check(r::random_bits(0, 0, 0, 0) == UINT64_C(0x6627e8d5e169c58d));
    std::set<uint64_t> draws;
    for (uint64_t cursor : {UINT64_C(0), UINT64_C(1), UINT64_C(1)<<32, UINT64_MAX})
        for (unsigned rank = 0; rank < 32; ++rank) for (unsigned domain = 0; domain < 2; ++domain) {
            check(draws.insert(r::random_bits(7, cursor, rank, domain)).second);
            double u = r::uniform(7, cursor, rank, domain);
            check(u > 0 && u < 1);
        }
    auto a = r::make_candidate(1, UINT64_C(1)<<50, 1, 0.5);
    auto b = a; b.rank = 0;
    auto error = r::candidate{}; error.status = 8;
    auto x = r::combine(r::combine(a, error), b);
    auto y = r::combine(a, r::combine(error, b));
    check(x.rank == 0 && x.status == 8 && x.index == (UINT64_C(1)<<50));
    check(y.rank == x.rank && y.status == x.status);
    check(r::combine(error, a).status == 8);
    check(!r::make_candidate(0, 0, 0, 0.5).valid);
    check(r::make_candidate(-1, 0, 0, 0.5).status);
    check(r::make_candidate(INFINITY, 0, 0, 0.5).status);
    check(r::make_candidate(NAN, 0, 0, 0.5).status);
    check(std::isfinite(r::make_candidate(std::numeric_limits<double>::denorm_min(),0,0,0.5).key));
    check(std::isfinite(r::make_candidate(std::numeric_limits<double>::max(),0,0,0.5).key));
    // Exercise different reduction parenthesizations, inert ranks, uneven and
    // extremely scaled finite masses; local samples use the separate domain.
    for (double scale : {1.0, 1e-300, 1e300}) {
        std::array<int,4> counts{};
        int local_ones = 0;
        constexpr int shots = 50000;
        for (int shot = 0; shot < shots; ++shot) {
            std::array<r::candidate,4> c;
            for (unsigned rank = 0; rank < 4; ++rank) {
                auto index = uint64_t(r::uniform(123,shot,rank,0) < 0.25);
                c[rank] = r::make_candidate(scale * (rank == 3 ? 0 : rank+1), index, rank,
                                           r::uniform(123,shot,rank,1));
            }
            auto winner = r::combine(r::combine(c[0],c[1]),r::combine(c[2],c[3]));
            auto other = r::combine(c[3],r::combine(c[2],r::combine(c[1],c[0])));
            check(winner.rank == other.rank && winner.index == other.index);
            ++counts[winner.rank]; local_ones += int(winner.index);
        }
        for (int rank = 0; rank < 3; ++rank) {
            double p = (rank+1)/6.0;
            check(std::abs(counts[rank]-shots*p) < 7*std::sqrt(shots*p*(1-p)));
        }
        check(counts[3] == 0);
        check(std::abs(local_ones-shots*0.25) < 7*std::sqrt(shots*0.25*0.75));
    }
    auto only = r::combine(r::candidate{}, a);
    check(only.valid && only.rank == a.rank);
    check(!r::combine(r::candidate{}, r::candidate{}).valid);
    std::cout << "measurement race tests passed\n";
}
