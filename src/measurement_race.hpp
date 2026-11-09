#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define QCS_RACE_HD __host__ __device__
#else
#define QCS_RACE_HD
#endif
namespace qcs::race_detail {
// Philox4x32-10: counter lanes encode the complete public cursor, rank and
// domain. Distinct tuples never alias an input block (no cursor multiplication).
QCS_RACE_HD inline uint64_t random_bits(uint64_t seed, uint64_t cursor, uint32_t rank, uint32_t domain) {
    uint32_t a = uint32_t(cursor), b = uint32_t(cursor >> 32), c = rank, d = domain;
    uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        uint64_t p = UINT64_C(0xd2511f53) * a, q = UINT64_C(0xcd9e8d57) * c;
        uint32_t na = uint32_t(q >> 32) ^ b ^ k0;
        uint32_t nc = uint32_t(p >> 32) ^ d ^ k1;
        b = uint32_t(q); d = uint32_t(p); a = na; c = nc;
        k0 += 0x9e3779b9u; k1 += 0xbb67ae85u;
    }
    return (uint64_t(a) << 32) | b;
}
QCS_RACE_HD inline double uniform(uint64_t seed, uint64_t cursor, uint32_t rank, uint32_t domain) {
    // 52-bit midpoint grid is strictly inside (0,1), including both endpoints.
    return (double(random_bits(seed, cursor, rank, domain) >> 12) + 0.5) * 0x1p-52;
}
struct candidate {
    double key = 0;
    uint64_t index = UINT64_MAX;
    uint32_t rank = UINT32_MAX;
    unsigned valid = 0, status = 0;
};
inline candidate make_candidate(double mass, uint64_t index, uint32_t rank, double u) {
    candidate out;
    if (!std::isfinite(mass) || mass < 0 || !(u > 0 && u < 1)) { out.status = 1; return out; }
    if (mass == 0) return out;
    out.key = std::log(-std::log(u)) - std::log(mass);
    out.index = index; out.rank = rank; out.valid = 1;
    return out;
}
// OR and lexicographic minimum are associative and commutative. Status is
// orthogonal to the winner: selecting a valid candidate cannot hide an error.
QCS_RACE_HD inline candidate combine(candidate a, const candidate& b) {
    unsigned status = a.status | b.status;
    if (b.valid && (!a.valid || b.key < a.key ||
        (b.key == a.key && (b.rank < a.rank || (b.rank == a.rank && b.index < a.index))))) a = b;
    a.status = status;
    return a;
}
} // namespace qcs::race_detail

#undef QCS_RACE_HD
