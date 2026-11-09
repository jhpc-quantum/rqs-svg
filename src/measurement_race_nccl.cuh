#pragma once
#include "measurement_race.hpp"
#include "measurement_tiled_cdf.cuh"

namespace qcs::race_detail {
// The local sample kernel and this kernel run after hierarchy construction on
// the same stream. No hierarchy or candidate is read back by the host.
__global__ void prepare_candidate(const tiled_cdf_detail::rank_summary* summary,
                                  const tiled_cdf_detail::sample_result* sample,
                                  candidate* output, uint64_t seed, uint64_t cursor,
                                  uint32_t rank, uint64_t local_states, unsigned failure) {
    candidate out{};
    out.status = failure;
    if (!failure) {
        unsigned status = summary->status | sample->status;
        if ((status & ~unsigned(tiled_cdf_detail::zero_mass)) ||
            !isfinite(summary->mass) || summary->mass < 0 ||
            (summary->mass > 0 && sample->index >= local_states)) out.status = 1;
        else if (summary->mass > 0) {
            double u = uniform(seed, cursor, rank, 1);
            out.key = log(-log(u)) - log(summary->mass);
            out.index = uint64_t(rank) * local_states + sample->index;
            out.rank = rank;
            out.valid = 1;
        }
    }
    *output = out;
}
// A serial metadata scan is intentional for the experimental baseline. NCCL
// AllGather order is rank order; combine also explicitly breaks ties by rank.
__global__ void select_candidate(const candidate* candidates, int ranks, candidate* result) {
    candidate winner{};
    for (int r = 0; r < ranks; ++r) winner = combine(winner, candidates[r]);
    if (!winner.valid) winner.status |= 1;
    *result = winner;
}
} // namespace qcs::race_detail
