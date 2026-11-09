#include "../src/measurement_race_nccl.cuh"
#include <cstdio>
#include <stdexcept>
namespace r = qcs::race_detail;
namespace t = qcs::tiled_cdf_detail;
static void check(bool ok) { if (!ok) throw std::runtime_error("device race regression"); }
int main() {
    r::candidate *c, *out;
    t::rank_summary* summary;
    t::sample_result* sample;
    t::check(cudaMalloc(&c, 4*sizeof(*c)));
    t::check(cudaMalloc(&out, sizeof(*out)));
    t::check(cudaMalloc(&summary, sizeof(*summary)));
    t::check(cudaMalloc(&sample, sizeof(*sample)));
    // Each status is validated even if a different rank wins; equal scores
    // select the lower rank, retaining global indices beyond 32 bits.
    r::candidate host[4]{};
    host[0] = r::make_candidate(1, UINT64_C(1)<<50, 0, 0.5);
    host[1] = host[0]; host[1].rank = 1;
    for (unsigned status : {0u, 16u}) {
        host[3].status = status;
        t::check(cudaMemcpy(c, host, sizeof(host), cudaMemcpyHostToDevice));
        r::select_candidate<<<1,1>>>(c,4,out);
        r::candidate result;
        t::check(cudaMemcpy(&result,out,sizeof(result),cudaMemcpyDeviceToHost));
        check(result.index == host[0].index && result.rank == 0 && result.status == status);
    }
    // All zero candidates must reject globally.
    t::check(cudaMemset(c,0,sizeof(host)));
    r::select_candidate<<<1,1>>>(c,4,out);
    r::candidate result;
    t::check(cudaMemcpy(&result,out,sizeof(result),cudaMemcpyDeviceToHost));
    check(result.status && !result.valid);
    for (double mass : {0.0, 1e-300, 1e300}) {
        t::rank_summary hs{mass, mass == 0 ? unsigned(t::zero_mass) : 0u, 0};
        t::sample_result hr{17, hs.status};
        t::check(cudaMemcpy(summary,&hs,sizeof(hs),cudaMemcpyHostToDevice));
        t::check(cudaMemcpy(sample,&hr,sizeof(hr),cudaMemcpyHostToDevice));
        r::prepare_candidate<<<1,1>>>(summary,sample,out,7,UINT64_MAX,3,UINT64_C(1)<<40,0);
        t::check(cudaMemcpy(&result,out,sizeof(result),cudaMemcpyDeviceToHost));
        check(!result.status && bool(result.valid) == (mass > 0));
        if (result.valid) {
            auto expected = r::make_candidate(mass,3*(UINT64_C(1)<<40)+17,3,
                                              r::uniform(7,UINT64_MAX,3,1));
            check(result.index == expected.index && result.rank == 3);
            check(std::abs(result.key-expected.key) < 1e-10);
        }
    }
    for (double mass : {double(NAN), double(INFINITY), -1.0}) {
        t::rank_summary hs{mass, 0, 0};
        t::sample_result hr{0, 0};
        t::check(cudaMemcpy(summary,&hs,sizeof(hs),cudaMemcpyHostToDevice));
        t::check(cudaMemcpy(sample,&hr,sizeof(hr),cudaMemcpyHostToDevice));
        r::prepare_candidate<<<1,1>>>(summary,sample,out,0,0,0,1,0);
        t::check(cudaMemcpy(&result,out,sizeof(result),cudaMemcpyDeviceToHost));
        check(result.status && !result.valid);
    }
    // Failure sentinels never dereference absent GPU hierarchy results.
    r::prepare_candidate<<<1,1>>>(nullptr,nullptr,out,0,0,0,1,8);
    t::check(cudaMemcpy(&result,out,sizeof(result),cudaMemcpyDeviceToHost));
    check(result.status == 8 && !result.valid);
    t::check(cudaFree(sample)); t::check(cudaFree(summary));
    t::check(cudaFree(out)); t::check(cudaFree(c));
    std::puts("device race tests passed");
}
