// Private access follows the existing SWAP harness; no production test hooks.
#ifndef QCS_EXCHANGE_SOURCE
#define QCS_EXCHANGE_SOURCE "../src/qcs.cu"
#endif
#include QCS_EXCHANGE_SOURCE
#include <complex>
#include <iostream>
#include <numeric>

using value = std::complex<double>;
using indices = std::vector<int>;
__global__ void consume_exchange(qcs::complex_t* state, uint64_t size)
{
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) state[i] = -state[i];
}
static uint64_t transpose(uint64_t index, const indices& b, const indices& g)
{
    for (size_t i = 0; i < b.size(); ++i)
        if (((index >> b[i]) ^ (index >> g[i])) & 1)
            index ^= (UINT64_C(1) << b[i]) | (UINT64_C(1) << g[i]);
    return index;
}
static void check(qcs::simulator_core& c, const std::vector<value>& physical,
                  const std::vector<value>& logical)
{
    std::vector<qcs::complex_t> actual(c.num_states_local);
    // Only the owner stream is joined: this tests the final copy-back dependency.
    ATLC_CHECK_CUDA(cudaMemcpyAsync, actual.data(), c.state_data_device,
        actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost, c.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
    int good = 1, all_good = 0;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        uint64_t p = uint64_t(c.proc_num) * actual.size() + i, l = 0;
        for (int bit = 0; bit < c.num_qubits; ++bit)
            l |= ((p >> bit) & 1) << c.perm_p2l[bit];
        value got(actual[i].real(), actual[i].imag());
        if (!std::isfinite(got.real()) || !std::isfinite(got.imag()) ||
            std::abs(got - physical[p]) > 1e-12 || std::abs(got - logical[l]) > 1e-12)
            good = 0;
        for (int bit = 0; bit < c.num_qubits; ++bit)
            if (c.perm_l2p[c.perm_p2l[bit]] != bit) good = 0;
    }
    ATLC_CHECK_MPI(MPI_Allreduce, &good, &all_good, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_good) throw std::runtime_error("physical/logical exchange mismatch or nonfinite value");
}
static void exchange(qcs::simulator_core& c, std::vector<value>& expected,
                     const indices& b, const indices& g)
{
    c.swap_target_local_list = b; c.swap_target_global_list = g;
    c.swap_target_local_logical_list.resize(b.size());
    c.swap_target_global_logical_list.resize(b.size());
    c.target_qubit_num_logical_list.clear();
    for (size_t i = 0; i < b.size(); ++i)
    {
        c.swap_target_local_logical_list[i] = c.perm_p2l[b[i]];
        c.swap_target_global_logical_list[i] = c.perm_p2l[g[i]];
    }
    c.exchange_qubits();
    c.update_qubit_permutation_mapping();
    auto before = expected;
    for (uint64_t p = 0; p < expected.size(); ++p)
        expected[p] = before[transpose(p, b, g)];
}
int main()
{
    auto* sim = qcs_simulator_create_cxx();
    try
    {
        auto& c = *sim->core;
        if (qcs::exchange_nvshmem_registered_put || (c.num_procs & (c.num_procs - 1)))
            throw std::runtime_error("requires NCCL and power-of-two ranks");
        const int n = 8;
        qcs_simulator_set_num_qubits_cxx(sim, n + c.log_num_procs);
        qcs_simulator_allocate_memory_cxx(sim);
        uint64_t allocated_capacity = c.swap_buffer_total_length;
        uint64_t checks = 0;
        for (uint64_t capacity : {1, 2, 6, 10})
        {
            if (allocated_capacity < capacity) throw std::runtime_error("insufficient test staging capacity");
            c.swap_buffer_total_length = capacity; // Same effective capacity on every rank, including mode 0.
            for (int k = 0; k <= std::min(3, c.log_num_procs); ++k)
            for (int layout = 0; layout < 3; ++layout)
            {
                c.reinitialize_mapping();
                indices b, g;
                for (int i = 0; i < k; ++i)
                {
                    b.push_back(layout == 0 ? 3 + i : (layout == 1 ? 1 + 2*i : n - 1 - 2*i));
                    // With >=16 ranks, layout 1 leaves an unexchanged rank bit between targets.
                    int rank_bit = layout == 1 && c.log_num_procs > k && i == k - 1 ? k : i;
                    g.push_back(n + rank_bit);
                }
                if (layout == 2) { std::reverse(b.begin(), b.end()); std::reverse(g.begin(), g.end()); }
                std::vector<value> physical(c.num_states), logical(c.num_states);
                for (uint64_t i = 0; i < physical.size(); ++i)
                    physical[i] = logical[i] = value(double(i + 1) / physical.size(),
                                                      double(int(i % 29) - 14) / 29);
                std::vector<qcs::complex_t> input(c.num_states_local);
                for (uint64_t i = 0; i < input.size(); ++i)
                {
                    auto v = physical[uint64_t(c.proc_num) * input.size() + i];
                    input[i] = qcs::complex_t(v.real(), v.imag());
                }
                ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, input.data(),
                    input.size() * sizeof(input[0]), cudaMemcpyHostToDevice, c.stream);
                exchange(c, physical, b, g);
                // Consume immediately, before any host or copy-stream synchronization.
                consume_exchange<<<(c.num_states_local + 127) / 128, 128, 0, c.stream>>>(
                    c.state_data_device, c.num_states_local);
                ATLC_CHECK_CUDA(cudaGetLastError);
                for (auto& v : physical) v = -v;
                for (auto& v : logical) v = -v;
                check(c, physical, logical); ++checks;
                exchange(c, physical, b, g); // Inverse, including metadata.
                check(c, physical, logical); ++checks;
                // Back-to-back calls exercise slot initialization across exchanges.
                exchange(c, physical, b, g);
                auto reversed_b = b, reversed_g = g;
                std::reverse(reversed_b.begin(), reversed_b.end());
                std::reverse(reversed_g.begin(), reversed_g.end());
                exchange(c, physical, reversed_b, reversed_g);
                check(c, physical, logical); ++checks;
            }
        }
        c.swap_buffer_total_length = allocated_capacity;
        if (!c.proc_num) std::cout << "PASS: " << checks << " physical/logical exchange checks; pipeline="
                                 << qcs::exchange_pipeline_copy_back << '\n';
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "exchange harness: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
