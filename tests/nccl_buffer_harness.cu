// Target-cluster checks and completed-work benchmark; never run implicitly.
#include "../src/qcs.cu"
#include <complex>

using value = std::complex<double>;
static void finish(qcs::simulator_core& core)
{
    ATLC_CHECK_CUDA(cudaStreamSynchronize, core.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, core.exchange_copy_stream);
}
static double maximum(double seconds)
{
    double result = 0;
    ATLC_CHECK_MPI(MPI_Allreduce, &seconds, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
}
static void check(qcs::simulator_core& c, const std::vector<value>& expected)
{
    finish(c);
    std::vector<qcs::complex_t> actual(c.num_states_local);
    ATLC_CHECK_CUDA(cudaMemcpy, actual.data(), c.state_data_device,
                    actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost);
    int good = 1;
    for (uint64_t i = 0; i < actual.size(); ++i)
    {
        uint64_t physical = uint64_t(c.proc_num) * actual.size() + i, logical = 0;
        for (int b = 0; b < c.num_qubits; ++b)
            logical |= ((physical >> b) & 1) << c.perm_p2l[b];
        if (std::abs(value(actual[i].real(), actual[i].imag()) - expected[logical]) > 1e-10)
            good = 0;
    }
    int all_good = 0;
    ATLC_CHECK_MPI(MPI_Allreduce, &good, &all_good, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_good) throw std::runtime_error("nonuniform complex-state comparison failed");
}
int main(int argc, char** argv)
{
    auto* sim = qcs_simulator_create_cxx();
    auto& c = *sim->core;
    try
    {
        if (qcs::exchange_nvshmem_registered_put || c.num_procs < 2 ||
            (c.num_procs & (c.num_procs - 1)))
            throw std::runtime_error("use NCCL exchange and a power-of-two rank count >= 2");
        const int local_bits = argc > 1 ? std::stoi(argv[1]) : 10;
        const uint64_t cap = argc > 2 ? std::stoull(argv[2]) : 6;
        const int repetitions = argc > 3 ? std::stoi(argv[3]) : 30;
        if (local_bits < 2 || local_bits > 20 || !cap || repetitions <= 0)
            throw std::runtime_error("arguments: local_bits [2,20], slot-total cap > 0, repetitions > 0");
        const int bits = local_bits + c.log_num_procs;
        qcs_simulator_set_num_qubits_cxx(sim, bits);
        for (int cycle = 0; cycle < 3; ++cycle)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            double begin = MPI_Wtime();
            c.allocate_memory(bits);
            finish(c);
            double initialization = maximum(MPI_Wtime() - begin);
            double allocation = maximum(c.allocation_seconds);
            double registration = maximum(c.registration_seconds);
            c.swap_buffer_total_length = std::min(cap, c.swap_buffer_total_length);
            std::vector<value> reference(c.num_states);
            for (uint64_t i = 0; i < reference.size(); ++i)
                reference[i] = value(double((i * 17) % 101) / 101, double((i * 13) % 97) / 97);
            std::vector<qcs::complex_t> initial(c.num_states_local);
            for (uint64_t i = 0; i < initial.size(); ++i)
            {
                auto v = reference[uint64_t(c.proc_num) * initial.size() + i];
                initial[i] = qcs::complex_t(v.real(), v.imag());
            }
            ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, initial.data(),
                initial.size() * sizeof(initial[0]), cudaMemcpyHostToDevice, c.stream);
            // Force the same physical exchange, twice restores the state. Low=0
            // exercises one-element regions; low=local_bits-1 exercises tails.
            for (int global = local_bits; global < bits; ++global)
            for (int low : {0, local_bits - 1, 1, local_bits - 1})
            {
                c.swap_target_local_list = {low};
                c.swap_target_global_list = {global};
                c.exchange_qubits();
                auto before = reference;
                for (uint64_t i = 0; i < reference.size(); ++i)
                {
                    uint64_t j = i;
                    if (((i >> low) ^ (i >> global)) & 1)
                        j ^= (UINT64_C(1) << low) | (UINT64_C(1) << global);
                    reference[i] = before[j];
                }
                check(c, reference);
                // Repeat exchanges; every pair restores the original state.
                for (int repeat = 0; repeat < 7; ++repeat)
                {
                    c.exchange_qubits();
                    if (!(repeat & 1)) check(c, before);
                    else check(c, reference);
                }
                reference = before;
            }
            // Exercise different chunk offsets and lengths within the allocation,
            // then restore capacity. Oracle state is unchanged after pairs.
            uint64_t saved_cap = c.swap_buffer_total_length;
            for (uint64_t changed : {UINT64_C(1), saved_cap})
            {
                c.swap_buffer_total_length = changed;
                c.exchange_qubits();
                c.exchange_qubits();
                check(c, reference);
            }
            c.reinitialize_mapping();
            check(c, reference);
            // Local and global gates exercise the normal mapping path.
            for (int round = 0; round < 2; ++round)
            {
                if (round)
                {
                    // Mapping reset is bookkeeping, not a physical permutation.
                    // Reload the original physical state and reset the oracle too.
                    finish(c);
                    c.reinitialize_mapping();
                    ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, initial.data(),
                        initial.size() * sizeof(initial[0]), cudaMemcpyHostToDevice, c.stream);
                    for (uint64_t i = 0; i < reference.size(); ++i)
                        reference[i] = value(double((i * 17) % 101) / 101, double((i * 13) % 97) / 97);
                    check(c, reference);
                }
                for (bit_num_t target : {bit_num_t(0), bit_num_t(bits - 1), bit_num_t(1)})
                {
                    qcs_simulator_gate_h_cxx(sim, &target, 1, nullptr, 0, nullptr, 0);
                    uint64_t mask = UINT64_C(1) << target;
                    for (uint64_t i = 0; i < reference.size(); ++i)
                        if (!(i & mask))
                        {
                            value a = reference[i], b = reference[i | mask];
                            reference[i] = (a + b) / std::sqrt(2.0);
                            reference[i | mask] = (a - b) / std::sqrt(2.0);
                        }
                    check(c, reference);
                }
            }
            c.swap_target_local_list = {local_bits - 1};
            c.swap_target_global_list = {local_bits};
            for (int warmup = 0; warmup < 5; ++warmup) c.exchange_qubits();
            finish(c);
            MPI_Barrier(MPI_COMM_WORLD);
            begin = MPI_Wtime();
            for (int rep = 0; rep < repetitions; ++rep) c.exchange_qubits();
            finish(c); // Includes copy-back on the independent stream.
            double elapsed = maximum(MPI_Wtime() - begin);
            if (!c.proc_num) printf("mode=%d cycle=%d initialization_s=%.9f allocation_s=%.9f registration_s=%.9f exchange_s=%.9f repetitions=%d cap=%llu\n",
                qcs::nccl_buffer_mode, cycle, initialization, allocation, registration,
                elapsed / repetitions, repetitions,
                static_cast<unsigned long long>(c.swap_buffer_total_length));
            c.free_memory();
        }
        qcs_simulator_destroy_cxx(sim);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "rank %d: %s\n", c.proc_num, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
