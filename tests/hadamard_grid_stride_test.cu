// Private state access permits comparison of every complex amplitude.
#include "../src/qcs.cu"

static std::vector<qcs::complex_t> snapshot(qcs_simulator *sim)
{
    std::vector<qcs::complex_t> state(sim->core->num_states_local);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, state.data(), sim->core->state_data_device,
                    state.size() * sizeof(state[0]), cudaMemcpyDeviceToHost, sim->core->stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, sim->core->stream);
    return state;
}

int main()
{
    auto *sim = qcs_simulator_create_cxx();
    try
    {
        if (sim->core->num_procs != 1) throw std::runtime_error("requires one GPU/rank");
        const auto configured_mode = qcs::hadamard_detail::selected_mode();
        if (sim->core->hadamard_launch_mode != configured_mode)
            throw std::runtime_error("startup launch mode was not cached");
        if (sim->core->gpu_sm_count <= 0)
            throw std::runtime_error("GPU SM count was not cached");
        setenv("QCS_H_LAUNCH_MODE", "invalid-after-initialization", 1);
        if (sim->core->hadamard_launch_mode != configured_mode)
            throw std::runtime_error("cached mode changed with environment");
        for (int n : {1, 2, 5, 10, 20})
        {
            qcs_simulator_set_num_qubits_cxx(sim, n);
            if (!qcs_simulator_allocate_memory(sim)) throw std::runtime_error("allocation failed");
            qcs_simulator_set_zero_state(sim);
            auto initial = snapshot(sim);
            const double scale = 1.0 / std::sqrt(double(initial.size()) * 2.0);
            for (size_t i = 0; i < initial.size(); ++i)
                initial[i] = qcs::complex_t(std::sin(double(i) + 0.3) * scale,
                                           std::cos(double(i) * 0.7) * scale);
            for (int target : {0, n - 1})
            {
                for (int operation = 0; operation < 3; ++operation)
                {
                    if (operation == 2 && n == 1) continue;
                    std::vector<qcs::complex_t> expected;
                    for (const char *mode : {"baseline", "A", "B", "grid-stride"})
                    {
                        // Private test override compares modes on one simulator;
                        // public configuration is read only during setup.
                        sim->core->hadamard_launch_mode = !std::strcmp(mode, "A") ? qcs::hadamard_detail::mode::a :
                            !std::strcmp(mode, "B") ? qcs::hadamard_detail::mode::b :
                            !std::strcmp(mode, "grid-stride") ? qcs::hadamard_detail::mode::grid_stride :
                            qcs::hadamard_detail::mode::baseline;
                        ATLC_CHECK_CUDA(cudaMemcpyAsync, sim->core->state_data_device, initial.data(),
                                        initial.size() * sizeof(initial[0]), cudaMemcpyHostToDevice, sim->core->stream);
                        // Build a nontrivial superposition, then exercise H, X,
                        // and controlled H under each mode.
                        sim->core->operate_gate(qcs::gate::hadamard(), {target}, {}, {});
                        if (operation == 0)
                            sim->core->operate_gate(qcs::gate::hadamard(), {target}, {}, {});
                        else if (operation == 1)
                            sim->core->operate_gate(qcs::gate::x(), {target}, {}, {});
                        else
                            sim->core->operate_gate(qcs::gate::hadamard(), {(target + 1) % n}, {}, {target});
                        auto actual = snapshot(sim);
                        if (expected.empty()) expected = actual;
                        for (size_t i = 0; i < actual.size(); ++i)
                            if (cuda::std::abs(actual[i] - expected[i]) > 1e-12)
                                throw std::runtime_error("baseline amplitude mismatch");
                    }
                }
            }
            sim->core->free_memory();
        }
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception &error)
    {
        fprintf(stderr, "%s\n", error.what());
        qcs_simulator_destroy_cxx(sim);
        return 1;
    }
}
