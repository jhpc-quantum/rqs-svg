// Single-GPU regression coverage; includes the core only to inspect amplitudes.
#include "../src/qcs.cu"
#include <array>
#include <complex>
#include <unistd.h>

static void require_initial_state(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

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
    qcs_simulator *sim = qcs_simulator_create_cxx();
    char original_directory[4096];
    char directory[] = "/tmp/qcs-initial-state-XXXXXX";
    try
    {
        require_initial_state(qcs_simulator_get_num_procs_cxx(sim) == 1,
                              "run this regression with one MPI rank and one GPU");
        require_initial_state(getcwd(original_directory, sizeof(original_directory)) != nullptr,
                              "getcwd failed");
        require_initial_state(mkdtemp(directory) != nullptr && chdir(directory) == 0,
                              "temporary directory failed");
        constexpr int n = 10;
        constexpr int count = 1 << n;
        std::vector<qcs::complex_t> loaded(count, qcs::complex_t(0));
        loaded[37] = 1;
        FILE *input = fopen("statevector_input.bin", "wb");
        require_initial_state(input != nullptr, "fixture open failed");
        require_initial_state(fwrite(loaded.data(), sizeof(loaded[0]), count, input) == count,
                              "fixture write failed");
        fclose(input);
        qcs_simulator_set_num_qubits_cxx(sim, n);
        qcs_simulator_set_num_clbits_cxx(sim, n);
        require_initial_state(!qcs_simulator_allocate_memory_with_initial_state(
            sim, static_cast<qcs_initial_state>(999)), "invalid enum accepted");
        const qcs_initial_state methods[] = {
            QCS_INITIAL_STATE_ZERO, QCS_INITIAL_STATE_FLAT, QCS_INITIAL_STATE_ENTANGLED,
            QCS_INITIAL_STATE_SEQUENTIAL, QCS_INITIAL_STATE_USE_CURAND,
            QCS_INITIAL_STATE_LOAD_STATEVECTOR
        };
        for (auto method : methods)
        {
            // Allocation must not invoke the file loader, even when selected.
            // Do not inspect unspecified allocation contents to test this contract.
            if (method == QCS_INITIAL_STATE_LOAD_STATEVECTOR)
                require_initial_state(rename("statevector_input.bin", "deferred.bin") == 0,
                                      "fixture rename failed");
            // Exercise the original C symbol as well as the explicit C entry point.
            require_initial_state(method == QCS_INITIAL_STATE_ZERO
                ? qcs_simulator_allocate_memory(sim)
                : qcs_simulator_allocate_memory_with_initial_state(sim, method),
                "allocation failed");
            if (method == QCS_INITIAL_STATE_LOAD_STATEVECTOR)
                require_initial_state(rename("deferred.bin", "statevector_input.bin") == 0,
                                      "fixture restore failed");
            require_initial_state(qcs_simulator_set_zero_state(sim),
                                  "explicit initialization failed");
            const auto initial = snapshot(sim);
            bool nonzero = false;
            for (int i = 0; i < count; ++i)
            {
                require_initial_state(std::isfinite(initial[i].real()) &&
                                      std::isfinite(initial[i].imag()), "nonfinite amplitude");
                nonzero |= cuda::std::abs(initial[i]) > 0;
                if (method == QCS_INITIAL_STATE_USE_CURAND) continue;
                // Preserve the current single-rank entangled initializer behavior.
                const double expected = method == QCS_INITIAL_STATE_FLAT ? 1.0 :
                    method == QCS_INITIAL_STATE_SEQUENTIAL ? double(i) :
                    method == QCS_INITIAL_STATE_LOAD_STATEVECTOR ? double(i == 37) :
                    double(i == 0);
                require_initial_state(cuda::std::abs(initial[i] - qcs::complex_t(expected)) < 1e-12,
                                      "wrong initial amplitudes");
            }
            require_initial_state(nonzero, "empty initial state");
            for (int shot = 0; shot < 3; ++shot)
            {
                // Leave measurement metadata, classical bits and amplitudes dirty.
                qcs_simulator_measure_to_clbit_cxx(sim, 0, 0);
                const bit_num_t target = 1;
                qcs_simulator_gate_x_cxx(sim, &target, 1, nullptr, 0, nullptr, 0);
                std::array<bit_num_t, n> mapping{};
                for (int i = 0; i < n; ++i) mapping[i] = n - i - 1;
                qcs_simulator_set_mapping_cxx(sim, mapping.data(), n);
                require_initial_state(qcs_simulator_reinitialize_mapping(sim) &&
                                      qcs_simulator_set_zero_state(sim) &&
                                      qcs_simulator_reset_clbits(sim), "shot reset failed");
                require_initial_state(sim->core->measured_mask_logical == 0 &&
                                      sim->core->pending_projection_mask_logical == 0,
                                      "measurement metadata not reset");
                const auto restored = snapshot(sim);
                for (int i = 0; i < count; ++i)
                    require_initial_state(cuda::std::abs(restored[i] - initial[i]) < 1e-12,
                                          "selected initializer not restored");
                require_initial_state(qcs_simulator_read_cxx(sim, 0) == 0,
                                      "classical bit not reset");
                require_initial_state(sim->core->perm_p2l[0] == n - 1,
                                      "mapping not reset");
            }
            sim->core->free_memory();
            ATLC_CHECK_CUDA(cudaStreamSynchronize, sim->core->stream);
        }
        unlink("statevector_input.bin");
        chdir(original_directory);
        rmdir(directory);
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception &error)
    {
        fprintf(stderr, "initial-state regression: %s\n", error.what());
        qcs_simulator_destroy_cxx(sim);
        return 1;
    }
}
