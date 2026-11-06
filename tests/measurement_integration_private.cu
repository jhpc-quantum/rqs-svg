// Run public API coverage and private RNG/metadata edge cases in one MPI lifetime.
#include "qcs.cu"
#include <stdexcept>

static void check_rank_sampling_metadata(qcs_simulator* sim)
{
    auto& c = *sim->core;
    // Explicit selection must override workload size, including cached requests.
    std::vector<int> dispatch_targets{0};
    if (!c.auto_measurement) {
        if (c.use_rank_sampling(dispatch_targets) != c.rank_sampled_measurement)
            throw std::runtime_error("explicit backend dispatch regression");
    } else if (c.use_rank_sampling(dispatch_targets)) {
        throw std::runtime_error("auto single-target dispatch regression");
    }
    if (!c.rank_sampled_measurement) return;
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("rank sampling metadata regression"); };
    c.initialize_selected_state();
    if (c.num_procs > 1) {
        // Direct physical preparation keeps the superposed bit global: gates
        // would otherwise be free to localize it before the measurement.
        std::vector<qcs::complex_t> state(c.num_states_local, qcs::complex_t(0, 0));
        state[0] = qcs::complex_t(1.0 / std::sqrt(double(c.num_procs)), 0);
        ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                        state.size() * sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
        int local = c.perm_p2l[0];
        bit_num_t global = c.perm_p2l[c.num_qubits_local];
        std::vector<bit_t> bit(1);
        c.measure_many({local}, bit);
        check(bit[0] == 0);
        check(c.measured_mask_logical == (UINT64_C(1) << local));
        qcs_pauli_operator px = QCS_PAULI_X;
        check(std::abs(c.expectation_pauli(&global, &px, 1, 1.0) - 1.0) < 1e-11);
        c.initialize_selected_state();
    }
    std::vector<int> targets{9, 2, 7, 0, 5, 1, 8, 3, 6, 4};
    std::vector<bit_t> result(targets.size());
    auto cursor = c.measurement_counter;
    c.measure_many(targets, result);
    check(c.measurement_counter == cursor + 1);
    check(c.measured_mask_logical == ((UINT64_C(1) << 10) - 1));
    check(c.pending_projection_mask_logical == c.measured_mask_logical);
    c.measure_many(targets, result);
    check(c.measurement_counter == cursor + 1);
    c.initialize_selected_state();
    c.measurement_counter = UINT64_MAX;
    c.measure_many(targets, result);
    check(c.measurement_counter_exhausted);
    c.measure_many(targets, result); // Recorded values work even after exhaustion.
    c.initialize_selected_state();
    bool rejected = false;
    try { c.measure_many(targets, result); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected && c.measured_mask_logical == 0);
    c.measurement_counter = cursor + 2;
    c.measurement_counter_exhausted = false;

    // Invalid effective norms must fail collectively without consuming a draw.
    for (double amplitude : {0.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
        std::vector<qcs::complex_t> state(c.num_states_local, qcs::complex_t(amplitude, 0));
        ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                        state.size() * sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
        auto before = c.measurement_counter;
        rejected = false;
        try { c.measure_many(targets, result); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected && c.measurement_counter == before && c.measured_mask_logical == 0);
    }
    c.initialize_selected_state();
}
#define QCS_PRIVATE_RANK_SAMPLING_CASES
#include "measurement_integration_test.cpp"
