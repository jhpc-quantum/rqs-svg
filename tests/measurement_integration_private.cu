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
    if (!c.rank_sampled_measurement && !c.tiled_cdf_measurement && !c.tiled_race_measurement) return;
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("rank sampling metadata regression"); };
    c.initialize_selected_state();
    {
        check(c.use_rank_sampling(dispatch_targets) == c.rank_sampled_measurement);
        c.measurement_counter = (UINT64_C(1) << 32) + 17;
        // Full-register correlated, nonuniform distribution spanning tiles and ranks.
        std::vector<int> all;
        for (int q = c.num_qubits - 1; q >= 0; --q) all.push_back(q);
        std::vector<bit_t> bits(all.size());
        int counts[4]{};
        const uint64_t basis[] = {0, 4097, c.num_states - 4098, c.num_states - 1};
        auto seed = c.rng_seed;
        c.rng_seed += c.proc_num; // Rank zero remains authoritative.
        for (int shot = 0; shot < 512; ++shot) {
            c.initialize_selected_state();
            std::vector<qcs::complex_t> state(c.num_states_local, qcs::complex_t(0, 0));
            for (int k = 0; k < 4; ++k)
                if (basis[k] / c.num_states_local == uint64_t(c.proc_num))
                    state[basis[k] % c.num_states_local] = qcs::complex_t(0, std::sqrt((k+1)/10.0));
            ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                state.size()*sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
            auto before = c.measurement_counter;
            c.measure_many(all, bits);
            check(c.measurement_counter == before + 1);
            uint64_t physical = 0;
            for (size_t j = 0; j < all.size(); ++j)
                physical |= uint64_t(bits[j]) << c.perm_l2p[all[j]];
            int k = 0;
            while (k < 4 && basis[k] != physical) ++k;
            check(k < 4);
            ++counts[k];
            uint64_t low, high;
            MPI_Allreduce(&physical, &low, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&physical, &high, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
            check(low == high);
            auto cached = bits;
            c.measure_many(all, bits);
            check(bits == cached && c.measurement_counter == before + 1);
        }
        for (int k = 0; k < 4; ++k) {
            double p = (k+1)/10.0;
            check(std::abs(counts[k]-512*p) < 7*std::sqrt(512*p*(1-p))+2);
        }
        c.rng_seed = seed;
        c.initialize_selected_state();
    }
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

    // A local pre-collective failure must reject on every rank and permit retry.
    c.initialize_selected_state();
    c.measurement_counter_exhausted = c.proc_num == c.num_procs - 1;
    auto before_local_failure = c.measurement_counter;
    std::fill(result.begin(), result.end(), bit_t(7));
    rejected = false;
    try { c.measure_many(targets, result); }
    catch (const std::runtime_error&) { rejected = true; }
    int failures = 0, local_failure = rejected;
    ATLC_CHECK_MPI(MPI_Allreduce, &local_failure, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    check(failures == c.num_procs);
    check(c.measurement_counter == before_local_failure && c.measured_mask_logical == 0);
    check(c.pending_projection_mask_logical == 0);
    check(std::all_of(result.begin(), result.end(), [](bit_t b) { return b == 7; }));
    c.measurement_counter_exhausted = false;
    c.measure_many(targets, result);
    check(c.measurement_counter == before_local_failure + 1);
    c.initialize_selected_state();

    if (c.tiled_race_measurement) {
        // One supported rank; finite effective masses near both double limits.
        // The smallest value stays normal to avoid GPU denormal mode dependence.
        for (double amplitude : {1e-150, 1e150}) {
            c.initialize_selected_state();
            std::vector<qcs::complex_t> state(c.num_states_local, qcs::complex_t(0, 0));
            if (c.proc_num == c.num_procs - 1) state[0] = qcs::complex_t(amplitude, 0);
            ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                state.size()*sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
            c.measure_many(targets, result);
            uint64_t physical = uint64_t(c.num_procs - 1) << c.num_qubits_local;
            for (size_t j = 0; j < targets.size(); ++j)
                check(result[j] == ((physical >> c.perm_l2p[targets[j]]) & 1));
        }
        c.initialize_selected_state();
    }

    // Invalid effective norms must fail collectively without consuming a draw.
    for (double amplitude : {0.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
        std::vector<qcs::complex_t> state(c.num_states_local, qcs::complex_t(0, 0));
        // Nonfinite norms originate on just one rank; other ranks remain valid.
        if (amplitude != 0.0 && c.proc_num != c.num_procs - 1)
            state[0] = qcs::complex_t(1, 0);
        else
            state[0] = qcs::complex_t(amplitude, 0);
        ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                        state.size() * sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
        auto before = c.measurement_counter;
        rejected = false;
        std::fill(result.begin(), result.end(), bit_t(7));
        try { c.measure_many(targets, result); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected && c.measurement_counter == before && c.measured_mask_logical == 0);
        check(c.pending_projection_mask_logical == 0);
        check(std::all_of(result.begin(), result.end(), [](bit_t b) { return b == 7; }));
    }
    c.initialize_selected_state();
}
#define QCS_PRIVATE_RANK_SAMPLING_CASES
#include "measurement_integration_test.cpp"
