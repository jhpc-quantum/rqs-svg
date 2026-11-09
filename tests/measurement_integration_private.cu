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
    if (c.auto_measurement) {
        // Dispatch must be independent of rank-local RNG and recorded outcomes.
        const auto seed = c.rng_seed;
        const auto value = c.measured_value_logical;
        const bool selected = c.use_rank_sampling(dispatch_targets);
        c.rng_seed += c.proc_num + 1;
        c.measured_value_logical ^= c.measured_mask_logical;
        const bool repeated = c.use_rank_sampling(dispatch_targets);
        c.rng_seed = seed;
        c.measured_value_logical = value;
        if (selected != repeated)
            throw std::runtime_error("auto depends on rank-local sampling data");
    }
    if (!c.rank_sampled_measurement && !c.tiled_cdf_measurement && !c.tiled_race_measurement && !c.tiled_race_nccl_measurement) return;
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
    if (c.num_procs > 1 && (c.rank_sampled_measurement || c.tiled_cdf_measurement)) {
        // Vary every cached global prefix and outcome: half the ranks through
        // a single eligible rank, with rank zero both eligible and excluded.
        // Reinitialization reuses persistent workspaces between these cases.
        for (int cached = 1; cached <= c.num_qubits - c.num_qubits_local; ++cached) {
            for (int outcome = 0; outcome < (1 << cached); ++outcome) {
                c.initialize_selected_state();
                uint64_t cached_mask = 0, cached_value = 0;
                for (int p = 0; p < cached; ++p) {
                    int q = c.perm_p2l[c.num_qubits_local + p];
                    cached_mask |= UINT64_C(1) << q;
                    if ((outcome >> p) & 1) cached_value |= UINT64_C(1) << q;
                    (((outcome >> p) & 1) ? c.measured_1_qubit_num_logical_list :
                        c.measured_0_qubit_num_logical_list).push_back(q);
                }
                c.measured_mask_logical = cached_mask;
                c.measured_value_logical = cached_value;
                c.pending_projection_mask_logical = cached_mask;
                bool active = (c.proc_num & ((1 << cached) - 1)) == outcome;
                std::vector<qcs::complex_t> state(c.num_states_local,
                    qcs::complex_t(active ? 0 : std::numeric_limits<double>::quiet_NaN(), 0));
                if (active) state[1] = qcs::complex_t(1, 0);
                ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                    state.size()*sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
                auto cursor = c.measurement_counter;
                std::vector<bit_t> bits(1, bit_t(7));
                // Recoverable failure on an excluded rank must still propagate.
                c.measurement_counter_exhausted = !active && c.proc_num == (outcome == 0 ? 1 : 0);
                bool rejected = false;
                try { c.measure_many({c.perm_p2l[0]}, bits); }
                catch (const std::runtime_error&) { rejected = true; }
                check(rejected && c.measurement_counter == cursor && bits[0] == 7);
                c.measurement_counter_exhausted = false;
                c.measure_many({c.perm_p2l[0]}, bits);
                check(bits[0] == 1 && c.measurement_counter == cursor + 1);
                check(c.measured_value_logical == (cached_value | (UINT64_C(1) << c.perm_p2l[0])));
            }
        }
        c.initialize_selected_state();
        auto mapping = c.initial_perm_p2l;
        auto seed = c.rng_seed;
        for (bool reversed : {false, true}) {
            for (int p = 0; p < c.num_qubits; ++p)
                c.initial_perm_p2l[p] = reversed ? c.num_qubits - 1 - p : p;
            c.reinitialize_mapping();
            std::vector<int> globals;
            for (int p = c.num_qubits - 1; p >= c.num_qubits_local; --p)
                globals.push_back(c.perm_p2l[p]);
            int local = c.perm_p2l[0];
            for (bool cached_local : {false, true}) for (bool cached_global : {false, true}) for (int mode = 0; mode < 6; ++mode) {
                // Compare the fast path to the unchanged local-index path with
                // the same counter, including rank-zero's authoritative draw.
                for (int shot = 0; shot < 32; ++shot) {
                    uint64_t cursor = c.measurement_counter;
                    std::vector<bit_t> fast;
                    for (bool reference : {false, true}) {
                        c.initialize_selected_state();
                        c.measurement_counter = cursor;
                        c.rng_seed = seed + c.proc_num;
                        std::vector<qcs::complex_t> state(c.num_states_local, qcs::complex_t(0, 0));
                        double amplitude = mode == 0 ? std::sqrt(double(c.proc_num + 1)) :
                            mode == 1 ? (c.proc_num == c.num_procs - 1 ? 1.0 : 0.0) :
                            mode == 2 ? 0.0 : mode == 3 ? std::numeric_limits<double>::quiet_NaN() :
                            mode == 4 ? std::numeric_limits<double>::infinity() : 1.0;
                        if (mode >= 3 && mode <= 4 && c.proc_num != c.num_procs - 1) amplitude = 1;
                        state[cached_local ? 1 : 0] = qcs::complex_t(amplitude, 0);
                        ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                            state.size()*sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
                        if (cached_local) {
                            c.measured_mask_logical = UINT64_C(1) << local;
                            c.measured_value_logical = c.measured_mask_logical;
                            c.pending_projection_mask_logical = c.measured_mask_logical;
                            c.measured_1_qubit_num_logical_list.push_back(local);
                        }
                        if (cached_global && globals.size() > 1) {
                            uint64_t bit = UINT64_C(1) << globals.back();
                            c.measured_mask_logical |= bit;
                            c.measured_value_logical |= bit;
                            c.pending_projection_mask_logical |= bit;
                            c.measured_1_qubit_num_logical_list.push_back(globals.back());
                        }
                        // A failure on a zero-mass rank must still reject globally.
                        c.measurement_counter_exhausted = mode == 5 && c.proc_num == 0;
                        auto targets = globals;
                        if (cached_local) targets.insert(targets.begin(), local);
                        if (reference) targets.push_back(c.perm_p2l[1]);
                        std::vector<bit_t> bits(targets.size(), bit_t(7));
                        auto mask = c.measured_mask_logical;
                        auto value = c.measured_value_logical;
                        auto ones = c.measured_1_qubit_num_logical_list;
                        auto zeros = c.measured_0_qubit_num_logical_list;
                        bool rejected = false;
                        try { c.measure_many(targets, bits); }
                        catch (const std::runtime_error&) { rejected = true; }
                        check(rejected == (mode >= 2));
                        if (rejected) {
                            check(c.measurement_counter == cursor && c.measured_mask_logical == mask);
                            check(c.measured_value_logical == value && c.pending_projection_mask_logical == mask);
                            check(c.measured_1_qubit_num_logical_list == ones && c.measured_0_qubit_num_logical_list == zeros);
                            check(std::all_of(bits.begin(), bits.end(), [](bit_t b) { return b == 7; }));
                        } else {
                            check(c.measurement_counter == cursor + 1);
                            if (cached_local) check(bits[0] == 1);
                            if (mode == 1)
                                for (size_t j = cached_local; j < globals.size() + cached_local; ++j) check(bits[j] == 1);
                            if (!reference) fast = bits;
                            else check(std::equal(fast.begin(), fast.end(), bits.begin()));
                            auto cached = bits;
                            c.measure_many(targets, bits);
                            check(bits == cached && c.measurement_counter == cursor + 1);
                        }
                        c.measurement_counter_exhausted = false;
                    }
                }
            }
        }
        c.rng_seed = seed;
        c.initial_perm_p2l = mapping;
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

    if (c.tiled_race_measurement || c.tiled_race_nccl_measurement) {
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
