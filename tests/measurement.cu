// Include the implementation so tests exercise measure_qubits<3> directly
// without adding a public API solely for testing.
#include "../src/qcs.cu"
#include <iostream>
#include <limits>

static void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void upload(qcs_simulator *sim, const std::vector<qcs::complex_t> &logical)
{
    auto &core = *sim->core;
    core.discard_measurement_state();
    std::vector<qcs::complex_t> local(core.num_states_local);
    for (uint64_t i = 0; i < core.num_states_local; ++i)
    {
        uint64_t physical = i | (uint64_t(core.proc_num) << core.num_qubits_local);
        uint64_t index = 0;
        for (int p = 0; p < core.num_qubits; ++p)
            index |= ((physical >> p) & 1) << core.perm_p2l[p];
        local[i] = logical[index];
    }
    ATLC_CHECK_CUDA(cudaMemcpyAsync, core.state_data_device, local.data(), local.size() * sizeof(local[0]), cudaMemcpyHostToDevice, core.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, core.stream);
}

static std::vector<qcs::complex_t> download(qcs_simulator *sim)
{
    auto &core = *sim->core;
    std::vector<qcs::complex_t> local(core.num_states_local);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, local.data(), core.state_data_device, local.size() * sizeof(local[0]), cudaMemcpyDeviceToHost, core.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, core.stream);
    return local;
}

static uint64_t bits(uint64_t value, const std::vector<int> &qubits)
{
    uint64_t index = 0;
    for (size_t i = 0; i < qubits.size(); ++i)
        index |= ((value >> i) & 1) << qubits[i];
    return index;
}

static void rank_agreement(uint64_t value)
{
    uint64_t lo, hi;
    ATLC_CHECK_MPI(MPI_Allreduce, &value, &lo, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    ATLC_CHECK_MPI(MPI_Allreduce, &value, &hi, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    require(lo == hi, "measurement differs between ranks");
}

static void check_basis(qcs_simulator *sim)
{
    auto &core = *sim->core;
    // All ordered nonadjacent/adjacent pairs: local, mixed and global when
    // the rank count permits. Test the C API, not just the internal template.
    for (int a = 0; a < core.num_qubits; ++a)
    for (int b = 0; b < core.num_qubits; ++b)
    {
        if (a == b) continue;
        bit_num_t qubits[] = {bit_num_t(a), bit_num_t(b)};
        for (int outcome = 0; outcome < 4; ++outcome)
        {
            std::vector<qcs::complex_t> state(core.num_states);
            state[bits(outcome, {a, b})] = 3; // deliberately not normalized
            upload(sim, state);
            auto const mapping = core.perm_l2p;
            auto const before = download(sim);
            bit_t result[2]{};
            require(qcs_simulator_measure_many(sim, qubits, 2, result), "basis API failed");
            require(result[0] + 2 * result[1] == outcome, "basis bit order");
            require(core.perm_l2p == mapping, "measurement changed mapping");
            require(download(sim) == before, "measurement eagerly projected state");
            uint64_t mask = (UINT64_C(1) << a) | (UINT64_C(1) << b);
            require(core.measured_mask_logical == mask && core.pending_projection_mask_logical == mask, "measurement masks");
            auto const engine = core.engine;
            auto *buffer = core.measure_norm_device;
            require(core.measure_qubits<2>({b, a}) == uint64_t((outcome >> 1) | ((outcome & 1) << 1)), "repeated measurement");
            require(core.engine == engine && core.measure_norm_device == buffer, "repeated measurement did work");
            bit_t single = 9;
            require(qcs_simulator_measure(sim, a, &single) && single == result[0], "single API consistency");
        }
    }
}

static void check_three(qcs_simulator *sim)
{
    auto &core = *sim->core;
    // These physical triples include every feasible local/global count for
    // 1, 2, 4 and 8 ranks; target ordering is intentionally nonmonotonic.
    for (auto physical : {std::array<int, 3>{2, 0, 1}, {5, 0, 2}, {4, 1, 5}, {5, 3, 4}})
    {
        std::array<int, 3> group;
        for (int i = 0; i < 3; ++i) group[i] = core.perm_p2l[physical[i]];
        std::vector<int> qubits(group.begin(), group.end());
        for (int outcome = 0; outcome < 8; ++outcome)
        {
            std::vector<qcs::complex_t> state(core.num_states);
            state[bits(outcome, qubits)] = 2;
            upload(sim, state);
            require(core.measure_qubits<3>(group) == uint64_t(outcome), "three-qubit basis order");
        }
        int ones = 0;
        for (int shot = 0; shot < 256; ++shot)
        {
            std::vector<qcs::complex_t> state(core.num_states);
            state[0] = state[bits(7, qubits)] = std::sqrt(0.5);
            upload(sim, state);
            auto outcome = core.measure_qubits<3>(group);
            rank_agreement(outcome);
            require(outcome == 0 || outcome == 7, "GHZ correlation");
            ones += outcome == 7;
            // A measured and an unmeasured qubit in one three-bit group.
            upload(sim, state);
            int first = core.measure_qubit(group[1]);
            require(core.measure_qubits<3>(group) == (first ? 7u : 0u), "partially measured GHZ");
        }
        require(ones > 80 && ones < 176, "GHZ sampling distribution");
    }
}

static void check_distributions(qcs_simulator *sim)
{
    auto &core = *sim->core;
    constexpr int shots = 2048;
    for (auto physical : {std::array<int, 2>{2, 0}, {5, 0}, {5, 4}})
    {
        std::vector<int> qubits{core.perm_p2l[physical[0]], core.perm_p2l[physical[1]]};
        for (auto weights : {std::array<double, 4>{0.5, 0, 0, 0.5}, // Bell
                             {0.25, 0.25, 0.25, 0.25},             // |++>
                             {0.1, 0.2, 0.3, 0.4}})
        {
            std::vector<qcs::complex_t> state(core.num_states);
            for (int o = 0; o < 4; ++o) state[bits(o, qubits)] = std::sqrt(7 * weights[o]);
            int counts[2][4]{};
            for (int mode = 0; mode < 2; ++mode)
            for (int shot = 0; shot < shots; ++shot)
            {
                upload(sim, state);
                uint64_t outcome;
                if (mode == 0)
                    outcome = core.measure_qubits<2>({qubits[0], qubits[1]});
                else
                {
                    int first = core.measure_qubit(qubits[0]);
                    outcome = first | (core.measure_qubit(qubits[1]) << 1);
                }
                require(weights[outcome] > 0, "zero-probability outcome");
                ++counts[mode][outcome];
            }
            for (int o = 0; o < 4; ++o)
            {
                for (int mode = 0; mode < 2; ++mode)
                    require(std::abs(counts[mode][o] / double(shots) - weights[o]) < 0.06, "incorrect distribution");
                require(std::abs(counts[0][o] - counts[1][o]) / double(shots) < 0.08, "sequential/joint disagreement");
            }
        }
    }
}

static void check_semantics(qcs_simulator *sim)
{
    auto &core = *sim->core;
    bit_num_t a = core.perm_p2l[5], b = core.perm_p2l[0], c = core.perm_p2l[2];
    std::vector<qcs::complex_t> state(core.num_states);
    state[0] = state[(1 << a) | (1 << b)] = std::sqrt(0.5);
    upload(sim, state);
    auto before = download(sim);
    uint64_t first = core.measure_qubits<2>({a, b});
    require(first == 0 || first == 3, "Bell measurement");
    require(download(sim) == before, "deferred projection changed amplitudes");
    core.operate_gate(qcs::gate::x(), {a}, {}, {});
    require((core.pending_projection_mask_logical & (UINT64_C(1) << a)) == 0, "gate did not release projection");
    require(core.measure_qubits<2>({a, b}) == (first ^ 1), "X after measurement");
    // A non-basis gate must release the recorded value and sample anew.
    int ones = 0;
    for (int shot = 0; shot < 256; ++shot)
    {
        core.operate_gate(qcs::gate::hadamard(), {a}, {}, {});
        ones += core.measure_qubit(a);
        require(core.measure_qubit(b) == int(first >> 1), "unrelated measured qubit changed");
    }
    require(ones > 80 && ones < 176, "H after measurement stayed deterministic");

    upload(sim, state);
    bit_num_t qubits[] = {a, b, c}, clbits[] = {4, 1, 3};
    bit_t results[] = {9, 9, 9};
    require(qcs_simulator_measure_many_to_clbits(sim, qubits, 3, clbits, 3, results), "classical API failed");
    require(results[0] == results[1] && results[2] == 0, "odd group correlation");
    for (int i = 0; i < 3; ++i)
        require(sim->clbits[clbits[i]] == bool(results[i]), "classical destination order");
    require(qcs_simulator_measure_many(sim, nullptr, 0, nullptr), "empty request");
    bit_num_t duplicate[] = {a, a};
    require(!qcs_simulator_measure_many(sim, duplicate, 2, results), "duplicate target accepted");
    duplicate[1] = core.num_qubits;
    require(!qcs_simulator_measure_many(sim, duplicate, 2, results), "out of range accepted");
    bool rejected = false;
    try { core.measure_qubits<3>({a, b, a}); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "template duplicate target accepted");

    for (double value : {0.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        state.assign(core.num_states, qcs::complex_t(0));
        state[0] = value;
        upload(sim, state);
        rejected = false;
        try { core.measure_qubits<2>({a, b}); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected && core.measured_mask_logical == 0, "invalid norm recorded a measurement");
    }
}

int main()
{
    qcs_simulator *sim = nullptr;
    try
    {
        sim = qcs_simulator_create_cxx();
        require(sim->core->num_procs == 1 || sim->core->num_procs == 2 || sim->core->num_procs == 4 || sim->core->num_procs == 8,
                "run with 1, 2, 4 or 8 MPI ranks");
        qcs_simulator_set_num_qubits_cxx(sim, 6);
        qcs_simulator_set_num_clbits_cxx(sim, 6);
        qcs_simulator_allocate_memory_cxx(sim);
        sim->core->engine.seed(1234);
        for (auto mapping : {std::vector<int>{0, 1, 2, 3, 4, 5}, {2, 5, 0, 4, 1, 3}})
        {
            sim->core->initial_perm_p2l = mapping;
            sim->core->reinitialize_mapping();
            check_basis(sim);
            check_three(sim);
            check_distributions(sim);
            check_semantics(sim);
        }
        if (sim->core->proc_num == 0) std::cout << "CUDA/MPI measurement checks passed\n";
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "measurement test failed: " << error.what() << '\n';
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
