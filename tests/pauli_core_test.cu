// Private core integration coverage; no public expectation API is added here.
#ifndef QCS_PAULI_SOURCE
#define QCS_PAULI_SOURCE "../src/qcs.cu"
#endif
#include QCS_PAULI_SOURCE
#include "pauli_reference.hpp"
#include <iostream>
#include <numeric>
#include <random>

using state = std::vector<pauli_test::complex>;
static unsigned checks = 0;
static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
template<class F> static void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid core request accepted");
}
static uint64_t logical_index(const qcs::simulator_core& c, uint64_t local)
{
    uint64_t physical = local | (uint64_t(c.proc_num) << c.num_qubits_local), result = 0;
    for (int p = 0; p < c.num_qubits; ++p) result |= ((physical >> p) & 1) << c.perm_p2l[p];
    return result;
}
static void upload(qcs::simulator_core& c, const state& values)
{
    c.discard_measurement_state();
    c.reinitialize_mapping();
    std::vector<qcs::complex_t> local(c.num_states_local);
    for (uint64_t i = 0; i < local.size(); ++i)
    {
        auto a = values[logical_index(c, i)];
        local[i] = qcs::complex_t(double(a.real()), double(a.imag()));
    }
    ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, local.data(), local.size() * sizeof(local[0]),
                    cudaMemcpyHostToDevice, c.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
}
static void check_state(qcs::simulator_core& c, const state& expected)
{
    std::vector<qcs::complex_t> local(c.num_states_local);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, local.data(), c.state_data_device, local.size() * sizeof(local[0]),
                    cudaMemcpyDeviceToHost, c.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
    double error = 0;
    for (uint64_t i = 0; i < local.size(); ++i)
    {
        auto actual = pauli_test::complex(local[i].real(), local[i].imag());
        if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag())) error = INFINITY;
        else error = std::max(error, double(std::abs(actual - expected[logical_index(c, i)])));
    }
    double worst;
    ATLC_CHECK_MPI(MPI_Allreduce, &error, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    require(worst <= 1e-12, "expectation changed logical state or missed projection");
    for (int q = 0; q < c.num_qubits; ++q)
        require(c.perm_p2l[c.perm_l2p[q]] == q, "inconsistent retained mapping");
}
static double compare(qcs_simulator* sim, const state& expected,
                      const std::vector<bit_num_t>& targets,
                      const std::vector<qcs_pauli_operator>& operators, double coefficient = 1)
{
    auto& c = *sim->core;
    auto mask = c.measured_mask_logical, value = c.measured_value_logical;
    auto counter = c.measurement_counter;
    bool exhausted = c.measurement_counter_exhausted;
    auto zeros = c.measured_0_qubit_num_logical_list, ones = c.measured_1_qubit_num_logical_list;
    auto zero_copy = c.measured_0_qubit_num_logical_list_copy, one_copy = c.measured_1_qubit_num_logical_list_copy;
    auto clbits = sim->clbits;
    auto* storage = c.measurement_workspace->data();
    auto capacity = c.measurement_workspace->capacity();
    double result;
    if (checks % 2)
        result = qcs_simulator_expectation_pauli_cxx(sim, targets.data(), operators.data(),
                                                   static_cast<bit_num_t>(targets.size()), coefficient);
    else
        require(qcs_simulator_expectation_pauli(sim, targets.data(), operators.data(),
                                                static_cast<bit_num_t>(targets.size()), coefficient, &result) == 1,
                "public C expectation failed");
    long double oracle = pauli_test::expectation(expected, targets, operators, coefficient);
    require(std::isfinite(result) && std::abs(result - oracle) <= 1e-12L + 5e-12L * std::abs(oracle),
            "core expectation/CPU oracle mismatch");
    double minimum, maximum;
    ATLC_CHECK_MPI(MPI_Allreduce, &result, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    ATLC_CHECK_MPI(MPI_Allreduce, &result, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    require(minimum == maximum, "expectation differs across ranks");
    require(c.measured_mask_logical == mask && c.measured_value_logical == value &&
            c.measurement_counter == counter && c.measurement_counter_exhausted == exhausted &&
            c.measured_0_qubit_num_logical_list == zeros && c.measured_1_qubit_num_logical_list == ones &&
            c.measured_0_qubit_num_logical_list_copy == zero_copy && c.measured_1_qubit_num_logical_list_copy == one_copy &&
            sim->clbits == clbits, "expectation modified measurement bookkeeping");
    require(c.pending_projection_mask_logical == 0, "projection remains pending");
    require(storage == c.measurement_workspace->data() && capacity == c.measurement_workspace->capacity(),
            "core expectation resized workspace");
    for (size_t i = 0; i < targets.size(); ++i)
        if (operators[i] != QCS_PAULI_I)
            require(c.perm_l2p[targets[i]] < c.num_qubits_local, "target was not localized");
    check_state(c, expected);
    ++checks;
    return result;
}
static void project(state& values, int qubit, int result)
{
    for (size_t i = 0; i < values.size(); ++i)
        if (int((i >> qubit) & 1) != result) values[i] = 0;
}
static void reference_x(state& values, int qubit)
{
    size_t bit = size_t(1) << qubit;
    for (size_t i = 0; i < values.size(); ++i) if (!(i & bit)) std::swap(values[i], values[i | bit]);
}
static void distributed_cases(qcs_simulator* sim)
{
    auto& c = *sim->core;
    std::mt19937_64 random(0x5041554c49);
    std::uniform_real_distribution<double> real(-1, 1);
    for (int layout = 0; layout < 5; ++layout)
    {
        c.initial_perm_p2l.resize(c.num_qubits);
        std::iota(c.initial_perm_p2l.begin(), c.initial_perm_p2l.end(), 0);
        if (layout == 1) std::reverse(c.initial_perm_p2l.begin(), c.initial_perm_p2l.end());
        if (layout > 1) std::shuffle(c.initial_perm_p2l.begin(), c.initial_perm_p2l.end(), random);
        state input(c.num_states);
        double scale = layout % 2 ? 1 : 1 / std::sqrt(double(c.num_states));
        for (auto& a : input) a = pauli_test::complex(scale * real(random), scale * real(random));
        // Reset before each operator to guarantee the chosen bit starts global
        // on distributed runs, including diagonal Z which must also localize.
        for (auto op : {QCS_PAULI_Z, QCS_PAULI_X, QCS_PAULI_Y})
        {
            upload(c, input);
            bit_num_t target = c.perm_p2l.back();
            compare(sim, input, {target}, {op}, -0.75);
        }
        upload(c, input);
        for (int trial = 0; trial < 40; ++trial)
        {
            std::vector<bit_num_t> targets(c.num_qubits);
            std::iota(targets.begin(), targets.end(), 0);
            std::shuffle(targets.begin(), targets.end(), random);
            targets.resize(trial % (c.num_qubits_local + 1));
            std::vector<qcs_pauli_operator> ops;
            for (size_t t = 0; t < targets.size(); ++t)
                ops.push_back(static_cast<qcs_pauli_operator>(random() % 4));
            compare(sim, input, targets, ops, trial % 3 == 0 ? -2 : (trial % 3 == 1 ? 0 : 1));
        }
        upload(c, input);
        std::vector<bit_num_t> maximum{static_cast<bit_num_t>(c.perm_p2l.back())};
        for (int p = 0; int(maximum.size()) < c.num_qubits_local; ++p)
            maximum.push_back(c.perm_p2l[p]);
        compare(sim, input, maximum, std::vector<qcs_pauli_operator>(maximum.size(), QCS_PAULI_Z));
        compare(sim, input, maximum, std::vector<qcs_pauli_operator>(maximum.size(), QCS_PAULI_X));
        for (int ys = 1; ys <= 4; ++ys)
        {
            std::vector<bit_num_t> targets(maximum.begin(), maximum.begin() + ys);
            compare(sim, input, targets, std::vector<qcs_pauli_operator>(ys, QCS_PAULI_Y));
        }
        // Fresh local/global pending projections, then request only an inside
        // target, only an outside target, or both. No renormalization in oracle.
        for (int selection = 0; selection < 3; ++selection)
        {
            upload(c, input);
            state projected = input;
            bit_num_t a = c.perm_p2l.back(), b = c.perm_p2l.front(), outside = c.perm_p2l[2];
            project(projected, a, qcs_simulator_measure_to_clbit_cxx(sim, a, 0));
            project(projected, b, qcs_simulator_measure_to_clbit_cxx(sim, b, 1));
            if (selection == 0) compare(sim, projected, {a}, {QCS_PAULI_Y});
            if (selection == 1) compare(sim, projected, {outside}, {QCS_PAULI_Z});
            if (selection == 2) compare(sim, projected, {b, a}, {QCS_PAULI_X, QCS_PAULI_Z});
            compare(sim, projected, {}, {});
        }
        state ghz(c.num_states);
        ghz.front() = ghz.back() = double(1 / std::sqrt(2.0));
        upload(c, ghz);
        compare(sim, ghz, {0, static_cast<bit_num_t>(c.num_qubits - 1)}, {QCS_PAULI_Z, QCS_PAULI_Z});
        compare(sim, ghz, maximum, std::vector<qcs_pauli_operator>(maximum.size(), QCS_PAULI_X));
        state bell(c.num_states);
        bell[0] = bell[1 | (size_t(1) << (c.num_qubits - 1))] = double(1 / std::sqrt(2.0));
        upload(c, bell);
        compare(sim, bell, {0, static_cast<bit_num_t>(c.num_qubits - 1)}, {QCS_PAULI_Y, QCS_PAULI_Y});
    }
}

int main()
{
    auto* sim = qcs_simulator_create_cxx();
    try
    {
        auto& c = *sim->core;
        require(c.num_procs == 1 || c.num_procs == 2 || c.num_procs == 4,
                "run on 1, 2 or 4 GPUs, one rank per GPU");
        rejects([&] { c.expectation_pauli(nullptr, nullptr, 0, 1); });
        constexpr int n = 8;
        qcs_simulator_set_num_qubits_cxx(sim, n);
        qcs_simulator_set_num_clbits_cxx(sim, n);
        qcs_simulator_allocate_memory_cxx(sim);
        state dense(size_t(1) << n);
        for (size_t i = 0; i < dense.size(); ++i)
            dense[i] = pauli_test::complex(double(1 + i % 11) / 100, double(int(i % 7) - 3) / 100);
        for (bool reverse : {false, true})
        {
            c.initial_perm_p2l.resize(n);
            std::iota(c.initial_perm_p2l.begin(), c.initial_perm_p2l.end(), 0);
            if (reverse) std::reverse(c.initial_perm_p2l.begin(), c.initial_perm_p2l.end());
            upload(c, dense);
            compare(sim, dense, {}, {}, -3);
            for (bit_num_t q = 0; q < n; ++q)
            for (auto op : {QCS_PAULI_I, QCS_PAULI_X, QCS_PAULI_Y, QCS_PAULI_Z})
                compare(sim, dense, {q}, {op}, -2.5);
            compare(sim, dense, {7, 0, 3}, {QCS_PAULI_Y, QCS_PAULI_X, QCS_PAULI_Z});
            compare(sim, dense, {0, 7}, {QCS_PAULI_X, QCS_PAULI_Y}, 0);
            // Validate before changing a pending projection or the mapping.
            upload(c, dense);
            state expected = dense;
            bit_num_t first = 0, second = 7;
            int a = qcs_simulator_measure_to_clbit_cxx(sim, first, 2);
            project(expected, first, a);
            int b = qcs_simulator_measure_to_clbit_cxx(sim, second, 3);
            project(expected, second, b);
            auto pending = c.pending_projection_mask_logical;
            auto mapping = c.perm_l2p;
            bit_num_t bad[] = {0, 0};
            qcs_pauli_operator bad_ops[] = {QCS_PAULI_X, QCS_PAULI_Y};
            rejects([&] { c.expectation_pauli(bad, bad_ops, 2, 1); });
            require(c.pending_projection_mask_logical == pending && c.perm_l2p == mapping,
                    "rejected request mutated state metadata");
            // Both measured bits are outside this request. Under either initial
            // mapping, one was a global bit on distributed runs.
            compare(sim, expected, {3}, {QCS_PAULI_X});
            compare(sim, expected, {}, {});
            compare(sim, expected, {7, 0}, {QCS_PAULI_Y, QCS_PAULI_Y});
            auto cursor = c.measurement_counter;
            require(qcs_simulator_measure_cxx(sim, first) == a && c.measurement_counter == cursor,
                    "expectation invalidated cached measurement");
            qcs_simulator_gate_x_cxx(sim, &first, 1, nullptr, 0, nullptr, 0);
            reference_x(expected, first);
            compare(sim, expected, {0, 7}, {QCS_PAULI_Z, QCS_PAULI_Z});
            require(qcs_simulator_measure_cxx(sim, first) == 1 - a, "gate after expectation used wrong mapping");
            project(expected, first, 1 - a);
            c.measurement_counter_exhausted = true;
            compare(sim, expected, {}, {}, 2); // Observation never draws randomness.
            c.measurement_counter_exhausted = false;
        }
        distributed_cases(sim);
        // Basis state leaves zero-contribution ranks; all must enter NCCL.
        state basis(size_t(1) << n);
        basis[0] = 1;
        upload(c, basis);
        compare(sim, basis, {0}, {QCS_PAULI_Z});
        compare(sim, basis, {7}, {QCS_PAULI_X});
        // Reallocate with one local qubit: small reductions plus global swaps.
        qcs_simulator_dispose_cxx(sim);
        c.initial_perm_p2l.clear();
        qcs_simulator_set_num_qubits_cxx(sim, c.log_num_procs + 1);
        qcs_simulator_allocate_memory_cxx(sim);
        state tiny(c.num_states);
        tiny.front() = tiny.back() = double(1 / std::sqrt(2.0));
        upload(c, tiny);
        compare(sim, tiny, {}, {}, -2);
        for (bit_num_t q = 0; q < c.num_qubits; ++q)
        for (auto op : {QCS_PAULI_X, QCS_PAULI_Y, QCS_PAULI_Z})
            compare(sim, tiny, {q}, {op});
        if (!c.proc_num) std::cout << "Pauli core: " << checks << " oracle/state checks passed\n";
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Pauli core: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
