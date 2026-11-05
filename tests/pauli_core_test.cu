// Private core integration coverage; no public expectation API is added here.
#include "../src/qcs.cu"
#include "pauli_reference.hpp"
#include <iostream>
#include <numeric>

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
        error = std::max(error, double(std::abs(actual - expected[logical_index(c, i)])));
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
    double result = c.expectation_pauli(targets.data(), operators.data(),
                                        static_cast<bit_num_t>(targets.size()), coefficient);
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
        // Basis state leaves zero-contribution ranks; all must enter NCCL.
        state basis(size_t(1) << n);
        basis[0] = 1;
        upload(c, basis);
        compare(sim, basis, {0}, {QCS_PAULI_Z});
        compare(sim, basis, {7}, {QCS_PAULI_X});
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
