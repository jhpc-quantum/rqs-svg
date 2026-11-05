// Public interfaces only. Run with one rank per GPU (1, 2, or 4 ranks).
#include <qcs.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

extern "C" int pauli_api_c_checks(qcs_simulator* sim);
static unsigned errors = 0;
static std::string last_error;
static void on_error(const char* message, size_t count)
{
    ++errors;
    last_error.assign(message, count);
}
static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
template<class F> static void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "C++ expectation did not throw");
}
static void near(double got, double expected)
{
    require(std::isfinite(got) && std::abs(got - expected) <= 1e-12 + 5e-12 * std::abs(expected),
            "public expectation mismatch");
}
int main()
{
    qcs_simulator* sim = nullptr;
    try
    {
        require(qcs_set_exception_callback(on_error) == 1, "callback setup failed");
        rejects([] { qcs_simulator_expectation_pauli_cxx(nullptr, nullptr, nullptr, 0, 1); });
        sim = qcs_simulator_create_cxx();
        rejects([&] { qcs_simulator_expectation_pauli_cxx(sim, nullptr, nullptr, 0, 1); });
        double result = 123;
        require(!qcs_simulator_expectation_pauli(sim, nullptr, nullptr, 0, 1, &result) && result == 123,
                "unallocated C request changed output");
        int ranks = qcs_simulator_get_num_procs_cxx(sim);
        require(ranks == 1 || ranks == 2 || ranks == 4, "use 1, 2 or 4 GPUs");
        constexpr int n = 8;
        qcs_simulator_set_num_qubits_cxx(sim, n);
        std::array<bit_num_t, n> mapping{};
        for (int p = 0; p < n; ++p) mapping[p] = n - p - 1;
        qcs_simulator_set_mapping_cxx(sim, mapping.data(), n);
        qcs_simulator_allocate_memory_cxx(sim);
        qcs_simulator_set_zero_state_cxx(sim);
        require(pauli_api_c_checks(sim), "C caller checks failed");
        const bit_num_t targets[] = {0, 7};
        const qcs_pauli_operator xx[] = {QCS_PAULI_X, QCS_PAULI_X};
        const qcs_pauli_operator yy[] = {QCS_PAULI_Y, QCS_PAULI_Y};
        const qcs_pauli_operator zz[] = {QCS_PAULI_Z, QCS_PAULI_Z};
        qcs_simulator_gate_h_cxx(sim, targets, 1, nullptr, 0, nullptr, 0);
        near(qcs_simulator_expectation_pauli_cxx(sim, targets, xx, 1, 2), 2);
        qcs_simulator_gate_s_cxx(sim, targets, 1, nullptr, 0, nullptr, 0);
        near(qcs_simulator_expectation_pauli_cxx(sim, targets, yy, 1, -3), -3);
        qcs_simulator_set_zero_state_cxx(sim);
        qcs_simulator_gate_h_cxx(sim, targets, 1, nullptr, 0, nullptr, 0);
        qcs_simulator_gate_x_cxx(sim, targets + 1, 1, nullptr, 0, targets, 1);
        near(qcs_simulator_expectation_pauli_cxx(sim, targets, xx, 2, 1), 1);
        require(qcs_simulator_expectation_pauli(sim, targets, yy, 2, 2, &result), "C Bell request failed");
        near(result, -2);
        near(qcs_simulator_expectation_pauli_cxx(sim, targets, zz, 2, 1), 1);

        auto invalid = [&](const bit_num_t* qubits, const qcs_pauli_operator* ops, bit_num_t count, double c) {
            result = 123;
            unsigned before = errors;
            require(qcs_simulator_expectation_pauli(sim, qubits, ops, count, c, &result) == 0 && result == 123,
                    "invalid C request succeeded or changed result");
            require(errors == before + 1 && !last_error.empty(), "C exception callback not called");
            rejects([&] { qcs_simulator_expectation_pauli_cxx(sim, qubits, ops, count, c); });
        };
        invalid(nullptr, xx, 1, 1);
        invalid(targets, nullptr, 1, 1);
        invalid(targets, xx, -1, 1);
        invalid(targets, xx, 9, 1);
        bit_num_t duplicate[] = {0, 0}, outside = 8, negative = -1;
        invalid(duplicate, xx, 2, 1);
        invalid(&outside, xx, 1, 1);
        invalid(&negative, xx, 1, 1);
        qcs_pauli_operator bad = static_cast<qcs_pauli_operator>(99);
        invalid(targets, &bad, 1, 1);
        invalid(targets, xx, 1, std::numeric_limits<double>::infinity());
        invalid(targets, xx, 1, std::numeric_limits<double>::quiet_NaN());
        std::array<bit_num_t, n> all{};
        std::array<qcs_pauli_operator, n> all_x{};
        for (int i = 0; i < n; ++i) { all[i] = i; all_x[i] = QCS_PAULI_X; }
        if (ranks > 1) invalid(all.data(), all_x.data(), n, 1);
        // Failures leave the Bell state usable, and identity includes its norm.
        near(qcs_simulator_expectation_pauli_cxx(sim, targets, xx, 2, 1), 1);
        qcs_simulator_set_flat_state_cxx(sim);
        near(qcs_simulator_expectation_pauli_cxx(sim, nullptr, nullptr, 0, 2), 512);
        int branch = qcs_simulator_measure_cxx(sim, 0);
        near(qcs_simulator_expectation_pauli_cxx(sim, nullptr, nullptr, 0, 1), 128);
        require(qcs_simulator_measure_cxx(sim, 0) == branch, "observation lost measurement record");
        near(qcs_simulator_expectation_pauli_cxx(sim, targets, zz, 1, 1), branch ? -128 : 128);
        if (!qcs_simulator_get_proc_num_cxx(sim)) std::cout << "Pauli public C/C++ API checks passed\n";
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Pauli public API: " << e.what() << '\n';
        return 1;
    }
}
