#include "../src/pauli_string.hpp"
#include "pauli_reference.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>

using namespace qcs::pauli_detail;
using pauli_test::complex;

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template<class F> static void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid input accepted");
}

static void near(long double actual, long double expected)
{
    require(std::abs(actual - expected) <= 1e-12L * (1 + std::abs(expected)), "expectation mismatch");
}

// Exercise the planned GPU arithmetic on the host; the oracle above uses
// full matrix applications instead. This also checks the conjugation order.
static long double masked_expectation(const std::vector<complex>& state,
                                      const PauliStringDescriptor& d)
{
    long double sum = 0;
    uint64_t count = d.flip_mask ? state.size() / 2 : state.size();
    uint64_t pivot = d.flip_mask ? pair_pivot(d.flip_mask) : 0;
    for (uint64_t k = 0; k < count; ++k)
    {
        uint64_t i = d.flip_mask ? pair_first_index(k, pivot) : k;
        complex phase = 1;
        for (unsigned y = 0; y < d.y_count_mod4; ++y) phase *= complex(0, 1);
        for (uint64_t bits = i & d.phase_mask; bits; bits &= bits - 1) phase = -phase;
        complex product = std::conj(state[i ^ d.flip_mask]) * state[i] * phase;
        sum += (d.flip_mask ? 2 : 1) * product.real();
    }
    return d.coefficient * sum;
}

static void validation()
{
    bit_num_t qubits[] = {0, 1, 2, 3};
    qcs_pauli_operator ops[] = {QCS_PAULI_X, QCS_PAULI_Y, QCS_PAULI_Z, QCS_PAULI_I};
    auto request = [&](int n, int local, bit_num_t count, double coefficient = 1) {
        return pauli_string(n, local, qubits, ops, count, coefficient);
    };
    for (int n : {-1, 0, 64, 65}) rejects([&] { request(n, 0, 0); });
    for (int local : {-1, 5}) rejects([&] { request(4, local, 0); });
    for (bit_num_t count : {-1, 5}) rejects([&] { request(4, 4, count); });
    rejects([&] { pauli_string(4, 4, nullptr, ops, 1, 1); });
    rejects([&] { pauli_string(4, 4, qubits, nullptr, 1, 1); });
    rejects([&] { request(4, 2, 3); });
    for (double c : {std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        rejects([&] { request(4, 4, 1, c); });
    for (bit_num_t q : {-1, 4, 64})
    {
        qubits[0] = q;
        rejects([&] { request(4, 4, 1); });
    }
    qubits[0] = 1;
    rejects([&] { request(4, 4, 2); });
    // Duplicate identity entries are rejected too.
    ops[0] = QCS_PAULI_I;
    rejects([&] { request(4, 4, 2); });
    qubits[0] = 0;
    ops[0] = static_cast<qcs_pauli_operator>(99);
    rejects([&] { request(4, 4, 1); });
    ops[0] = QCS_PAULI_X;
    auto p = request(4, 3, 4, -2);
    require(p.terms().size() == 3, "identity was not removed");
    auto d = p.descriptor({2, 0, 1, 3});
    require(d.coefficient == -2 && d.flip_mask == 5 && d.phase_mask == 3 && d.y_count_mod4 == 1,
            "incorrect mapped descriptor");
    qubits[0] = 3;
    ops[0] = QCS_PAULI_I;
    auto owned = p.descriptor({2, 0, 1, 3});
    require(owned.flip_mask == d.flip_mask && owned.phase_mask == d.phase_mask,
            "validated terms borrowed caller storage");
    qubits[0] = 0;
    ops[0] = QCS_PAULI_X;
    for (const std::vector<int>& mapping : {std::vector<int>{0, 1}, {0, 1, 2, 4},
                                           {0, 1, 2, -1}, {0, 1, 1, 3}, {3, 0, 1, 2}})
        rejects([&] { p.descriptor(mapping); });
    // An identity on a global bit needs no localization or local capacity.
    qcs_pauli_operator identity = QCS_PAULI_I;
    bit_num_t global = 3;
    auto id = pauli_string(4, 0, &global, &identity, 1, 3).descriptor({0, 1, 2, 3});
    require(!id.flip_mask && !id.phase_mask && !id.y_count_mod4, "bad identity descriptor");
    auto empty = pauli_string(1, 0, nullptr, nullptr, 0, 1).descriptor({0});
    require(!empty.flip_mask && !empty.phase_mask, "bad empty string");
    // Check the largest representable simulator bit without allocating a state.
    bit_num_t high = 62;
    qcs_pauli_operator y = QCS_PAULI_Y;
    std::vector<int> mapping(63);
    std::iota(mapping.begin(), mapping.end(), 0);
    auto high_d = pauli_string(63, 63, &high, &y, 1, 1).descriptor(mapping);
    require(high_d.flip_mask == (UINT64_C(1) << 62) && high_d.phase_mask == high_d.flip_mask,
            "high bit was truncated");
    require(pair_pivot(UINT64_C(1) << 63) == (UINT64_C(1) << 63), "64-bit pivot failed");
    rejects([] { pair_pivot(0); });
    rejects([] { validate_simulator_state(nullptr, nullptr); });
    // Opaque pointer is never dereferenced by allocation validation.
    int storage = 0;
    auto* simulator = reinterpret_cast<const qcs_simulator*>(&storage);
    rejects([&] { validate_simulator_state(simulator, nullptr); });
    validate_simulator_state(simulator, &storage);
    rejects([&] { validate_simulator_state(nullptr, &storage); });
}

static void pairs()
{
    for (unsigned n = 1; n <= 9; ++n)
    {
        uint64_t dim = UINT64_C(1) << n;
        for (uint64_t flip = 1; flip < dim; ++flip)
        {
            std::vector<unsigned> visits(dim, 0);
            uint64_t count = 0, pivot = pair_pivot(flip);
            require(pivot <= flip && flip - pivot < pivot, "incorrect pivot");
            for (uint64_t k = 0; k < dim / 2; ++k)
            {
                uint64_t i = pair_first_index(k, pivot), j = i ^ flip;
                require(i < j && j < dim, "invalid pair");
                require(++visits[i] == 1 && ++visits[j] == 1, "duplicate pair");
                ++count;
            }
            require(count == dim / 2 && std::all_of(visits.begin(), visits.end(),
                    [](unsigned v) { return v == 1; }), "missing pair");
        }
    }
    for (unsigned b : {31u, 32u, 61u, 62u})
    {
        uint64_t pivot = UINT64_C(1) << b;
        for (uint64_t k : {UINT64_C(0), pivot - 1, pivot, (UINT64_C(1) << 62) - 1})
        {
            uint64_t i = pair_first_index(k, pivot);
            require(!(i & pivot) && i < (i ^ (pivot | 1)), "wide pair index failed");
        }
    }
}

static void accuracy()
{
    near(pauli_test::expectation({1, 0}, {0}, {QCS_PAULI_Z}), 1);
    near(pauli_test::expectation({0, 1}, {0}, {QCS_PAULI_Z}), -1);
    near(pauli_test::expectation({1, 1}, {0}, {QCS_PAULI_X}), 2);
    near(pauli_test::expectation({1, complex(0, 1)}, {0}, {QCS_PAULI_Y}), 2);
    near(pauli_test::expectation({1, complex(0, -1)}, {0}, {QCS_PAULI_Y}), -2);
    near(pauli_test::expectation({1, complex(0, 1)}, {}, {}, -3), -6);
    std::vector<complex> bell = {1, 0, 0, 1};
    near(pauli_test::expectation(bell, {0, 1}, {QCS_PAULI_X, QCS_PAULI_X}), 2);
    near(pauli_test::expectation(bell, {0, 1}, {QCS_PAULI_Y, QCS_PAULI_Y}), -2);
    std::mt19937_64 random(12345);
    std::uniform_real_distribution<long double> distribution(-1, 1);
    for (int n = 1; n <= 4; ++n)
    {
        std::vector<complex> state(size_t(1) << n);
        for (auto& a : state) a = complex(distribution(random), distribution(random));
        std::vector<bit_num_t> qubits(n);
        std::vector<int> mapping(n);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::iota(mapping.begin(), mapping.end(), 0);
        for (unsigned encoding = 0; encoding < (1u << (2 * n)); ++encoding)
        {
            std::vector<qcs_pauli_operator> ops(n);
            for (int q = 0; q < n; ++q) ops[q] = static_cast<qcs_pauli_operator>((encoding >> (2 * q)) & 3);
            for (double c : {-2.5, 0.0, 1.0})
            {
                auto d = pauli_string(n, n, qubits.data(), ops.data(), n, c).descriptor(mapping);
                near(masked_expectation(state, d), pauli_test::expectation(state, qubits, ops, c));
                // Reverse mapping tests use logical state order independently.
                std::reverse(mapping.begin(), mapping.end());
                std::vector<complex> physical(state.size());
                for (size_t i = 0; i < state.size(); ++i)
                {
                    size_t dest = 0;
                    for (int q = 0; q < n; ++q) dest |= ((i >> q) & 1) << mapping[q];
                    physical[dest] = state[i];
                }
                auto remapped = pauli_string(n, n, qubits.data(), ops.data(), n, c).descriptor(mapping);
                near(masked_expectation(physical, remapped), pauli_test::expectation(state, qubits, ops, c));
                std::reverse(mapping.begin(), mapping.end());
            }
        }
    }
    // Non-contiguous targets, four Ys (modulo wraps to zero), and GHZ.
    std::vector<complex> ghz(64);
    ghz.front() = ghz.back() = 1 / std::sqrt(2.0L);
    std::vector<bit_num_t> targets{5, 0, 3, 2};
    std::vector<qcs_pauli_operator> ops(4, QCS_PAULI_Y);
    auto d = pauli_string(6, 6, targets.data(), ops.data(), 4, 1).descriptor({0, 1, 2, 3, 4, 5});
    require(d.y_count_mod4 == 0, "Y count did not wrap");
    near(masked_expectation(ghz, d), pauli_test::expectation(ghz, targets, ops));
    near(pauli_test::expectation(ghz, {0, 1, 2, 3, 4, 5}, std::vector<qcs_pauli_operator>(6, QCS_PAULI_X)), 1);
}

int main()
{
    static_assert(std::is_trivially_copyable<PauliStringDescriptor>::value);
    validation();
    pairs();
    accuracy();
    std::cout << "Pauli string host tests passed (validation, mappings, pairs, phases, CPU oracle)\n";
}
