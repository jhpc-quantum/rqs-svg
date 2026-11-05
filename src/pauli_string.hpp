#pragma once

#include "../include/qcs.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qcs::pauli_detail
{
struct PauliStringDescriptor
{
    double coefficient;
    uint64_t flip_mask;
    uint64_t phase_mask;
    uint8_t y_count_mod4;
};

struct term
{
    bit_num_t qubit;
    qcs_pauli_operator op;
};

// Own the validated input so localization cannot invalidate caller storage.
class pauli_string
{
    double coefficient_;
    int num_qubits_;
    int num_local_;
    std::vector<term> terms_;

public:
    pauli_string(int num_qubits, int num_local, const bit_num_t* qubits,
                 const qcs_pauli_operator* operators, bit_num_t count, double coefficient)
        : coefficient_(coefficient), num_qubits_(num_qubits), num_local_(num_local)
    {
        // 2^64 cannot be represented as a uint64_t state-vector dimension.
        if (num_qubits < 1 || num_qubits > 63 || num_local < 0 || num_local > num_qubits)
            throw std::runtime_error("Pauli string: invalid simulator dimensions");
        if (count < 0 || count > num_qubits)
            throw std::runtime_error("Pauli string: invalid term count");
        if (count && (!qubits || !operators))
            throw std::runtime_error("Pauli string: missing input array");
        if (!std::isfinite(coefficient))
            throw std::runtime_error("Pauli string: coefficient must be finite");
        uint64_t seen = 0;
        for (int i = 0; i < count; ++i)
        {
            int q = qubits[i];
            if (q < 0 || q >= num_qubits)
                throw std::runtime_error("Pauli string: qubit out of range");
            uint64_t bit = UINT64_C(1) << q;
            if (seen & bit)
                throw std::runtime_error("Pauli string: duplicate qubit");
            seen |= bit;
            switch (operators[i])
            {
            case QCS_PAULI_I: break;
            case QCS_PAULI_X:
            case QCS_PAULI_Y:
            case QCS_PAULI_Z: terms_.push_back({qubits[i], operators[i]}); break;
            default: throw std::runtime_error("Pauli string: invalid operator");
            }
        }
        if (terms_.size() > static_cast<size_t>(num_local))
            throw std::runtime_error("Pauli string: non-identity support exceeds local qubits");
    }

    double coefficient() const { return coefficient_; }
    const std::vector<term>& terms() const { return terms_; }

    // Call only after localization. Mapping is logical -> physical and remains
    // owned by the simulator; no exchange or mapping mutation happens here.
    PauliStringDescriptor descriptor(const std::vector<int>& logical_to_physical) const
    {
        if (logical_to_physical.size() != static_cast<size_t>(num_qubits_))
            throw std::runtime_error("Pauli string: mapping size mismatch");
        uint64_t seen = 0;
        for (int p : logical_to_physical)
        {
            if (p < 0 || p >= num_qubits_)
                throw std::runtime_error("Pauli string: physical bit out of range");
            uint64_t bit = UINT64_C(1) << p;
            if (seen & bit)
                throw std::runtime_error("Pauli string: mapping is not a permutation");
            seen |= bit;
        }
        PauliStringDescriptor d{coefficient_, 0, 0, 0};
        for (const auto& t : terms_)
        {
            int p = logical_to_physical[t.qubit];
            if (p >= num_local_)
                throw std::runtime_error("Pauli string: target must be localized before descriptor construction");
            uint64_t bit = UINT64_C(1) << p;
            if (t.op == QCS_PAULI_X || t.op == QCS_PAULI_Y) d.flip_mask |= bit;
            if (t.op == QCS_PAULI_Y || t.op == QCS_PAULI_Z) d.phase_mask |= bit;
            if (t.op == QCS_PAULI_Y) d.y_count_mod4 = (d.y_count_mod4 + 1) % 4;
        }
        return d;
    }
};

// The API checks the simulator pointer before obtaining core state storage.
// Allocation does not prove initialization: an explicit state setter is still
// a caller precondition, as with existing gates and measurements.
inline void validate_simulator_state(const qcs_simulator* simulator, const void* state)
{
    if (!simulator) throw std::runtime_error("Pauli string: null simulator");
    if (!state) throw std::runtime_error("Pauli string: state storage is not allocated");
}

inline uint64_t pair_pivot(uint64_t flip_mask)
{
    if (!flip_mask) throw std::runtime_error("Pauli string: diagonal operator has no pair pivot");
    uint64_t pivot = 1;
    while (flip_mask >>= 1) pivot <<= 1;
    return pivot;
}

// Preconditions: pivot is the highest set bit of a nonzero local flip mask;
// k < local_dim/2. Insert a zero at that bit, guaranteeing i < (i ^ flip_mask).
inline constexpr uint64_t pair_first_index(uint64_t k, uint64_t pivot)
{
    uint64_t lower = pivot - 1;
    return (k & lower) | ((k & ~lower) << 1);
}
} // namespace qcs::pauli_detail
