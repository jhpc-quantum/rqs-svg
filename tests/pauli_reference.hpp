#pragma once

#include "../include/qcs.h"
#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

namespace pauli_test
{
using complex = std::complex<long double>;

// Independent oracle: apply explicit 2x2 matrices to a full CPU vector, then
// take its inner product with the original vector. No descriptor or pair masks.
inline long double expectation(const std::vector<complex>& state,
                               const std::vector<bit_num_t>& qubits,
                               const std::vector<qcs_pauli_operator>& operators,
                               long double coefficient = 1)
{
    if (state.empty() || (state.size() & (state.size() - 1)) || qubits.size() != operators.size())
        throw std::runtime_error("invalid Pauli reference input");
    auto transformed = state;
    for (size_t t = 0; t < qubits.size(); ++t)
    {
        if (qubits[t] < 0 || qubits[t] >= 63)
            throw std::runtime_error("reference qubit out of range");
        uint64_t width = UINT64_C(1) << qubits[t];
        if (width >= state.size()) throw std::runtime_error("reference qubit outside state");
        std::array<complex, 4> matrix;
        switch (operators[t])
        {
        case QCS_PAULI_I: matrix = {1, 0, 0, 1}; break;
        case QCS_PAULI_X: matrix = {0, 1, 1, 0}; break;
        case QCS_PAULI_Y: matrix = {0, complex(0, -1), complex(0, 1), 0}; break;
        case QCS_PAULI_Z: matrix = {1, 0, 0, -1}; break;
        default: throw std::runtime_error("invalid reference operator");
        }
        for (size_t base = 0; base < state.size(); base += 2 * width)
        for (size_t offset = 0; offset < width; ++offset)
        {
            size_t a = base + offset, b = a + width;
            complex x = transformed[a], y = transformed[b];
            transformed[a] = matrix[0] * x + matrix[1] * y;
            transformed[b] = matrix[2] * x + matrix[3] * y;
        }
    }
    complex result = 0;
    for (size_t i = 0; i < state.size(); ++i) result += std::conj(state[i]) * transformed[i];
    return coefficient * result.real();
}
} // namespace pauli_test
