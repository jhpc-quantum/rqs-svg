/* Compiled as C and linked into the public API integration executable. */
#include <qcs.h>

int pauli_api_c_checks(qcs_simulator *sim)
{
    const bit_num_t qubits[] = {0, 7};
    const qcs_pauli_operator z[] = {QCS_PAULI_Z, QCS_PAULI_Z};
    const qcs_pauli_operator identity[] = {QCS_PAULI_I, QCS_PAULI_I};
    const double coefficients[] = {2.0, -3.0, 0.0};
    double result = 123.0;
    for (unsigned i = 0; i < sizeof(coefficients) / sizeof(coefficients[0]); ++i)
    {
        if (qcs_simulator_expectation_pauli(sim, qubits, z, 2, coefficients[i], &result) != 1 ||
            result != coefficients[i]) return 0;
    }
    if (!qcs_simulator_expectation_pauli(sim, NULL, NULL, 0, -2.0, &result) || result != -2.0) return 0;
    if (!qcs_simulator_expectation_pauli(sim, qubits, identity, 2, 3.0, &result) || result != 3.0) return 0;
    result = 123.0;
    if (qcs_simulator_expectation_pauli(NULL, qubits, z, 2, 1.0, &result) != 0 || result != 123.0) return 0;
    if (qcs_simulator_expectation_pauli(sim, qubits, z, 2, 1.0, NULL) != 0) return 0;
    return 1;
}
