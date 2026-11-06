#include <qcs.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static bit_num_t num_qubits;
static bit_num_t num_clbits;

void circuit_init(qcs_simulator *sim)
{
    const char *const num_qubits_str = getenv("NUM_QUBITS");
    if (num_qubits_str == NULL || num_qubits_str[0] == '\0')
    {
        fprintf(stderr, "NUM_QUBITS is empty\n");
        exit(EXIT_FAILURE);
    }

    char *endptr;
    unsigned long parsed_num_qubits =
        strtoul(num_qubits_str, &endptr, 10);

    if (endptr == num_qubits_str || parsed_num_qubits > INT_MAX)
    {
        fprintf(stderr, "strtoul on NUM_QUBITS failed\n");
        exit(EXIT_FAILURE);
    }

    num_qubits = (bit_num_t)parsed_num_qubits;
    num_clbits = (bit_num_t)parsed_num_qubits;

    qcs_simulator_set_num_qubits(sim, num_qubits);
    qcs_simulator_set_num_clbits(sim, num_clbits);
}

void circuit_run(qcs_simulator *sim)
{
    bit_num_t *qubit_num_list = NULL;
    bit_num_t *clbit_num_list = NULL;
    bit_t *results = NULL;

    if (num_qubits > 0)
    {
        qubit_num_list =
            malloc(sizeof(*qubit_num_list) * (size_t)num_qubits);

        clbit_num_list =
            malloc(sizeof(*clbit_num_list) * (size_t)num_clbits);

        results =
            malloc(sizeof(*results) * (size_t)num_qubits);

        if (qubit_num_list == NULL ||
            clbit_num_list == NULL ||
            results == NULL)
        {
            fprintf(stderr, "malloc failed\n");

            free(qubit_num_list);
            free(clbit_num_list);
            free(results);

            exit(EXIT_FAILURE);
        }

        for (bit_num_t i = 0; i < num_qubits; i++)
        {
            qubit_num_list[i] = i;
            clbit_num_list[i] = i;
        }
    }

    if (!qcs_simulator_measure_many_to_clbits(
            sim,
            qubit_num_list,
            num_qubits,
            clbit_num_list,
            num_clbits,
            results))
    {
        fprintf(stderr,
                "qcs_simulator_measure_many_to_clbits failed\n");

        free(qubit_num_list);
        free(clbit_num_list);
        free(results);

        exit(EXIT_FAILURE);
    }

    free(qubit_num_list);
    free(clbit_num_list);
    free(results);
}
