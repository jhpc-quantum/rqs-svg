#pragma once

#include <cstdint>
#include <type_traits>

#ifndef __CUDACC__
#define QCS_GATE_HD
#else
#define QCS_GATE_HD __host__ __device__
#endif

namespace qcs
{
    constexpr int max_num_qubits_local = 34;

    struct kernel_input_qnlist_struct
    {
        int num_target_qubits;
        int num_negative_control_qubits;
        int num_positive_control_qubits;
        uint64_t is_measured_bits;
        uint64_t measured_value_bits;
        int qubit_num_list[2 * max_num_qubits_local];

        // Entries = operands + positive controls + targets = 2 * operands - negative controls.
        static constexpr QCS_GATE_HD bool valid_counts(int targets, int negative, int positive)
        {
            return targets >= 0 && negative >= 0 && positive >= 0 &&
                   targets <= max_num_qubits_local && negative <= max_num_qubits_local &&
                   positive <= max_num_qubits_local &&
                   targets + negative + positive <= max_num_qubits_local;
        }

        QCS_GATE_HD int get_num_operand_qubits() const
        {
            return this->num_positive_control_qubits + this->num_negative_control_qubits + this->num_target_qubits;
        }

        QCS_GATE_HD int const *get_operand_qubit_num_list_sorted() const
        {
            return this->qubit_num_list;
        }

        QCS_GATE_HD int *get_operand_qubit_num_list_sorted()
        {
            return this->qubit_num_list;
        }

        QCS_GATE_HD int const *get_positive_control_qubit_num_list() const
        {
            return this->qubit_num_list + this->get_num_operand_qubits();
        }

        QCS_GATE_HD int *get_positive_control_qubit_num_list()
        {
            return this->qubit_num_list + this->get_num_operand_qubits();
        }

        QCS_GATE_HD int const *get_target_qubit_num_list() const
        {
            return this->qubit_num_list + 2 * this->num_positive_control_qubits + this->num_negative_control_qubits + this->num_target_qubits;
        }

        QCS_GATE_HD int *get_target_qubit_num_list()
        {
            return this->qubit_num_list + 2 * this->num_positive_control_qubits + this->num_negative_control_qubits + this->num_target_qubits;
        }
    }; /* kernel_input_qnlist_struct */

    static_assert(std::is_trivially_copyable<kernel_input_qnlist_struct>::value,
                  "Gate parameters must be trivially copyable");
    static_assert(sizeof(kernel_input_qnlist_struct) == 304, "Unexpected gate parameter layout");
    // Reserve room for the gate object or SWAP launch counters under the legacy 4 KiB limit.
    static_assert(sizeof(kernel_input_qnlist_struct) + 2 * sizeof(uint64_t) <= 4096,
                  "SWAP parameters exceed the CUDA kernel parameter limit");
}

#undef QCS_GATE_HD
