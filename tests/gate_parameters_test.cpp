#include "../src/gate_parameters.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>

int main()
{
    using parameters = qcs::kernel_input_qnlist_struct;
    constexpr int limit = qcs::max_num_qubits_local;
    for (int targets = -1; targets <= limit + 1; ++targets)
    for (int negative = -1; negative <= limit + 1; ++negative)
    for (int positive = -1; positive <= limit + 1; ++positive)
    {
        bool const valid = targets >= 0 && negative >= 0 && positive >= 0 &&
                           targets + negative + positive <= limit;
        assert(parameters::valid_counts(targets, negative, positive) == valid);
        if (!valid) continue;
        parameters args{};
        args.num_target_qubits = targets;
        args.num_negative_control_qubits = negative;
        args.num_positive_control_qubits = positive;
        int const operands = targets + negative + positive;
        int const entries = operands + positive + targets;
        assert(entries <= 2 * limit);
        assert(args.get_num_operand_qubits() == operands);
        assert(args.get_positive_control_qubit_num_list() == args.qubit_num_list + operands);
        assert(args.get_target_qubit_num_list() == args.qubit_num_list + operands + positive);
        for (int i = 0; i < operands; ++i) args.get_operand_qubit_num_list_sorted()[i] = i;
        for (int i = 0; i < positive; ++i) args.get_positive_control_qubit_num_list()[i] = 100 + i;
        for (int i = 0; i < targets; ++i) args.get_target_qubit_num_list()[i] = 200 + i;
        args.is_measured_bits = UINT64_C(1) << 33;
        args.measured_value_bits = UINT64_C(1) << 32;
        parameters copy{};
        std::memcpy(&copy, &args, sizeof(args));
        parameters const &view = copy;
        for (int i = 0; i < operands; ++i) assert(view.get_operand_qubit_num_list_sorted()[i] == i);
        for (int i = 0; i < positive; ++i) assert(view.get_positive_control_qubit_num_list()[i] == 100 + i);
        for (int i = 0; i < targets; ++i) assert(view.get_target_qubit_num_list()[i] == 200 + i);
        for (int i = entries; i < 2 * limit; ++i) assert(view.qubit_num_list[i] == 0);
        assert(view.is_measured_bits == (UINT64_C(1) << 33));
        assert(view.measured_value_bits == (UINT64_C(1) << 32));
    }
    assert(!parameters::valid_counts(std::numeric_limits<int>::max(), 0, 0));
    std::cout << "Gate parameter layout/capacity tests passed (" << sizeof(parameters) << " bytes)\n";
}
