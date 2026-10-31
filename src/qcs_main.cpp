#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include <atlc/format.hpp>

#include <qcs.h>
#include "qcs_args.h"

static std::vector<bit_num_t> parse_mapping_csv(std::string const &mapping_text)
{
    if (mapping_text.empty())
    {
        throw std::runtime_error("mapping string must not be empty");
    }

    std::vector<bit_num_t> mapping;
    std::stringstream ss(mapping_text);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty())
        {
            throw std::runtime_error("mapping contains an empty entry");
        }
        mapping.push_back(std::stoi(token));
    }

    return mapping;
}

static std::vector<bit_num_t> invert_mapping(std::vector<bit_num_t> const &perm_p2l, bit_num_t const num_qubits)
{
    if ((int)perm_p2l.size() != num_qubits)
    {
        throw std::runtime_error(atlc::format("mapping size %d does not match num_qubits %d", (int)perm_p2l.size(), num_qubits));
    }

    std::vector<bit_num_t> perm_l2p(num_qubits, -1);
    for (int physical_qubit_num = 0; physical_qubit_num < num_qubits; physical_qubit_num++)
    {
        int const logical_qubit_num = perm_p2l[physical_qubit_num];
        if (logical_qubit_num < 0 || logical_qubit_num >= num_qubits)
        {
            throw std::runtime_error(atlc::format("mapping value %d is out of range [0, %d)", logical_qubit_num, num_qubits));
        }
        if (perm_l2p[logical_qubit_num] != -1)
        {
            throw std::runtime_error(atlc::format("mapping value %d appears multiple times", logical_qubit_num));
        }
        perm_l2p[logical_qubit_num] = physical_qubit_num;
    }

    return perm_l2p;
}

int main(int argc, char **argv)
{
    gengetopt_args_info parsed_options;
    if (cmdline_parser(argc, argv, &parsed_options) != 0)
    {
        throw std::runtime_error("failed to parse command-line options");
    }
    std::unique_ptr<gengetopt_args_info, void (*)(gengetopt_args_info *)> parsed_options_cleanup(&parsed_options, cmdline_parser_free);

    if (parsed_options.inputs_num != 1)
    {
        throw std::runtime_error("user circuit shared object path is required");
    }

    qcs_set_exception_callback([](char const *message, size_t length)
    {
        fprintf(stderr, "[qcs pid=%ld] ", (long)getpid());
        fwrite(message, 1, length, stderr);
        fputc('\n', stderr);
        fflush(stderr);
    });

    qcs_simulator *sim_ptr = nullptr;
    if (!qcs_simulator_create(&sim_ptr))
    {
        throw std::runtime_error("qcs_simulator_create failed");
    }
    std::unique_ptr<qcs_simulator, int (*)(qcs_simulator *)> sim(sim_ptr, qcs_simulator_destroy);

    std::string const usercircuit_so_path = parsed_options.inputs[0];
    int const num_samples = parsed_options.num_samples_arg;
    std::string const output_statevector_path = parsed_options.output_statevector_given > 0
                                                    ? parsed_options.output_statevector_arg
                                                    : "";
    if (num_samples <= 0)
    {
        throw std::runtime_error("num_samples must be greater than 0");
    }

    char const *const usercircuit_so_abspath = realpath(usercircuit_so_path.c_str(), NULL);
    if (usercircuit_so_abspath == NULL)
    {
        throw std::runtime_error("realpath returned NULL");
    }
    std::unique_ptr<char, void (*)(void *)> usercircuit_so_abspath_cleanup((char *)usercircuit_so_abspath, free);

    void *usercircuit_dl = dlopen(usercircuit_so_abspath, RTLD_LAZY);
    if (usercircuit_dl == NULL)
    {
        throw std::runtime_error("dlopen failed");
    }
    std::unique_ptr<void, int (*)(void *)> usercircuit_dl_cleanup(usercircuit_dl, dlclose);

    auto circuit_init = reinterpret_cast<void (*)(qcs_simulator *)>(dlsym(usercircuit_dl, "circuit_init"));
    auto circuit_run = reinterpret_cast<void (*)(qcs_simulator *)>(dlsym(usercircuit_dl, "circuit_run"));

    circuit_init(sim.get());

    std::vector<bit_num_t> mapping;
    if (parsed_options.mapping_given > 0)
    {
        mapping = parse_mapping_csv(parsed_options.mapping_arg);
    }
    else
    {
        bit_t num_qubits = 0;
        if (!qcs_simulator_get_num_qubits(sim.get(), &num_qubits))
        {
            throw std::runtime_error("qcs_simulator_get_num_qubits failed");
        }
        mapping.resize(num_qubits);
        for (bit_num_t qubit_num = 0; qubit_num < num_qubits; qubit_num++)
        {
            mapping[qubit_num] = qubit_num;
        }
    }

    if (parsed_options.reversed_mapping_flag)
    {
        bit_t num_qubits = 0;
        if (!qcs_simulator_get_num_qubits(sim.get(), &num_qubits))
        {
            throw std::runtime_error("qcs_simulator_get_num_qubits failed");
        }
        mapping = invert_mapping(mapping, num_qubits);
    }
    qcs_simulator_set_mapping(sim.get(), mapping.data(), mapping.size());

    qcs_simulator_allocate_memory(sim.get());

    bit_t event_1 = 0;
    bit_t event_2 = 0;
    if (!qcs_simulator_event_create(sim.get(), &event_1) || !qcs_simulator_event_create(sim.get(), &event_2))
    {
        throw std::runtime_error("qcs_simulator_event_create failed");
    }

    for (int sample_num = 0; sample_num < num_samples; sample_num++)
    {

        qcs_simulator_event_record(sim.get(), event_1);

        circuit_run(sim.get());

        qcs_simulator_event_record(sim.get(), event_2);

        double elapsed_time = 0.0;
        if (!qcs_simulator_event_get_elapsed_time(sim.get(), event_1, event_2, &elapsed_time))
        {
            throw std::runtime_error("qcs_simulator_event_get_elapsed_time failed");
        }

        bit_t num_clbits = 0;
        if (!qcs_simulator_get_num_clbits(sim.get(), &num_clbits))
        {
            throw std::runtime_error("qcs_simulator_get_num_clbits failed");
        }
        std::vector<char> clbits_string(num_clbits + 1);
        qcs_simulator_get_clbits_string(sim.get(), clbits_string.data());
        bit_t proc_num = 0;
        if (!qcs_simulator_get_proc_num(sim.get(), &proc_num))
        {
            throw std::runtime_error("qcs_simulator_get_proc_num failed");
        }
        if (proc_num == 0)
        {
            fprintf(stdout, "{\"sample_num\": %d, \"clbits\": \"%s\", \"elapsed_time\": %.18g}\n", sample_num, clbits_string.data(), elapsed_time);
            fflush(stdout);
        }

        if (sample_num != num_samples - 1)
        {
            qcs_simulator_reinitialize_mapping(sim.get());
            qcs_simulator_set_zero_state(sim.get());
            qcs_simulator_reset_clbits(sim.get());
            qcs_simulator_reset_measurement_state(sim.get());
        }
    }

    if (!output_statevector_path.empty())
    {
        qcs_simulator_save_statevector(sim.get(), output_statevector_path.c_str());
    }

    return 0;
}
