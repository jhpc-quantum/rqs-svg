#include <qcs.h>

#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static void print_usage(const char *program_name)
{
    fprintf(stderr,
            "Usage: %s --num-qubits NUM_QUBITS [--num-samples NUM_SAMPLES]\n"
            "\n"
            "Options:\n"
            "  --num-qubits NUM_QUBITS    Number of qubits in the GHZ circuit\n"
            "  --num-samples NUM_SAMPLES  Number of circuit executions (default: 1)\n"
            "  -h, --help                 Print this help message\n",
            program_name);
}

static int parse_positive_int(const char *option_name, const char *value)
{
    if (value == NULL || value[0] == '\0')
    {
        fprintf(stderr, "%s requires a positive integer\n", option_name);
        exit(EXIT_FAILURE);
    }

    errno = 0;
    char *endptr;
    long parsed_value = strtol(value, &endptr, 10);
    if (errno != 0 || endptr == value || *endptr != '\0' || parsed_value <= 0 || parsed_value > INT_MAX)
    {
        fprintf(stderr, "%s requires a positive integer in the range [1, %d]\n", option_name, INT_MAX);
        exit(EXIT_FAILURE);
    }

    return (int)parsed_value;
}

int main(int argc, char **argv)
{
    bit_num_t num_qubits = 0;
    int num_samples = 1;

    static const struct option long_options[] = {
        {"num-qubits", required_argument, NULL, 'q'},
        {"num-samples", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0},
    };

    int option;
    while ((option = getopt_long(argc, argv, "h", long_options, NULL)) != -1)
    {
        switch (option)
        {
        case 'q':
            num_qubits = parse_positive_int("--num-qubits", optarg);
            break;
        case 's':
            num_samples = parse_positive_int("--num-samples", optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (optind != argc)
    {
        fprintf(stderr, "unexpected positional argument: %s\n", argv[optind]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (num_qubits == 0)
    {
        fprintf(stderr, "--num-qubits is required\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bit_num_t num_clbits = 0;

    qcs_simulator *sim = NULL;
    if (!qcs_simulator_create(&sim))
    {
        fprintf(stderr, "failed to create qcs simulator\n");
        return EXIT_FAILURE;
    }

    qcs_simulator_set_num_qubits(sim, num_qubits);
    qcs_simulator_set_num_clbits(sim, num_clbits);
    qcs_simulator_allocate_memory(sim);
    qcs_simulator_set_zero_state(sim);

    bit_t event_1 = 0;
    bit_t event_2 = 0;
    if (!qcs_simulator_event_create(sim, &event_1) || !qcs_simulator_event_create(sim, &event_2))
    {
        fprintf(stderr, "failed to create qcs events\n");
        qcs_simulator_destroy(sim);
        return EXIT_FAILURE;
    }

    char *const clbits = malloc((size_t)num_clbits + 1);
    if (clbits == NULL)
    {
        fprintf(stderr, "failed to allocate clbits buffer\n");
        qcs_simulator_destroy(sim);
        return EXIT_FAILURE;
    }

    for (int sample_num = 0; sample_num < num_samples; sample_num++)
    {
        qcs_simulator_event_record(sim, event_1);

        for (bit_num_t qubit_num = 0; qubit_num < num_qubits; qubit_num++)
        {
            bit_num_t target[] = {qubit_num};
            qcs_simulator_gate_h(sim, target, 1, NULL, 0, NULL, 0);
        }
        qcs_simulator_event_record(sim, event_2);
        double elapsed_time = 0.0;
        qcs_simulator_event_get_elapsed_time(sim, event_1, event_2, &elapsed_time);

        qcs_simulator_get_clbits_string(sim, clbits);
        bit_t proc_num = 0;
        qcs_simulator_get_proc_num(sim, &proc_num);
        if (proc_num == 0)
        {
            fprintf(stdout, "{\"sample_num\": %d, \"clbits\": \"%s\", \"elapsed_time\": %.18g}\n", sample_num, clbits, elapsed_time);
            fflush(stdout);
        }

        if (sample_num != num_samples - 1)
        {
            qcs_simulator_reinitialize_mapping(sim);
            qcs_simulator_set_zero_state(sim);
            qcs_simulator_reset_clbits(sim);
        }
    }

    free(clbits);
    qcs_simulator_destroy(sim);
    return EXIT_SUCCESS;
}
