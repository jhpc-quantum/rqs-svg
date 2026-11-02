// Public API integration coverage. Run with one GPU per MPI rank (1, 2, 4, 8).
#include <qcs.h>
#include <mpi.h>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <unistd.h>

static void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static void x(qcs_simulator* sim, bit_num_t q)
{
    qcs_simulator_gate_x_cxx(sim, &q, 1, nullptr, 0, nullptr, 0);
}

int main()
{
    qcs_simulator* sim = qcs_simulator_create_cxx();
    try
    {
        constexpr int n = 10;
        const int rank = qcs_simulator_get_proc_num_cxx(sim);
        const int ranks = qcs_simulator_get_num_procs_cxx(sim);
        require(ranks > 0 && ranks <= 8 && (ranks & (ranks - 1)) == 0,
                "use 1, 2, 4 or 8 ranks with one GPU per rank");
        qcs_simulator_set_num_qubits_cxx(sim, n);
        qcs_simulator_set_num_clbits_cxx(sim, n);
        std::array<bit_num_t, n> mapping{};
        for (int i = 0; i < n; ++i) mapping[i] = n - 1 - i;
        qcs_simulator_set_mapping_cxx(sim, mapping.data(), n);
        qcs_simulator_allocate_memory_cxx(sim);

        // Reverse mapping places logical zero on a rank bit with MPI. No gate
        // precedes this measurement: it exercises L=0, then inactive ranks.
        qcs_simulator_set_zero_state_cxx(sim);
        bit_t first = 1;
        require(qcs_simulator_measure(sim, 0, &first) && first == 0,
                "single/global-only measurement");
        const bit_num_t mixed[] = {9, 0, 1};
        bit_t mixed_result[3]{};
        require(qcs_simulator_measure_many(sim, mixed, 3, mixed_result), "mixed group");
        for (auto b : mixed_result) require(b == 0, "inactive rank or recorded bit");

        // A nonuniform basis state detects caller ordering and classical mapping.
        // Nine targets exercise two full groups and a remainder, growing and
        // reusing the workspace after the single-qubit reduction above.
        qcs_simulator_set_zero_state_cxx(sim);
        for (bit_num_t q : {bit_num_t(2), bit_num_t(5), bit_num_t(9)}) x(sim, q);
        const bit_num_t targets[] = {9, 2, 7, 0, 5, 1, 8, 3, 6};
        const bit_num_t clbits[] = {1, 8, 4, 9, 0, 6, 2, 7, 3};
        bit_t results[9]{};
        require(qcs_simulator_measure_many_to_clbits(sim, targets, 9, clbits, 9, results),
                "measure_many_to_clbits");
        for (int i = 0; i < 9; ++i)
        {
            bit_t expected = targets[i] == 2 || targets[i] == 5 || targets[i] == 9;
            require(results[i] == expected, "unsorted target order");
            require(qcs_simulator_read_cxx(sim, clbits[i]) == expected, "classical destination order");
        }
        bit_t repeated[9]{};
        require(qcs_simulator_measure_many(sim, targets, 9, repeated), "repeat group");
        for (int i = 0; i < 9; ++i) require(repeated[i] == results[i], "repeat stability");

        // Every possible smaller remainder, with fresh metadata between shots.
        for (int count = 5; count <= 10; ++count)
        {
            qcs_simulator_set_zero_state_cxx(sim);
            x(sim, 2);
            std::vector<bit_num_t> request(count);
            std::vector<bit_t> bits(count);
            for (int i = 0; i < count; ++i) request[i] = count - 1 - i;
            qcs_simulator_measure_many_cxx(sim, request.data(), count, bits.data());
            for (int i = 0; i < count; ++i)
                require(bits[i] == (request[i] == 2), "fresh shot/remainder width");
        }

        // Bell correlations with a global endpoint under the reverse mapping.
        qcs_simulator_set_zero_state_cxx(sim);
        const bit_num_t bell_root = 0, bell_other = 9;
        qcs_simulator_gate_h_cxx(sim, &bell_root, 1, nullptr, 0, nullptr, 0);
        qcs_simulator_gate_x_cxx(sim, &bell_other, 1, nullptr, 0, &bell_root, 1);
        const int bell_branch = qcs_simulator_measure_cxx(sim, bell_root);
        require(qcs_simulator_measure_cxx(sim, bell_other) == bell_branch, "Bell correlation");
        int min_branch, max_branch;
        MPI_Allreduce(&bell_branch, &min_branch, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&bell_branch, &max_branch, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        require(min_branch == max_branch, "rank agreement");

        // Biased GHZ has only two correlated outcomes. After measuring one bit,
        // all later fresh bins must exclude its discarded, still pending branch.
        qcs_simulator_set_zero_state_cxx(sim);
        const bit_num_t root = 9;
        constexpr double theta = 0.7;
        qcs_simulator_gate_ry_cxx(sim, theta, &root, 1, nullptr, 0, nullptr, 0);
        for (bit_num_t q = 0; q < 9; ++q)
            qcs_simulator_gate_x_cxx(sim, &q, 1, nullptr, 0, &root, 1);
        const int branch = qcs_simulator_measure_cxx(sim, root);
        const bit_num_t rest[] = {8, 9, 2, 5, 0, 7, 1, 6, 3, 4};
        bit_t correlated[10]{};
        require(qcs_simulator_measure_many(sim, rest, 10, correlated), "correlated groups");
        for (auto b : correlated) require(b == branch, "pending control correlation");
        x(sim, root); // Releases this target's deferred projection.
        require(qcs_simulator_measure_cxx(sim, root) == 1 - branch, "gate after measurement");

        // Save materializes remaining projections without normalizing the branch.
        std::array<char, 256> filename{};
        if (rank == 0)
            std::snprintf(filename.data(), filename.size(), "/tmp/qcs-measurement-%ld.bin", long(getpid()));
        MPI_Bcast(filename.data(), filename.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
        qcs_simulator_save_statevector_cxx(sim, filename.data());
        if (rank == 0)
        {
            FILE* file = std::fopen(filename.data(), "rb");
            require(file != nullptr, "open saved state");
            std::vector<std::complex<double>> state(1 << n);
            const size_t count = std::fread(state.data(), sizeof(state[0]), state.size(), file);
            std::fclose(file);
            std::remove(filename.data());
            require(count == state.size(), "saved state size");
            const unsigned selected = (branch ? (1u << n) - 1 : 0) ^ (1u << root);
            const double amplitude = branch ? std::sin(theta / 2) : std::cos(theta / 2);
            for (unsigned i = 0; i < state.size(); ++i)
                require(std::abs(state[i] - (i == selected ? amplitude : 0.0)) < 1e-11,
                        "saved projected branch and norm");
        }
        qcs_simulator_reset_cxx(sim, root);
        require(qcs_simulator_measure_cxx(sim, root) == 0, "reset after measurement");
        require(qcs_simulator_measure_many(sim, nullptr, 0, nullptr), "empty request");
        require(qcs_simulator_measure_many_to_clbits(sim, nullptr, 0, nullptr, 0, nullptr),
                "empty classical request");
        const bit_num_t duplicate[] = {2, 2}, invalid[] = {n};
        require(!qcs_simulator_measure_many(sim, duplicate, 2, results), "duplicate target rejection");
        require(!qcs_simulator_measure_many(sim, invalid, 1, results), "range rejection");
        require(!qcs_simulator_measure_many(sim, nullptr, 1, results), "null list rejection");
        require(!qcs_simulator_measure_many(sim, targets, -1, results), "negative count rejection");
        require(!qcs_simulator_measure_many(sim, targets, 1, nullptr), "null result rejection");
        require(!qcs_simulator_measure_many_to_clbits(sim, targets, 2, clbits, 1, results),
                "classical count rejection");
        require(!qcs_simulator_measure_many_to_clbits(sim, targets, 2, duplicate, 2, results),
                "duplicate classical destination rejection");
        if (rank == 0) std::puts("measurement integration passed");
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "measurement integration: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
