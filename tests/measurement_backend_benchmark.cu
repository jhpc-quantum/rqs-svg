// Private access follows pauli_benchmark.cu; no diagnostic oracle in timed paths.
#include "../src/qcs.cu"
#include <iomanip>
#include <iostream>

__global__ void prepare_measurement_state(qcs::complex_t* state, uint64_t count,
                                         uint64_t offset, double scale, int kind) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * blockDim.x) {
        uint64_t g = offset + i;
        double weight = kind == 0 ? (g == 0 ? 1.0 : 0.0) :
                        kind == 1 ? scale : scale * double(1 + g % 17) / 17;
        if (kind == 3) weight = (g % 257 == 0) ? scale : 0;
        if (kind == 4) weight = scale * (offset == 0 ? 16.0 : 1.0);
        state[i] = qcs::complex_t(weight, 0);
    }
}
static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[size_t(p * (v.size() - 1))];
}
int main(int argc, char** argv) {
    auto* sim = qcs_simulator_create_cxx();
    try {
        auto& c = *sim->core;
        int n = argc > 1 ? std::stoi(argv[1]) : 33;
        int warmups = argc > 2 ? std::stoi(argv[2]) : 5;
        int repetitions = argc > 3 ? std::stoi(argv[3]) : 30;
        bool global_only = argc > 4 && std::string(argv[4]) == "global-only";
        if (global_only && c.num_procs == 1)
            throw std::runtime_error("global-only benchmark requires multiple ranks");
        if (n < c.log_num_procs + 1 || n > 33 || warmups < 1 || repetitions < 1 ||
            c.num_procs > 128 || (c.num_procs & (c.num_procs - 1)))
            throw std::runtime_error("expected qubits <=33, >=1 local qubit, positive warmups/repetitions, 1..128 power-of-two ranks");
        qcs_simulator_set_num_qubits_cxx(sim, n);
        qcs_simulator_set_num_clbits_cxx(sim, n);
        qcs_simulator_allocate_memory_cxx(sim);
        MPI_Comm shared;
        ATLC_CHECK_MPI(MPI_Comm_split_type, MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared);
        int local_rank;
        ATLC_CHECK_MPI(MPI_Comm_rank, shared, &local_rank);
        int leader = local_rank == 0, nodes;
        ATLC_CHECK_MPI(MPI_Allreduce, &leader, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        ATLC_CHECK_MPI(MPI_Comm_free, &shared);
        cudaDeviceProp prop{};
        ATLC_CHECK_CUDA(cudaGetDeviceProperties, &prop, c.gpu_id);
        int driver_version = 0, runtime_version = 0, nccl_version = 0;
        ATLC_CHECK_CUDA(cudaDriverGetVersion, &driver_version);
        ATLC_CHECK_CUDA(cudaRuntimeGetVersion, &runtime_version);
        ATLC_CHECK_NCCL(ncclGetVersion, &nccl_version);
        char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
        int mpi_version_length = 0;
        ATLC_CHECK_MPI(MPI_Get_library_version, mpi_version, &mpi_version_length);
        char hostname[MPI_MAX_PROCESSOR_NAME];
        int hostname_length = 0;
        ATLC_CHECK_MPI(MPI_Get_processor_name, hostname, &hostname_length);
        // Each rank records its placement; stderr is retained by the runner.
        std::cerr << "rank=" << c.proc_num << " host=" << hostname
                  << " local_rank=" << local_rank << " device=" << c.gpu_id << '\n';
        if (!c.proc_num) {
            std::cerr << "CUDA_driver=" << driver_version << " CUDA_runtime=" << runtime_version
                      << " NCCL=" << nccl_version << " MPI="
                      << std::string(mpi_version, mpi_version_length) << '\n';
            std::cerr << "GPU=" << prop.name << " ranks=" << c.num_procs << " nodes=" << nodes << '\n';
            std::cout << "backend,qubits,gpus,nodes,state,targets,layout,deferred,mapping,api,warmups,repetitions,first_call_ms,p10_ms,median_ms,p90_ms,samples_ms\n";
        }
        std::vector<int> counts{n,1,2,3,4,5,7,8,9,15,16,17,24,32};
        std::sort(counts.begin(), counts.end(), std::greater<int>());
        counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
        const char* states[] = {"zero", "uniform", "nonuniform", "sparse", "skewed"};
        for (int kind = 0; kind < 5; ++kind) for (int count : counts)
        for (int mapping = 0; mapping < 3; ++mapping)
        for (int layout = 0; layout < 4; ++layout) for (int deferred = 0; deferred < 4; ++deferred)
        for (int api = 0; api < 2; ++api) {
            if (global_only && (layout != 1 || deferred == 2)) continue;
            if (count > n || (layout == 0 && count > c.num_qubits_local) ||
                (layout == 1 && count > c.log_num_procs) ||
                (layout == 2 && (count < 2 || !c.log_num_procs || count > n)) ||
                (layout == 3 && count != n)) continue;
            std::vector<bit_num_t> targets, locals, globals;
            // Even/odd order gives separated, noncontiguous subsets when feasible.
            for (int parity = 0; parity < 2; ++parity) {
                for (int p = parity; p < c.num_qubits_local; p += 2) locals.push_back(p);
                for (int p = parity; p < c.log_num_procs; p += 2) globals.push_back(c.num_qubits_local + p);
            }
            if (layout == 0) targets = locals;
            else if (layout == 1) targets = globals;
            else {
                for (size_t i = 0; i < std::max(locals.size(), globals.size()); ++i) {
                    if (i < globals.size()) targets.push_back(globals[i]);
                    if (i < locals.size()) targets.push_back(locals[i]);
                }
            }
            targets.resize(count);
            // Exercise identity, rotation and reversal across the rank boundary.
            c.initial_perm_p2l.resize(n);
            for (int p = 0; p < n; ++p) c.initial_perm_p2l[p] = mapping == 2 ? n - 1 - p : mapping == 1 ? (p + 1) % n : p;
            c.reinitialize_mapping();
            for (auto& q : targets) q = c.perm_p2l[q];
            std::vector<bit_t> result(count);
            std::vector<bit_num_t> clbits(count);
            for (int j = 0; j < count; ++j) clbits[j] = n - 1 - j;
            std::vector<double> samples;
            double first_call = 0;
            for (int iteration = -warmups - 1; iteration < repetitions; ++iteration) {
                c.discard_measurement_state();
                c.reinitialize_mapping();
                prepare_measurement_state<<<256,256,0,c.stream>>>(c.state_data_device,
                    c.num_states_local, c.num_states_local * c.proc_num,
                    1.0 / std::sqrt(double(c.num_states)), kind);
                ATLC_CHECK_CUDA(cudaGetLastError);
                // Install the same zero-outcome branch for every backend. All test
                // distributions have positive mass at index zero. No RNG in setup.
                std::vector<bit_num_t> cached;
                if (deferred == 2) cached = targets;
                if (deferred == 1) cached.assign(targets.begin(), targets.begin() + std::max(1, count / 2));
                // Condition on low/high local physical bits, even outside the request.
                if (deferred == 3) {
                    cached.push_back(c.perm_p2l[0]);
                    if (c.num_qubits_local > 1) cached.push_back(c.perm_p2l[c.num_qubits_local - 1]);
                }
                for (auto q : cached) {
                    uint64_t bit = UINT64_C(1) << q;
                    c.measured_mask_logical |= bit;
                    c.pending_projection_mask_logical |= bit;
                    c.measured_0_qubit_num_logical_list.push_back(q);
                }
                ATLC_CHECK_CUDA(cudaDeviceSynchronize);
                ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
                double elapsed;
                {
                    QCS_MEASURE_RANGE("measurement_api");
                    auto start = std::chrono::steady_clock::now();
                    if (api == 0)
                        qcs_simulator_measure_many_cxx(sim, targets.data(), count, result.data());
                    else
                        qcs_simulator_measure_many_to_clbits_cxx(sim, targets.data(), count,
                            clbits.data(), count, result.data());
                    ATLC_CHECK_CUDA(cudaDeviceSynchronize);
                    elapsed = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                }
                double maximum;
                ATLC_CHECK_MPI(MPI_Allreduce, &elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (iteration == -warmups - 1) first_call = maximum;
                if (iteration >= 0) samples.push_back(maximum);
            }
            if (!c.proc_num) {
                const char* layouts[] = {"local", "global", "mixed", "full"};
                const char* modes[] = {"none", "partial", "all", "local_low_high"};
                std::cout << (c.tiled_race_nccl_measurement ? "tiled_race_nccl" : c.tiled_race_measurement ? "tiled_race" : c.tiled_cdf_measurement ? "tiled_cdf" : c.auto_measurement ? "auto" : c.rank_sampled_measurement ? "rank_sampled" : "grouped") << ',' << n << ','
                    << c.num_procs << ',' << nodes << ',' << states[kind] << ',' << count << ',' << layouts[layout]
                    << ',' << modes[deferred] << ',' << (mapping == 2 ? "reversed" : mapping == 1 ? "rotated" : "identity") << ',' << (api ? "measure_many_to_clbits" : "measure_many") << ',' << warmups << ',' << repetitions << std::setprecision(12);
                std::cout << ',' << first_call;
                for (double p : {0.1,0.5,0.9}) std::cout << ',' << percentile(samples,p);
                std::cout << ",\"";
                for (size_t i=0; i<samples.size(); ++i) std::cout << (i ? ";" : "") << samples[i];
                std::cout << "\"\n" << std::flush;
            }
        }
        qcs_simulator_destroy_cxx(sim);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"measurement benchmark: %s\n",e.what());
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
