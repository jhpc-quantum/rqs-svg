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
        if (!c.proc_num) {
            std::cerr << "GPU=" << prop.name << " ranks=" << c.num_procs << " nodes=" << nodes << '\n';
            std::cout << "backend,qubits,gpus,nodes,state,targets,layout,deferred,warmups,repetitions,p10_ms,median_ms,p90_ms,samples_ms\n";
        }
        std::vector<int> counts{1,2,4,8,16,24,32,n};
        std::sort(counts.begin(), counts.end());
        counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
        const char* states[] = {"zero", "uniform", "nonuniform"};
        for (int kind = 0; kind < 3; ++kind) for (int count : counts)
        for (int layout = 0; layout < 4; ++layout) for (int deferred = 0; deferred < 3; ++deferred) {
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
            std::vector<bit_t> result(count);
            std::vector<double> samples;
            for (int iteration = -warmups; iteration < repetitions; ++iteration) {
                c.discard_measurement_state();
                c.reinitialize_mapping();
                prepare_measurement_state<<<256,256,0,c.stream>>>(c.state_data_device,
                    c.num_states_local, c.num_states_local * c.proc_num,
                    1.0 / std::sqrt(double(c.num_states)), kind);
                ATLC_CHECK_CUDA(cudaGetLastError);
                // partial = one cached target plus fresh targets; all = cached API call.
                if (deferred == 1) qcs_simulator_measure_cxx(sim, targets.front());
                if (deferred == 2) qcs_simulator_measure_many_cxx(sim, targets.data(), count, result.data());
                ATLC_CHECK_CUDA(cudaDeviceSynchronize);
                ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
                QCS_MEASURE_RANGE("measurement_api");
                auto start = std::chrono::steady_clock::now();
                qcs_simulator_measure_many_cxx(sim, targets.data(), count, result.data());
                ATLC_CHECK_CUDA(cudaDeviceSynchronize);
                double elapsed = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                double maximum;
                ATLC_CHECK_MPI(MPI_Allreduce, &elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (iteration >= 0) samples.push_back(maximum);
            }
            if (!c.proc_num) {
                const char* layouts[] = {"local", "global", "mixed", "full"};
                const char* modes[] = {"none", "partial", "all"};
                std::cout << (c.rank_sampled_measurement ? "rank_sampled" : "grouped") << ',' << n << ','
                    << c.num_procs << ',' << nodes << ',' << states[kind] << ',' << count << ',' << layouts[layout]
                    << ',' << modes[deferred] << ',' << warmups << ',' << repetitions << std::setprecision(12);
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
