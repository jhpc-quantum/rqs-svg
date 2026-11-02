// Compiles a private, instrumented copy of either revision, never libqcs.
#include "backend.hpp"
#include "qcs.cu"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static double max_error = 0;
static uint64_t vectors_checked = 0;

// Independent physical enumeration. Do not use reducer descriptors or packing.
static void check_probabilities(qcs::simulator_core& c, int n, const int* targets,
                                const double* local, const double* global) {
    std::vector<qcs::complex_t> state(c.num_states_local);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, state.data(), c.state_data_device,
                   state.size() * sizeof(state[0]), cudaMemcpyDeviceToHost, c.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
    std::vector<long double> expected(1 << n), correction(1 << n), total(1 << n);
    for (uint64_t i = 0; i < c.num_states_local; ++i) {
        uint64_t physical = i | (uint64_t(c.proc_num) << c.num_qubits_local), logical = 0;
        for (int p = 0; p < c.num_qubits; ++p)
            logical |= ((physical >> p) & 1) << c.perm_p2l[p];
        if ((logical & c.measured_mask_logical) != c.measured_value_logical) continue;
        unsigned bin = 0;
        for (int j = 0; j < n; ++j) bin |= ((logical >> targets[j]) & 1) << j;
        long double re = state[i].real(), im = state[i].imag();
        long double y = re * re + im * im - correction[bin];
        long double next = expected[bin] + y;
        correction[bin] = (next - expected[bin]) - y;
        expected[bin] = next;
    }
    ATLC_CHECK_MPI(MPI_Allreduce, expected.data(), total.data(), 1 << n,
                   MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < (1 << n); ++i) {
        for (int kind = 0; kind < 2; ++kind) {
            const long double want = kind ? total[i] : expected[i];
            const double got = kind ? global[i] : local[i];
            long double error = std::abs(got - want);
            max_error = std::max(max_error, double(error));
            require(std::isfinite(got) && error <= 1e-12L + 5e-12L * std::abs(want),
                    "distributed CPU probability oracle mismatch");
            if (want == 0) require(got == 0, "excluded rank/bin must contribute exact zero");
        }
    }
    ++vectors_checked;
}

static void measure(qcs_simulator* sim, const std::vector<bit_num_t>& targets) {
    std::vector<bit_t> result(targets.size());
    qcs_simulator_measure_many_cxx(sim, targets.data(), targets.size(), result.data());
    for (auto bit : result) {
        int value = bit, low, high;
        ATLC_CHECK_MPI(MPI_Allreduce, &value, &low, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        ATLC_CHECK_MPI(MPI_Allreduce, &value, &high, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        require(low == high, "ranks sampled different results");
    }
}

#if MEASUREMENT_GPU_SAMPLING
#include "measurement_sampling_cases.hpp"
#endif

static std::vector<qcs::complex_t> prepared(qcs::simulator_core& c) {
    std::vector<qcs::complex_t> state(c.num_states_local);
    // Exactly reproducible, unequal masses; global scale bounds the total norm.
    const double scale = 1.0 / std::sqrt(double(c.num_states));
    for (uint64_t i = 0; i < c.num_states_local; ++i) {
        uint64_t global = i + c.num_states_local * c.proc_num;
        state[i] = qcs::complex_t(scale * (1 + global % 17) / 17,
                                 scale * (int(global % 13) - 6) / 13);
    }
    return state;
}
static void restore(qcs::simulator_core& c, const std::vector<qcs::complex_t>& state) {
    c.discard_measurement_state();
    ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                   state.size() * sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
}

static void regression(qcs_simulator* sim) {
    auto& c = *sim->core;
    auto state = prepared(c);
#if MEASUREMENT_GPU_SAMPLING
    qcs::measurement_detail::sample_result expected_sample{};
    measurement_probe::result_observer = [&](unsigned outcome, unsigned status) {
        require(outcome == expected_sample.outcome && status == static_cast<unsigned>(expected_sample.status),
                "production result differs from diagnostic sampler");
    };
#endif
    measurement_probe::observer = [&](int n, const int* t, const double* l, const double* g) {
        check_probabilities(c, n, t, l, g);
#if MEASUREMENT_GPU_SAMPLING
        sampling_cases::fixture f(c.stream);
        std::vector<double> weights(g, g + (1 << n));
        switch (n) {
#define CHECK_SAMPLE(N) case N: f.run<N>(weights, -1, qcs::measurement_detail::sample_status::success, 0, c.measurement_counter, c.rng_seed); break
            CHECK_SAMPLE(1); CHECK_SAMPLE(2); CHECK_SAMPLE(3); CHECK_SAMPLE(4);
#undef CHECK_SAMPLE
        }
        expected_sample = f.last_result;
#endif
    };
    // Every feasible N/L, with unsorted local/global physical positions and a
    // permutation crossing the local/global boundary. A second pass includes
    // both a prior local control and a prior global control excluding ranks.
    for (int n = 1; n <= 4; ++n) for (int l = 0; l <= n; ++l) {
        if (n - l > c.log_num_procs || l > c.num_qubits_local) continue;
        std::vector<bit_num_t> targets;
        const int positions[] = {0, c.num_qubits_local - 1, 2, 4};
        for (int j = 0; j < l; ++j) targets.push_back(c.perm_p2l[positions[j]]);
        for (int j = 0; j < n - l; ++j) targets.push_back(c.perm_p2l[c.num_qubits_local + j]);
        std::reverse(targets.begin(), targets.end());
        for (int controls = 0; controls < 2; ++controls) {
            restore(c, state);
            if (controls) {
                measure(sim, {bit_num_t(c.perm_p2l[1])}); // outside group
                if (c.log_num_procs) measure(sim, {bit_num_t(c.perm_p2l[c.num_qubits_local])});
                measure(sim, {targets.front()}); // inside group, pending projection
            }
            measure(sim, targets);
            measure(sim, targets); // cached outcome, no oracle invocation expected
        }
    }
    for (int count = 5; count <= 11; ++count) {
        restore(c, state);
        std::vector<bit_num_t> targets;
        for (int i = count - 1; i >= 0; --i) targets.push_back(i);
        measure(sim, targets); // all remainder widths, multiple complete groups
    }
    measurement_probe::observer = {};
    measurement_probe::result_observer = {};
    double error;
    ATLC_CHECK_MPI(MPI_Allreduce, &max_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!c.proc_num) std::cout << "oracle_vectors_per_rank=" << vectors_checked
                             << " max_absolute_error=" << std::setprecision(17) << error << '\n';
}

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[size_t(fraction * (values.size() - 1))];
}

static void benchmark(qcs_simulator* sim, int warmups, int repetitions) {
    auto& c = *sim->core;
    auto state = prepared(c);
    cudaDeviceProp prop{};
    ATLC_CHECK_CUDA(cudaGetDeviceProperties, &prop, c.gpu_id);
    int runtime, driver;
    ATLC_CHECK_CUDA(cudaRuntimeGetVersion, &runtime);
    ATLC_CHECK_CUDA(cudaDriverGetVersion, &driver);
    char mpi[MPI_MAX_LIBRARY_VERSION_STRING]; int length;
    ATLC_CHECK_MPI(MPI_Get_library_version, mpi, &length);
    if (!c.proc_num) {
        std::cerr << "GPU=" << prop.name << " ranks=" << c.num_procs
                  << " CUDA_runtime=" << runtime << " driver=" << driver
                  << " compiler=" << __VERSION__
                  << " nvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
                  << '.' << __CUDACC_VER_BUILD__ << " MPI=" << std::string(mpi, length) << '\n';
        std::cout << "backend,qubits,local_states,active_states,N,layout,control,S,G,dynamic_shared,static_shared,workspace_bytes,warmups,repetitions,kernel_p10_ms,kernel_median_ms,kernel_p90_ms,measurement_p10_ms,measurement_median_ms,measurement_p90_ms\n";
    }
    ATLC_CHECK_CUDA(cudaEventCreate, &measurement_probe::start);
    ATLC_CHECK_CUDA(cudaEventCreate, &measurement_probe::stop);
    for (int n = 1; n <= 4; ++n) for (int layout = 0; layout < 4; ++layout)
    for (int controlled = 0; controlled < 3; ++controlled) {
        if (layout == 3 && !c.log_num_procs) continue;
        if (controlled == 2 && (!c.log_num_procs || layout == 3)) continue;
        std::vector<bit_num_t> targets;
        for (int j = 0; j < n; ++j) {
            int p = layout == 0 ? j : layout == 1 ? c.num_qubits_local - 1 - j : 2 * j;
            if (layout == 3 && j == 0) p = c.num_qubits_local;
            targets.push_back(c.perm_p2l[p]);
        }
        auto conditioned = state;
        const int control_physical = controlled == 2 ? c.num_qubits_local : 5;
        if (controlled) for (uint64_t i = 0; i < c.num_states_local; ++i) {
            uint64_t physical = i | (uint64_t(c.proc_num) << c.num_qubits_local);
            if ((physical >> control_physical) & 1) conditioned[i] = qcs::complex_t(0, 0);
        }
        std::vector<double> kernels, measurements;
        for (int iteration = -warmups - 1; iteration < repetitions; ++iteration) {
            restore(c, conditioned);
            if (controlled) {
                // State preparation makes this branch deterministically zero.
                bit_num_t q = c.perm_p2l[control_physical];
                require(qcs_simulator_measure_cxx(sim, q) == 0, "control setup differs");
            }
            ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
            ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
            measurement_probe::enabled = true;
            measurement_probe::recorded = false;
            std::vector<bit_t> result(n);
            auto start = std::chrono::steady_clock::now();
            qcs_simulator_measure_many_cxx(sim, targets.data(), n, result.data());
            double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            measurement_probe::enabled = false;
            float kernel = 0;
            if (measurement_probe::recorded) {
                ATLC_CHECK_CUDA(cudaEventSynchronize, measurement_probe::stop);
                ATLC_CHECK_CUDA(cudaEventElapsedTime, &kernel, measurement_probe::start, measurement_probe::stop);
            }
            double kernel_local = kernel, kernel_max, elapsed_max;
            ATLC_CHECK_MPI(MPI_Allreduce, &kernel_local, &kernel_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            ATLC_CHECK_MPI(MPI_Allreduce, &elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (iteration >= 0) { kernels.push_back(kernel_max); measurements.push_back(elapsed_max); }
        }
        size_t workspace = 0, dynamic = 0, stat = 0; unsigned g = 0, s = 0;
#if MEASUREMENT_CUSTOM
        workspace = c.measurement_workspace->capacity() * sizeof(double);
        g = 8 * prop.multiProcessorCount; s = 256; dynamic = size_t(1 << n) * s * sizeof(double);
        cudaFuncAttributes attr{};
        // All widths share the same compiled static scratch shape; query N=4.
        ATLC_CHECK_CUDA(cudaFuncGetAttributes, &attr, reinterpret_cast<const void*>(qcs::measurement_detail::first_stage<4>));
        stat = attr.sharedSizeBytes;
#else
        workspace = c.cub_temp_buffer_device_size + c.measure_norm_device_size;
#endif
        if (!c.proc_num) {
            const char* layouts[] = {"low", "high", "noncontiguous", "mixed"};
            std::cout << (MEASUREMENT_CUSTOM ? "custom" : "CUB") << ',' << c.num_qubits << ','
                      << c.num_states_local << ',' << (c.num_states_local >> (controlled == 1)) << ',' << n << ','
                      << layouts[layout] << ',' << (controlled == 2 ? "first_global=0" : controlled ? "physical5=0" : "none") << ','
                      << s << ',' << g << ',' << dynamic << ',' << stat << ',' << workspace << ',' << warmups << ',' << repetitions;
            for (const auto& v : {kernels, measurements}) for (double p : {0.1, 0.5, 0.9})
                std::cout << ',' << std::setprecision(9) << percentile(v, p);
            std::cout << '\n';
        }
    }
    ATLC_CHECK_CUDA(cudaEventDestroy, measurement_probe::start);
    ATLC_CHECK_CUDA(cudaEventDestroy, measurement_probe::stop);
}

int main(int argc, char** argv) {
    qcs_simulator* sim = qcs_simulator_create_cxx();
    try {
        bool bench = argc > 1 && std::string(argv[1]) == "benchmark";
        int qubits = argc > 2 ? std::stoi(argv[2]) : 12;
        int warmups = argc > 3 ? std::stoi(argv[3]) : 5;
        int repetitions = argc > 4 ? std::stoi(argv[4]) : 30;
        require(qubits >= 12 && qubits <= 30 && warmups >= 0 && repetitions > 0, "invalid harness arguments");
        require(sim->core->num_procs == 1 || sim->core->num_procs == 2 || sim->core->num_procs == 4,
                "harness supports 1, 2 or 4 ranks, one GPU per rank");
        qcs_simulator_set_num_qubits_cxx(sim, qubits);
        qcs_simulator_set_num_clbits_cxx(sim, qubits);
        std::vector<bit_num_t> mapping(qubits);
        for (int p = 0; p < qubits; ++p) mapping[p] = qubits - p - 1;
        qcs_simulator_set_mapping_cxx(sim, mapping.data(), qubits);
        qcs_simulator_allocate_memory_cxx(sim);
        if (bench) benchmark(sim, warmups, repetitions);
        else {
#if MEASUREMENT_GPU_SAMPLING
            sampling_cases::run(sim);
            sampling_cases::lifecycle(sim);
#endif
            regression(sim);
        }
        qcs_simulator_destroy_cxx(sim);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "measurement harness: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
