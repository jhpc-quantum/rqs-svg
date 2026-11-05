// Benchmark-only baseline and private access; production code is unmodified.
#include "../src/qcs.cu"
#include <iomanip>
#include <iostream>

namespace pd = qcs::pauli_detail;

__global__ void prepare_benchmark_state(pd::state_type* state, uint64_t count, uint64_t offset,
                                       double scale)
{
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * blockDim.x)
    {
        uint64_t logical = offset + i; // Identity mapping restored before launch.
        state[i] = pd::state_type(scale * double(1 + logical % 17) / 17,
                                 scale * double(int(logical % 13) - 6) / 13);
    }
}

// Same product/phase arithmetic and block reduction as the production kernel,
// but scan the full index domain and select pairs with an i < j branch.
__global__ void scan_pairs(const pd::state_type* state, uint64_t count,
                           pd::PauliStringDescriptor d, double* partials)
{
    __shared__ double scratch[pd::block_size / 32];
    double sum = 0;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += uint64_t(gridDim.x) * blockDim.x)
    {
        uint64_t j = i ^ d.flip_mask;
        if (i < j)
        {
            auto a = state[i], b = state[j];
            double value;
            if (d.y_count_mod4 & 1) value = b.imag() * a.real() - b.real() * a.imag();
            else value = b.real() * a.real() + b.imag() * a.imag();
            if (d.y_count_mod4 & 2) value = -value;
            value *= 2;
            sum += (__popcll(i & d.phase_mask) & 1) ? -value : value;
        }
    }
    double total = qcs::measurement_detail::block_sum(sum, scratch);
    if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

static double maximum(double value)
{
    double result;
    ATLC_CHECK_MPI(MPI_Allreduce, &value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
}
static void restore(qcs::simulator_core& c, unsigned blocks)
{
    c.discard_measurement_state();
    c.reinitialize_mapping();
    prepare_benchmark_state<<<blocks, pd::block_size, 0, c.stream>>>(
        c.state_data_device, c.num_states_local, c.num_states_local * c.proc_num,
        1 / std::sqrt(double(c.num_states)));
    ATLC_CHECK_CUDA(cudaGetLastError);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{
    auto* sim = qcs_simulator_create_cxx();
    try
    {
        auto& c = *sim->core;
        int n = argc > 1 ? std::stoi(argv[1]) : 20;
        int warmups = argc > 2 ? std::stoi(argv[2]) : 5;
        int repetitions = argc > 3 ? std::stoi(argv[3]) : 30;
        if (n < c.log_num_procs + 2 || n > 30 || warmups < 0 || repetitions < 1)
            throw std::runtime_error("arguments: total_qubits (at least two local, <=30), warmups, repetitions");
        qcs_simulator_set_num_qubits_cxx(sim, n);
        qcs_simulator_allocate_memory_cxx(sim);
        int device, runtime, driver, nccl;
        ATLC_CHECK_CUDA(cudaGetDevice, &device);
        ATLC_CHECK_CUDA(cudaRuntimeGetVersion, &runtime);
        ATLC_CHECK_CUDA(cudaDriverGetVersion, &driver);
        ATLC_CHECK_NCCL(ncclGetVersion, &nccl);
        cudaDeviceProp prop{};
        ATLC_CHECK_CUDA(cudaGetDeviceProperties, &prop, device);
        char host[MPI_MAX_PROCESSOR_NAME], mpi[MPI_MAX_LIBRARY_VERSION_STRING], pci[32];
        int host_len, mpi_len;
        ATLC_CHECK_MPI(MPI_Get_processor_name, host, &host_len);
        ATLC_CHECK_MPI(MPI_Get_library_version, mpi, &mpi_len);
        ATLC_CHECK_CUDA(cudaDeviceGetPCIBusId, pci, sizeof(pci), device);
        for (int rank = 0; rank < c.num_procs; ++rank)
        {
            ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
            if (rank == c.proc_num)
                std::cerr << "rank=" << rank << " host=" << std::string(host, host_len)
                    << " gpu=" << prop.name << " pci=" << pci << " SMs=" << prop.multiProcessorCount
                    << " compute=" << prop.major << '.' << prop.minor << " CUDA_runtime=" << runtime
                    << " CUDA_driver=" << driver << " NCCL=" << nccl << " MPI=" << std::string(mpi, mpi_len) << '\n';
        }
        if (!c.proc_num) std::cerr << "NVCC=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__
            << '.' << __CUDACC_VER_BUILD__ << " host_compiler=" << __VERSION__ << '\n';
        auto geometry = pd::resources(device, c.num_states_local);
        struct workload { const char* name; std::vector<bit_num_t> qubits; std::vector<qcs_pauli_operator> ops; bool global; };
        bit_num_t high = c.num_qubits_local - 1, mid = c.num_qubits_local / 2;
        std::vector<workload> cases = {
            {"diagonal", {high, 0}, {QCS_PAULI_Z, QCS_PAULI_Z}, false},
            {"pivot-low", {0}, {QCS_PAULI_X}, false},
            {"pivot-middle", {mid}, {QCS_PAULI_Y}, false},
            {"pivot-high", {high}, {QCS_PAULI_X}, false},
            {"sparse", {high, 0}, {QCS_PAULI_Y, QCS_PAULI_X}, false}};
        if (c.num_procs > 1) cases.push_back({"global-X", {bit_num_t(n - 1)}, {QCS_PAULI_X}, true});
        cudaEvent_t start, middle, stop;
        ATLC_CHECK_CUDA(cudaEventCreate, &start);
        ATLC_CHECK_CUDA(cudaEventCreate, &middle);
        ATLC_CHECK_CUDA(cudaEventCreate, &stop);
        if (!c.proc_num) std::cout << "qubits,ranks,case,path,method,sample,blocks,scratch_bytes,stage1_ms,stage2_ms,wall_ms,result\n";
        for (const auto& w : cases)
        {
            double api_result = 0;
            // Full production API timings include any SWAP, NCCL and readback.
            for (int sample = -warmups; sample < repetitions; ++sample)
            {
                restore(c, geometry.blocks);
                double begin = MPI_Wtime();
                double result = qcs_simulator_expectation_pauli_cxx(sim, w.qubits.data(), w.ops.data(), w.qubits.size(), 1);
                if (!std::isfinite(result)) throw std::runtime_error("nonfinite API result");
                api_result = result;
                double elapsed = maximum((MPI_Wtime() - begin) * 1000);
                if (sample >= 0 && !c.proc_num)
                    std::cout << n << ',' << c.num_procs << ',' << w.name << ",api,generated," << sample
                        << ",0," << c.measurement_workspace->capacity() * sizeof(double) << ",,,"
                        << std::setprecision(17) << elapsed << ',' << result << '\n';
            }
            if (w.global) continue;
            restore(c, geometry.blocks);
            auto d = pd::pauli_string(n, c.num_qubits_local, w.qubits.data(), w.ops.data(), w.qubits.size(), 1).descriptor(c.perm_l2p);
            auto config = pd::resources(device, pd::validate_descriptor(c.num_states_local, d));
            auto scratch = c.measurement_workspace->borrow_scalar(config.blocks, c.stream);
            // Interleave methods per sample to limit drift. Both use the same G,
            // scratch, input and final reduction; only index enumeration differs.
            for (int sample = -warmups; sample < repetitions; ++sample)
            {
                double generated = 0;
                for (int method = 0; method < (d.flip_mask ? 2 : 1); ++method)
                {
                    restore(c, geometry.blocks);
                    double begin = MPI_Wtime();
                    ATLC_CHECK_CUDA(cudaEventRecord, start, c.stream);
                    if (method) scan_pairs<<<config.blocks, pd::block_size, 0, c.stream>>>(c.state_data_device, c.num_states_local, d, scratch.partials);
                    else if (d.flip_mask) pd::first_stage<true><<<config.blocks, pd::block_size, 0, c.stream>>>(c.state_data_device, c.num_states_local, d, scratch.partials);
                    else pd::first_stage<false><<<config.blocks, pd::block_size, 0, c.stream>>>(c.state_data_device, c.num_states_local, d, scratch.partials);
                    ATLC_CHECK_CUDA(cudaGetLastError);
                    ATLC_CHECK_CUDA(cudaEventRecord, middle, c.stream);
                    pd::second_stage<<<1, pd::block_size, 0, c.stream>>>(scratch.partials, config.blocks, scratch.output);
                    ATLC_CHECK_CUDA(cudaGetLastError);
                    ATLC_CHECK_CUDA(cudaEventRecord, stop, c.stream);
                    ATLC_CHECK_NCCL(ncclAllReduce, scratch.output, scratch.output, 1, ncclDouble, ncclSum, c.nccl_comm, c.stream);
                    double result;
                    ATLC_CHECK_CUDA(cudaMemcpyAsync, &result, scratch.output, sizeof(result), cudaMemcpyDeviceToHost, c.stream);
                    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
                    double wall = maximum((MPI_Wtime() - begin) * 1000);
                    float first, second;
                    ATLC_CHECK_CUDA(cudaEventElapsedTime, &first, start, middle);
                    ATLC_CHECK_CUDA(cudaEventElapsedTime, &second, middle, stop);
                    double first_max = maximum(first), second_max = maximum(second);
                    if (std::abs(result - api_result) > 1e-12 + 5e-12 * std::abs(api_result))
                        throw std::runtime_error("kernel/API result mismatch");
                    if (!std::isfinite(result)) throw std::runtime_error("nonfinite benchmark result");
                    if (!method) generated = result;
                    else if (std::abs(result - generated) > 1e-12 + 5e-12 * std::abs(generated))
                        throw std::runtime_error("baseline/generated result mismatch");
                    if (sample >= 0 && !c.proc_num)
                        std::cout << n << ',' << c.num_procs << ',' << w.name << ",kernels,"
                            << (method ? "scan-branch" : "generated") << ',' << sample << ',' << config.blocks << ','
                            << c.measurement_workspace->capacity() * sizeof(double) << ',' << std::setprecision(17)
                            << first_max << ',' << second_max << ',' << wall << ',' << result << '\n';
                }
            }
        }
        ATLC_CHECK_CUDA(cudaEventDestroy, start);
        ATLC_CHECK_CUDA(cudaEventDestroy, middle);
        ATLC_CHECK_CUDA(cudaEventDestroy, stop);
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Pauli benchmark: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
