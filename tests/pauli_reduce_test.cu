#include "../src/pauli_reduce.cuh"
#include "pauli_reference.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace pd = qcs::pauli_detail;
using State = pd::state_type;

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template<class F> static void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid reducer input accepted");
}

struct fixture
{
    cudaStream_t stream{};
    State* state = nullptr;
    double* scratch = nullptr;
    uint64_t capacity = 0;
    uint64_t cases = 0;
    static constexpr uint64_t max_dim = UINT64_C(1) << 20;
    ~fixture()
    {
        if (state) cudaFreeAsync(state, stream);
        if (scratch) cudaFreeAsync(scratch, stream);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
    void init()
    {
        ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
        int device;
        ATLC_CHECK_CUDA(cudaGetDevice, &device);
        capacity = pd::resources(device, max_dim).blocks + 1;
        ATLC_CHECK_CUDA(cudaMallocAsync, &state, max_dim * sizeof(State), stream);
        ATLC_CHECK_CUDA(cudaMallocAsync, &scratch, (capacity + 1) * sizeof(double), stream);
        require(pd::resources(device, 1).blocks == 1, "small state launched unnecessary blocks");
        uint64_t work = 257;
        require(pd::resources(device, work).blocks == 2, "small tail grid was not clamped");
    }
    void compare(const std::vector<pauli_test::complex>& input,
                 const std::vector<bit_num_t>& qubits,
                 const std::vector<qcs_pauli_operator>& ops, double coefficient = 1)
    {
        int n = 0;
        for (size_t size = input.size(); size > 1; size >>= 1) ++n;
        // A one-amplitude local slice is permitted even with no local qubits.
        int total = std::max(1, n);
        std::vector<int> mapping(total);
        std::iota(mapping.begin(), mapping.end(), 0);
        auto d = pd::pauli_string(total, n, qubits.data(), ops.data(),
                                  static_cast<bit_num_t>(qubits.size()), coefficient).descriptor(mapping);
        std::vector<State> upload;
        std::vector<pauli_test::complex> rounded;
        for (const auto& a : input)
        {
            // Oracle sees exactly the double amplitudes uploaded to the GPU.
            double re = static_cast<double>(a.real()), im = static_cast<double>(a.imag());
            upload.emplace_back(re, im);
            rounded.emplace_back(re, im);
        }
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state, upload.data(), upload.size() * sizeof(State),
                        cudaMemcpyHostToDevice, stream);
        ATLC_CHECK_CUDA(cudaMemsetAsync, scratch, 0xff, (capacity + 1) * sizeof(double), stream);
        auto* output = pd::enqueue(state, upload.size(), d, scratch, capacity, stream);
        require(output == scratch + capacity - 1, "output location changed");
        double raw = 0;
        uint64_t guard = 0;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, &raw, output, sizeof(raw), cudaMemcpyDeviceToHost, stream);
        ATLC_CHECK_CUDA(cudaMemcpyAsync, &guard, scratch + capacity, sizeof(guard), cudaMemcpyDeviceToHost, stream);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        long double expected = pauli_test::expectation(rounded, qubits, ops);
        require(std::isfinite(raw) && std::abs(raw - expected) <= 1e-12L + 5e-12L * std::abs(expected),
                "GPU/CPU Pauli expectation mismatch");
        require(guard == UINT64_MAX, "reducer overwrote workspace guard");
        // Coefficient must be left untouched until the future global sum.
        require(std::abs(coefficient * raw - coefficient * expected) <=
                1e-12L + 5e-12L * std::abs(coefficient * expected), "coefficient result mismatch");
        ++cases;
    }
};

static void invalid_inputs(fixture& f)
{
    pd::PauliStringDescriptor d{1, 0, 0, 0};
    rejects([&] { pd::validate_descriptor(0, d); });
    rejects([&] { pd::validate_descriptor(3, d); });
    rejects([&] { pd::validate_descriptor(2, {1, 2, 0, 0}); });
    rejects([&] { pd::validate_descriptor(2, {1, 0, 2, 0}); });
    rejects([&] { pd::validate_descriptor(2, {1, 1, 1, 0}); });
    rejects([&] { pd::validate_descriptor(2, {1, 0, 0, 4}); });
    rejects([&] { pd::validate_descriptor(2, {std::numeric_limits<double>::infinity(), 0, 0, 0}); });
    rejects([&] { pd::enqueue(nullptr, 2, d, f.scratch, f.capacity, f.stream); });
    rejects([&] { pd::enqueue(f.state, 2, d, nullptr, f.capacity, f.stream); });
    rejects([&] { pd::enqueue(f.state, 2, d, f.scratch, 1, f.stream); });
    require(pd::validate_descriptor(UINT64_C(1) << 63, {1, UINT64_C(1) << 62, 0, 0}) ==
            (UINT64_C(1) << 62), "wide work count truncated");
}

static void reduction_tail(fixture& f)
{
    // More partials than threads, even on GPUs where 4*SM_count <= 256.
    constexpr unsigned count = 1025;
    std::vector<double> partials(count);
    long double expected = 0;
    for (unsigned i = 0; i < count; ++i) expected += partials[i] = (int(i % 7) - 3) * 0.125;
    double* device = nullptr;
    ATLC_CHECK_CUDA(cudaMallocAsync, &device, (count + 1) * sizeof(double), f.stream);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, device, partials.data(), count * sizeof(double), cudaMemcpyHostToDevice, f.stream);
    pd::second_stage<<<1, pd::block_size, 0, f.stream>>>(device, count, device + count);
    ATLC_CHECK_CUDA(cudaGetLastError);
    double result = 0;
    ATLC_CHECK_CUDA(cudaMemcpyAsync, &result, device + count, sizeof(result), cudaMemcpyDeviceToHost, f.stream);
    ATLC_CHECK_CUDA(cudaFreeAsync, device, f.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, f.stream);
    require(result == expected, "Stage 2 lost partials beyond one block");
}

int main()
{
    try
    {
        fixture f;
        f.init();
        invalid_inputs(f);
        reduction_tail(f);
        f.compare({2}, {}, {}, -3);
        f.compare({1, pauli_test::complex(0, 1)}, {0}, {QCS_PAULI_Y});
        f.compare({1, pauli_test::complex(0, -1)}, {0}, {QCS_PAULI_Y});
        std::mt19937_64 random(12345);
        std::uniform_real_distribution<double> distribution(-1, 1);
        for (int n = 1; n <= 4; ++n)
        {
            std::vector<pauli_test::complex> input(size_t(1) << n);
            for (auto& a : input) a = {distribution(random), distribution(random)};
            std::vector<bit_num_t> qubits(n);
            std::iota(qubits.begin(), qubits.end(), 0);
            for (unsigned encoding = 0; encoding < (1u << (2 * n)); ++encoding)
            {
                std::vector<qcs_pauli_operator> ops(n);
                for (int q = 0; q < n; ++q) ops[q] = static_cast<qcs_pauli_operator>((encoding >> (2 * q)) & 3);
                f.compare(input, qubits, ops, encoding % 2 ? -2.5 : 0);
            }
        }
        f.compare({1, 0, 0, 1}, {0, 1}, {QCS_PAULI_X, QCS_PAULI_X});
        f.compare({1, 0, 0, 1}, {0, 1}, {QCS_PAULI_Y, QCS_PAULI_Y});
        std::vector<pauli_test::complex> ghz(64);
        ghz.front() = ghz.back() = 1 / std::sqrt(2.0L);
        f.compare(ghz, {0, 1, 2, 3, 4, 5}, std::vector<qcs_pauli_operator>(6, QCS_PAULI_X));
        f.compare(ghz, {5, 0, 3, 2}, std::vector<qcs_pauli_operator>(4, QCS_PAULI_Y));
        // Large grid-stride domain, sparse/non-contiguous masks, varied pivots,
        // and repeated reuse across diagonal/non-diagonal launches.
        for (int n : {9, 12, 20})
        {
            std::vector<pauli_test::complex> input(size_t(1) << n);
            double scale = 1 / std::sqrt(double(input.size()));
            for (auto& a : input) a = {scale * distribution(random), scale * distribution(random)};
            f.compare(input, {}, {});
            for (int q : {0, n / 2, n - 1})
            {
                f.compare(input, {static_cast<bit_num_t>(q)}, {QCS_PAULI_Z});
                f.compare(input, {static_cast<bit_num_t>(q)}, {QCS_PAULI_X});
                f.compare(input, {static_cast<bit_num_t>(q)}, {QCS_PAULI_Y});
            }
            f.compare(input, {static_cast<bit_num_t>(n - 1), 0}, {QCS_PAULI_Y, QCS_PAULI_X});
            for (int repetition = 0; repetition < 12 && n < 20; ++repetition)
            {
                std::vector<bit_num_t> qubits(n);
                std::vector<qcs_pauli_operator> ops(n);
                std::iota(qubits.begin(), qubits.end(), 0);
                for (auto& op : ops) op = static_cast<qcs_pauli_operator>(random() % 4);
                f.compare(input, qubits, ops);
            }
        }
        std::cout << "Pauli reducer: " << f.cases << " GPU/oracle cases passed\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Pauli reducer: " << e.what() << '\n';
        return 1;
    }
}
