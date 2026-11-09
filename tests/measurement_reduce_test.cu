#include "../src/measurement_reduce.cuh"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

namespace md = qcs::measurement_detail;
using State = md::state_type;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void rejects(F action)
{
    bool rejected = false;
    try { action(); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid descriptor/resource size was accepted");
}

// Independent oracle: enumerate PHYSICAL indices, filter a mask, and use
// compensated long-double accumulation. It never calls pack/physical_index.
template<int N>
std::array<long double, (1 << N)> oracle(const std::vector<State>& state,
                                        const md::descriptor<N>& d)
{
    std::array<long double, (1 << N)> sums{}, corrections{};
    for (uint64_t i = 0; i < state.size(); ++i)
    {
        if ((i & d.control_mask) != d.control_value) continue;
        unsigned outcome = static_cast<unsigned>(d.fixed_outcome);
        for (int j = 0; j < d.local_targets; ++j)
            outcome |= unsigned((i >> d.local_target_physical[j]) & 1) << d.local_outcome_bits[j];
        long double re = state[i].real(), im = state[i].imag();
        long double term = re * re + im * im - corrections[outcome];
        long double next = sums[outcome] + term;
        corrections[outcome] = (next - sums[outcome]) - term;
        sums[outcome] = next;
    }
    return sums;
}

struct device_state
{
    cudaStream_t stream;
    State* data = nullptr;
    uint64_t capacity = 0;
    explicit device_state(cudaStream_t s) : stream(s) {}
    ~device_state() { if (data) cudaFreeAsync(data, stream); }
    void upload(const std::vector<State>& values)
    {
        if (values.size() > capacity)
        {
            State* next = nullptr;
            ATLC_CHECK_CUDA(cudaMallocAsync, &next, md::allocation_bytes(values.size(), sizeof(State)), stream);
            if (data) ATLC_CHECK_CUDA(cudaFreeAsync, data, stream);
            data = next;
            capacity = values.size();
        }
        ATLC_CHECK_CUDA(cudaMemcpyAsync, data, values.data(), values.size() * sizeof(State),
                       cudaMemcpyHostToDevice, stream);
    }
};

uint64_t cases = 0;
template<int N>
std::array<double, (1 << N)> compare(md::workspace& workspace, device_state& input,
                                    const std::vector<State>& state,
                                    const md::descriptor<N>& d, bool poison = true)
{
    input.upload(state);
    workspace.reserve<N>(input.stream);
    if (poison)
        ATLC_CHECK_CUDA(cudaMemsetAsync, workspace.data(), 0xff,
                       md::allocation_bytes(workspace.capacity(), sizeof(double)), input.stream);
    auto* old_data = workspace.data();
    auto old_capacity = workspace.capacity();
    auto* result = workspace.enqueue(input.data, state.size(), d, input.stream);
    require(workspace.data() == old_data && workspace.capacity() == old_capacity,
            "steady-state invocation unexpectedly grew workspace");
    std::array<double, (1 << N)> gpu{};
    ATLC_CHECK_CUDA(cudaMemcpyAsync, gpu.data(), result, sizeof(gpu), cudaMemcpyDeviceToHost, input.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, input.stream);
    auto expected = oracle(state, d);
    for (unsigned bin = 0; bin < gpu.size(); ++bin)
    {
        // Positive squared norms: abs + relative tolerance, no CUB bitwise rule.
        constexpr long double absolute = 1e-12L, relative = 5e-12L;
        if (!std::isfinite(gpu[bin]) ||
            std::abs(static_cast<long double>(gpu[bin]) - expected[bin]) >
                absolute + relative * std::abs(expected[bin]))
            throw std::runtime_error("GPU/oracle mismatch in case " + std::to_string(cases) +
                                     ", N=" + std::to_string(N) + ", bin=" + std::to_string(bin));
        if (expected[bin] == 0) require(gpu[bin] == 0, "zero bin not overwritten exactly");
    }
    // Reducer contract includes read-only state, even for zero/unnormalized input.
    std::vector<State> after(state.size());
    ATLC_CHECK_CUDA(cudaMemcpyAsync, after.data(), input.data, state.size() * sizeof(State),
                   cudaMemcpyDeviceToHost, input.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, input.stream);
    for (size_t i = 0; i < state.size(); ++i)
        require(after[i] == state[i], "reducer modified the state");
    ++cases;
    return gpu;
}

std::vector<State> nonuniform(size_t count)
{
    std::vector<State> result(count);
    for (size_t i = 0; i < count; ++i)
        result[i] = State((double(i % 131) - 61.0) / 19.0,
                          (double((i * 17) % 97) - 41.0) / 13.0);
    return result;
}

template<int N>
void fixtures(md::workspace& workspace, device_state& input, size_t large_count)
{
    for (int local = 0; local <= N; ++local)
    {
        // Targets are intentionally noncontiguous; both control values are set.
        const int physical[] = {7, 2, 5, 0};
        md::descriptor<N> d;
        d.local_targets = local;
        d.control_mask = (UINT64_C(1) << 1) | (UINT64_C(1) << 6);
        d.control_value = UINT64_C(1) << 6;
        uint64_t variable = 0;
        for (int j = 0; j < local; ++j)
        {
            d.local_target_physical[j] = physical[j];
            d.local_outcome_bits[j] = N - 1 - j;
            variable |= UINT64_C(1) << d.local_outcome_bits[j];
        }
        d.fixed_outcome = ((UINT64_C(1) << N) - 1) & ~variable;
        compare(workspace, input, nonuniform(256), d);
        auto large = nonuniform(large_count);
        compare(workspace, input, large, d, false); // preceding output is finite, different data
        // Reverse physical target order while retaining output bit positions.
        std::reverse(d.local_target_physical.begin(), d.local_target_physical.begin() + local);
        d.control_value = UINT64_C(1) << 1; // also exercise the opposite control polarity
        compare(workspace, input, large, d);
        d.fixed_outcome = 0;
        std::vector<State> sparse(256);
        sparse[2] = State(3, -4); // eligible, norm=25, unnormalized
        sparse[3] = State(-2, 7); // another eligible state
        sparse[0] = State(100, 100); // rejected by a control outside the group
        compare(workspace, input, sparse, d);
        std::fill(sparse.begin(), sparse.end(), State(0, 0));
        sparse[2] = State(0, -2); // basis state and exact impossible bins
        compare(workspace, input, sparse, d);
        std::fill(sparse.begin(), sparse.end(), State(0, 0));
        compare(workspace, input, sparse, d, false); // must clear ALL previous results
        compare(workspace, input, sparse, d); // NaN-poisoned partials/output

        // Smallest feasible domain for each L: 1,2,4,8,16 amplitudes.
        md::descriptor<N> tiny;
        tiny.local_targets = local;
        for (int j = 0; j < local; ++j)
        {
            tiny.local_target_physical[j] = local - j - 1;
            tiny.local_outcome_bits[j] = j;
        }
        tiny.fixed_outcome = ((UINT64_C(1) << N) - 1) & ~((UINT64_C(1) << local) - 1);
        compare(workspace, input, nonuniform(size_t(1) << local), tiny);
        if (local == 0)
        {
            tiny.control_mask = 255;
            tiny.control_value = 173;
            compare(workspace, input, nonuniform(256), tiny); // exactly ONE eligible amplitude
        }
    }
}

// Consecutive groups shrink the live stride in the same poisoned/reused buffer.
// Permuted physical order makes controls noncontiguous during the sweep.
template<int N>
void shrinking_groups(md::workspace& workspace, device_state& input)
{
    auto state = nonuniform(4096);
    md::descriptor<N> d;
    const int order[] = {11, 0, 9, 2, 7, 4, 5, 6, 3, 8, 1, 10};
    for (int begin = 0; begin < 12; begin += N)
    {
        d.local_targets = std::min(N, 12 - begin);
        for (int j = 0; j < d.local_targets; ++j)
        {
            d.local_target_physical[j] = order[begin + j];
            d.local_outcome_bits[j] = j;
        }
        compare(workspace, input, state, d, begin == 0);
        for (int j = 0; j < d.local_targets; ++j)
        {
            uint64_t bit = UINT64_C(1) << d.local_target_physical[j];
            d.control_mask |= bit;
            if ((begin + j) % 2) d.control_value |= bit;
        }
    }
    d.local_targets = 0;
    compare(workspace, input, state, d, false); // one remaining active element
}

// Check dispatch itself as well as values: capture after reserve so allocation
// and resource discovery are outside capture. Never execute the captured graph.
template<int N>
void single_block_dispatch(md::workspace& workspace, device_state& input)
{
    auto state = nonuniform(4096);
    input.upload(state);
    auto config = workspace.reserve<N>(input.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, input.stream);
    // Noncontiguous physical targets, reversed outcome positions, fixed bits,
    // and prior controls of both polarities. Sweep across the one-block edge.
    for (int local = 0; local <= N; ++local)
    for (unsigned active_bits = static_cast<unsigned>(local); active_bits <= 10; ++active_bits)
    {
        const int order[] = {9, 2, 7, 0, 11, 4, 1, 10, 3, 8, 5, 6};
        md::descriptor<N> d;
        d.local_targets = local;
        d.fixed_outcome = ((UINT64_C(1) << N) - 1) &
                          ~(((UINT64_C(1) << local) - 1) << (N - local));
        for (int j = 0; j < local; ++j)
        {
            d.local_target_physical[j] = order[j];
            d.local_outcome_bits[j] = N - 1 - j;
        }
        for (unsigned j = active_bits; j < 12; ++j)
        {
            d.control_mask |= UINT64_C(1) << order[j];
            if (j % 2) d.control_value |= UINT64_C(1) << order[j];
        }
        auto packed = md::pack(d, state.size());
        unsigned blocks = md::measurement_blocks(packed.active_count, config.blocks);
        cudaGraph_t graph = nullptr;
        ATLC_CHECK_CUDA(cudaStreamBeginCapture, input.stream, cudaStreamCaptureModeThreadLocal);
        auto* output = workspace.enqueue(input.data, state.size(), d, input.stream);
        ATLC_CHECK_CUDA(cudaStreamEndCapture, input.stream, &graph);
        size_t nodes = 0;
        ATLC_CHECK_CUDA(cudaGraphGetNodes, graph, nullptr, &nodes);
        ATLC_CHECK_CUDA(cudaGraphDestroy, graph);
        require(nodes == (blocks == 1 ? 1u : 2u), "unexpected reduction launch count");
        require(output == workspace.data() + (UINT64_C(1) << N) * config.blocks,
                "registered output offset changed");
        auto actual = compare(workspace, input, state, d);
        // Explicit pre-optimization path, including poisoned output storage.
        ATLC_CHECK_CUDA(cudaMemsetAsync, workspace.data(), 0xff,
                       workspace.capacity() * sizeof(double), input.stream);
        md::first_stage<N><<<blocks, md::block_size, config.dynamic_shared, input.stream>>>(
            input.data, packed, workspace.data());
        ATLC_CHECK_CUDA(cudaGetLastError);
        md::second_stage<N><<<(1u << N), md::block_size, 0, input.stream>>>(
            workspace.data(), output, blocks);
        ATLC_CHECK_CUDA(cudaGetLastError);
        std::array<double, (1 << N)> baseline{};
        ATLC_CHECK_CUDA(cudaMemcpyAsync, baseline.data(), output, sizeof(baseline),
                       cudaMemcpyDeviceToHost, input.stream);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, input.stream);
        require(actual == baseline, "optimized/baseline bins differ");
    }
}

void analytic_order(md::workspace& workspace, device_state& input)
{
    std::vector<State> state(8);
    for (unsigned i = 0; i < state.size(); ++i) state[i] = State(i + 1, 0);
    md::descriptor<2> d;
    d.local_targets = 2;
    d.local_target_physical = {0, 2};
    d.local_outcome_bits = {0, 1};
    auto a = compare(workspace, input, state, d);
    std::swap(d.local_target_physical[0], d.local_target_physical[1]);
    auto b = compare(workspace, input, state, d);
    require(a == std::array<double, 4>{10, 20, 74, 100}, "analytic forward histogram");
    require(b == std::array<double, 4>{10, 74, 20, 100}, "analytic reversed histogram");
}

void host_index_checks()
{
    // Bounded checks of a huge logical domain, with no huge GPU allocation.
    md::descriptor<4> d;
    d.local_targets = 2;
    d.local_target_physical = {39, 3, 0, 0};
    d.local_outcome_bits = {3, 0, 0, 0};
    d.fixed_outcome = 4;
    d.control_mask = (UINT64_C(1) << 35) | (UINT64_C(1) << 5) | 1;
    d.control_value = (UINT64_C(1) << 35) | 1;
    auto packed = md::pack(d, UINT64_C(1) << 40);
    require(packed.active_count == (UINT64_C(1) << 37), "active count truncated to 32 bits");
    const uint64_t probes[] = {0, 1, 31, 32, (UINT64_C(1) << 32) - 1,
                              UINT64_C(1) << 32, (UINT64_C(1) << 36),
                              packed.active_count - 1};
    for (auto p : probes)
    {
        auto physical = packed.physical_index(p);
        require(physical < (UINT64_C(1) << 40), "physical mapping out of bounds");
        require((physical & d.control_mask) == d.control_value, "mapping lost control bits");
        uint64_t roundtrip = 0;
        unsigned bit = 0;
        for (unsigned i = 0; i < 40; ++i)
            if (!(d.control_mask & (UINT64_C(1) << i)))
                roundtrip |= ((physical >> i) & 1) << bit++;
        require(roundtrip == p, "packed/physical round trip");
        if (p + 1 < packed.active_count)
            require(physical < packed.physical_index(p + 1), "mapping is not monotone");
    }
    // Exhaustively compare small packed domains to independent physical enumeration.
    for (uint64_t mask = 0; mask < 256; ++mask)
    {
        md::descriptor<1> small;
        small.control_mask = mask;
        for (uint64_t value : {UINT64_C(0), mask, mask & UINT64_C(0x55)})
        {
            small.control_value = value;
            auto p = md::pack(small, 256);
            uint64_t ordinal = 0;
            for (uint64_t i = 0; i < 256; ++i)
                if ((i & mask) == value)
                    require(p.physical_index(ordinal++) == i, "packed enumeration mismatch");
            require(ordinal == p.active_count, "packed cardinality mismatch");
        }
    }
    rejects([&] { md::pack(d, 0); });
    rejects([&] { md::pack(d, 3); });
    rejects([&] { md::allocation_bytes(UINT64_C(1) << 63, sizeof(State)); });
    rejects([&] { md::checked_product(UINT64_MAX, 2); });
    auto bad = d;
    bad.local_targets = 5;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.local_target_physical[0] = 35;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.local_target_physical[0] = 3;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.local_target_physical[0] = 64;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.local_outcome_bits[0] = -1;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.local_outcome_bits[0] = 4;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.local_outcome_bits[0] = 0;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.fixed_outcome |= 8;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.fixed_outcome = 16;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.control_mask |= UINT64_C(1) << 40;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
    bad = d; bad.control_value |= 4;
    rejects([&] { md::pack(bad, UINT64_C(1) << 40); });
}

// Optional paired microbenchmark. GPU batch timing includes launch gaps; host
// timing includes one stream completion per reducer API call, not NCCL/sampling.
template<int N>
void benchmark(cudaStream_t stream)
{
    md::workspace workspace(stream);
    device_state input(stream);
    auto config = workspace.reserve<N>(stream);
    for (size_t count : {size_t(1), size_t(16), size_t(256), size_t(512), size_t(1) << 20})
    {
        auto state = nonuniform(count);
        input.upload(state);
        md::descriptor<N> d;
        d.fixed_outcome = (1u << N) - 1;
        auto packed = md::pack(d, count);
        unsigned blocks = md::measurement_blocks(packed.active_count, config.blocks);
        double* output = workspace.data() + (UINT64_C(1) << N) * config.blocks;
        auto launch = [&](bool optimized) {
            if (optimized) workspace.enqueue(input.data, count, d, stream);
            else
            {
                // Match enqueue's host metadata and workspace checks too.
                auto baseline_packed = md::pack(d, count);
                auto baseline_config = workspace.reserve<N>(stream);
                unsigned baseline_blocks = md::measurement_blocks(
                    baseline_packed.active_count, baseline_config.blocks);
                md::first_stage<N><<<baseline_blocks, md::block_size, baseline_config.dynamic_shared, stream>>>(
                    input.data, baseline_packed, workspace.data());
                ATLC_CHECK_CUDA(cudaGetLastError);
                md::second_stage<N><<<(1u << N), md::block_size, 0, stream>>>(
                    workspace.data(), output, baseline_blocks);
                ATLC_CHECK_CUDA(cudaGetLastError);
            }
        };
        cudaEvent_t start, stop;
        ATLC_CHECK_CUDA(cudaEventCreate, &start);
        ATLC_CHECK_CUDA(cudaEventCreate, &stop);
        for (bool optimized : {false, true})
        {
            for (int i = 0; i < 20; ++i) launch(optimized);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            constexpr int repeats = 200;
            ATLC_CHECK_CUDA(cudaEventRecord, start, stream);
            for (int i = 0; i < repeats; ++i) launch(optimized);
            ATLC_CHECK_CUDA(cudaEventRecord, stop, stream);
            ATLC_CHECK_CUDA(cudaEventSynchronize, stop);
            float milliseconds = 0;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &milliseconds, start, stop);
            auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i)
            {
                launch(optimized);
                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            }
            double host_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - begin).count() / repeats;
            std::cout << N << ',' << count << ',' << blocks << ',' << optimized
                      << ',' << milliseconds * 1000 / repeats << ',' << host_us << '\n';
        }
        ATLC_CHECK_CUDA(cudaEventDestroy, start);
        ATLC_CHECK_CUDA(cudaEventDestroy, stop);
    }
    workspace.close();
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--benchmark")
        {
            cudaStream_t stream;
            ATLC_CHECK_CUDA(cudaStreamCreateWithFlags, &stream, cudaStreamNonBlocking);
            std::cout << "width,active_count,blocks,optimized,gpu_batch_us,host_completion_us\n";
            benchmark<1>(stream);
            benchmark<2>(stream);
            benchmark<3>(stream);
            benchmark<4>(stream);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
            return 0;
        }
        host_index_checks();
        cudaStream_t stream = nullptr;
        ATLC_CHECK_CUDA(cudaStreamCreateWithFlags, &stream, cudaStreamNonBlocking);
        {
            md::workspace workspace(stream);
            device_state input(stream);
            auto config = workspace.reserve<1>(stream);
            // Power-of-two physical domain with active size exceeding a full
            // grid stride, including a final stride (partial on most devices).
            uint64_t large = 256;
            uint64_t stride = md::checked_product(config.blocks, md::block_size);
            while (large / 4 <= stride)
                large = md::checked_product(large, 2);
            shrinking_groups<1>(workspace, input);
            shrinking_groups<2>(workspace, input);
            shrinking_groups<3>(workspace, input);
            shrinking_groups<4>(workspace, input);
            single_block_dispatch<1>(workspace, input);
            single_block_dispatch<2>(workspace, input);
            single_block_dispatch<3>(workspace, input);
            single_block_dispatch<4>(workspace, input);
            analytic_order(workspace, input);
            fixtures<1>(workspace, input, large);
            fixtures<2>(workspace, input, large);
            fixtures<3>(workspace, input, large);
            fixtures<4>(workspace, input, large);
            // Reuse grown storage with smaller N, then larger N and zero input.
            fixtures<1>(workspace, input, 512);
            md::descriptor<4> zero;
            zero.fixed_outcome = 13;
            compare(workspace, input, std::vector<State>(1), zero);
            cudaStream_t other = nullptr;
            ATLC_CHECK_CUDA(cudaStreamCreateWithFlags, &other, cudaStreamNonBlocking);
            rejects([&] { workspace.reserve<1>(other); });
            ATLC_CHECK_CUDA(cudaStreamDestroy, other);
            workspace.close();
            workspace.close(); // explicit teardown is idempotent
            compare(workspace, input, nonuniform(1), zero); // safe reinitialization

            // Simulator mode borrows one fixed slice. Closing it must leave
            // both the backing allocation and the adjacent swap slice alive.
            auto largest = md::resources<4>(workspace.device());
            size_t bytes = md::allocation_bytes(
                md::checked_product(16, uint64_t(largest.blocks) + 1), sizeof(double));
            bytes = (bytes + 255) / 256 * 256 + 256;
            void* backing = nullptr;
            ATLC_CHECK_CUDA(cudaMalloc, &backing, bytes);
            qcs::buffer_arena arena(backing, bytes);
            {
                md::workspace borrowed(stream, arena);
                void* swap = arena.take(256);
                ATLC_CHECK_CUDA(cudaMemsetAsync, swap, 0x5a, 256, stream);
                auto* original = borrowed.data();
                analytic_order(borrowed, input);
                fixtures<1>(borrowed, input, 512);
                fixtures<2>(borrowed, input, 512);
                fixtures<3>(borrowed, input, 512);
                fixtures<4>(borrowed, input, 512);
                require(borrowed.data() == original, "borrowed workspace moved");
                borrowed.close();
                borrowed.close();
                std::array<unsigned char, 256> guard{};
                ATLC_CHECK_CUDA(cudaMemcpyAsync, guard.data(), swap, guard.size(), cudaMemcpyDeviceToHost, stream);
                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                require(std::all_of(guard.begin(), guard.end(), [](unsigned char v) { return v == 0x5a; }),
                        "measurement overwrote adjacent swap storage");
                ATLC_CHECK_CUDA(cudaMemsetAsync, backing, 0, bytes, stream);
            }
            ATLC_CHECK_CUDA(cudaFreeAsync, backing, stream);
        }
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
        std::cout << "measurement reducer: " << cases << " GPU/oracle cases passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "measurement reducer: " << error.what() << '\n';
        return 1;
    }
}
