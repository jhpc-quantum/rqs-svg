#include "../src/measurement_reduce.cuh"
#include <algorithm>
#include <cmath>
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

int main()
{
    try
    {
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
