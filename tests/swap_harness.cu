// Private amplitude access; no production test hooks or legacy instantiations.
#ifndef QCS_SWAP_SOURCE
#define QCS_SWAP_SOURCE "../src/qcs.cu"
#endif
#include QCS_SWAP_SOURCE
#include <complex>
#include <iomanip>
#include <iostream>
#include <numeric>

using amplitude = std::complex<double>;
using qubits = std::vector<bit_num_t>;
static uint64_t checks = 0;
static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
static uint64_t logical_index(const qcs::simulator_core& c, uint64_t local)
{
    uint64_t physical = local + uint64_t(c.proc_num) * c.num_states_local, logical = 0;
    for (int p = 0; p < c.num_qubits; ++p)
        logical |= ((physical >> p) & 1) << c.perm_p2l[p];
    return logical;
}
static amplitude fixture(uint64_t i, uint64_t count)
{
    return amplitude(double(1 + i % 17) / 17, double(int(i % 13) - 6) / 13) /
           std::sqrt(double(count));
}
static std::vector<amplitude> initial(qcs::simulator_core& c, int kind, uint64_t basis)
{
    std::vector<amplitude> result(c.num_states);
    double norm = 0;
    for (uint64_t i = 0; i < result.size(); ++i)
    {
        result[i] = kind ? amplitude(i == basis, 0) : fixture(i, c.num_states);
        norm += std::norm(result[i]);
    }
    for (auto& v : result) v /= std::sqrt(norm);
    return result;
}
static void load(qcs::simulator_core& c, const std::vector<amplitude>& state)
{
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
    c.discard_measurement_state();
    c.reinitialize_mapping();
    std::vector<qcs::complex_t> local(c.num_states_local);
    for (uint64_t i = 0; i < local.size(); ++i)
    {
        auto v = state[logical_index(c, i)];
        local[i] = qcs::complex_t(v.real(), v.imag());
    }
    ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, local.data(),
                   local.size() * sizeof(local[0]), cudaMemcpyHostToDevice, c.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
}
// Sequential composition of individual transpositions is independent of the
// GPU's simultaneous bit exchange. Controls are disjoint from every target.
static void reference_swap(std::vector<amplitude>& state, const qubits& targets,
                           const qubits& negative, const qubits& positive)
{
    for (size_t pair = 0; pair < targets.size(); pair += 2)
    {
        auto before = state;
        for (uint64_t i = 0; i < state.size(); ++i)
        {
            bool active = true;
            for (auto q : negative) active &= !((i >> q) & 1);
            for (auto q : positive) active &= bool((i >> q) & 1);
            if (!active) continue;
            uint64_t j = i;
            if (((i >> targets[pair]) ^ (i >> targets[pair + 1])) & 1)
                j ^= (UINT64_C(1) << targets[pair]) | (UINT64_C(1) << targets[pair + 1]);
            state[i] = before[j];
        }
    }
}
static void check(qcs::simulator_core& c, const std::vector<amplitude>& expected)
{
    std::vector<qcs::complex_t> state(c.num_states_local);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, state.data(), c.state_data_device,
                   state.size() * sizeof(state[0]), cudaMemcpyDeviceToHost, c.stream);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
    double error = 0, norm = 0;
    for (uint64_t i = 0; i < state.size(); ++i)
    {
        uint64_t logical = logical_index(c, i);
        amplitude got(state[i].real(), state[i].imag());
        // Observe the effective state without changing deferred projection.
        if (((logical ^ c.measured_value_logical) & c.pending_projection_mask_logical) != 0)
            got = 0;
        if (!std::isfinite(got.real()) || !std::isfinite(got.imag())) error = INFINITY;
        error = std::max(error, std::abs(got - expected[logical]));
        norm += std::norm(got);
    }
    double worst = 0, total_norm = 0, want_norm = 0;
    ATLC_CHECK_MPI(MPI_Allreduce, &error, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    ATLC_CHECK_MPI(MPI_Allreduce, &norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (auto v : expected) want_norm += std::norm(v);
    require(worst < 2e-12, "SWAP amplitude mismatch/nonfinite result");
    require(std::abs(total_norm - want_norm) < 2e-11, "SWAP norm mismatch");
    ++checks;
}
static void operate(qcs_simulator* sim, const qubits& t, const qubits& n, const qubits& p, bool generic)
{
    using function = decltype(&qcs_simulator_gate_swap);
    const function api[] = {qcs_simulator_gate_swap, qcs_simulator_gate_swap1,
        qcs_simulator_gate_swap2, qcs_simulator_gate_swap3, qcs_simulator_gate_swap4};
    require(api[generic ? 0 : t.size() / 2](sim, t.data(), t.size(), n.data(), n.size(), p.data(), p.size()),
            "SWAP public API failed");
}
static void measure(qcs_simulator* sim, std::vector<amplitude>& expected, bit_num_t target)
{
    int outcome = qcs_simulator_measure_cxx(sim, target), low = 0, high = 0;
    ATLC_CHECK_MPI(MPI_Allreduce, &outcome, &low, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ATLC_CHECK_MPI(MPI_Allreduce, &outcome, &high, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    require(low == high, "measurement outcome differs across ranks");
    for (uint64_t i = 0; i < expected.size(); ++i)
        if (int((i >> target) & 1) != outcome) expected[i] = 0;
    require(sim->core->pending_projection_mask_logical & (UINT64_C(1) << target),
            "measurement did not leave a pending projection");
}
static qubits targets_for(int swaps, int layout, int bits)
{
    qubits order(bits - 2);
    std::iota(order.begin(), order.end(), 0);
    if (layout == 1) std::reverse(order.begin(), order.end());
    if (layout == 2)
    {
        // Interleave distant targets and reverse alternating pairs.
        for (int i = 0; i < bits - 2; ++i)
            order[i] = i % 2 ? bits - 3 - i / 2 : i / 2;
    }
    order.resize(2 * swaps);
    return order;
}
static void regression(qcs_simulator* sim)
{
    auto& c = *sim->core;
    for (int mapping = 0; mapping < 2; ++mapping)
    {
        qubits permutation(c.num_qubits);
        for (int p = 0; p < c.num_qubits; ++p)
            permutation[p] = mapping ? c.num_qubits - 1 - p : p;
        qcs_simulator_set_mapping_cxx(sim, permutation.data(), permutation.size());
        for (int swaps = 1; swaps <= 4; ++swaps)
        for (int layout = 0; layout < 3; ++layout)
        for (int control = 0; control < 4; ++control)
        for (int kind = 0; kind < 3; ++kind)
        {
            auto t = targets_for(swaps, layout, c.num_qubits);
            qubits n, p;
            if (control & 1) n.push_back(c.num_qubits - 2);
            if (control & 2) p.push_back(c.num_qubits - 1);
            uint64_t basis = UINT64_C(1) << t[0];
            if (kind == 1) { for (auto q : p) basis |= UINT64_C(1) << q; }
            if (kind == 2) { for (auto q : n) basis |= UINT64_C(1) << q; }
            auto expected = initial(c, kind, basis);
            load(c, expected);
            check(c, expected);
            // Repeated in-place permutations catch corruption; check every step.
            for (int repeat = 0; repeat < 5; ++repeat)
            {
                operate(sim, t, n, p, repeat % 2);
                reference_swap(expected, t, n, p);
                check(c, expected);
            }
        }
        for (int swaps = 1; swaps <= 4; ++swaps)
        for (int scenario = 0; scenario < 5; ++scenario)
        {
            auto t = targets_for(swaps, 2, c.num_qubits);
            auto expected = initial(c, 0, 0);
            load(c, expected);
            qubits n, p;
            bit_num_t measured = scenario < 2 ? t[0] : bit_num_t(c.num_qubits - 1);
            measure(sim, expected, measured);
            if (scenario == 1) c.materialize_pending_projection(c.pending_projection_mask_logical);
            if (scenario >= 2 && scenario < 4)
            {
                bool value = bool(c.measured_value_logical & (UINT64_C(1) << measured));
                if (scenario == 3) value = !value; // measured control rejects gate
                (value ? p : n).push_back(measured);
            }
            check(c, expected);
            operate(sim, t, n, p, false);
            reference_swap(expected, t, n, p);
            check(c, expected);
            if (scenario < 2)
                require(!(c.measured_mask_logical & (UINT64_C(1) << measured)),
                        "SWAP target retains stale measurement metadata");
            c.materialize_pending_projection(c.pending_projection_mask_logical);
            check(c, expected);
        }
    }
    if (!c.proc_num) std::cout << "SWAP regression passed: vectors=" << checks << '\n';
}
// Focused numerical cases for larger vectors, including an optional 26-local-
// qubit run whose uncontrolled launch crosses the former 65535-block boundary.
static void large_regression(qcs_simulator* sim)
{
    auto& c = *sim->core;
    for (int swaps : {1, 4})
    {
        auto t = targets_for(swaps, 2, c.num_qubits_local);
        auto expected = initial(c, 0, 0);
        load(c, expected);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            operate(sim, t, {}, {}, true);
            reference_swap(expected, t, {}, {});
            check(c, expected);
        }
    }
    if (!c.proc_num) std::cout << "SWAP large regression passed: vectors=" << checks << '\n';
}
static double maximum(double value)
{
    double result = 0;
    ATLC_CHECK_MPI(MPI_Allreduce, &value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
}
static void benchmark(qcs_simulator* sim, int warmups, int repetitions)
{
    auto& c = *sim->core;
    std::vector<qcs::complex_t> state(c.num_states_local);
    for (uint64_t i = 0; i < state.size(); ++i)
    {
        auto v = fixture(i + uint64_t(c.proc_num) * c.num_states_local, c.num_states);
        state[i] = qcs::complex_t(v.real(), v.imag());
    }
    if (!c.proc_num) std::cout << "qubits,ranks,swaps,layout,control,warmups,repetitions,median_ms,p10_ms,p90_ms\n";
    for (int swaps = 1; swaps <= 4; ++swaps)
    for (int layout = 0; layout < 3; ++layout)
    for (int control = 0; control < 4; ++control)
    {
        c.discard_measurement_state();
        c.reinitialize_mapping();
        ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, state.data(),
                       state.size() * sizeof(state[0]), cudaMemcpyHostToDevice, c.stream);
        auto t = targets_for(swaps, layout, c.num_qubits);
        qubits n, p;
        if (control & 1) n.push_back(c.num_qubits - 2);
        if (control & 2) p.push_back(c.num_qubits - 1);
        // Settle localization before timing the steady-state public API path.
        for (int i = 0; i < warmups + 1; ++i) operate(sim, t, n, p, false);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
        std::vector<double> times;
        for (int i = 0; i < repetitions; ++i)
        {
            ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
            double begin = MPI_Wtime();
            operate(sim, t, n, p, false);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, c.stream);
            times.push_back(maximum((MPI_Wtime() - begin) * 1000));
        }
        std::sort(times.begin(), times.end());
        auto percentile = [&](double f) { return times[size_t(f * (times.size() - 1))]; };
        if (!c.proc_num) std::cout << c.num_qubits << ',' << c.num_procs << ',' << swaps << ','
            << layout << ',' << control << ',' << warmups << ',' << repetitions << ','
            << std::setprecision(10) << percentile(.5) << ',' << percentile(.1) << ',' << percentile(.9) << '\n';
    }
}
int main(int argc, char** argv)
{
    auto* sim = qcs_simulator_create_cxx();
    try
    {
        auto& c = *sim->core;
        bool bench = argc > 1 && std::string(argv[1]) == "benchmark";
        bool large = argc > 1 && std::string(argv[1]) == "large";
        require(argc <= 1 || bench || large || std::string(argv[1]) == "regression", "unknown mode");
        int bits = argc > 2 ? std::stoi(argv[2]) : (large ? 20 : 12);
        int warmups = argc > 3 ? std::stoi(argv[3]) : 5;
        int repetitions = argc > 4 ? std::stoi(argv[4]) : 30;
        require(bits >= 12 && bits <= (bench ? 30 : (large ? 26 : 20)) && warmups >= 0 && repetitions > 0,
                "arguments: regression|large|benchmark qubits warmups repetitions (regression <=20, large <=26)");
        require(c.num_procs > 0 && !(c.num_procs & (c.num_procs - 1)), "power-of-two ranks required");
        require(bits - c.log_num_procs >= 8, "SWAP4 requires at least eight local qubits");
        qcs_simulator_set_num_qubits_cxx(sim, bits);
        qcs_simulator_allocate_memory_cxx(sim);
        if (bench) benchmark(sim, warmups, repetitions);
        else if (large) large_regression(sim);
        else
        {
            regression(sim);
#ifdef QCS_SWAP_LAUNCH_HPP
            // Keep the same numerical/measurement/MPI matrix, but force many
            // launches and non-power-of-two blocks with a partial final block.
            auto const original = c.swap_launch_config;
            c.swap_launch_config = qcs::swap_detail::launch_limits(37,
                original.threads, original.threads, original.threads,
                std::min<unsigned int>(3, original.blocks));
            regression(sim);
            // Fix all local bits except the two targets: only four ordinals,
            // even when the allocated vector is much larger than one block.
            c.swap_launch_config = original;
            c.initial_perm_p2l.resize(c.num_qubits);
            std::iota(c.initial_perm_p2l.begin(), c.initial_perm_p2l.end(), 0);
            auto expected = initial(c, 0, 0);
            load(c, expected);
            qubits t{0, 1}, n, p;
            for (int q = 2; q < c.num_qubits_local; ++q)
                (q % 2 ? p : n).push_back(q);
            operate(sim, t, n, p, true);
            reference_swap(expected, t, n, p);
            check(c, expected);
            if (!c.proc_num) std::cout << "SWAP small-work regression passed\n";
#endif
        }
        qcs_simulator_destroy_cxx(sim);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "SWAP harness: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
