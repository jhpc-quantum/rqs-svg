// Private harness only: the production kernel specialization has no diagnostics.
#pragma once
#include <limits>
#include <cstring>

namespace sampling_cases {
namespace md = qcs::measurement_detail;

inline void agreement(const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::vector<unsigned char> low(size), high(size);
    ATLC_CHECK_MPI(MPI_Allreduce, bytes, low.data(), int(size), MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
    ATLC_CHECK_MPI(MPI_Allreduce, bytes, high.data(), int(size), MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
    require(low == high, "sampler diagnostics disagree across ranks");
}

struct fixture {
    double* weights = nullptr;
    md::sample_result* result = nullptr;
    md::sample_diagnostics* diagnostics = nullptr;
    cudaStream_t stream;
    md::sample_result last_result{};
    explicit fixture(cudaStream_t s) : stream(s) {
        ATLC_CHECK_CUDA(cudaMalloc, &weights, 16 * sizeof(double));
        ATLC_CHECK_CUDA(cudaMalloc, &result, sizeof(*result));
        ATLC_CHECK_CUDA(cudaMalloc, &diagnostics, sizeof(*diagnostics));
    }
    ~fixture() { cudaFree(weights); cudaFree(result); cudaFree(diagnostics); }
    template<int N>
    md::sample_diagnostics run(const std::vector<double>& w, double u,
                               md::sample_status status, unsigned outcome = 0,
                               uint64_t counter = 0, uint64_t seed = 0x123456789abcdef0ULL) {
        ATLC_CHECK_CUDA(cudaMemcpyAsync, weights, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice, stream);
        md::sample_outcome<N, true><<<1, 32, 0, stream>>>(weights, seed, counter, result, diagnostics, u);
        ATLC_CHECK_CUDA(cudaGetLastError);
        md::sample_result r{};
        md::sample_diagnostics d{};
        ATLC_CHECK_CUDA(cudaMemcpyAsync, &r, result, sizeof(r), cudaMemcpyDeviceToHost, stream);
        ATLC_CHECK_CUDA(cudaMemcpyAsync, &d, diagnostics, sizeof(d), cudaMemcpyDeviceToHost, stream);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        last_result = r;
        agreement(&r, sizeof(r));
        agreement(d.cumulative, (1 << N) * sizeof(double));
        agreement(&d.raw, sizeof(d.raw));
        agreement(&d.uniform, sizeof(d.uniform));
        require(r.status == status, "sampler error status");
        if (status == md::sample_status::success) {
            require(r.outcome < w.size(), "outcome outside weight buffer");
            require(w[r.outcome] > 0, "selected zero weight");
            if (u >= 0) require(r.outcome == outcome, "controlled threshold outcome");
            long double sum = 0;
            for (int i = 0; i < (1 << N); ++i) {
                sum += w[i];
                require(std::abs(static_cast<long double>(d.cumulative[i]) - sum) <=
                        5e-15L * sum, "CUB prefix versus long double oracle");
            }
        }
        return d;
    }
};

template<int N>
void numerical(fixture& f) {
    constexpr unsigned bins = 1 << N;
    const double top = std::nextafter(1.0, 0.0);
    const auto ok = md::sample_status::success;
    for (unsigned hot = 0; hot < bins; ++hot) {
        std::vector<double> w(bins); w[hot] = 8;
        for (double u : {0.0, 0.25, top}) f.run<N>(w, u, ok, hot);
    }
    std::vector<double> w(bins);
    unsigned left = N == 1 ? 0 : 1, right = N == 1 ? 1 : 3;
    w[left] = 2; w[right] = 6;
    f.run<N>(w, 0, ok, left);
    f.run<N>(w, std::nextafter(0.25, 0.0), ok, left);
    f.run<N>(w, 0.25, ok, right);
    f.run<N>(w, std::nextafter(0.25, 1.0), ok, right);
    f.run<N>(w, top, ok, right);
    // Every interior boundary, with exactly representable integer prefixes.
    for (unsigned i = 0; i < bins; ++i) w[i] = 1;
    for (unsigned i = 1; i < bins; ++i) {
        double boundary = double(i) / bins;
        f.run<N>(w, std::nextafter(boundary, 0.0), ok, i - 1);
        f.run<N>(w, boundary, ok, i);
        f.run<N>(w, std::nextafter(boundary, 1.0), ok, i);
    }
    w.assign(bins, 0); w[left] = 0x1p-900; w[right] = 0x1p900;
    f.run<N>(w, 0, ok, left); f.run<N>(w, top, ok, right);
    // A one-subnormal total rounds u*total up to total at the upper end.
    w.assign(bins, 0); w[left] = std::numeric_limits<double>::denorm_min();
    f.run<N>(w, top, ok, left);
    w.assign(bins, 0); f.run<N>(w, 0, md::sample_status::invalid_total);
    w[0] = w[1] = std::numeric_limits<double>::max();
    f.run<N>(w, 0, md::sample_status::invalid_total);
    for (double bad : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
        w.assign(bins, 1); w[bins - 1] = bad;
        f.run<N>(w, 0, md::sample_status::invalid_weight);
    }
}

inline void run(qcs_simulator* sim) {
    auto& c = *sim->core;
    fixture f(c.stream);
    numerical<1>(f); numerical<2>(f); numerical<3>(f); numerical<4>(f);
    for (uint64_t seed : {uint64_t(0), UINT64_C(0xfedcba9876543210)})
    for (uint64_t counter : {uint64_t(0), UINT64_C(1) << 32,
                             (UINT64_C(1) << 32) + 17, UINT64_MAX}) {
        auto a = f.run<1>({1, 1}, -1, md::sample_status::success, 0, counter, seed);
        auto b = f.run<1>({1, 1}, -1, md::sample_status::success, 0, counter, seed);
        require(std::memcmp(&a.raw, &b.raw, sizeof(a.raw)) == 0 && a.uniform == b.uniform,
                "seed/counter RNG repeatability");
        uint64_t mantissa = (uint64_t(a.raw.x) << 21) | (uint64_t(a.raw.y) >> 11);
        require(a.uniform == double(mantissa) * 0x1p-53 && a.uniform >= 0 && a.uniform < 1,
                "raw RNG to uniform mapping");
        if (counter > UINT32_MAX) {
            auto truncated = f.run<1>({1, 1}, -1, md::sample_status::success, 0, uint32_t(counter), seed);
            require(std::memcmp(&a.raw, &truncated.raw, sizeof(a.raw)) != 0, "counter truncated to 32 bits");
        }
    }
}

inline void measurement_order(qcs_simulator* sim) {
    auto& c = *sim->core;
    qcs_simulator_set_zero_state_cxx(sim);
    std::vector<bit_num_t> targets(c.num_qubits), clbits(c.num_qubits);
    std::vector<bit_t> bits(c.num_qubits);
    for (int i = 0; i < c.num_qubits; ++i) {
        targets[i] = i;
        clbits[i] = c.num_qubits - 1 - i;
    }
    std::vector<int> expected;
    for (int p = c.num_qubits_local - 1; p >= 0; --p)
        expected.push_back(c.perm_p2l[p]);
    for (int p = c.num_qubits - 1; p >= c.num_qubits_local; --p)
        expected.push_back(c.perm_p2l[p]);
    qcs_simulator_measure_many_to_clbits_cxx(
        sim, targets.data(), c.num_qubits, clbits.data(), c.num_qubits, bits.data());
    require(c.measured_0_qubit_num_logical_list == expected,
            "measurement order must be local high-to-low, then global high-to-low");
}

inline void lifecycle(qcs_simulator* sim) {
    auto& c = *sim->core;
    require(c.measurement_counter == 0 && !c.measurement_counter_exhausted,
            "new simulator counter must start at zero");
    c.measurement_counter = (UINT64_C(1) << 32) + 9;
    uint64_t expected = c.measurement_counter;
    auto check = [&] {
        require(c.measurement_counter == expected && !c.measurement_counter_exhausted,
                "measurement counter lifecycle");
        agreement(&c.measurement_counter, sizeof(c.measurement_counter));
    };
    qcs_simulator_set_zero_state_cxx(sim); check();
    measure(sim, {}); check();
    qcs_simulator_measure_many_to_clbits_cxx(sim, nullptr, 0, nullptr, 0, nullptr); check();
    measure(sim, {0}); ++expected; check();
    measure(sim, {0}); check();
    measure(sim, {0, 1}); ++expected; check();
    // Under reverse mapping the first control excludes half the ranks.
    measure(sim, {2, 3, 4, 5, 6}); expected += 2; check();
    const bit_num_t targets[] = {8, 11, 7, 10, 9};
    const bit_num_t clbits[] = {4, 3, 2, 1, 0};
    bit_t bits[5]{};
    qcs_simulator_measure_many_to_clbits_cxx(sim, targets, 5, clbits, 5, bits);
    expected += 2; check();
    qcs_simulator_measure_many_to_clbits_cxx(sim, targets, 5, clbits, 5, bits); check();
    auto* data = c.measurement_workspace->data();
    auto capacity = c.measurement_workspace->capacity();
    qcs_simulator_reset_measurement_state_cxx(sim); check();
    measure(sim, {0, 1, 2, 3}); ++expected; check();
    require(data == c.measurement_workspace->data() && capacity == c.measurement_workspace->capacity(),
            "workspace allocated again after warmup");
    // The exact reset sequence in qcs_main.cpp between shots.
    qcs_simulator_reinitialize_mapping_cxx(sim);
    qcs_simulator_set_zero_state_cxx(sim);
    qcs_simulator_reset_clbits_cxx(sim);
    qcs_simulator_reset_measurement_state_cxx(sim); check();
    measure(sim, {0}); ++expected; check();
    // Invalid total/weight must use the public error path without consuming RNG.
    std::vector<qcs::complex_t> invalid(c.num_states_local, qcs::complex_t(0, 0));
    for (bool nan : {false, true}) {
        qcs_simulator_reset_measurement_state_cxx(sim);
        if (nan) invalid[0] = qcs::complex_t(std::numeric_limits<double>::quiet_NaN(), 0);
        ATLC_CHECK_CUDA(cudaMemcpyAsync, c.state_data_device, invalid.data(),
                       invalid.size() * sizeof(invalid[0]), cudaMemcpyHostToDevice, c.stream);
        bit_t rejected = 9;
        require(!qcs_simulator_measure(sim, 0, &rejected), "invalid distribution C error path");
        check();
    }
    c.measurement_counter = UINT64_MAX;
    qcs_simulator_set_zero_state_cxx(sim);
    measure(sim, {0});
    require(c.measurement_counter == UINT64_MAX && c.measurement_counter_exhausted,
            "final counter must exhaust without wrapping");
    measure(sim, {0}); measure(sim, {});
    bit_t result = 9;
    require(!qcs_simulator_measure(sim, 1, &result), "exhausted counter C error path");
    require(c.measurement_counter == UINT64_MAX && c.measurement_counter_exhausted,
            "exhaustion must persist");
    // Private fixture restores its cursor for later harness cases only.
    c.measurement_counter = expected;
    c.measurement_counter_exhausted = false;
}
} // namespace sampling_cases
