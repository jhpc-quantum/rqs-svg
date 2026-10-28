// Compiled by measurement_host.py after extracting the production helpers.
static void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <int N, int L = N>
static qcs::MeasureNorm<N> load_norm(int locals, const std::array<int, N> &positions,
                                    uint64_t fixed, uint64_t count)
{
    if (locals != L)
    {
        if constexpr (L > 0)
            return load_norm<N, L - 1>(locals, positions, fixed, count);
        throw std::runtime_error("bad local target count");
    }
    qcs::cubUtility::IndirectLoad<N, L> loader{positions, fixed};
    qcs::MeasureNorm<N> result{};
    for (uint64_t i = 0; i < count; ++i)
        result = qcs::cubUtility::MeasureNormAdd<N>()(result, loader(i));
    return result;
}

template <int N>
static void check_loaders()
{
    // Every ordered target combination on six physical qubits, across
    // simulated 1/2/4/8-rank partitions and all prior-measurement subsets.
    constexpr int width = 6;
    std::vector<qcs::complex_t> state(1 << width);
    for (size_t i = 0; i < state.size(); ++i)
        state[i] = qcs::complex_t(int(i % 7) - 3, int(i % 11) - 5);
    std::array<int, N> targets{};
    auto visit = [&](auto &&self, int depth) -> void {
        if (depth != N)
        {
            for (int q = 0; q < width; ++q)
            {
                if (std::find(targets.begin(), targets.begin() + depth, q) != targets.begin() + depth)
                    continue;
                targets[depth] = q;
                self(self, depth + 1);
            }
            return;
        }
        for (int local_width = 3; local_width <= width; ++local_width)
        for (unsigned prior_mask = 0; prior_mask < (1u << N); ++prior_mask)
        {
            // Include a previous measurement outside the current target list.
            unsigned mask = 0;
            for (int i = 0; i < N; ++i)
                if (prior_mask & (1u << i))
                    mask |= 1u << targets[i];
            for (int q = 0; q < width; ++q)
                if (std::find(targets.begin(), targets.end(), q) == targets.end())
                {
                    mask |= 1u << q;
                    break;
                }
            for (unsigned values : {0u, mask, mask & 0x15u})
            {
                qcs::MeasureNorm<N> expected{}, actual{};
                for (unsigned index = 0; index < state.size(); ++index)
                {
                    if ((index & mask) != values)
                        continue;
                    unsigned outcome = 0;
                    for (int i = 0; i < N; ++i)
                        outcome |= ((index >> targets[i]) & 1) << i;
                    expected[outcome] += std::norm(state[index]);
                }
                for (unsigned rank = 0; rank < (1u << (width - local_width)); ++rank)
                {
                    if (((rank << local_width) & mask) != (values & ~((1u << local_width) - 1)))
                        continue;
                    auto *args = reinterpret_cast<qcs::kernel_input_qnlist_struct *>(qcs::kernel_input_constant);
                    std::vector<int> local_targets, positive, operands;
                    std::array<int, N> positions{};
                    uint64_t fixed = 0;
                    for (int i = 0; i < N; ++i)
                    {
                        int q = targets[i];
                        if (mask & (1u << q))
                            fixed |= uint64_t((values >> q) & 1) << i;
                        else if (q >= local_width)
                            fixed |= uint64_t((rank >> (q - local_width)) & 1) << i;
                        else
                        {
                            positions[local_targets.size()] = i;
                            local_targets.push_back(q);
                        }
                    }
                    operands = local_targets;
                    for (int q = 0; q < local_width; ++q)
                        if (mask & (1u << q))
                        {
                            operands.push_back(q);
                            if (values & (1u << q)) positive.push_back(q);
                        }
                    std::sort(operands.begin(), operands.end());
                    args->num_target_qubits = local_targets.size();
                    args->num_positive_control_qubits = positive.size();
                    args->num_negative_control_qubits = operands.size() - local_targets.size() - positive.size();
                    args->is_measured_bits = args->measured_value_bits = 0;
                    std::copy(operands.begin(), operands.end(), args->get_operand_qubit_num_list_sorted());
                    std::copy(positive.begin(), positive.end(), args->get_positive_control_qubit_num_list());
                    std::copy(local_targets.begin(), local_targets.end(), args->get_target_qubit_num_list());
                    qcs::kernel_common_constant.state_data_device = state.data() + (rank << local_width);
                    auto partial = load_norm<N>(local_targets.size(), positions, fixed, 1u << (local_width - operands.size()));
                    actual = qcs::cubUtility::MeasureNormAdd<N>()(actual, partial);
                }
                require(actual == expected, "joint probability/index mapping mismatch");
            }
        }
    };
    visit(visit, 0);
}

static void check_sampling()
{
    using namespace qcs::cubUtility;
    qcs::MeasureNorm<2> norm{0, 2, 0, 3};
    require(measure_norm_sum<2>(norm) == 5, "unnormalized sum");
    require(select_measure_outcome<2>(norm, 0) == 1, "leading zero bin");
    require(select_measure_outcome<2>(norm, 2) == 3, "zero bin at boundary");
    require(select_measure_outcome<2>({0, 2, 3, 0}, 5) == 2, "rounded endpoint");
    for (auto bad : {qcs::MeasureNorm<2>{}, qcs::MeasureNorm<2>{1, -1, 1, 0},
                     qcs::MeasureNorm<2>{1, NAN, 0, 0}, qcs::MeasureNorm<2>{1, INFINITY, 0, 0},
                     qcs::MeasureNorm<2>{1e308, 1e308, 0, 0}})
    {
        bool threw = false;
        try { measure_norm_sum<2>(bad); }
        catch (const std::runtime_error &) { threw = true; }
        require(threw, "invalid norm accepted");
    }
    std::mt19937_64 engine(1234);
    std::uniform_real_distribution<double> draw(0, 5);
    int counts[4]{};
    for (int shot = 0; shot < 20000; ++shot)
        ++counts[select_measure_outcome<2>(norm, draw(engine))];
    require(counts[0] == 0 && counts[2] == 0, "sampled zero bin");
    require(std::abs(counts[1] / 20000.0 - 0.4) < 0.02, "sampling distribution");
}

static void check_orchestration()
{
    qcs::simulator_core core{};
    core.num_qubits = core.num_qubits_local = 6;
    core.num_states_local = 64;
    core.log_block_size_max = 9;
    core.perm_l2p = {3, 5, 1, 4, 0, 2};
    core.qcs_kernel_input_constant_addr = reinterpret_cast<qcs::kernel_input_qnlist_struct *>(qcs::kernel_input_constant);
    core.engine.seed(1234);
    std::vector<qcs::complex_t> state(64);
    core.state_data_device = state.data();
    qcs::kernel_common_constant.state_data_device = state.data();
    qcs_simulator sim{&core, 6, 6, std::vector<bool>(6)};
    auto reset = [&]() {
        core.discard_measurement_state();
        std::fill(state.begin(), state.end(), qcs::complex_t(0));
    };
    auto physical = [&](uint64_t logical) {
        uint64_t result = 0;
        for (int q = 0; q < 6; ++q)
            result |= ((logical >> q) & 1) << core.perm_l2p[q];
        return result;
    };
    // Grow from a single-bit output to a pair and then a triple. The same
    // reduction code must query scratch storage for every configuration.
    state[0] = 1;
    require(core.measure_qubit(0) == 0, "single-bit warmup");
    require(core.measure_norm_device_size == sizeof(qcs::MeasureNorm<1>), "single output size");
    reset();
    state[0] = 1;
    require(core.measure_qubits<2>({0, 1}) == 0, "two-bit warmup");
    require(core.measure_norm_device_size == sizeof(qcs::MeasureNorm<2>), "pair output size");
    for (int expected = 0; expected < 8; ++expected)
    {
        reset();
        uint64_t logical = ((expected & 1) << 5) | (((expected >> 1) & 1) << 1) | (((expected >> 2) & 1) << 3);
        state[physical(logical)] = 3;
        auto before = state;
        int count = reductions;
        require(core.measure_qubits<3>({5, 1, 3}) == uint64_t(expected), "three-bit orchestration");
        require(reductions == count + 1, "more than one joint reduction");
        require(state == before, "measurement mutated state");
        require(core.pending_projection_mask_logical == ((1u << 5) | (1u << 1) | (1u << 3)), "pending mask");
        require(core.measured_value_logical == logical, "recorded value");
        auto engine = core.engine;
        require(core.measure_qubits<3>({5, 1, 3}) == uint64_t(expected), "repeat outcome");
        require(reductions == count + 1 && core.engine == engine, "repeat reduced or sampled");
    }
    require(core.measure_norm_device_size == sizeof(qcs::MeasureNorm<3>), "result buffer size");
    for (int shot = 0; shot < 64; ++shot)
    {
        reset();
        state[0] = state[physical((1u << 5) | (1u << 1) | (1u << 3))] = 1;
        int first = core.measure_qubit(1);
        require(core.measure_qubits<3>({5, 1, 3}) == (first ? 7u : 0u), "prior measurement not conditioned");
    }
    reset();
    std::fill(state.begin(), state.end(), qcs::complex_t(1));
    bit_num_t qubits[]{5, 1, 3, 0, 4}, clbits[]{0, 3, 2, 4, 1};
    bit_t results[5]{};
    int count = reductions, collective_count = allreduces, broadcast_count = broadcasts;
    qcs_simulator_measure_many_to_clbits_cxx(&sim, qubits, 5, clbits, 5, results);
    require(reductions == count + 3 && allreduces == collective_count + 3 && broadcasts == broadcast_count + 3,
            "grouping did not use two pairs and a single remainder");
    for (int i = 0; i < 5; ++i)
        require(sim.clbits[clbits[i]] == bool(results[i]), "classical output mapping");
    reset();
    std::fill(state.begin(), state.end(), qcs::complex_t(1));
    int allocation_count = allocations;
    qcs_simulator_measure_many_cxx(&sim, qubits, 5, results);
    require(allocations == allocation_count, "buffers were not reused");
    auto mask = core.measured_mask_logical;
    bit_num_t invalid[]{1, 1};
    bool threw = false;
    try { qcs_simulator_measure_many_cxx(&sim, invalid, 2, results); }
    catch (const std::runtime_error &) { threw = true; }
    require(threw && core.measured_mask_logical == mask, "validation modified measurement state");
    reset();
    threw = false;
    try { core.measure_qubits<2>({1, 3}); }
    catch (const std::runtime_error &) { threw = true; }
    require(threw && core.measured_mask_logical == 0, "invalid distribution recorded result");
    cudaFreeAsync(core.cub_temp_buffer_device, 0);
    cudaFreeAsync(core.measure_norm_device, 0);
}

int main()
{
    check_loaders<1>();
    check_loaders<2>();
    check_loaders<3>();
    check_loaders<4>();
    check_sampling();
    check_orchestration();
    std::cout << "Host measurement checks passed (N=1..4; simulated partitions, prior controls and orchestration)\n";
}
