#!/usr/bin/env python3
"""Check localization invariants on the host; CUDA/NCCL calls are stubbed."""

from pathlib import Path
import os
import shlex
import subprocess
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "src" / "qcs.cu"


def main():
    source = SOURCE.read_text()
    start = source.index("        void determine_qubit_exchange_mapping()")
    end = source.index("        void check_control_qubit_num_physical()", start)
    methods = source[start:end]
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

// Exercise the host control flow, without emulating asynchronous transport.
static int transport_calls = 0;
static bool fail_transport = false;
static void transport_call()
{
    ++transport_calls;
    if (fail_transport)
        throw std::runtime_error("injected transport failure");
}
#define ATLC_CHECK_NCCL(...) transport_call()
#define ATLC_CHECK_CUDA(...) transport_call()

struct Core
{
    int num_qubits_local = 0;
    uint64_t proc_num = 0;
    uint64_t swap_buffer_total_length = 1;
    std::vector<int> perm_p2l, perm_l2p;
    std::vector<int> target_qubit_num_logical_list, target_qubit_num_physical_list;
    std::vector<int> swap_target_global_list, swap_target_local_list;
    std::vector<int> swap_target_global_logical_list, swap_target_local_logical_list;
''' + methods + r'''
};

static int physical_index(int logical_index, const std::vector<int>& l2p)
{
    int index = 0;
    for (int q = 0; q < static_cast<int>(l2p.size()); ++q)
        index |= ((logical_index >> q) & 1) << l2p[q];
    return index;
}

static void check_case(Core c)
{
    const auto old_p2l = c.perm_p2l;
    const auto old_l2p = c.perm_l2p;
    c.determine_qubit_exchange_mapping();
    assert(c.perm_p2l == old_p2l && c.perm_l2p == old_l2p);
    assert(c.swap_target_global_list.size() == c.swap_target_local_list.size());
    std::vector<int> endpoints;
    for (size_t i = 0; i < c.swap_target_global_list.size(); ++i)
    {
        int g = c.swap_target_global_list[i], l = c.swap_target_local_list[i];
        assert(g >= c.num_qubits_local && g < static_cast<int>(old_p2l.size()));
        assert(l >= 0 && l < c.num_qubits_local);
        assert(c.swap_target_global_logical_list[i] == old_p2l[g]);
        assert(c.swap_target_local_logical_list[i] == old_p2l[l]);
        for (int target : c.target_qubit_num_logical_list)
            assert(old_l2p[target] != l);
        endpoints.push_back(g);
        endpoints.push_back(l);
    }
    std::sort(endpoints.begin(), endpoints.end());
    assert(std::adjacent_find(endpoints.begin(), endpoints.end()) == endpoints.end());
    c.exchange_qubits();
    assert(c.perm_p2l == old_p2l && c.perm_l2p == old_l2p);
    c.update_qubit_permutation_mapping();
    for (size_t p = 0; p < c.perm_p2l.size(); ++p)
        assert(c.perm_l2p[c.perm_p2l[p]] == static_cast<int>(p));
    for (size_t i = 0; i < c.target_qubit_num_logical_list.size(); ++i)
    {
        int p = c.target_qubit_num_physical_list[i];
        assert(p >= 0 && p < c.num_qubits_local);
        assert(p == c.perm_l2p[c.target_qubit_num_logical_list[i]]);
    }
    // Moving amplitude bits by the planned exchanges must preserve logical states.
    for (int logical = 0; logical < (1 << old_p2l.size()); ++logical)
    {
        int moved = physical_index(logical, old_l2p);
        for (size_t i = 0; i < c.swap_target_global_list.size(); ++i)
        {
            int g = c.swap_target_global_list[i], l = c.swap_target_local_list[i];
            if (((moved >> g) & 1) != ((moved >> l) & 1))
                moved ^= (1 << g) | (1 << l);
        }
        assert(moved == physical_index(logical, c.perm_l2p));
    }
    // Repeated localization and empty measurement targets clear previous work.
    const auto updated_l2p = c.perm_l2p;
    transport_calls = 0;
    c.localize_target_qubits();
    assert(c.perm_l2p == updated_l2p && transport_calls == 0);
    assert(c.swap_target_global_list.empty() && c.swap_target_local_list.empty());
    assert(c.swap_target_global_logical_list.empty() && c.swap_target_local_logical_list.empty());
    c.target_qubit_num_logical_list.clear();
    c.localize_target_qubits();
    assert(c.target_qubit_num_physical_list.empty() && transport_calls == 0);
}

int main()
{
    size_t cases = 0;
    for (int n = 0; n <= 5; ++n)
    {
        std::vector<int> permutation(n);
        std::iota(permutation.begin(), permutation.end(), 0);
        do
        {
            for (int local = 0; local <= n; ++local)
            {
                Core base;
                base.num_qubits_local = local;
                base.perm_p2l = permutation;
                base.perm_l2p.resize(n);
                for (int p = 0; p < n; ++p)
                    base.perm_l2p[permutation[p]] = p;
                auto enumerate = [&](auto&& self) -> void {
                    check_case(base);
                    ++cases;
                    if (static_cast<int>(base.target_qubit_num_logical_list.size()) == local)
                        return;
                    for (int q = 0; q < n; ++q)
                    {
                        auto& targets = base.target_qubit_num_logical_list;
                        if (std::find(targets.begin(), targets.end(), q) != targets.end())
                            continue;
                        targets.push_back(q);
                        self(self);
                        targets.pop_back();
                    }
                };
                enumerate(enumerate);
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
    Core c;
    c.num_qubits_local = 3;
    c.perm_p2l = c.perm_l2p = {0, 1, 2, 3};
    c.target_qubit_num_logical_list = {2, 3};
    fail_transport = true;
    bool failed = false;
    try { c.localize_target_qubits(); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed && c.perm_l2p == c.perm_p2l);
    assert((c.perm_l2p == std::vector<int>{0, 1, 2, 3}));
    fail_transport = false;
    c.localize_target_qubits();
    assert((c.target_qubit_num_physical_list == std::vector<int>{2, 1}));
    transport_calls = 0;
    const auto old_l2p = c.perm_l2p;
    c.target_qubit_num_logical_list = {0, 1, 2, 3};
    failed = false;
    try { c.localize_target_qubits(); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed && transport_calls == 0 && c.perm_l2p == old_l2p);
    std::cout << cases << " localization cases passed; transport is stubbed\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="qcs-localize-test-") as directory:
        cpp = Path(directory) / "localize.cpp"
        binary = Path(directory) / "localize"
        cpp.write_text(harness)
        compiler = shlex.split(os.environ.get("CXX", "g++"))
        subprocess.run(compiler + ["-std=c++17", "-O1", "-g",
                                  "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                                  str(cpp), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
