#!/usr/bin/env python3
"""Run production measurement code with host CUDA types and synchronous stubs.

This checks arithmetic, index mapping and orchestration, not real CUB/CUDA/MPI.
No copied implementation: uniquely delimited source fragments are compiled.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'src/qcs.cu').read_text()


def fragment(start, end):
    assert source.count(start) == 1, start
    assert source.count(end) == 1, end
    return source.split(start, 1)[1].split(end, 1)[0]


preamble = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#define __host__
#define __device__
#define __constant__
namespace cuda { namespace std = ::std; }
namespace qcs {
'''
code = preamble
code += fragment('    struct simulator_core;', '    __device__ __host__ complex_t multiply_i(')
code += '    struct kernel_common_struct' + fragment('    struct kernel_common_struct', '    namespace gate\n')
code += '    namespace cubUtility\n    {' + fragment('    namespace cubUtility\n    {', '    } /* cubUtility */') + '}\n}\n'
code += (root / 'tests/measurement_host_runtime.hpp').read_text()
code += 'namespace qcs { enum class initstate_enum {}; struct simulator_core {\n'
code += fragment('        /* begin simulator variables */', '        /* end simulator variables */')
code += fragment('        }; /* localize_target_qubits */', '        void update_measured_list()')
code += '        void discard_measurement_state()' + fragment('        void discard_measurement_state()', '        void materialize_pending_projection(')
code += '        template <int N, int L = N>' + fragment('        template <int N, int L = N>', '        template <typename GateType>\n        void operate_gate(')
code += r'''
    void localize_target_qubits() {
        target_qubit_num_physical_list.clear();
        for (int q : target_qubit_num_logical_list) {
            if (perm_l2p[q] >= num_qubits_local)
                throw std::runtime_error("measurement tried to exchange a global target");
            target_qubit_num_physical_list.push_back(perm_l2p[q]);
        }
    }
}; }
'''
code += 'struct qcs_simulator\n{' + fragment('struct qcs_simulator\n{', 'static void qcs_simulator_init(qcs_simulator *sim);')
code += 'static std::vector<int> qcs_validate_measure_many(' + fragment('static std::vector<int> qcs_validate_measure_many(', 'int qcs_simulator_read_cxx(')
code += (root / 'tests/measurement_host.cpp').read_text()
with tempfile.TemporaryDirectory(prefix='qcs-measurement-') as directory:
    cpp = Path(directory) / 'measurement.cpp'
    exe = Path(directory) / 'measurement'
    cpp.write_text(code)
    command = shlex.split(os.environ.get('CXX', 'g++'))
    command += ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-unknown-pragmas', '-Wno-sign-compare']
    command += shlex.split(os.environ.get('MEASUREMENT_HOST_CXXFLAGS', ''))
    subprocess.run(command + [str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
