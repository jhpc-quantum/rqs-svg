"""Host-only regression checks; MPI/NCCL stubs do not validate cluster runtime."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NcclBufferSetupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        source = (ROOT / 'src/qcs.cu').read_text()
        start = source.index('            MPI_Comm_size', source.index('void setup_after_mpi()'))
        end = source.index('            atlc::group_by_hostname', start)
        # Compile the actual production validation block, without CUDA dependencies.
        stub = r'''
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
constexpr int nccl_buffer_mode = QCS_NCCL_BUFFER_MODE;
constexpr int NCCL_VERSION_CODE = 23102;
constexpr int MPI_COMM_WORLD = 0, MPI_INT = 0, MPI_MIN = 0, MPI_MAX = 1;
int calls = 0, set_calls = 0;
std::string scenario;
bool exchange_nvshmem_registered_put = false;
int MPI_Comm_size(int, int* n) { *n = 2; return 0; }
int MPI_Comm_rank(int, int* n) { *n = scenario == "nonroot" ? 1 : 0; return 0; }
int MPI_Allreduce(const int* in, int* out, int, int, int, int) {
    ++calls;
    *out = *in;
    if (scenario == "mixed" && calls == 2) *out = (nccl_buffer_mode + 1) % 3;
    if (scenario == "peer-setenv-failure" && calls == 3) *out = 0;
    if (scenario == "peer-unsupported" && calls == 4) *out = 0;
    return 0;
}
int ncclGetVersion(int* version) {
    *version = scenario == "old-version" ? 23000 : 23102;
    return 0;
}
int test_setenv(const char* name, const char* value, int overwrite) {
    ++set_calls;
    if (scenario == "setenv-failure" || scenario == name) return -1;
    return ::setenv(name, value, overwrite);
}
#define setenv test_setenv
#define ATLC_CHECK_MPI(fn, ...) fn(__VA_ARGS__)
#define ATLC_CHECK_NCCL(fn, ...) fn(__VA_ARGS__)
void setup() {
    int num_procs, proc_num;
'''
        tail = r'''
}
int main(int argc, char** argv) {
    scenario = argc > 1 ? argv[1] : "normal";
    exchange_nvshmem_registered_put = scenario == "nvshmem";
    std::string result = "ok";
    try { setup(); }
    catch (const std::runtime_error& error) { result = error.what(); }
    const char* value = std::getenv("NCCL_PXN_DISABLE");
    const char* channels = std::getenv("NCCL_NCHANNELS_PER_NET_PEER");
    const char* chunk = std::getenv("NCCL_P2P_NET_CHUNKSIZE");
    std::cout << result << '\n' << (value ? value : "<unset>") << '\n'
              << calls << '\n' << set_calls << '\n'
              << (channels ? channels : "<unset>") << '\n'
              << (chunk ? chunk : "<unset>") << '\n';
}
'''
        path = cls.directory / 'setup.cpp'
        path.write_text(stub + source[start:end] + tail)
        for mode in range(3):
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            f'-DQCS_NCCL_BUFFER_MODE={mode}', str(path),
                            '-o', str(cls.directory / f'setup-{mode}')], check=True)

    def run_setup(self, mode, value, scenario='normal', tuning=(None, None)):
        env = os.environ.copy()
        env.pop('NCCL_PXN_DISABLE', None)
        if value is not None:
            env['NCCL_PXN_DISABLE'] = value
        for name, setting in zip(('NCCL_NCHANNELS_PER_NET_PEER',
                                  'NCCL_P2P_NET_CHUNKSIZE'), tuning):
            env.pop(name, None)
            if setting is not None:
                env[name] = setting
        output = subprocess.check_output(
            [str(self.directory / f'setup-{mode}'), scenario], env=env, text=True)
        result, actual, calls, set_calls, channels, chunk = output.splitlines()
        return result, actual, int(calls), int(set_calls), channels, chunk

    def test_environment_matrix(self):
        for mode in range(3):
            for value in (None, '', '1', '0', 'other', ' 1 '):
                with self.subTest(mode=mode, value=value):
                    result, actual, calls, set_calls, _, _ = self.run_setup(mode, value)
                    defaulted = mode == 2 and not value
                    self.assertEqual(actual, '1' if defaulted else
                                     '<unset>' if value is None else value)
                    self.assertEqual(set_calls, 2 + int(defaulted))
                    self.assertEqual(calls, 4)
                    if mode == 2 and value not in (None, '', '1'):
                        self.assertIn('requires NCCL_PXN_DISABLE=1', result)
                    else:
                        self.assertEqual(result, 'ok')

    def test_tuning_defaults_and_overrides(self):
        for mode in range(3):
            for scenario in ('normal', 'nonroot'):
                for channels in (None, '', '4', 'custom'):
                    for chunk in (None, '', '131072', 'custom'):
                        with self.subTest(mode=mode, scenario=scenario, channels=channels, chunk=chunk):
                            result, _, calls, set_calls, actual_channels, actual_chunk = self.run_setup(
                                mode, '1', scenario, tuning=(channels, chunk))
                            self.assertEqual(result, 'ok')
                            self.assertEqual(actual_channels, channels or '8')
                            self.assertEqual(actual_chunk, chunk or '262144')
                            self.assertEqual(calls, 4)
                            self.assertEqual(set_calls, int(not channels) + int(not chunk))

    def test_each_default_failure(self):
        for mode in range(3):
            names = ['NCCL_NCHANNELS_PER_NET_PEER', 'NCCL_P2P_NET_CHUNKSIZE']
            if mode == 2:
                names.append('NCCL_PXN_DISABLE')
            for name in names:
                for value in (None, ''):
                    with self.subTest(mode=mode, name=name, value=value):
                        result, _, calls, set_calls, _, _ = self.run_setup(
                            mode, value, name, tuning=(value, value))
                        self.assertIn('Failed to set NCCL environment defaults', result)
                        self.assertEqual(calls, 3)
                        # A later successful setenv must not mask an earlier failure.
                        self.assertEqual(set_calls, 2 + int(mode == 2))

    def test_collective_failures(self):
        for scenario in ('setenv-failure', 'peer-setenv-failure'):
            for value in (None, ''):
                with self.subTest(scenario=scenario, value=value):
                    for mode in range(3):
                        result, _, calls, _, _, _ = self.run_setup(mode, value, scenario)
                        self.assertIn('Failed to set NCCL environment defaults', result)
                        self.assertEqual(calls, 3)  # Stop before configuration validation.
        for scenario in ('peer-unsupported', 'old-version', 'nvshmem'):
            for mode in (1, 2):
                with self.subTest(scenario=scenario, mode=mode):
                    result, _, calls, _, _, _ = self.run_setup(mode, None, scenario)
                    self.assertIn('require NCCL >= 2.31.2', result)
                    self.assertEqual(calls, 4)
        for mode in range(3):
            result, actual, calls, set_calls, _, _ = self.run_setup(mode, None, 'mixed')
            self.assertIn('differs across ranks', result)
            self.assertEqual((actual, calls, set_calls), ('<unset>', 2, 0))
        for scenario in ('old-version', 'nvshmem'):
            self.assertEqual(self.run_setup(0, None, scenario)[0], 'ok')

    def test_build_modes(self):
        for mode in (None, 0, 1, 2):
            command = ['make', '--no-print-directory', '-nB', 'sharedlibrary', 'nccl-buffer-harness']
            if mode is not None:
                command.append(f'NCCL_BUFFER_MODE={mode}')
            env = os.environ.copy()
            env.pop('NCCL_BUFFER_MODE', None)
            output = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
            selected = 2 if mode is None else mode
            self.assertIn(f'-DQCS_NCCL_BUFFER_MODE={selected}', output)
            self.assertIn(f'bin/nccl-buffer-{selected}', output)
        # Preprocess the real fallback definition without installed NCCL headers.
        (self.directory / 'nccl.h').write_text('')
        output = subprocess.check_output(
            ['g++', '-E', '-dM', '-x', 'c++', '-I', str(self.directory),
             str(ROOT / 'src/nccl_buffer.cuh')], text=True)
        self.assertIn('#define QCS_NCCL_BUFFER_MODE 2\n', output)


if __name__ == '__main__':
    unittest.main()
