#!/usr/bin/env python3
"""Copy and instrument a source tree; never edit either production checkout."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--metadata', choices=('constant', 'global'), default='constant')
a = parser.parse_args()
source = (a.source / 'src/qcs.cu').read_text()

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f'unsupported source: expected exactly one occurrence of {old!r}')
    return text.replace(old, new, 1)

source = replace_once(source, 'constexpr bool use_global_memory_for_constants = false;',
                      'constexpr bool use_global_memory_for_constants = ' +
                      ('true;' if a.metadata == 'global' else 'false;'))
gpu_sampling = 'measurement_workspace->enqueue_sample<N>' in source
if gpu_sampling:
    collective = 'ATLC_CHECK_NCCL(ncclAllReduce, norm_device, norm_device, 1 << N, ncclDouble, ncclSum, nccl_comm, stream);'
    source = replace_once(source, collective,
        'if (measurement_probe::observer) measurement_probe::capture_local(N, norm_device, stream);\n            ' +
        collective + '\n            if (measurement_probe::observer) measurement_probe::observe(N, qubits.data(), norm_device, stream);')
    readback = '            if (result.status == measurement_detail::sample_status::invalid_weight)'
    source = replace_once(source, readback,
        '            if (measurement_probe::result_observer) measurement_probe::result_observer(result.outcome, static_cast<unsigned>(result.status));\n' + readback)
else:
    collective = 'ATLC_CHECK_MPI(MPI_Allreduce, local_norm.data(), global_norm.data(), 1 << N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);'
    source = replace_once(source, collective, collective + '\n            if (measurement_probe::observer) measurement_probe::observer(N, qubits.data(), local_norm.data(), global_norm.data());')
custom = 'measurement_workspace->enqueue<N>' in source
a.output.mkdir(parents=True, exist_ok=True)
for name in ('work_buffer.hpp', 'work_buffer.cuh', 'nccl_buffer.cuh', 'swap_launch.hpp'):
    header_path = a.source / 'src' / name
    if header_path.exists():
        (a.output / name).write_text(header_path.read_text())
if custom:
    header = (a.source / 'src/measurement_reduce.cuh').read_text()
    start = '        first_stage<N><<<config.blocks, block_size, config.dynamic_shared, stream>>>(state, packed, data_);'
    end = '        second_stage<N><<<config.blocks, block_size, 0, stream>>>(data_, output);\n        ATLC_CHECK_CUDA(cudaGetLastError);'
    header = replace_once(header, start, '        measurement_probe::begin(stream);\n' + start)
    header = replace_once(header, end, end + '\n        measurement_probe::end(stream);')
    (a.output / 'measurement_reduce.cuh').write_text(header)
else:
    call = '            ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, cub_temp_buffer_device, cub_temp_buffer_device_size, in_it, output, count, add, zero, stream);'
    source = replace_once(source, call, '            measurement_probe::begin(stream);\n' + call + '\n            measurement_probe::end(stream);')
(a.output / 'qcs.cu').write_text('#include "' + str((Path(__file__).resolve().parent.parent / 'tests/measurement_probe.hpp')) + '"\n' + source)
(a.output / 'backend.hpp').write_text('#define MEASUREMENT_CUSTOM ' + str(int(custom)) + '\n#define MEASUREMENT_GPU_SAMPLING ' + str(int(gpu_sampling)) + '\n')
