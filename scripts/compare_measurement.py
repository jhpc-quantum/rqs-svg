#!/usr/bin/env python3
"""Build isolated custom/CUB probes and record a reproducible paired run."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

BASELINE = '1ce55aff510934fd1502abf933e8c26a1db79133'
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--baseline', default=BASELINE)
p.add_argument('--ranks', type=int, choices=(1, 2, 4), default=1)
p.add_argument('--qubits', type=int, nargs='+', default=[12, 24])
p.add_argument('--warmups', type=int, default=5)
p.add_argument('--repetitions', type=int, default=30)
p.add_argument('--metadata', choices=('constant', 'global'), default='constant')
p.add_argument('--mpiexec', default='mpiexec', help='launcher and optional flags, shell quoted')
a = p.parse_args()
root = Path(__file__).resolve().parent.parent
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)

def capture(command):
    return subprocess.check_output(command, cwd=root, text=True).strip()

revision = capture(['git', 'rev-parse', 'HEAD'])
baseline = capture(['git', 'rev-parse', a.baseline + '^{commit}'])
manifest = {
    'reviewed_baseline': BASELINE, 'baseline': baseline, 'custom': revision,
    'working_tree': capture(['git', 'status', '--porcelain']),
    'arguments': vars(a) | {'output': str(out)},
    'environment': {key: os.environ[key] for key in
                    ('CUDA_VISIBLE_DEVICES', 'NVCC', 'CFLAGS_VENDOR', 'LDFLAGS_VENDOR',
                     'GENCODE_FLAGS', 'LDLIBS', 'NVSHMEM', 'NVSHMEM_HOME') if key in os.environ},
    'config_mk': (root / 'config.mk').read_text() if (root / 'config.mk').exists() else None,
    'commands': [],
}

def save():
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

save()
# git archive preserves tracked contents; the baseline is never placed over HEAD.
with tempfile.TemporaryDirectory(prefix='qcs-cub-baseline-') as temporary:
    baseline_tree = Path(temporary)
    archive = baseline_tree / 'baseline.tar'
    with archive.open('wb') as file:
        subprocess.run(['git', 'archive', baseline], cwd=root, stdout=file, check=True)
    with tarfile.open(archive) as files:
        files.extractall(baseline_tree, filter='data')
    for name, source in [('custom', root), ('CUB', baseline_tree)]:
        build = out / ('build-' + name)
        command = ['make', 'measurement-harness', f'MEASUREMENT_METADATA={a.metadata}',
                   f'MEASUREMENT_HARNESS_SOURCE={source}', f'MEASUREMENT_HARNESS_DIR={build}']
        manifest['commands'].append(command)
        save()
        with (out / (name + '-build.log')).open('w') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        for qubits in a.qubits:
            command = shlex.split(a.mpiexec) + ['-n', str(a.ranks), str(build / 'measurement_harness'),
                       'benchmark', str(qubits), str(a.warmups), str(a.repetitions)]
            manifest['commands'].append(command)
            save()
            with (out / f'{name}-{qubits}.csv').open('w') as csv_file, (out / f'{name}-{qubits}.log').open('w') as log:
                subprocess.run(command, cwd=root, stdout=csv_file, stderr=log, check=True)

# Preserve raw timings and make regressions visible without selecting cases.
with (out / 'comparison.csv').open('w') as report:
    columns = ['qubits', 'N', 'layout', 'control', 'kernel_CUB_over_custom',
               'measurement_CUB_over_custom', 'kernel_regression', 'measurement_regression']
    writer = csv.DictWriter(report, fieldnames=columns)
    writer.writeheader()
    for qubits in a.qubits:
        with (out / f'custom-{qubits}.csv').open() as file:
            custom = {(r['N'], r['layout'], r['control']): r for r in csv.DictReader(file)}
        with (out / f'CUB-{qubits}.csv').open() as file:
            for cub in csv.DictReader(file):
                key = (cub['N'], cub['layout'], cub['control'])
                current = custom.pop(key)
                ratios = {}
                for metric in ('kernel', 'measurement'):
                    old = float(cub[metric + '_median_ms'])
                    new = float(current[metric + '_median_ms'])
                    if old <= 0 or new <= 0:
                        raise RuntimeError('nonpositive timing; increase workload or repetitions')
                    ratios[metric + '_CUB_over_custom'] = old / new
                    ratios[metric + '_regression'] = new > old
                writer.writerow(dict(qubits=qubits, N=key[0], layout=key[1], control=key[2], **ratios))
        if custom:
            raise RuntimeError('baseline/custom benchmark case sets differ')
