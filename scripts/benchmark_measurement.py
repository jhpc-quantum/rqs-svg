#!/usr/bin/env python3
"""Run paired backend benchmarks using the existing Makefile infrastructure."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ranks', type=int, choices=(1,2,4,8,16,32,64,128), required=True)
p.add_argument('--qubits', type=int, default=33)
p.add_argument('--warmups', type=int, default=5)
p.add_argument('--repetitions', type=int, default=30)
p.add_argument('--mpiexec', default='mpiexec', help='launcher prefix, e.g. srun --mpi=pmix')
p.add_argument('--binary', type=Path, default=Path('bin/measurement_backend_benchmark'))
p.add_argument('--no-build', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent.parent
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
manifest = {'arguments': vars(a) | {'output': str(out), 'binary': str(a.binary)},
            'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
            'environment': {}, 'commands': []}
# Environment manifests can contain secrets: record only relevant tuning/build keys.
manifest['environment'] = {k:v for k,v in os.environ.items() if k.startswith(('NCCL_', 'CUDA_', 'SLURM_', 'OMP_')) or k in ('GENCODE_FLAGS','CFLAGS_VENDOR','LDFLAGS_VENDOR','NVSHMEM')}
manifest['config_mk'] = (root/'config.mk').read_text() if (root/'config.mk').exists() else None

def run(command, **kwargs):
    manifest['commands'].append(command)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run(command, cwd=root, check=True, **kwargs)

if not a.no_build:
    with (out/'build.log').open('w') as log:
        run(['make', str(a.binary)], stdout=log, stderr=subprocess.STDOUT)
for backend in ('grouped','rank_sampled','auto'):
    with (out/(backend+'.csv')).open('w') as data, (out/(backend+'.log')).open('w') as log:
        run(shlex.split(a.mpiexec)+['-n',str(a.ranks),str((root/a.binary).resolve()),str(a.qubits),str(a.warmups),str(a.repetitions)],
            env=os.environ | {'QCS_MEASUREMENT_BACKEND':backend}, stdout=data, stderr=log)
keys = ['qubits','gpus','nodes','state','targets','layout','deferred','warmups','repetitions']
with (out/'grouped.csv').open() as f:
    grouped = {tuple(r[k] for k in keys):r for r in csv.DictReader(f)}
with (out/'comparison.csv').open('w') as f, (out/'rank_sampled.csv').open() as source:
    writer = csv.DictWriter(f, fieldnames=keys+['grouped_median_ms','rank_sampled_median_ms','grouped_over_rank_sampled'])
    writer.writeheader()
    for row in csv.DictReader(source):
        old = grouped.pop(tuple(row[k] for k in keys))
        g, r = float(old['median_ms']), float(row['median_ms'])
        writer.writerow({k:row[k] for k in keys} | {'grouped_median_ms':g,'rank_sampled_median_ms':r,'grouped_over_rank_sampled':g/r if r else ''})
    if grouped:
        raise RuntimeError('backend case sets differ')
