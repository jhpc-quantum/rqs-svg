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
p.add_argument('--backends', nargs='+', choices=('grouped','rank_sampled','tiled_cdf','auto'),
               default=['grouped','rank_sampled','tiled_cdf','auto'])
p.add_argument('--no-build', action='store_true')
a = p.parse_args()
if len(set(a.backends)) != len(a.backends):
    p.error('duplicate backends')
root = Path(__file__).resolve().parent.parent
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
manifest = {'arguments': vars(a) | {'output': str(out), 'binary': str(a.binary)},
            'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
            'environment': {}, 'commands': [],
            'working_tree': subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True),
            'diff': subprocess.check_output(['git','diff','HEAD'], cwd=root, text=True)}
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
for backend in a.backends:
    with (out/(backend+'.csv')).open('w') as data, (out/(backend+'.log')).open('w') as log:
        run(shlex.split(a.mpiexec)+['-n',str(a.ranks),str((root/a.binary).resolve()),str(a.qubits),str(a.warmups),str(a.repetitions)],
            env=os.environ | {'QCS_MEASUREMENT_BACKEND':backend}, stdout=data, stderr=log)
keys = ['qubits','gpus','nodes','state','targets','layout','deferred','mapping','warmups','repetitions']
tables = {}
for backend in a.backends:
    with (out/(backend+'.csv')).open() as source:
        table = {}
        for row in csv.DictReader(source):
            key = tuple(row[k] for k in keys)
            if row['backend'] != backend or key in table:
                raise RuntimeError('incorrect backend label or duplicate workload')
            table[key] = row
        if not table or (tables and table.keys() != next(iter(tables.values())).keys()):
            raise RuntimeError('empty or mismatched backend case sets')
        tables[backend] = table
# Preserve the original comparison when its two backends were selected.
if 'grouped' in tables and 'rank_sampled' in tables:
    with (out/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys+['grouped_median_ms','rank_sampled_median_ms','grouped_over_rank_sampled'])
        writer.writeheader()
        for key, row in tables['grouped'].items():
            g, r = float(row['median_ms']), float(tables['rank_sampled'][key]['median_ms'])
            writer.writerow(dict(zip(keys,key)) | {'grouped_median_ms':g,'rank_sampled_median_ms':r,'grouped_over_rank_sampled':g/r if r else ''})
with (out/'pairwise.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=keys+['baseline','candidate','baseline_median_ms','candidate_median_ms','baseline_over_candidate'])
    writer.writeheader()
    for i, baseline in enumerate(a.backends):
        for candidate in a.backends[i+1:]:
            for key, row in tables[baseline].items():
                b, c = float(row['median_ms']), float(tables[candidate][key]['median_ms'])
                writer.writerow(dict(zip(keys,key)) | {'baseline':baseline,'candidate':candidate,
                    'baseline_median_ms':b,'candidate_median_ms':c,'baseline_over_candidate':b/c if c else ''})
