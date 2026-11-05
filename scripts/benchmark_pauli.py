#!/usr/bin/env python3
"""Run the Pauli timing sweep and retain raw CSV, logs, and percentile summaries."""
import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def summarize(paths, output, repetitions):
    groups = {}
    for path in paths:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f"empty timing CSV: {path}")
            for row in rows:
                key = tuple(row[k] for k in ('qubits', 'ranks', 'case', 'path', 'method'))
                groups.setdefault(key, []).append(row)
    fields = ['qubits', 'ranks', 'case', 'path', 'method', 'samples']
    metrics = ('stage1_ms', 'stage2_ms', 'wall_ms')
    fields += [f'{metric}_{p}' for metric in metrics for p in ('p10', 'median', 'p90')]
    summaries = {}
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            if sorted(int(row['sample']) for row in rows) != list(range(repetitions)):
                raise ValueError(f'incomplete or duplicate samples for {key}')
            summary = dict(zip(fields[:5], key)) | {'samples': len(rows)}
            for metric in metrics:
                values = sorted(float(row[metric]) for row in rows if row[metric])
                if any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError(f'invalid timing for {key}')
                if values:
                    summary[metric + '_p10'] = values[int(.1 * (len(values) - 1))]
                    summary[metric + '_median'] = statistics.median(values)
                    summary[metric + '_p90'] = values[int(.9 * (len(values) - 1))]
            summaries[key] = summary
            writer.writerow(summary)
    with (output / 'ratios.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['qubits', 'ranks', 'case', 'stage1_scan_over_generated', 'wall_scan_over_generated'])
        for key, scan in sorted(summaries.items()):
            if key[-1] != 'scan-branch':
                continue
            generated = summaries[key[:-1] + ('generated',)]
            ratios = [scan[m] / generated[m] if generated[m] else ''
                      for m in ('stage1_ms_median', 'wall_ms_median')]
            writer.writerow([*key[:3], *ratios])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory for results and binary')
    parser.add_argument('--qubits', type=int, nargs='+', default=[12, 20, 24])
    parser.add_argument('--ranks', type=int, nargs='+', choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--buffer-mode', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--make-arg', action='append', default=[])
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1 or any(n < 4 or n > 30 for n in args.qubits):
        parser.error('require 4..30 total qubits, nonnegative warmups, and positive repetitions')
    if len(set(args.qubits)) != len(args.qubits) or len(set(args.ranks)) != len(args.ranks):
        parser.error('sizes and ranks must be distinct')
    reserved = {'PAULI_BENCHMARK', 'NCCL_BUFFER_MODE', 'NVSHMEM'}
    if any('=' not in a or a.split('=', 1)[0] in reserved for a in args.make_arg):
        parser.error('--make-arg must assign a non-output/non-mode variable')
    launcher = shlex.split(args.mpiexec)
    if not launcher:
        parser.error('empty launcher')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'planned' if args.dry_run else 'running', 'commands': [],
              'arguments': vars(args) | {'output': str(out)},
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'working_tree': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
              'source': str(ROOT)}

    def save():
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(command, stdout, stderr):
        entry = {'argv': command, 'status': 'planned' if args.dry_run else 'running',
                 'stdout': str(stdout), 'stderr': str(stderr)}
        report['commands'].append(entry)
        save()
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        try:
            with stdout.open('w') as output, stderr.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=log, check=False)
            entry['returncode'] = result.returncode
            entry['status'] = 'passed' if result.returncode == 0 else 'failed'
        except OSError as error:
            entry['status'] = 'failed'
            entry['error'] = str(error)
        save()
        if entry['status'] == 'failed':
            raise RuntimeError(f'command failed; see {stderr} and {stdout}')

    try:
        binary = out / 'pauli_benchmark'
        run(['make', '-B', *args.make_arg, f'PAULI_BENCHMARK={binary}',
             f'NCCL_BUFFER_MODE={args.buffer_mode}', 'NVSHMEM=0', str(binary)],
            out / 'build.stdout', out / 'build.stderr')
        paths = []
        for ranks in args.ranks:
            for n in args.qubits:
                raw = out / f'ranks-{ranks}-qubits-{n}.csv'
                run([*launcher, '-n', str(ranks), str(binary), str(n), str(args.warmups), str(args.repetitions)],
                    raw, raw.with_suffix('.log'))
                paths.append(raw)
        if not args.dry_run:
            summarize(paths, out, args.repetitions)
        report['status'] = 'planned' if args.dry_run else 'passed'
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        print(error)
        return 1
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
