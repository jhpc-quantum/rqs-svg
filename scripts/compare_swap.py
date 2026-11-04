#!/usr/bin/env python3
"""Opt-in clean build, symbol, regression and runtime SWAP comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import tarfile
import time

from check_swap_instantiations import audit, require_dedicated

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='new directory, retained with all source/logs')
    p.add_argument('--baseline', default='9f833b5^', help='revision before dedicated SWAP')
    p.add_argument('--current', default='HEAD', help='committed revision; working-tree edits are excluded')
    p.add_argument('--make-arg', action='append', default=[], help='identical make assignment for both revisions')
    p.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc --forward-unknown-to-host-compiler'))
    p.add_argument('--nvcc-time', action='store_true', help='enable NVCC --time CSV output if supported')
    p.add_argument('--build-repetitions', type=int, default=3)
    p.add_argument('--ranks', type=int, choices=(1, 2, 4), default=1)
    p.add_argument('--qubits', nargs='+', type=int, default=[12, 20, 24])
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--repetitions', type=int, default=30)
    p.add_argument('--mpiexec', default='mpiexec')
    p.add_argument('--regression-threshold', type=float, default=1.10,
                   help='flag current/baseline median ratios above this value')
    a = p.parse_args()
    if len(set(a.qubits)) != len(a.qubits):
        p.error('qubit sizes must be distinct')
    if (a.build_repetitions < 1 or a.warmups < 0 or a.repetitions < 1 or
            any(q < 12 or q > 30 for q in a.qubits) or a.regression_threshold <= 1):
        p.error('invalid repetition, size or regression threshold')
    # Prevent optional assignments from accidentally selecting different outputs.
    forbidden = ('NVCC', 'LIBQCS_SO', 'SWAP_HARNESS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')
    if any(arg.split('=', 1)[0] in forbidden or '=' not in arg for arg in a.make_arg):
        p.error('make arguments must be assignments other than reserved compiler/output variables')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def capture(command):
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()

    manifest = {'arguments': vars(a) | {'output': str(out)},
                'working_tree_excluded': capture(['git', 'status', '--porcelain']),
                'revisions': {name: capture(['git', 'rev-parse', ref + '^{commit}'])
                              for name, ref in [('baseline', a.baseline), ('current', a.current)]},
                'environment': dict(os.environ), 'commands': [], 'builds': [], 'audit': {}}
    # Record only build/runtime variables, avoiding credentials in unrelated env.
    manifest['environment'] = {k: v for k, v in manifest['environment'].items()
                               if k in ('CUDA_VISIBLE_DEVICES', 'CFLAGS_VENDOR', 'LDFLAGS_VENDOR',
                                        'GENCODE_FLAGS', 'LDLIBS', 'NVSHMEM', 'NVSHMEM_HOME',
                                        'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'PATH',
                                        'LD_LIBRARY_PATH', 'CPATH', 'CXX') or
                               k.startswith(('NCCL_', 'OMPI_', 'CUDA_CACHE_'))}
    config = ROOT / 'config.mk'
    manifest['config_mk'] = config.read_text() if config.exists() else None

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(command, cwd, log, timed=False):
        manifest['commands'].append({'argv': command, 'cwd': str(cwd), 'log': str(log)})
        save()
        memory = Path(str(log) + '.peak-rss-kib')
        actual = ['/usr/bin/time', '-f', '%M', '-o', str(memory)] + command if timed else command
        start = time.monotonic()
        with log.open('w') as stream:
            subprocess.run(actual, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return {'wall_seconds': time.monotonic() - start,
                'peak_rss_kib': int(memory.read_text().strip()) if timed else None}

    save()
    run(shlex.split(a.nvcc) + ['--version'], ROOT, out / 'nvcc-version.log')
    run(['nvidia-smi'], ROOT, out / 'gpu.log')
    run(shlex.split(a.mpiexec) + ['--version'], ROOT, out / 'mpi-version.log')
    # Use the exact same harness and build recipes for both snapshots. The
    # standalone harness includes that snapshot's qcs.cu without editing it.
    harness = (ROOT / 'tests/swap_harness.cu').read_bytes()
    (out / 'swap_harness.cu').write_bytes(harness)
    rows = []
    for name in ('baseline', 'current'):
        tree = out / name
        tree.mkdir()
        archive = out / (name + '.tar')
        with archive.open('wb') as stream:
            subprocess.run(['git', 'archive', manifest['revisions'][name]], cwd=ROOT, stdout=stream, check=True)
        with tarfile.open(archive) as files:
            files.extractall(tree, filter='data')
        if config.exists():
            shutil.copyfile(config, tree / 'config.mk')
        (tree / 'tests/swap_harness.cu').write_bytes(harness)
        extra_make = tree / 'SwapComparison.mk'
        extra_make.write_text(
            '.PHONY: swap-comparison-harness\n'
            'swap-comparison-harness:\n'
            '\t$(NVCC) $(NVCC_CFLAGS) tests/swap_harness.cu -o bin/swap-comparison-harness $(NVCC_LDFLAGS)\n')
        (tree / 'bin').mkdir(exist_ok=True)
        base = ['make', '--no-print-directory', '-j1', '-B', '-f', 'Makefile', '-f', str(extra_make)] + a.make_arg
        for iteration in range(a.build_repetitions):
            for target in ('sharedlibrary', 'swap-comparison-harness'):
                label = f'{name}-{target}-{iteration}'
                compiler = a.nvcc
                if a.nvcc_time:
                    compiler += ' --time ' + shlex.quote(str(out / (label + '.nvcc-time.csv')))
                result = run(base + ['NVCC=' + compiler, 'LIBQCS_SO=lib/libqcs.so', target], tree, out / (label + '.log'), timed=True)
                binary = tree / ('lib/libqcs.so' if target == 'sharedlibrary' else 'bin/swap-comparison-harness')
                result.update(variant=name, target=target, iteration=iteration, binary_bytes=binary.stat().st_size)
                manifest['builds'].append(result)
                save()
        evidence = audit(tree, tree / 'lib/libqcs.so')
        manifest['audit'][name] = evidence
        save()
        if name == 'current':
            require_dedicated(evidence, require_symbols=True)
        elif not any(e.get('legacy_matches') for e in evidence['symbols'].values()):
            raise RuntimeError('baseline has no observed legacy SWAP symbols; select the original revision')
        launcher = shlex.split(a.mpiexec) + ['-n', str(a.ranks), str(tree / 'bin/swap-comparison-harness')]
        run(launcher + ['regression', '12'], tree, out / (name + '-regression.log'))
        for bits in a.qubits:
            log = out / f'{name}-{bits}.log'
            run(launcher + ['benchmark', str(bits), str(a.warmups), str(a.repetitions)], tree, log)
            # Simulator/MPI diagnostics may share stdout; retain the full log and
            # extract only the harness's numeric CSV rows for comparison.
            data = [line for line in log.read_text().splitlines() if line.startswith('qubits,') or
                    (line and line[0].isdigit() and len(line.split(',')) == 10)]
            csv_path = out / f'{name}-{bits}.csv'
            csv_path.write_text('\n'.join(data) + '\n')
            parsed = list(csv.DictReader(data))
            if len(parsed) != 48:
                raise RuntimeError(f'{name}/{bits}: expected 48 benchmark cases, got {len(parsed)}')
            rows.extend(dict(variant=name, **row) for row in parsed)
    keys = ('qubits', 'ranks', 'swaps', 'layout', 'control')
    old = {tuple(r[k] for k in keys): r for r in rows if r['variant'] == 'baseline'}
    new = {tuple(r[k] for k in keys): r for r in rows if r['variant'] == 'current'}
    if old.keys() != new.keys():
        raise RuntimeError('benchmark case sets differ')
    with (out / 'comparison.csv').open('w') as stream:
        fields = list(keys) + ['baseline_ms', 'current_ms', 'current_over_baseline', 'flagged_regression']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in old:
            before, after = float(old[key]['median_ms']), float(new[key]['median_ms'])
            if before <= 0 or after <= 0:
                raise RuntimeError('nonpositive runtime measurement')
            writer.writerow(dict(zip(keys, key)) | {'baseline_ms': before, 'current_ms': after,
                'current_over_baseline': after / before,
                'flagged_regression': after / before > a.regression_threshold})
    with (out / 'build-comparison.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'target', 'median_wall_seconds', 'max_peak_rss_kib', 'binary_bytes'])
        for name in ('baseline', 'current'):
            for target in ('sharedlibrary', 'swap-comparison-harness'):
                builds = [b for b in manifest['builds'] if b['variant'] == name and b['target'] == target]
                writer.writerow([name, target, statistics.median(b['wall_seconds'] for b in builds),
                                 max(b['peak_rss_kib'] for b in builds), builds[-1]['binary_bytes']])
    manifest['status'] = 'completed; inspect regression logs, phase CSVs and flagged runtime cases'
    save()


if __name__ == '__main__':
    main()
