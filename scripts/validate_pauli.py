#!/usr/bin/env python3
"""Build and run the Pauli and existing regression matrix with retained logs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new directory for builds/logs/report')
    parser.add_argument('--ranks', type=int, nargs='+', choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument('--buffer-modes', type=int, nargs='+', choices=(0, 1, 2))
    parser.add_argument('--nvshmem', action='store_true', help='optional NVSHMEM exchange; requires mode 0')
    parser.add_argument('--sanitizers', action='store_true', help='also run memcheck/racecheck/synccheck')
    parser.add_argument('--mpiexec', default='mpiexec', help='launcher and optional arguments')
    parser.add_argument('--make-arg', action='append', default=[], help='vendor/toolchain make assignments')
    parser.add_argument('--dry-run', action='store_true', help='record commands as planned without running them')
    args = parser.parse_args()
    modes = args.buffer_modes if args.buffer_modes is not None else ([0] if args.nvshmem else [0, 1, 2])
    if args.nvshmem and modes != [0]:
        parser.error('NVSHMEM requires --buffer-modes 0')
    reserved = {'NVSHMEM', 'NCCL_BUFFER_MODE', 'MEASUREMENT_METADATA', 'MEASUREMENT_HARNESS_DIR',
                'LIBQCS_SO', 'PAULI_CORE_METADATA_TEST', 'PAULI_API_TEST', 'PAULI_API_C_OBJECT',
                'PAULI_REDUCE_TEST', 'MEASUREMENT_REDUCE_TEST', 'INITIAL_STATE_TEST', 'SWAP_HARNESS'}
    if any('=' not in value or value.split('=', 1)[0] in reserved for value in args.make_arg):
        parser.error('--make-arg must be an assignment to a non-matrix/output variable')
    launcher = shlex.split(args.mpiexec)
    if not launcher:
        parser.error('empty MPI launcher')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'dry_run': args.dry_run, 'commands': [], 'status': 'planned' if args.dry_run else 'running',
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'working_tree': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
              'source': str(ROOT), 'ranks': args.ranks, 'buffer_modes': modes,
              'nvshmem': args.nvshmem, 'sanitizers': args.sanitizers}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(command):
        entry = {'argv': command, 'status': 'planned' if args.dry_run else 'running'}
        report['commands'].append(entry)
        logfile = output / f'{len(report["commands"]):04d}.log'
        entry['log'] = str(logfile)
        save()
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        try:
            with logfile.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            entry['returncode'] = result.returncode
            entry['status'] = 'passed' if result.returncode == 0 else 'failed'
        except OSError as error:
            logfile.write_text(str(error) + '\n')
            entry['status'] = 'failed'
            entry['error'] = str(error)
        save()
        if entry['status'] == 'failed':
            raise RuntimeError(f'command failed; see {logfile}')

    try:
        for mode in modes:
            for metadata in ('constant', 'global'):
                folder = output / f'mode-{mode}-{metadata}'
                names = {'LIBQCS_SO': 'libqcs.so', 'PAULI_CORE_METADATA_TEST': 'pauli_core_test',
                         'PAULI_API_TEST': 'pauli_api_test', 'PAULI_API_C_OBJECT': 'pauli_api_c.o',
                         'PAULI_REDUCE_TEST': 'pauli_reduce_test', 'MEASUREMENT_REDUCE_TEST': 'measurement_reduce_test',
                         'INITIAL_STATE_TEST': 'initial_state_test', 'SWAP_HARNESS': 'swap_harness'}
                paths = {key: str(folder / name) for key, name in names.items()}
                assignments = [f'{key}={value}' for key, value in paths.items()]
                assignments += [f'MEASUREMENT_HARNESS_DIR={folder / "measurement"}',
                                f'MEASUREMENT_METADATA={metadata}', f'NCCL_BUFFER_MODE={mode}',
                                f'NVSHMEM={int(args.nvshmem)}']
                targets = [paths[key] for key in names if key != 'PAULI_API_C_OBJECT']
                targets += [str(folder / 'measurement' / 'measurement_harness'),
                            str(folder / 'measurement' / 'measurement_integration_test')]
                run(['make', '-B', *args.make_arg, *assignments, *targets])
                # Standalone tests require one device/rank only.
                for key in ('PAULI_REDUCE_TEST', 'MEASUREMENT_REDUCE_TEST', 'INITIAL_STATE_TEST'):
                    run([*launcher, '-n', '1', paths[key]])
                for ranks in args.ranks:
                    binaries = [(paths['PAULI_CORE_METADATA_TEST'], []), (paths['PAULI_API_TEST'], []),
                                (paths['SWAP_HARNESS'], ['regression', '12']),
                                (str(folder / 'measurement' / 'measurement_harness'), ['regression', '12']),
                                (str(folder / 'measurement' / 'measurement_integration_test'), [])]
                    for binary, extra in binaries:
                        run([*launcher, '-n', str(ranks), binary, *extra])
                    if args.sanitizers:
                        for tool in ('memcheck', 'racecheck', 'synccheck'):
                            run([*launcher, '-n', str(ranks), 'compute-sanitizer', '--error-exitcode', '1',
                                 '--tool', tool, paths['PAULI_CORE_METADATA_TEST']])
                if args.sanitizers:
                    for tool in ('memcheck', 'racecheck', 'synccheck'):
                        run([*launcher, '-n', '1', 'compute-sanitizer', '--error-exitcode', '1',
                             '--tool', tool, paths['PAULI_REDUCE_TEST']])
        report['status'] = 'planned' if args.dry_run else 'passed'
    except RuntimeError as error:
        report['status'] = 'failed'
        print(error)
        return 1
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
