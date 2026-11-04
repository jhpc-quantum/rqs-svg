#!/usr/bin/env python3
"""Explicit source/symbol audit; never invoked by the normal production build."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def audit(source, binary=None):
    source = Path(source)
    text = (source / 'src/qcs.cu').read_text()
    definition = re.search(r'template\s*<\s*unsigned int num_swaps\s*>\s*struct swap\b', text)
    legacy = re.compile(r'(?:qcs::)?gate::swap\s*<')
    calls = []
    for path in (source / 'src').rglob('*'):
        if path.suffix in ('.cu', '.cuh', '.cpp', '.hpp', '.h', '.c'):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if legacy.search(line):
                    calls.append(f'{path.relative_to(source)}:{number}:{line.strip()}')
    wrappers = {}
    for suffix in ('', '1', '2', '3', '4'):
        name = 'qcs_simulator_gate_swap' + suffix
        body = re.search(r'extern \"C\" int ' + name + r'\([^\n]*\)\n\{(.*?)\n\}', text, re.S)
        wrappers[name] = bool(body and 'qcs_simulator_gate_swap_cxx(sim,' in body.group(1))
    result = {'legacy_definition_present': bool(definition), 'legacy_references': calls,
              'public_wrappers_use_common_dispatch': wrappers,
              'dedicated_kernel_present': '__global__ void cuda_swap(' in text,
              'dedicated_host_dispatch_present': 'sim->core->operate_swap(targets, negctrls, ctrls);' in text,
              'symbols': {}}
    if binary:
        commands = {'nm': ['nm', '-C', '--defined-only', str(binary)],
                    'cuobjdump': ['cuobjdump', '--dump-elf-symbols', str(binary)]}
        for name, command in commands.items():
            if not shutil.which(command[0]):
                result['symbols'][name] = {'status': 'unavailable'}
                continue
            raw = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            if name == 'cuobjdump':
                raw = subprocess.check_output(['c++filt'], input=raw, text=True)
            Path(str(binary) + '.' + name + '.txt').write_text(raw)
            result['symbols'][name] = {
                'status': 'inspected',
                'legacy_matches': [line for line in raw.splitlines() if legacy.search(line)],
                'dedicated_matches': [line for line in raw.splitlines() if 'cuda_swap(' in line],
            }
    return result


def require_dedicated(result, require_symbols=False):
    if (not result['legacy_definition_present'] or result['legacy_references'] or
            not result['dedicated_kernel_present'] or not result['dedicated_host_dispatch_present'] or
            not all(result['public_wrappers_use_common_dispatch'].values())):
        raise RuntimeError('dedicated SWAP source audit failed')
    for evidence in result['symbols'].values():
        if evidence.get('legacy_matches'):
            raise RuntimeError('legacy SWAP template symbol found')
    if require_symbols and not any(e.get('dedicated_matches') for e in result['symbols'].values()):
        raise RuntimeError('no dedicated CUDA SWAP symbol observed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--require-dedicated', action='store_true')
    args = parser.parse_args()
    result = audit(args.source, args.binary)
    print(json.dumps(result, indent=2))
    if args.require_dedicated:
        require_dedicated(result, bool(args.binary))


if __name__ == '__main__':
    main()
