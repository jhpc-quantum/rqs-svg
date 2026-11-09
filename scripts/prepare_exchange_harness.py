#!/usr/bin/env python3
"""Make a private non-pipelined source variant for exchange correctness tests."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.source.read_text()
old = 'constexpr bool exchange_pipeline_copy_back = true;'
if source.count(old) != 1:
    raise SystemExit('expected exactly one pipeline setting')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(source.replace(old, 'constexpr bool exchange_pipeline_copy_back = false;', 1))
