"""Host-only runner regression; synthetic timings are not GPU measurements."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRunnerTest(unittest.TestCase):
    def test_api_cases_remain_distinct_and_manifest_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            launcher = directory / 'launcher.py'
            launcher.write_text('''import csv, os, sys
w = csv.writer(sys.stdout)
w.writerow('backend qubits gpus nodes state targets layout deferred mapping api warmups repetitions first_call_ms p10_ms median_ms p90_ms samples_ms'.split())
for api in ('measure_many', 'measure_many_to_clbits'):
    w.writerow([os.environ['QCS_MEASUREMENT_BACKEND'], 20, 1, 1, 'zero', 3, 'local', 'local_low_high', 'reversed', api, 1, 3, 9, 1, 2, 3, '1;2;3'])
''')
            output = directory / 'results'
            subprocess.run([sys.executable, str(ROOT / 'scripts/benchmark_measurement.py'),
                            '--output', str(output), '--ranks', '1', '--no-build',
                            '--binary', str(launcher), '--mpiexec',
                            f'{sys.executable} {launcher}'], check=True, cwd=ROOT)
            with (output / 'pairwise.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 12)  # Six pairs, two API cases each.
            self.assertEqual({row['api'] for row in rows},
                             {'measure_many', 'measure_many_to_clbits'})
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['binary_sha256'],
                             hashlib.sha256(launcher.read_bytes()).hexdigest())
            self.assertEqual(len(manifest['commands']), 4)
            self.assertIn('probes', manifest)
            self.assertIn('build_preview', manifest)


if __name__ == '__main__':
    unittest.main()
