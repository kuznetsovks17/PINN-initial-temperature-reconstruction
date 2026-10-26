"""Run lightweight numerical checks; optional short neural check is in notebook03."""
from pathlib import Path
import json
from inverse_study.common import ROOT, DATA, sha256_file, environment
from inverse_study.numerics import run_self_checks
from inverse_study.qr_audit import run_qr_self_checks

def main():
    manifest = json.loads((DATA / 'SOURCE_MANIFEST.json').read_text())
    copied = [r for r in manifest if 'destination' in r]
    for row in copied:
        path = ROOT / row['destination']
        assert sha256_file(path) == row['sha256'], f'Source file changed: {path}'
    print(json.dumps({'environment': environment(), 'source_files_verified': len(copied),
                      'numerics': run_self_checks(), 'QR_1d': run_qr_self_checks()}, indent=2))

if __name__ == '__main__':
    main()
