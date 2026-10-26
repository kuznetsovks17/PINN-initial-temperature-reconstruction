#!/usr/bin/env python3
"""Fast read-only repository sanity checks; does not import TensorFlow."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'data/SOURCE_MANIFEST.json').read_text())
for r in manifest:
    if 'destination' not in r:continue
    fn=ROOT/r['destination']
    assert fn.is_file(),fn
    hash_value=hashlib.sha256(fn.read_bytes()).hexdigest()
    assert r['sha256']==hash_value,f'Input SHA256 mismatch: {fn}'
nb=list(ROOT.glob('*.ipynb'))
assert len(nb)==10,len(nb)
assert not (ROOT/'START_PRODUCTION.ipynb').exists()
assert not (ROOT/'06_Sampling_exposure_and_dimensions.ipynb').exists()
assert not list(ROOT.rglob('*Zone.Identifier*'))
# Local compile/test steps may create ignored .pyc files; source archives exclude them.
for p in nb:
    obj=json.loads(p.read_text())
    assert all((cell.get('execution_count') is None and not cell.get('outputs'))
               for cell in obj['cells'] if cell['cell_type']=='code'),p
print(f'PASS: {len(nb)} output-free notebooks, {len(manifest)} hash-verified source inputs, no batch launcher or 3D extension.')
