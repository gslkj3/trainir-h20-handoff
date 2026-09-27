"""CPU-only gate for the five-case shared candidate contract."""

import hashlib
import ast
import json
import os
import sys
from pathlib import Path

MEG = Path.cwd()
sys.path.insert(0,str(MEG))
from dtsir_common16.common5_space import candidates, megatron_parallel, require_equal, signature

GALV = Path(os.environ.get('GALV_REPO',
    '/data/run01/LEGACY_USER/wjy/dependencies/Hetu-Galvatron-dtsir'))
names = ('llama7b_2k','llama2_7b_4k','llama2_13b_4k',
         'llama3_8b_8k','qwen3_14b_4k')
# Fail on missing/stale upload dependencies before requesting an allocation.
for root, files in ((MEG,('dtsir_common16/run.py','dtsir_common16/entry.py',
    'dtsir_common16/common5_space.py','dtsir_common16/worker.sh','dtsir_common16/env.sh',
    'paired16/network.py','paired16/probe.py','paired16/probe_worker.sh',
    'paired16/all5_exact.sbatch','paired16/collect_exact.py')),
    (GALV,('dtsir_galvatron6/run_six.py','dtsir_galvatron6/common5_space.py',
    'dtsir_galvatron6/worker.sh','dtsir_galvatron6/env.sh'))):
    for relative in files:
        path=root/relative
        if not path.is_file(): raise RuntimeError(f'Missing upload/dependency: {path}')
        if path.suffix=='.py': ast.parse(path.read_text(),filename=str(path))
        if path.suffix in ('.sh','.sbatch') and b'\r\n' in path.read_bytes():
            raise RuntimeError(f'Windows CRLF line endings: convert {path} to LF before submission')
meg = json.loads((MEG/'dtsir_common16/cases.json').read_text())
galv = {c['id']:c for c in json.loads((GALV/'dtsir_galvatron6/cases.json').read_text())}
left = (MEG/'dtsir_common16/common5_space.py').read_bytes()
right = (GALV/'dtsir_galvatron6/common5_space.py').read_bytes()
if hashlib.sha256(left).digest() != hashlib.sha256(right).digest():
    raise RuntimeError('The two copies of common5_space.py differ')

for name in names:
    m, g = meg[name], galv[name]
    launcher=MEG/'dtsir_common16/launchers'/f'{name}.sh'
    if not launcher.is_file(): raise RuntimeError(f'Missing launcher: {launcher}')
    for a,b in (('num_layers','layers'),('hidden_size','hidden'),
                ('ffn_hidden_size','ffn'),('num_attention_heads','heads'),
                ('kv_heads','kv'),('seq_length','seq'),
                ('global_batch_size','gbs')):
        if m[a] != g[b]:
            raise RuntimeError(f'{name}: mismatched {a}/{b}: {m[a]} != {g[b]}')
    checks = (
        ('Galvatron dtype', g['dtype'], m['dtype']),
        ('Galvatron max_tp', g['max_tp'], 8),
        ('Megatron mbs', m['mbs'], [1,2,4,8]),
        ('Galvatron RMSNorm eps', g['eps'], 1e-5),
        ('dataset prefix', g['data'], m['data_path']),
    )
    differences = [f'{field}: actual={actual!r}, expected={expected!r}'
                   for field,actual,expected in checks if actual != expected]
    if differences:
        raise RuntimeError(f'{name}: ' + '; '.join(differences)
                           + f'. Check {GALV}/dtsir_galvatron6/cases.json and '
                             f'{MEG}/dtsir_common16/cases.json')
    rows = candidates(layers=g['layers'],hidden=g['hidden'],heads=g['heads'],
                      kv_heads=g['kv'],global_batch=g['gbs'])
    expected_count = 48 if name == 'llama3_8b_8k' else 52
    if len(rows) != expected_count:
        raise RuntimeError(f'{name}: got {len(rows)}, expected {expected_count}')
    previous = MEG/'mm_logs/paired16_20260921_211352_megatron'/name/'search/search.json'
    if previous.is_file():
        seen = [p for p in json.loads(previous.read_text())['screened'] if p[1] <= 8]
        require_equal(seen,[megatron_parallel(row,g['gbs']) for row in rows],
                      f'{name}: previous Megatron enumeration')
    print(f'{name}: {len(rows)} candidates, SHA256 {signature(rows)}')
print('PASS: common candidate contract; Galvatron strategy support is also '
      'checked on the login node, along with native Profile command generation.')
