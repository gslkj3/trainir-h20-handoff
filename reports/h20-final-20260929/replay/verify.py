"""Validate published result values and every archived file; standard library only."""
import hashlib,json,math,statistics,tarfile
from pathlib import Path
p=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
for record in read(p/'MANIFEST.json'):
 f=p/record['path'];assert f.stat().st_size==record['bytes'] and sha(f.read_bytes())==record['sha256'],f
results=read(p/'results.json');pairs=read(p/'comparison.json');assert len(results)==32 and len(pairs)==16
assert len({(r['case'],r['space'],r['system']) for r in results})==32
for archive in read(p/'evidence/index.json'):
 f=p/archive['archive'];assert sha(f.read_bytes())==archive['sha256']
 records=read(p/'evidence'/f"{archive['case']}.manifest.json")
 with tarfile.open(f,'r:gz') as t:
  files={m.name:m for m in t.getmembers()};assert len(files)==len(records)
  for r in records:
   assert files[r['path']].isfile() and sha(t.extractfile(r['path']).read())==r['sha256']
  for row in [r for r in results if r['case']==archive['case']]:
   s=json.load(t.extractfile('measurements/'+row['evidence_id']+'/training_summary.json'))
   assert math.isclose(s['mean_iteration_s'],row['mean_iteration_s'],rel_tol=1e-10)
   assert math.isclose(statistics.mean(s['iteration_s'][5:10]),row['mean_iteration_s'],rel_tol=1e-10)
for r in pairs:
 d=next(x for x in results if (x['case'],x['space'],x['system'])==(r['case'],r['space'],'devastator'))
 g=next(x for x in results if (x['case'],x['space'],x['system'])==(r['case'],r['space'],'galvatron'))
 assert d['mean_iteration_s']==r['devastator_iteration_s'] and g['mean_iteration_s']==r['galvatron_iteration_s']
 assert math.isclose(r['speedup'],g['mean_iteration_s']/d['mean_iteration_s'],rel_tol=1e-10)
 if r['equal_by_reuse']:assert d['evidence_id']==g['evidence_id'] and r['speedup']==1
timings=read(p/'search_timing.json');assert len(timings)==32
for r in timings:
 f=p/r['evidence'];assert sha(f.read_bytes())==r['evidence_sha256']
 t=read(f);assert t['search_e2e_seconds']==r['search_e2e_seconds']>0
 row=next(x for x in results if (x['case'],x['space'],x['system'])==(r['case'],r['space'],r['system']))
 assert row['search_e2e_seconds']==r['search_e2e_seconds']
print('PASS: 16 comparisons, 32 result rows, 32 measured search timings, all files and archives verified.')
