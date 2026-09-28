"""Derived plot tables only. Original native predictions retain their units and provenance."""
import csv
import json
from h20_campaign import *
root=DEFAULT_ROOT;rows=[];profiles=[]
for path in sorted((root/'runs').glob('*/*/*/*')):
 if not path.is_dir():continue
 case,system,space,attempt=path.relative_to(root/'runs').parts
 base=dict(case=case,system=system,space=space,attempt=attempt,run=str(path))
 p=path/'search.json'
 if system=='devastator' and p.exists():
  data=json.loads(p.read_text())
  for status in ('evaluated','rejected'):
   for i,r in enumerate(data[status]):
    row=dict(base,candidate_index=i,candidate_status=status,**dict(zip(('dp','pp','cp','up','tp','sp','ep','mbs','chunks'),r['parallel'])),kv_heads=r['strategy']['Hybrid_MHA_MQA'][1],strategy=json.dumps(r['strategy'],sort_keys=True),predicted_cost=r.get('predicted_cost'),predicted_cost_unit='microseconds (native operator clock)',predicted_feasible=r.get('predicted_feasible'),predicted_peak_bytes=r.get('predicted_peak_memory'),reason=r.get('reason'))
    rows.append(row)
  profiles.append(dict(base,kind='native_operator_profile',**data['stats'],**data['timings']))
 if system=='galvatron':
  p=path/'candidate_evaluations.json'
  tasks=json.loads(p.read_text()) if p.exists() else []
  p=path/'candidate_evaluations.jsonl'
  if p.exists():
   for line in p.read_text().splitlines():
    try:tasks.append(json.loads(line))
    except json.JSONDecodeError:pass # Live final line is incomplete; rerun after completion.
  for i,r in enumerate(tasks):
   result=r['result'];row=dict(base,candidate_index=i,predicted_feasible=result.get('throughput',-1)>0,predicted_cost=result.get('time_cost'),predicted_cost_unit='seconds',predicted_samples_per_second=result.get('throughput'),memory_mib=json.dumps(result.get('memory_cost')),strategy=str(result.get('strategy_list')))
   if 'candidate' in r:row.update(dict(zip(('dp','pp','tp','mbs'),r['candidate'])),cp=1,up=1,chunks=r['chunks'])
   else:row['native_task_arguments']=json.dumps(r.get('arguments'));row['native_task_keywords']=json.dumps(r.get('keywords'))
   rows.append(row)
 for p in sorted((path/'stages').glob('*.json')):
  if p.stem.startswith(('computation_','memory_')):
   d=json.loads(p.read_text());profiles.append(dict(base,kind=p.stem,status=d['status'],wall_seconds=d.get('seconds'),log_sha256=d.get('log_sha256')))
for name,data in [('candidates',rows),('profile_accounting',profiles)]:
 keys=list(dict.fromkeys(k for row in data for k in row))
 with (root/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
print(dict(candidates=len(rows),profile_records=len(profiles)))
