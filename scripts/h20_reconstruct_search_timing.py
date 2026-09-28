"""Recover complete pre-training intervals from original log creation timestamps.

Only for early workflows before the process-lifetime clock was added. No raw
logs/timings are changed, and reconstructed boundaries are explicitly labelled.
"""
import subprocess
from datetime import datetime
from h20_campaign import *
root=DEFAULT_ROOT
for run in sorted((root/'runs').glob('*/*/*/*')):
 if not run.is_dir() or (run/'search_timing.json').exists():continue
 case,system,space,attempt=run.relative_to(root/'runs').parts
 end=run/'stages/selected_training.log'
 if not end.exists():continue
 starts=[]
 if case=='llama7b_2k' and space=='common' and attempt=='01':
  starts=[root/('llama7b-meg-common-01.log' if system=='devastator' else 'llama7b-galv-common-01.log')]
 else:
  for status in (root/'batches').glob('*/status.json'):
   data=json.loads(status.read_text())
   if any(j.get('case')==case and j.get('system')==system and j.get('space')==space and j.get('command') for j in data['jobs']):
    f=status.parent/(case+'-'+system+'.log')
    if f.exists():starts.append(f)
 if len(starts)!=1:continue
 start=starts[0]
 def birth(path):
  value=subprocess.check_output(['stat','-c','%w',str(path)],text=True).strip();assert value!='-',path
  return value,datetime.strptime(value[:26]+value[-5:],'%Y-%m-%d %H:%M:%S.%f%z').timestamp()
 st,s=birth(start);et,e=birth(end);assert e>s
 save(run/'search_timing.json',dict(search_e2e_seconds=e-s,method='Reconstructed from original driver stdout log birth to selected-training stdout log birth',outcome='selected',start_log=str(start),end_log=str(end),start_birth_utc=st,end_birth_utc=et,scope='All elapsed workflow time between these boundaries; includes profile/search startup and preparation/processing. Excludes earlier communication calibration and later selected training.',precision_note='Filesystem creation boundaries, not original monotonic instrumentation. Logs were created immediately before subprocess launch; tiny launch-boundary offsets remain.'))
 print(case,system,space,round(e-s,6))
