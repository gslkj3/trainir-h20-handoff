"""Serial campaign queue; each attempt remains immutable and failures are retained."""
import argparse
import subprocess
from h20_campaign import *
from h20_workloads import cases
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--space',choices=['common','full'],required=True);p.add_argument('--cases',nargs='+');p.add_argument('--systems',nargs='+',choices=['devastator','galvatron'],default=['devastator','galvatron']);p.add_argument('--tag',required=True);a=p.parse_args()
folder=a.root/'batches'/a.tag;folder.mkdir(parents=True,exist_ok=False);jobs=[]
for name in a.cases or cases(a.root):
 assert name in cases(a.root)
 for system in a.systems:
  if (a.root/'STOP_AFTER_STAGE').exists():save(folder/'status.json',dict(status='stopped_at_job_boundary',jobs=jobs));raise SystemExit(0)
  out=a.root/'runs'/name/system/a.space/'01'
  if out.exists():jobs.append(dict(case=name,system=system,status='existing_attempt_preserved',path=str(out)));continue
  script='h20_run_megatron.py' if system=='devastator' else 'h20_run_galvatron.py'
  python='/usr/bin/python3' if system=='devastator' else str(PYTHONS['galvatron'])
  env=environment('megatron' if system=='devastator' else 'galvatron',folder/(name+'-'+system))
  cmd=[python,str(REPO/'scripts'/script),'--root',str(a.root),'--case',name,'--space',a.space]
  record=dict(case=name,system=system,space=a.space,command=cmd,status='running',start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  jobs.append(record);save(folder/'status.json',dict(status='running',jobs=jobs));t=time.monotonic()
  with (folder/(name+'-'+system+'.log')).open('w') as f:
   result=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
  record.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed',seconds=time.monotonic()-t)
  save(folder/'status.json',dict(status='running',jobs=jobs))
  subprocess.run(['/usr/bin/python3',str(REPO/'scripts/h20_summarize_campaign.py')],cwd=REPO,check=True)
save(folder/'status.json',dict(status='finished',jobs=jobs))
