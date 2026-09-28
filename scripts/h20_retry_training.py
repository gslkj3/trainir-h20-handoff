"""Replay a recorded selected training in a fresh evidence directory; preserve original search."""
import argparse
import shutil
from h20_campaign import *
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--retry',required=True);p.add_argument('--galvatron-sequence-parallel',action='store_true');a=p.parse_args()
r=a.run.resolve();o=r/'training_retries'/a.retry;o.mkdir(parents=True,exist_ok=False)
old=json.loads((r/'stages/selected_training.json').read_text());system='megatron' if 'devastator' in r.parts else 'galvatron'
env=environment(system,o);env.update(old['environment']);cmd=old['command']
if system=='megatron':env['H20_RUN_OUT']=str(o)
else:
 cmd[cmd.index('--out')+1]=str(o)
 if a.galvatron_sequence_parallel:cmd+=['--','runtime.train.sequence_parallel=True']
save(o/'provenance.json',dict(original_run=str(r),original_training_stage=old,script_hashes={f.name:sha(f) for f in (REPO/'scripts').glob('h20_*.py')},reason='Training-only integration repair; unchanged selected candidate, profile and search.',galvatron_sequence_parallel_override=a.galvatron_sequence_parallel))
try:
 stage(o,'selected_training',cmd,old['cwd'],env,7200);save(o/'status.json',dict(status='completed'))
except Exception as e:
 save(o/'status.json',dict(status='failed',error=repr(e)));raise
