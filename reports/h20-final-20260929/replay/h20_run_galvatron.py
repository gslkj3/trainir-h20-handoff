"""Run one native cold profile/search/train workflow; all evidence isolated per attempt."""
import argparse
import shutil
import time
from h20_campaign import *
from h20_workloads import cases,galv_runtime
from h20_galvatron_search import prepare,yaml_save
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--case',required=True);p.add_argument('--space',choices=['common','full'],required=True);p.add_argument('--attempt',default='01');a=p.parse_args()
c=cases(a.root)[a.case];out=a.root/'runs'/a.case/'galvatron'/a.space/a.attempt;out.mkdir(parents=True,exist_ok=False)
gpt=out/'gpt';gpt.mkdir();(gpt/'scripts').mkdir();(gpt/'configs').mkdir()
for f in ('train_dist.py','search_dist.py'):shutil.copy2(GALV/'galvatron/models/gpt'/f,gpt/f)
r=galv_runtime(c,out,a.root)
policy_path=a.root/'galvatron_profile_policy.json'
memory_policy=json.loads(policy_path.read_text()).get('cases',{}).get(a.case) if policy_path.exists() else None
communication_dir=a.root/'hardware/galvatron/hardware_configs'
communication_files={f.name:sha(f) for f in communication_dir.glob('*.json')}
assert len(communication_files)==4,'Expected completed four-file Galvatron hardware calibration'
save(out/'contract.json',dict(case=c,runtime=r,space=a.space,memory_profile_policy=memory_policy,communication_files=communication_files,protocol=json.loads((a.root/'protocol.json').read_text()),scripts={f.name:sha(f) for f in (REPO/'scripts').glob('h20_*.py')}))
try:
 for kind in ('computation','memory'):
  pa,prof,jobs=prepare(c,r,out,kind,memory_policy=memory_policy)
  save(out/(kind+'_jobs.json'),jobs)
  for i,(argv,native_env) in enumerate(jobs):
   name=f'{kind}_{i:03d}';env=environment('galvatron',out/name);env.update(native_env)
   cmd=launch('galvatron',[REPO/'scripts/h20_galvatron_entry.py','--source',gpt/'train_dist.py','--config',argv[0],'--mode','profile','--']+argv[1:],1 if kind=='computation' else 8)
   stage(out,name,cmd,gpt,env,3600)
  pa.profile_flow_control='data_only';t=time.perf_counter();prof.process_profiled_data();save(out/(kind+'_processing.json'),dict(seconds=time.perf_counter()-t))
 cmd=launch('galvatron',[REPO/'scripts/h20_galvatron_search.py','--root',a.root,'--case',a.case,'--space',a.space,'--out',out])
 stage(out,'search',cmd,gpt,environment('galvatron',out/'search'),14400)
 selected=list((out/'selected').glob('galvatron_config_*.json'))
 if not selected:
  finish_search_timing(out,'no_feasible_candidate')
  save(out/'status.json',dict(status='no_feasible_candidate'));raise SystemExit(0)
 assert len(selected)==1
 winner=json.loads(selected[0].read_text());assert winner['global_bsz']==c['global_batch_size'] and winner['world_size']==8
 if a.space=='common':
  chosen=json.loads((out/'selected_candidate.json').read_text())['candidate'];dp,pp,tp,mbs=chosen
  assert winner['pp_deg']==pp and winner['vtp']==tp and winner['vsp']==0 and winner['chunks']==c['global_batch_size']//(dp*mbs)
  for key,value in [('tp_sizes_enc',tp),('dp_types_enc',0),('use_sp',0),('checkpoint',0)]:assert set(map(int,winner[key].split(',')))=={value}
  assert list(map(int,winner['pp_division'].split(',')))==[c['num_layers']//pp]*pp
  assert winner['embed_sdp']==0 and winner['default_dp_type']=='ddp'
  r['train']['micro_batch_size']=mbs;r['train']['sequence_parallel']=True
 r['parallel']['vocab_sdp']=winner['embed_sdp']
 r['parallel']['galvatron_config_path']=str(selected[0]);r['train']['chunks']=winner['chunks']
 runtime=out/'selected_runtime.yaml';yaml_save(runtime,{'runtime':r})
 cmd=launch('galvatron',[REPO/'scripts/h20_galvatron_entry.py','--source',gpt/'train_dist.py','--config',runtime,'--mode','train','--out',out],8)
 stage(out,'selected_training',cmd,gpt,environment('galvatron',out/'training'),7200)
 save(out/'status.json',dict(status='completed'))
except Exception as e:
 if not (out/'search_timing.json').exists():finish_search_timing(out,'failed_before_selected_training')
 save(out/'status.json',dict(status='failed',error=repr(e)));raise
