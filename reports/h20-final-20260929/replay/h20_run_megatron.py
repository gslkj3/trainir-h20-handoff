"""One isolated full-size Devastator search and selected training attempt."""
import argparse
import shutil
from h20_campaign import *
from h20_workloads import cases,meg_arguments
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--case',required=True);p.add_argument('--space',choices=['common','full'],required=True);p.add_argument('--attempt',default='01');p.add_argument('--search-only',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
c=cases(a.root)[a.case];out=a.out or a.root/'runs'/a.case/'devastator'/a.space/a.attempt;out.mkdir(parents=True,exist_ok=False)
evidence=out/'native_evidence';(evidence/'comm_data').mkdir(parents=True);(evidence/'calc_data').mkdir()
shutil.copy2(a.root/'hardware/devastator/comm_data/profile_comm.json',evidence/'comm_data/profile_comm.json')
save(evidence/'calc_data/data.json',{})
save(out/'contract.json',dict(case=c,space=a.space,protocol=json.loads((a.root/'protocol.json').read_text()),scripts={f.name:sha(f) for f in (REPO/'scripts').glob('h20_*.py')},communication_sha256=sha(evidence/'comm_data/profile_comm.json')))
env=environment('megatron',out);env.update(H20_CAMPAIGN_ROOT=str(a.root),H20_RUN_OUT=str(out),H20_CASE=a.case,H20_SPACE=a.space,H20_PHASE='search',DTSIR_IR_FIXED='1',DTSIR_MML_LOGS=str(evidence),DTSIR_IR_OUT=str(out),DTSIR_LOG_DIR=str(out/'native_logs'))
cmd=launch('megatron',[REPO/'scripts/h20_megatron_entry.py']+meg_arguments(c,out),8)
try:
 stage(out,'search',cmd,MEG,env,14400)
 result=json.loads((out/'search.json').read_text());best=result['best']
 if best is None:
  finish_search_timing(out,'no_feasible_candidate')
  save(out/'status.json',dict(status='no_feasible_candidate'));raise SystemExit(0)
 save(out/'selected_candidate.json',best)
 if a.search_only:
  finish_search_timing(out,'search_only_profile_refresh')
  save(out/'status.json',dict(status='profile_refresh_completed',training_executed=False))
  raise SystemExit(0)
 from h20_equal_common_reuse import decide
 reuse=decide(a.root,c,a.space,best)
 save(out/'common_equal_config_reuse_decision.json',reuse)
 if reuse['eligible']:
  finish_search_timing(out,'selected_equal_common_config_reused')
  save(out/'timing_reuse.json',reuse)
  save(out/'status.json',dict(status='reused_equal_common_config',megatron_training_executed=False))
  raise SystemExit(0)
 env.pop('DTSIR_IR_FIXED');env.update(H20_PHASE='train',DTSIR_EXPERIMENT='measure',DTSIR_MEASURE_CANDIDATE_JSON=json.dumps(best))
 stage(out,'selected_training',cmd,MEG,env,7200)
 save(out/'status.json',dict(status='completed'))
except Exception as e:
 if not (out/'search_timing.json').exists():finish_search_timing(out,'failed_before_selected_training')
 save(out/'status.json',dict(status='failed',error=repr(e)));raise
