"""Native Galvatron profiler/search with isolated cold evidence and literal common candidates."""
import argparse
import copy
import math
import shlex
import shutil
import time
import yaml
from h20_campaign import *
from h20_workloads import cases,galv_runtime
from common_space import candidates,candidate_signature,require_equal

def yaml_save(path,data):Path(path).write_text(yaml.safe_dump(data,sort_keys=False))
def prepare(c,r,out,kind,memory_policy=None):
 from galvatron.core.profiler.args_schema import GalvatronModelProfilerArgs
 from galvatron.core.profiler.model_profiler import ModelProfiler
 pr=copy.deepcopy(r);template=out/'model_template.yaml';yaml_save(template,{})
 pr['model'].update(model_config_path=str(template),set_layernum_manually=1,set_seqlen_manually=1,initialize_on_meta=0)
 pr['train'].update(train_iters=20,chunks=1);pr['parallel']['async_grad_reduce']=False
 pr['profile'].update(profile=1,exit_after_profiling=1);pr['data']['use_random_dataset']=True
 path=out/(kind+'_runtime.yaml');yaml_save(path,{'runtime':pr})
 pa=GalvatronModelProfilerArgs(profile_type=kind,profile_mode='static',profile_unit='all',profile_flow_control='scripts_only',profile_mixed_precision=c['dtype'],profile_fixed_batch_size=1 if kind=='computation' else 8,profile_fixed_seq_length_list=[c['seq_length']],profile_layernum_min=1,profile_layernum_max=2,profile_max_tp_deg=max(row[2] for row in candidates(c)),profile_dp_type='ddp',runtime_yaml_template_path=str(path))
 if kind=='memory' and memory_policy and memory_policy['mode']=='sequence':
  pa.profile_mode='sequence'
  pa.profile_min_seq_length=memory_policy['min_seq_length'];pa.profile_max_seq_length=memory_policy['max_seq_length']
  assert 0<pa.profile_min_seq_length<=pa.profile_max_seq_length<=c['seq_length']
 for k,v in r['model'].items():setattr(pa.model_info,k,v)
 pa.model_info.model_config_path=str(template)
 save(out/(kind+'_profiler_args.json'),pa.model_dump())
 os.environ.update(NUM_NODES='1',NUM_GPUS_PER_NODE='8',RUNTIME_LAUNCHER='H20_NATIVE_ENTRY')
 gpt=out/'gpt';prof=ModelProfiler(pa);prof.set_profiler_launcher(str(gpt),r['model']['model_size']);prof.launch_profiling_scripts()
 jobs=[]
 for line in (gpt/'scripts'/(kind+'_profile_scripts_all.sh')).read_text().splitlines():
  if 'H20_NATIVE_ENTRY' not in line:continue
  ts=shlex.split(line.split('2>&1 | tee',1)[0]);i=ts.index('H20_NATIVE_ENTRY');env=dict(t.split('=',1) for t in ts[:i]);argv=ts[i+1:]
  vals=dict(t.split('=',1) for t in argv[1:]);pp=int(vals.get('runtime.parallel.pp_deg',1));tp=int(vals.get('runtime.parallel.global_tp_deg',1));world=1 if kind=='computation' else 8
  assert world%(pp*tp)==0 and int(vals['runtime.train.global_batch_size'])%((world//(pp*tp))*int(vals['runtime.train.chunks']))==0,line
  jobs.append((argv,env))
 assert jobs
 return pa,prof,jobs

def search(c,r,out,root,space):
 from galvatron.core.search_engine.args_schema import GalvatronSearchArgs
 from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
 from galvatron.utils.hf_config_adapter import model_layer_configs,model_name
 a=GalvatronSearchArgs()
 for k,v in r['model'].items():setattr(a.model_info,k,v)
 a.parallelism_info.mixed_precision=c['dtype'];a.parallelism_info.default_dp_type='ddp';a.parallelism_info.pipeline_type='pipedream_flush'
 a.common_train_info.seq_length=c['seq_length'];a.common_train_info.global_batch_size=c['global_batch_size'];a.common_train_info.sequence_parallel=True
 a.hardware_info.num_nodes=1;a.hardware_info.num_gpus_per_node=8;a.hardware_info.memory_constraint=json.loads((root/'protocol.json').read_text())['memory_limit_gib'];a.batch_size_info.settle_bsz=c['global_batch_size']
 for k in ('disable_ckpt','disable_fsdp','disable_cp','disable_sp','disable_embedding_lmhead_sp'):setattr(a.search_space_info,k,1 if space=='common' else 0)
 # Native serializer drops CP; no faithful native CP search-to-runtime roundtrip.
 a.search_space_info.disable_cp=1
 a.search_space_info.max_tp_deg=max(row[2] for row in candidates(c));a.search_space_info.max_pp_deg=8
 a.search_space_info.max_sp_deg=max(d for d in (1,2,4,8) if c['num_attention_heads']%d==0)
 a.options_info.fine_grained_mode=0 if space=='common' else 1;a.options_info.parallel_search=False
 a.options_info.output_config_path=str(out/'selected');a.options_info.log_dir=str(out/'search_logs')
 a.profiling_info.memory_profiling_path=str(out/'gpt/configs');a.profiling_info.time_profiling_path=str(out/'gpt/configs')
 profile_args=out/'memory_profiler_args.json'
 if profile_args.exists():a.profiling_info.memory_profile_mode=json.loads(profile_args.read_text())['profile_mode']
 for k in ('allreduce_bandwidth_config_path','p2p_bandwidth_config_path','overlap_coe_path','sp_time_path'):setattr(a.profiling_info,k,str(root/'hardware/galvatron/hardware_configs'))
 (out/'selected').mkdir();save(out/'search_args.json',a.model_dump())
 eng=GalvatronSearchEngine(a);eng.set_search_engine_info(path=str(out/'gpt'),model_layer_configs=model_layer_configs(a),model_name=model_name(a));eng.initialize_search_engine(show_all_strategy_list=False)
 layers=eng.layer_strategy_list;emb=eng.embedding_lmhead_strategy_list
 save(out/'space_definition.json',dict(space=space,args=a.model_dump(),layer_strategies=list(map(str,layers)),embedding_strategies=list(map(str,emb)),GQA='fixed native KV',CP='fixed 1: native strategy serializer drops CP',embedding_sdp='native selection passed explicitly to runtime vocab_sdp',mbs='1,2,4,8' if space=='common' else 'native global batch chunk divisors',pipeline='native uniform layer partition, pipedream_flush'))
 records=[];winner=None;t=time.perf_counter()
 def clean(x):
  if isinstance(x,float) and not math.isfinite(x):return None
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [clean(v) for v in x]
  return x
 native_save=eng.save_results
 def save_selected(optimal,bsz,chunks):
  save(out/'native_selected_result.json',dict(optimal=clean(optimal),global_batch_size=bsz,chunks=chunks))
  return native_save(optimal,bsz,chunks)
 eng.save_results=save_selected
 if space=='common':
  rows=candidates(c)
  supported={(s.dp_size,s.pp_size,s.tp_size,m) for s in layers if s.cp_size==s.sp_size==1 and not s.checkpoint and str(getattr(s.dp_type,'name',s.dp_type)).upper()=='DDP' for m in (1,2,4,8) if c['num_layers']%s.pp_size==0 and c['global_batch_size']%(s.dp_size*m)==0 and c['global_batch_size']//(s.dp_size*m)>=s.pp_size}
  require_equal(supported,rows,'Galvatron common supported')
  for dp,pp,tp,mbs in rows:
   chunks=c['global_batch_size']//(dp*mbs)
   eng.layer_strategy_list=[s for s in layers if (s.dp_size,s.pp_size,s.tp_size,s.sp_size,s.cp_size)==(dp,pp,tp,1,1) and not s.checkpoint and str(getattr(s.dp_type,'name',s.dp_type)).upper()=='DDP']
   eng.embedding_lmhead_strategy_list=[s for s in emb if (s.dp_size,s.pp_size,s.tp_size,s.sp_size,s.cp_size)==(dp,pp,tp,1,1) and str(getattr(s.dp_type,'name',s.dp_type)).upper()=='DDP']
   assert len(eng.layer_strategy_list)==len(eng.embedding_lmhead_strategy_list)==1
   result=eng.search_for_single_task(c['global_batch_size'],chunks,pp,tp,'tp_with_sp')
   rec=dict(candidate=[dp,pp,tp,mbs],chunks=chunks,result=clean(result));records.append(rec)
   save(out/'candidate_evaluations.json',records)
   throughput=float(result['throughput'])
   if math.isfinite(throughput) and throughput>0 and (winner is None or throughput>winner[0]):winner=(throughput,result,chunks,[dp,pp,tp,mbs])
  require_equal([x['candidate'] for x in records],rows,'Galvatron evaluated coverage')
  if winner:eng.save_results(winner[1],c['global_batch_size'],winner[2])
  save(out/'selected_candidate.json',dict(candidate=winner[3] if winner else None,common_candidate_sha256=candidate_signature(rows)))
 else:
  native=eng.search_for_single_task
  def observed(*aa,**kw):
   result=native(*aa,**kw);records.append(dict(arguments=aa,keywords=kw,result=clean(result)))
   with (out/'candidate_evaluations.jsonl').open('a') as f:f.write(json.dumps(records[-1],default=str,allow_nan=False)+'\n')
   return result
  eng.search_for_single_task=observed
  eng.parallelism_optimization()
 save(out/'search_summary.json',dict(status='completed',tasks=len(records),seconds=time.perf_counter()-t))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--case',required=True);p.add_argument('--space',choices=['common','full'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 c=cases(a.root)[a.case];search(c,galv_runtime(c,a.out,a.root),a.out,a.root,a.space)
