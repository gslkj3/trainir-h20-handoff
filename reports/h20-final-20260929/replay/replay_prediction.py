"""Replay one published 30 GiB/s ranking using archived candidates/profiles; no GPU training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,sys,types,shutil,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True,help='Extracted prediction/CASE/SPACE directory');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise SystemExit('Use a new output filename')
def read(p):return json.loads(p.read_text())
root=a.inputs.resolve();expected=read(root/'ranking.json');search=read(root/'search_input.json');cache=read(root/'profile.json')
source=Path(__file__).with_name('test_parallel_model.py')
for path,key in [(source,'source_sha256'),(root/'search_input.json','input_sha256'),(root/'profile.json','profile_sha256')]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[key],path
mod=types.ModuleType('published_predictor');mod.__file__=str(source);sys.modules[mod.__name__]=mod
exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
mod.TPDS_RUNTIME.config.experiment='cross_model_ablation';mod.TPDS_RUNTIME.config.variant='full';mod.TPDS_RUNTIME.reset()
mod.FileJSONHandler._save_to_json=lambda *args,**kwargs:None
mod.ModelConfig._tpds_profile_value=lambda self,name,key,measure:cache[name][key]
scratch=tempfile.TemporaryDirectory(prefix='h20-replay-')
work=Path(scratch.name)
(work/'calc_data').mkdir()
shutil.copy2(root/'profile.json',work/'calc_data/data.json')
shutil.copytree(root/'native_evidence/comm_data',work/'comm_data')
groups={}
for c in search['evaluated']:groups.setdefault(json.dumps(c['strategy'],sort_keys=True),[]).append(c)
rows=[]
for st in read(root/'space_definition.json')['strategies']:
 args=types.SimpleNamespace(**copy.deepcopy(search['effective_args']))
 args.recompute_granularity,args.recompute_modules=copy.deepcopy(st['ReCompute']);args.use_distributed_optimizer=st['DistributedOptimizer']
 args.num_layers_per_virtual_pipeline_stage=st['VirtualPipe'];args.virtual_pipeline_model_parallel_size=None;args.overlap_p2p_comm=st['VirtualPipe'] is not None
 args.group_query_attention,args.num_query_groups=st['Hybrid_MHA_MQA'];args.sequence_parallel=True
 mod.TPDS_RUNTIME.note_strategy(args);model=mod.GPT(args,str(work),search_level=4);model.device_name='NVIDIA H20'
 for c in groups.get(json.dumps(st,sort_keys=True),[]):
  value,_=model.costmodel_create([c['parallel']]);rows.append(dict(parallel=c['parallel'],strategy=st,predicted_cost=value))
rows.sort(key=lambda x:(x['predicted_cost'],json.dumps(x['parallel'])))
assert len(rows)==len(expected['rows'])
for x,y in zip(rows,expected['rows']):
 assert x['parallel']==y['parallel'] and x['strategy']==y['strategy']
 assert abs(x['predicted_cost']-y['predicted_cost'])<0.1,(x,y)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='passed',candidates=len(rows),best=rows[0],rows=rows),indent=2)+'\n')
print('Verified',len(rows),'candidates; winner and all costs match.')
