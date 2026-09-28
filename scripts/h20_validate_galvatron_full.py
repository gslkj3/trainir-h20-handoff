"""Bounded native eight-GPU validation of full-space dimensions before searching them."""
import copy
import argparse
import shutil
from h20_campaign import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
base=json.loads(Path('/opt/hbv/trainir-h20-validation-20260927/galvatron-qwen3-tp2-bf16-02/runtime.yaml').read_text())
from galvatron.utils.strategy_utils import LayerStrategy,DPType,strategy_list2config
jobs=[('ulysses4-kv2',dict(tp=1,sp=4,dp=1,ckpt=False,zero=False)),('zero3-checkpoint',dict(tp=2,sp=1,dp=2,ckpt=True,zero=True)),('heterogeneous',None),('zero3-with-vocab',dict(tp=2,sp=1,dp=2,ckpt=True,zero=True,vocab_sdp=1))]
for name,params in jobs:
 out=a.out/name;out.mkdir();cfg=copy.deepcopy(base);r=cfg['runtime']
 r['model']['model_config_path']=None;r['model']['num_layers']=4;r['model']['num_query_groups']=2
 vocab_sdp=(params or {}).get('vocab_sdp',0)
 r['parallel'].update(reduce_in_fp32=True,entropy_in_fp32=True,pipeline_type='pipedream_flush',vocab_sdp=vocab_sdp)
 r['data']['data_cache_path']=str(out/'dataset-cache')
 strategies=[]
 for layer in range(4):
  v=params or dict(tp=1 if layer<2 else 2,sp=1,dp=4 if layer<2 else 2,ckpt=layer%2==1,zero=layer%2==1)
  strategies.append(LayerStrategy(pp_size=2,tp_size=v['tp'],sp_size=v['sp'],cp_size=1,dp_size=v['dp'],checkpoint=v['ckpt'],dp_type=DPType.ZERO3 if v['zero'] else DPType.DDP))
 selected=strategy_list2config(strategies);selected.update(global_bsz=8,chunks=2,pp_division='2,2',pipeline_type='pipedream_flush',default_dp_type='ddp',vtp=2,vsp=0,vocab_sdp=vocab_sdp)
 save(out/'selected.json',selected);r['parallel']['galvatron_config_path']=str(out/'selected.json');r['train']['chunks']=2
 save(out/'runtime.yaml',cfg)
 cmd=launch('galvatron',[REPO/'scripts/h20_galvatron_validation.py',out/'runtime.yaml',out],8)
 stage(out,'validation',cmd,GALV,environment('galvatron',out),900)
 ranks=[json.loads(f.read_text()) for f in out.glob('rank[0-7].json')];assert len(ranks)==8
 if params and params['zero']:
  layouts=[json.loads((out/f'fsdp_rank{i}.json').read_text()) for i in range(8)]
  assert all(any('FULL_SHARD' in m['sharding_strategy'] and m['group_size']>1 for m in layout) for layout in layouts), 'Zero3 must actually shard parameters on every rank'
 save(out/'status.json',dict(status='completed',scope='Small-model real-input functional validation, excluded from benchmark timings'))
save(a.out/'status.json',dict(status='completed',cases=[j[0] for j in jobs]))
