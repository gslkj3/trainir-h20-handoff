"""Run via a generated copy of the user's working launcher. No core edits."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path.cwd()))

def canonical(record):
    return json.dumps({'parallel':record['parallel'], 'strategy':record['strategy']}, sort_keys=True)

def strategy_grid(level, kv_heads, heads, vpp_sizes):
    import itertools
    for rec, shard, vpp in itertools.product(
        [False, True] if level >= 1 else [False],
        [False, True] if level >= 2 else [False],
        [None]+vpp_sizes if level >= 3 else [None]):
        yield {'ReCompute':['selective',['mlp']] if rec else [None,None],
               'DistributedOptimizer':shard, 'VirtualPipe':vpp,
               'Hybrid_MHA_MQA':[kv_heads != heads, kv_heads]}

def search(args):
    import torch
    import torch.distributed as dist
    from test_parallel_model import GPT, TPDS_RUNTIME as rt, apply_candidate_from_env
    plan=json.loads(os.environ['DTSIR_SPACE_PLAN'])
    target=Path(os.environ['DTSIR_IR_OUT'])
    if dist.get_rank() == 0:
        if args.world_size != 4 or args.num_experts is not None:
            raise RuntimeError('This pilot requires four GPUs and a dense model.')
        if (args.num_layers,args.hidden_size,args.ffn_hidden_size,args.num_attention_heads,args.seq_length,args.global_batch_size)!=(8,4096,11008,32,2048,256):
            raise RuntimeError('Pilot requires the established 8-layer 4096/11008/32-head model, sequence 2048 and global batch 256; formal workloads require a reviewed plan.')
        if plan['kv_heads'] > args.num_attention_heads or args.num_attention_heads % plan['kv_heads']:
            raise RuntimeError('Invalid fixed KV head count')
        evidence=os.environ['DTSIR_MML_LOGS']
        rt.refresh(evidence)
        rt.config.experiment='space_ladder'
        rt.config.variant='full'
        rt.config.capture_candidates=True
        rt.config.capture_rejected=True
        rt.config.candidate_limit=0
        rt.config.strategy_limit=0
        rt.config.max_mbs=max(plan['mbs'])
        rt.config.profile_seed_from_existing=True
        rt.reset()
        original=GPT._tpds_enumerate_structural_candidates
        def restricted(model):
            rows=[]
            for s in original(model):
                if s[2]!=1 or s[3]!=1 or s[6]!=1 or s[7] not in plan['mbs']:
                    continue
                s=list(s); s[5]=1  # Fixed SP-off policy in every tier.
                rows.append(s)
            return rows
        GPT._tpds_enumerate_structural_candidates=restricted
        screened=[]
        start=time.perf_counter()
        for i, strategy in enumerate(strategy_grid(plan['level'],plan['kv_heads'],args.num_attention_heads,plan['vpp_sizes'])):
            a=copy.deepcopy(args)
            a.recompute_granularity, a.recompute_modules=strategy['ReCompute']
            a.use_distributed_optimizer=strategy['DistributedOptimizer']
            a.num_layers_per_virtual_pipeline_stage=strategy['VirtualPipe']
            a.virtual_pipeline_model_parallel_size=None
            a.group_query_attention,a.num_query_groups=strategy['Hybrid_MHA_MQA']
            rt.note_strategy(a)
            model=GPT(a,mmlogs_path=evidence,search_level=4)
            possible=restricted(model)
            vpp=strategy['VirtualPipe']
            if vpp is not None:
                possible=[s for s in possible if s[1]>1 and a.num_layers%(s[1]*vpp)==0 and a.num_layers>s[1]*vpp]
            screened.extend({'parallel':s,'strategy':copy.deepcopy(strategy)} for s in possible)
            feasible=model.search_space_create()
            print(f"P{plan['level']} strategy={i} feasible={len(feasible)} {strategy}",flush=True)
            if feasible:
                model.costmodel_create(feasible)
            for (name,key),value in rt.profile_overlay.items():
                model.map_manager.data_map.setdefault(name,{})[key]=value
            model.map_manager._save_to_json()
        elapsed=time.perf_counter()-start
        records=rt.candidate_records
        if any(not __import__('math').isfinite(float(r['predicted_cost'])) for r in records):
            raise RuntimeError('Non-finite predicted cost')
        best=min(records,key=lambda r:(r['predicted_cost'],canonical(r))) if records else None
        if best:
            best=copy.deepcopy(best)
            best['candidate_id']=f"P{plan['level']}_best"
            # Do not silently canonicalize an invalid VPP winner on lowering.
            from test_parallel_model import _tpds_canonicalize_measurement_record
            if canonical(_tpds_canonicalize_measurement_record(best,args.num_layers)) != canonical(best):
                raise RuntimeError('Winner changes under backend canonicalization')
        payload=rt.result_dict({'search_seconds':elapsed,'plan':plan,'best':best,
            'screened':screened,'evaluated':records,'rejected':rt.rejected_records,
            'model':{k:getattr(args,k,None) for k in ('num_layers','hidden_size','ffn_hidden_size','num_attention_heads','seq_length','global_batch_size','fp16','bf16','seed')},
            'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
            'scope':'Controlled warm-start four-GPU pilot; selective MLP recomputation only; fixed GQA and SP-off; no total-memory error claim.'})
        (target/'search.json').write_text(json.dumps(payload,indent=2))
    dist.barrier()

if __name__ == '__main__':
    if '--distributed-timeout-minutes' not in sys.argv:
        sys.argv += ['--distributed-timeout-minutes','180']
    if os.environ.get('DTSIR_SPACE_SEARCH')=='1':
        import dtsir_collect
        dtsir_collect.predict_fixed=search
    runpy.run_path(str(Path.cwd()/'pretrain_gpt.py'),run_name='__main__')
