import copy
import itertools
import json
import math
import os
from pathlib import Path
import runpy
import time
import torch
import torch.distributed as dist
from h20_campaign import *
from h20_workloads import cases,megatron_semantic_adapter
from h20_training_evidence import sample,finish,model_evidence
from common_space import candidates as common_candidates,require_equal,candidate_signature

root=Path(os.environ['H20_CAMPAIGN_ROOT']);out=Path(os.environ['H20_RUN_OUT'])
c=cases(root)[os.environ['H20_CASE']];space_name=os.environ['H20_SPACE']
protocol=json.loads((root/'protocol.json').read_text())

def strategies():
    native=c['native_kv_heads'];heads=c['num_attention_heads'];layers=c['num_layers']
    if space_name=='common':
        yield dict(ReCompute=[None,None],DistributedOptimizer=False,VirtualPipe=None,Hybrid_MHA_MQA=[native!=heads,native]);return
    groups=[k for k in range(min(8,native),heads+1) if heads%k==0]
    vpps=[None]+[n for n in range(1,layers//2+1) if layers%n==0]
    for kv,rec,shard,vpp in itertools.product(groups,[False,True],[False,True],vpps):
        yield dict(ReCompute=['selective',['mlp']] if rec else [None,None],DistributedOptimizer=shard,
            VirtualPipe=vpp,Hybrid_MHA_MQA=[kv!=heads,kv])

def search(args):
    from test_parallel_model import GPT,TPDS_RUNTIME as rt,_tpds_canonicalize_measurement_record
    if dist.get_rank()==0:
        evidence=os.environ['DTSIR_MML_LOGS'];rt.refresh(evidence)
        rt.config.experiment='h20_'+space_name;rt.config.variant='full'
        rt.config.capture_candidates=True;rt.config.capture_rejected=True
        rt.config.candidate_limit=0;rt.config.strategy_limit=0;rt.config.max_mbs=8
        rt.config.profile_seed_from_existing=False;rt.reset()
        original=GPT._tpds_enumerate_structural_candidates
        expected=[[dp,pp,1,1,tp,tp,1,mbs,c['global_batch_size']//(dp*mbs)] for dp,pp,tp,mbs in common_candidates(c)]
        all_strategies=list(strategies())
        save(out/'space_definition.json',dict(space=space_name,strategies=all_strategies,
            fixed=dict(world_size=8,mbs=[1,2,4,8],sp='TP',ep=1),
            gqa_min=min(8,c['native_kv_heads']),cp_up='1' if space_name=='common' else 'legal factors of 8',
            recompute='none' if space_name=='common' else 'none or selective MLP',
            termination='exhaust finite strategy/candidate list',memory_limit_gib=protocol['memory_limit_gib']))
        def restricted(model):
            rows=[s for s in original(model) if s[7] in (1,2,4,8) and s[6]==1]
            if space_name=='common':
                rows=[s for s in rows if s[2]==s[3]==1]
                require_equal(rows,expected,'Devastator common candidates')
            vpp=getattr(model.args,'num_layers_per_virtual_pipeline_stage',None)
            if vpp:
                rows=[s for s in rows if s[1]>1 and args.num_layers%(s[1]*vpp)==0
                      and args.num_layers>s[1]*vpp and s[8]%s[1]==0]
            return rows
        GPT._tpds_enumerate_structural_candidates=restricted
        begun=time.perf_counter();screened=[];strategy_status=[]
        for index,st in enumerate(all_strategies):
            aa=copy.deepcopy(args)
            aa.recompute_granularity,aa.recompute_modules=st['ReCompute']
            aa.use_distributed_optimizer=st['DistributedOptimizer'];aa.num_layers_per_virtual_pipeline_stage=st['VirtualPipe']
            aa.virtual_pipeline_model_parallel_size=None
            aa.overlap_p2p_comm=st['VirtualPipe'] is not None
            aa.group_query_attention,aa.num_query_groups=st['Hybrid_MHA_MQA'];aa.sequence_parallel=True
            rt.note_strategy(aa)
            model=GPT(aa,mmlogs_path=evidence,search_level=4)
            structural=restricted(model)
            screened.extend(dict(parallel=s,strategy=st) for s in structural)
            feasible=model.search_space_create(precent=protocol['memory_limit_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
            if feasible:model.costmodel_create(feasible)
            for (name,key),value in rt.profile_overlay.items():model.map_manager.data_map.setdefault(name,{})[key]=value
            model.map_manager._save_to_json()
            strategy_status.append(dict(index=index,strategy=st,structural=len(structural),feasible=len(feasible),elapsed_s=time.perf_counter()-begun))
            save(out/'strategy_progress.json',strategy_status)
            print('H20_STRATEGY',index,len(all_strategies),'structural',len(structural),'feasible',len(feasible),flush=True)
        save(out/'native_memory_templates.json',[dict(key=key,model=value[0],analysis=value[1]) for key,value in rt.memory_template_cache.items()])
        records=rt.candidate_records
        assert all(math.isfinite(float(r['predicted_cost'])) for r in records)
        if space_name=='common':require_equal([r['parallel'] for r in records]+[r['parallel'] for r in rt.rejected_records],expected,'evaluated+rejected coverage')
        assert rt.stats['profile_seed_hits']==0
        best=copy.deepcopy(min(records,key=lambda r:(r['predicted_cost'],json.dumps(r['parallel'])))) if records else None
        if best:
            best['candidate_id']=c['id']+'_'+space_name+'_winner'
            canonical=_tpds_canonicalize_measurement_record(best,args.num_layers)
            assert canonical['parallel']==best['parallel'] and canonical['strategy']==best['strategy']
        save(out/'search.json',rt.result_dict(dict(case=c['id'],space=space_name,best=best,screened=screened,
            evaluated=records,rejected=rt.rejected_records,search_seconds=time.perf_counter()-begun,
            common_candidate_sha256=candidate_signature(common_candidates(c)) if space_name=='common' else None,
            effective_args=vars(args))))
    dist.barrier();dist.destroy_process_group()

megatron_semantic_adapter()
if os.environ['H20_PHASE']=='search':
    import dtsir_collect
    dtsir_collect.predict_fixed=search
else:
    from megatron.training import training,get_args
    rows=[];state={};native_setup=training.setup_model_and_optimizer
    def setup(*aa,**kw):
        models,opt,scheduler=native_setup(*aa,**kw)
        model_evidence(out,models)
        params=[p for m in models for p in m.parameters()]
        assert all(g.get('wd_mult',1)==1 for g in opt.param_groups)
        state.update(optimizer=opt,parameters=params,before=sample(params))
        return models,opt,scheduler
    training.setup_model_and_optimizer=setup
    native_step=training.train_step
    def step(*aa,**kw):
        torch.cuda.synchronize();t=time.perf_counter();result=native_step(*aa,**kw)
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        loss_dict=result[0]
        loss=loss_dict.get('lm loss')
        row=dict(iteration=len(rows)+1,seconds=seconds,loss=None if loss is None else float(loss),skipped=int(result[1]),
                 grad_norm=None if result[5] is None else float(result[5]),
                 memory_allocated_bytes=torch.cuda.memory_allocated(),memory_reserved_bytes=torch.cuda.memory_reserved())
        rows.append(row)
        with (out/f'rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return result
    training.train_step=step
runpy.run_path(str(MEG/'pretrain_gpt.py'),run_name='__main__')
if os.environ['H20_PHASE']=='train':
    finish(out,rows,state['before'],state['parameters'],vars(get_args()),state['optimizer'],c['global_batch_size'],c['seq_length'])
    dist.destroy_process_group()
