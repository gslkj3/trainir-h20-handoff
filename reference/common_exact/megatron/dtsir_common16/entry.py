import copy
import json
import os
from pathlib import Path
import runpy
import sys
import time
import math

sys.path.insert(0,str(Path.cwd()))

def strategy(spec):
    return {'ReCompute':[None,None],'VirtualPipe':None,'DistributedOptimizer':False,
            'Hybrid_MHA_MQA':[spec['kv_heads']!=spec['num_attention_heads'],spec['kv_heads']]}

def search(args):
    import torch
    import torch.distributed as dist
    from test_parallel_model import GPT,TPDS_RUNTIME as rt
    from common5_space import candidates, megatron_parallel, require_equal, signature
    spec=json.loads(os.environ['DTSIR_COMMON_SPEC'])
    out=Path(os.environ['DTSIR_IR_OUT'])
    if dist.get_rank()==0:
        nodes=spec.get('nodes',2);gpus=spec.get('gpus_per_node',8)
        if args.world_size!=nodes*gpus or torch.cuda.device_count()!=gpus:
            raise RuntimeError('World size / visible device count disagrees with test specification')
        for k in ('num_layers','hidden_size','ffn_hidden_size','num_attention_heads','seq_length','global_batch_size'):
            if getattr(args,k)!=spec[k]: raise RuntimeError(f'Model mismatch: {k}')
        if args.norm_epsilon != 1e-5:
            raise RuntimeError('RMSNorm epsilon disagrees with shared 1e-5 configuration')
        if not getattr(args,spec['dtype']): raise RuntimeError('Precision mismatch')
        if args.num_experts is not None: raise RuntimeError('Dense cases only')
        evidence=os.environ['DTSIR_MML_LOGS']
        rt.refresh(evidence);rt.config.experiment='common16';rt.config.variant='full'
        rt.config.capture_candidates=True;rt.config.capture_rejected=True
        rt.config.candidate_limit=0;rt.config.strategy_limit=0
        rt.config.max_mbs=max(spec['mbs']);rt.config.profile_seed_from_existing=False
        rt.reset();rt.note_strategy(args)
        original=GPT._tpds_enumerate_structural_candidates
        expected=candidates(layers=spec['num_layers'],hidden=spec['hidden_size'],
            heads=spec['num_attention_heads'],kv_heads=spec['kv_heads'],
            global_batch=spec['global_batch_size'])
        expected_parallel=[megatron_parallel(row,spec['global_batch_size']) for row in expected]
        def restricted(model):
            rows=[];seen=set()
            for s in original(model):
                if s[1]<=8 and s[2]==1 and s[3]==1 and s[6]==1 and s[4] in (1,2,4,8) and s[7] in spec['mbs']:
                    s=list(s);s[5]=s[4]
                    if tuple(s) not in seen:
                        seen.add(tuple(s));rows.append(s)
            require_equal(rows,expected_parallel,'Megatron structural candidates')
            return rows
        GPT._tpds_enumerate_structural_candidates=restricted
        t=time.perf_counter()
        # Planning supports TP-SP, whose numeric degree is carried per candidate.
        # Do not inherit SP=False from the TP1 boot configuration for all candidates.
        analysis_args=copy.deepcopy(args)
        analysis_args.sequence_parallel=True
        model=GPT(analysis_args,mmlogs_path=evidence,search_level=4)
        screened=restricted(model)
        memory_fraction=(28 * 1024**3)/torch.cuda.get_device_properties(0).total_memory
        if not 0 < memory_fraction < 1:
            raise RuntimeError('The common 28-GiB cap is invalid for this GPU')
        feasible=model.search_space_create(precent=memory_fraction)
        if feasible: model.costmodel_create(feasible)
        elapsed=time.perf_counter()-t
        records=rt.candidate_records
        require_equal([r['parallel'] for r in records]+[r['parallel'] for r in rt.rejected_records],
                      expected_parallel,'Megatron evaluated/rejected candidate coverage')
        if rt.stats['profile_seed_hits']:
            raise RuntimeError('Cold operator profiling unexpectedly used old measurements')
        if any(not math.isfinite(float(r['predicted_cost'])) for r in records): raise RuntimeError('Non-finite predicted cost')
        best=copy.deepcopy(min(records,key=lambda r:(r['predicted_cost'],json.dumps(r['parallel'])))) if records else None
        if best: best['candidate_id']=os.environ['DTSIR_COMMON_CASE']+'_best'
        for (name,key),value in rt.profile_overlay.items(): model.map_manager.data_map.setdefault(name,{})[key]=value
        model.map_manager._save_to_json()
        payload=rt.result_dict({'spec':spec,'best':best,'screened':screened,'evaluated':records,'rejected':rt.rejected_records,
            'common_candidate_count':len(expected),'common_candidate_sha256':signature(expected),
            'memory_limit_gib':28,
            'search_seconds':elapsed,'args':vars(args),
            'protocol':'Shared DP/TP/PP/MBS candidate contract; uniform PP; TP-SP coupled; 28-GiB cap; fixed model and GQA.'})
        (out/'search.json').write_text(json.dumps(payload,indent=2,default=str))
        print('SEARCH COMPLETED',len(screened),len(records),best,flush=True)
    dist.barrier()

if __name__=='__main__':
    import test_parallel_model as model_module
    original_apply=model_module.apply_candidate_from_env
    def apply(args):
        args=original_apply(args)
        expected_sp=args.tensor_model_parallel_size>1
        if args.sequence_parallel != expected_sp:
            raise RuntimeError('Candidate SP must follow TP: expected '+str(expected_sp))
        # Fixed execution policy in all five cases. Preserve model-specific math.
        args.overlap_grad_reduce=False
        args.overlap_param_gather=False
        args.recompute_granularity=None;args.recompute_modules=None
        args.use_distributed_optimizer=False
        args.num_layers_per_virtual_pipeline_stage=None
        args.virtual_pipeline_model_parallel_size=None
        root=Path(os.environ['DTSIR_IR_OUT'])
        (root/f"rank{os.environ.get('RANK','0')}_effective.json").write_text(json.dumps(vars(args),indent=2,default=str))
        return args
    model_module.apply_candidate_from_env=apply
    if os.environ.get('DTSIR_COMMON_PHASE')=='search':
        import dtsir_collect
        dtsir_collect.predict_fixed=search
    if '--distributed-timeout-minutes' not in sys.argv: sys.argv+=['--distributed-timeout-minutes','180']
    runpy.run_path(str(Path.cwd()/'pretrain_gpt.py'),run_name='__main__')
