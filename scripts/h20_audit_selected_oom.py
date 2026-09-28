"""Audit selected-training CUDA OOM as an observed failure, never as a speed result."""
import ast
import json
import math
import re
from datetime import datetime
from h20_campaign import MEG, sha
from common_space import candidates, require_equal


def audit_selected_oom(root, run, case, system, space):
    assert system=='devastator' and space=='full'
    search=json.loads((run/'search.json').read_text())
    selected=json.loads((run/'selected_candidate.json').read_text())
    assert selected==search['best'] and selected['predicted_feasible']
    assert selected['predicted_cost']==min(x['predicted_cost'] for x in search['evaluated'])
    native=[x for x in search['evaluated']+search['rejected']
        if x['strategy']==dict(ReCompute=[None,None],DistributedOptimizer=False,VirtualPipe=None,
            Hybrid_MHA_MQA=[case['native_kv_heads']!=case['num_attention_heads'],case['native_kv_heads']])
        and x['parallel'][2]==x['parallel'][3]==1]
    require_equal([(x['parallel'][0],x['parallel'][1],x['parallel'][4],x['parallel'][7])
        for x in native],candidates(case),'OOM full common subset')
    assert any(x['parallel'][2]>1 for x in search['screened'])
    assert any(x['parallel'][3]>1 for x in search['screened'])
    kvs={x['strategy']['Hybrid_MHA_MQA'][1] for x in search['screened']}
    assert min(kvs)==min(8,case['native_kv_heads']) and max(kvs)==case['num_attention_heads']
    assert search['stats'].get('profile_seed_hits',0)==0
    stages={name:json.loads((run/'stages'/f'{name}.json').read_text())
        for name in ('search','selected_training')}
    for name,stage in stages.items():
        assert sha(run/'stages'/f'{name}.log')==stage['log_sha256']
    assert stages['search']['status']=='completed' and stages['search']['returncode']==0
    train=stages['selected_training']
    assert train['status']=='failed' and train['returncode'] not in (0,124)
    assert json.loads(train['environment']['DTSIR_MEASURE_CANDIDATE_JSON'])==selected
    contract=json.loads((run/'contract.json').read_text())
    comm=root/'hardware/devastator/comm_data/profile_comm.json'
    assert sha(comm)==contract['communication_sha256']==sha(run/'native_evidence/comm_data/profile_comm.json')
    assert comm.stat().st_mtime<datetime.fromisoformat(stages['search']['start_utc'].replace('Z','+00:00')).timestamp()
    timing=json.loads((run/'search_timing.json').read_text())
    assert timing['outcome']=='selected' and math.isfinite(timing['search_e2e_seconds'])
    assert timing['search_e2e_seconds']>=stages['search']['seconds']
    log=(run/'stages/selected_training.log').read_text()
    assert 'torch.OutOfMemoryError: CUDA out of memory.' in log
    assert 'Variable._execution_engine.run_backward' in log
    classification=json.loads((run/'failure_classification.json').read_text())
    memory=classification['oom_memory_report']
    assert memory['raw_exception'] in log and memory['unit']=='GiB'
    patterns=dict(requested_allocation=r'Tried to allocate ([\d.]+) GiB',
        device_total_capacity=r'total capacity of ([\d.]+) GiB',
        device_free=r'of which ([\d.]+) GiB is free',
        process_memory_including_non_pytorch=r'process has ([\d.]+) GiB memory in use',
        pytorch_allocated=r'([\d.]+) GiB is allocated by PyTorch',
        pytorch_reserved_but_unallocated=r'([\d.]+) GiB is reserved by PyTorch but unallocated')
    for key,pattern in patterns.items():
        assert float(re.search(pattern,memory['raw_exception']).group(1))==memory[key]
    args=dict(re.findall(r'^\s{2}([a-zA-Z0-9_]+)\s+\.{2,}\s+(.*?)\s*$',log,re.M))
    dp,pp,cp,up,tp,sp,ep,mbs,chunks=selected['parallel'];strategy=selected['strategy']
    expected={k:case[k] for k in ('num_layers','hidden_size','ffn_hidden_size','num_attention_heads',
        'seq_length','global_batch_size','rotary_base','norm_epsilon','qk_layernorm',
        'add_qkv_bias','untie_embeddings_and_output_weights')}
    expected.update(padded_vocab_size=case['declared_vocab_size'],train_iters=10,
        data_parallel_size=dp,pipeline_model_parallel_size=pp,tensor_model_parallel_size=tp,
        context_parallel_size=cp*up,expert_model_parallel_size=ep,micro_batch_size=mbs,
        sequence_parallel=sp>1,num_query_groups=strategy['Hybrid_MHA_MQA'][1],
        use_distributed_optimizer=strategy['DistributedOptimizer'],
        num_layers_per_virtual_pipeline_stage=strategy['VirtualPipe'],
        virtual_pipeline_model_parallel_size=None if strategy['VirtualPipe'] is None else case['num_layers']//(pp*strategy['VirtualPipe']),
        recompute_granularity=strategy['ReCompute'][0],recompute_modules=strategy['ReCompute'][1],
        bf16=case['dtype']=='bf16',fp16=case['dtype']=='fp16')
    for key,value in expected.items():
        assert args[key]==str(value),(key,args[key],value)
    assert dp*pp*cp*up*tp==8 and dp*mbs*chunks==case['global_batch_size']
    assert str(MEG/case['inputs']['data_prefix']) in ast.literal_eval(args['data_path'])
    assert args['tokenizer_model']==str(MEG/case['inputs']['galvatron_tokenizer'])
    completed=[]
    for rank in range(8):
        rows=[json.loads(line) for line in (run/f'rank{rank}.jsonl').read_text().splitlines()]
        assert 0<len(rows)<10 and [x['iteration'] for x in rows]==list(range(1,len(rows)+1))
        assert all(x['skipped']==0 and x['seconds']>0 and
            all(math.isfinite(x[k]) for k in ('seconds','loss','grad_norm') if x.get(k) is not None) for x in rows)
        assert json.loads((run/f'model_rank{rank}.json').read_text())['local_parameter_elements']>0
        completed.append(len(rows))
    assert len(set(completed))==1
    assert not (run/'training_summary.json').exists()
    assert not list(run.glob('training_retries/*/training_summary.json'))
    return dict(verified=True,classification='selected_training_cuda_oom',
        completed_steps_per_rank=completed,failed_iteration=completed[0]+1,
        search_e2e_seconds=timing['search_e2e_seconds'],selected_config_sha256=sha(run/'selected_candidate.json'),
        failure_log_sha256=train['log_sha256'],
        oom_memory_report=memory,
        interpretation='Native selected winner OOMed during real training under the fixed campaign runtime. No ten-step performance result. This does not prove all full-space configurations are infeasible; fragmentation is possible but not established by this log.')
