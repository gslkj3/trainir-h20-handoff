"""Audit allocator replays independently from the formal default-allocator tables."""
import csv
import json
import math
import re
from pathlib import Path
from h20_campaign import DEFAULT_ROOT, save, sha

root=DEFAULT_ROOT
folder=root/'diagnostics/allocator-expandable-01'
records=[]
metrics=[]
for name,stage_name in [('megatron_selected_training','selected_training'),('galvatron_memory_001','memory_001')]:
    out=folder/name
    record=dict(diagnostic=name,verified=False,status='pending',evidence=str(out))
    try:
        if not (out/'resolution.json').exists():
            record['status']='running' if out.exists() else 'pending'
            records.append(record)
            continue
        resolution=json.loads((out/'resolution.json').read_text())
        record['status']=resolution['status']
        provenance=json.loads((out/'provenance.json').read_text())
        original=Path(provenance['original_run'])
        old=provenance['original_stage']
        stage=json.loads((out/'stages'/f'{stage_name}.json').read_text())
        log_path=out/'stages'/f'{stage_name}.log'
        log=log_path.read_text()
        assert sha(log_path)==stage['log_sha256']
        assert sha(original/'stages'/f'{stage_name}.log')==provenance['original_log_sha256']==old['log_sha256']
        assert stage['environment']['PYTORCH_CUDA_ALLOC_CONF']=='expandable_segments:True'
        assert not stage['environment'].get('PYTORCH_ALLOC_CONF')
        copied=out/'original_sources'
        for source in copied.glob('*.py'):
            assert sha(source)==sha(Path(old['code_snapshot'])/source.name)
        normalized=[]
        for arg in stage['command']:
            if arg.startswith(str(copied)+'/'):
                candidates=[x for x in old['command'] if Path(x).name==Path(arg).name]
                assert len(candidates)==1
                arg=candidates[0]
            elif name.startswith('galvatron'):
                arg=arg.replace(str(out),str(original))
            normalized.append(arg)
        assert normalized==old['command'],'Workload command changed'
        if resolution['status']!='passed':
            assert stage['returncode']!=0
            record.update(verified=True,classification='allocator_change_did_not_resolve',log_sha256=sha(log_path))
            records.append(record)
            continue
        assert stage['returncode']==0 and stage['status']=='completed'
        assert 'CUDA out of memory' not in log and 'expandable_segments not supported' not in log
        if name.startswith('megatron'):
            assert stage['environment']['DTSIR_MEASURE_CANDIDATE_JSON']==old['environment']['DTSIR_MEASURE_CANDIDATE_JSON']
            assert sha(out/'selected_candidate.json')==sha(original/'selected_candidate.json')
            original_args=dict(re.findall(r'^\s{2}([a-zA-Z0-9_]+)\s+\.{2,}\s+(.*?)\s*$',(original/'stages/selected_training.log').read_text(),re.M))
            keys=('num_layers','hidden_size','ffn_hidden_size','num_attention_heads','seq_length',
                'global_batch_size','micro_batch_size','train_iters','num_query_groups',
                'data_parallel_size','pipeline_model_parallel_size','tensor_model_parallel_size',
                'context_parallel_size','sequence_parallel','expert_model_parallel_size',
                'num_layers_per_virtual_pipeline_stage','virtual_pipeline_model_parallel_size',
                'use_distributed_optimizer','recompute_granularity','recompute_modules',
                'qk_layernorm','add_qkv_bias','untie_embeddings_and_output_weights',
                'rotary_base','norm_epsilon','padded_vocab_size','bf16','fp16','data_path','tokenizer_model',
                'lr','min_lr','adam_beta1','adam_beta2','adam_eps','weight_decay','clip_grad','seed')
            ranks=[json.loads((out/f'rank{i}.json').read_text()) for i in range(8)]
            assert {x['rank'] for x in ranks}==set(range(8)) and {x['device'] for x in ranks}==set(range(8))
            for rank in ranks:
                i=rank['rank']
                assert json.loads((out/f'model_rank{i}.json').read_text())==json.loads((original/f'model_rank{i}.json').read_text())
                for key in keys:
                    assert str(rank['effective'][key])==original_args[key],(i,key)
                assert rank['changed_sample_elements']>0 and rank['optimizer_parameter_dtypes']==['torch.float32']
                steps=rank['iterations']
                assert [x['iteration'] for x in steps]==list(range(1,11))
                for step in steps:
                    assert step['skipped']==0 and step['seconds']>0
                    assert all(math.isfinite(step[k]) for k in ('seconds','loss','grad_norm') if step.get(k) is not None)
            assert all(any(x['iterations'][i]['loss'] is not None for x in ranks) for i in range(10))
            times=[max(x['iterations'][i]['seconds'] for x in ranks) for i in range(10)]
            summary=json.loads((out/'training_summary.json').read_text())
            assert summary['iteration_s']==times
            assert math.isclose(summary['mean_iteration_s'],sum(times[5:])/5,rel_tol=1e-12)
            metrics.append(dict(case='qwen3_14b_4k',system='devastator',space='full',
                allocator='expandable_segments:True',mean_iteration_s=summary['mean_iteration_s'],
                tokens_per_second=summary['tokens_per_second'],peak_allocated_bytes=summary['peak_allocated_bytes'],
                peak_reserved_bytes=summary['peak_reserved_bytes'],evidence=str(out),formal_default_allocator_result=False))
        else:
            assert 'iter_5_after_backward' in log
            match=re.search(r'Already written profiled memory into config file (.*?)!',log)
            assert match and Path(match.group(1)).is_relative_to(out)
            profile=Path(match.group(1))
            assert profile.is_file() and json.loads(profile.read_text())
        record.update(verified=True,classification='memory_management_sensitive_oom',
            log_sha256=sha(log_path),exact_fragmentation_mechanism_proven=False)
    except Exception as exc:
        record['issue']=repr(exc)
    records.append(record)
save(root/'allocator_diagnostic_audit.json',dict(records=records,
    all_diagnostics_verified=all(x['verified'] for x in records),
    note='A pass demonstrates allocator-sensitive feasibility for the unchanged workload. It does not prove exact fragmentation mechanism or global memory-predictor accuracy. Original OOM remains recorded.'))
save(root/'allocator_diagnostic_metrics.json',metrics)
keys=['case','system','space','allocator','mean_iteration_s','tokens_per_second','peak_allocated_bytes','peak_reserved_bytes','evidence','formal_default_allocator_result']
with (root/'allocator_diagnostic_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(metrics)
print('Allocator diagnostic audit:',[(x['diagnostic'],x['status'],x['verified']) for x in records])
