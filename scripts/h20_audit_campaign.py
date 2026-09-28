"""Evidence audit: never infer completion from status labels alone."""
import json
import math
from datetime import datetime
from pathlib import Path
from h20_campaign import *
from common_space import candidates,require_equal,candidate_signature
root=DEFAULT_ROOT;cases=json.loads((root/'cases8.json').read_text())['cases'];rows=[]
for c in cases:
 for system in ('devastator','galvatron'):
  for space in ('common','full'):
   record=dict(case=c['id'],system=system,space=space,verified=False,checks=[],issues=[])
   runs=sorted((root/'runs'/c['id']/system/space).glob('*'))
   if not runs:record['issues'].append('No attempt');rows.append(record);continue
   run=runs[-1];record['run']=str(run)
   try:
    if space=='common':
     if system=='devastator':
      d=json.loads((run/'search.json').read_text());observed=[(r['parallel'][0],r['parallel'][1],r['parallel'][4],r['parallel'][7]) for r in d['evaluated']+d['rejected']]
      assert d['stats'].get('profile_seed_hits',0)==0
     else:observed=[r['candidate'] for r in json.loads((run/'candidate_evaluations.json').read_text())]
     require_equal(observed,candidates(c),'actual common coverage');record['common_candidate_sha256']=candidate_signature(observed)
     record['checks'].append('Actual common candidate set equals declared workload candidates, without duplicates')
    if space=='full':
     definition=json.loads((run/'space_definition.json').read_text());assert definition['space']=='full'
     if system=='devastator':
      d=json.loads((run/'search.json').read_text());native=[r for r in d['evaluated']+d['rejected'] if r['strategy']==dict(ReCompute=[None,None],DistributedOptimizer=False,VirtualPipe=None,Hybrid_MHA_MQA=[c['native_kv_heads']!=c['num_attention_heads'],c['native_kv_heads']]) and r['parallel'][2]==r['parallel'][3]==1]
      require_equal([(r['parallel'][0],r['parallel'][1],r['parallel'][4],r['parallel'][7]) for r in native],candidates(c),'Actual full common subset')
      assert any(r['parallel'][2]>1 for r in d['screened']) and any(r['parallel'][3]>1 for r in d['screened'])
      values={r['strategy']['Hybrid_MHA_MQA'][1] for r in d['screened']};assert min(values)==min(8,c['native_kv_heads']) and max(values)==c['num_attention_heads']
      assert d['stats'].get('profile_seed_hits',0)==0
     else:
      summary=json.loads((run/'search_summary.json').read_text());assert summary['status']=='completed' and summary['tasks']>0
      assert definition['args']['search_space_info']['disable_cp']==1 # Native export limitation is separately recorded.
     record['checks'].append('Full space definition and completed search present; native constraints checked')
    stage=json.loads((run/'stages/search.json').read_text());assert stage['status']=='completed' and stage['returncode']==0
    assert sha(run/'stages/search.log')==stage['log_sha256']
    if system=='devastator':
     communication=root/'hardware/devastator/comm_data/profile_comm.json'
     contract=json.loads((run/'contract.json').read_text())
     assert sha(communication)==contract['communication_sha256']==sha(run/'native_evidence/comm_data/profile_comm.json')
     assert communication.stat().st_mtime<datetime.fromisoformat(stage['start_utc'].replace('Z','+00:00')).timestamp()
     record['communication_sha256']=contract['communication_sha256']
     record['checks'].append('Run-local communication model matches this campaign calibration and calibration file predates search')
    else:
     contract=json.loads((run/'contract.json').read_text())
     comm_dir=root/'hardware/galvatron/hardware_configs'
     comm_files=list(comm_dir.glob('*.json'));assert len(comm_files)==4
     assert all(f.stat().st_mtime<datetime.fromisoformat(stage['start_utc'].replace('Z','+00:00')).timestamp() for f in comm_files)
     if 'communication_files' in contract:
      assert {f.name:sha(f) for f in comm_files}==contract['communication_files']
      record['checks'].append('Galvatron hardware hashes match launch contract; all calibration files predate search')
     else:
      record['checks'].append('Galvatron calibration file timestamps predate search; historical launch contract did not capture calibration hashes')
    timing=json.loads((run/'search_timing.json').read_text())
    elapsed=timing['search_e2e_seconds']
    assert math.isfinite(elapsed) and elapsed>0
    assert timing['outcome']=='selected'
    if 'start_boot_ticks' in timing:
     expected=timing['end_boot_seconds']-timing['start_boot_ticks']/timing['clock_ticks_per_second']
     assert math.isclose(elapsed,expected,abs_tol=1e-6)
    else:
     assert 'Reconstructed' in timing['method'] and timing.get('precision_note')
     assert Path(timing['start_log']).is_file() and Path(timing['end_log'])==run/'stages/selected_training.log'
    profile_stage_seconds=0.0
    if system=='galvatron' and (run/'memory_profiler_args.json').exists():
     memory_args=json.loads((run/'memory_profiler_args.json').read_text())
     compute_args=json.loads((run/'computation_profiler_args.json').read_text())
     search_args=json.loads((run/'search_args.json').read_text())
     assert compute_args['profile_mode']=='static' and compute_args['profile_fixed_seq_length_list']==[c['seq_length']]
     assert search_args['common_train_info']['seq_length']==c['seq_length']
     assert search_args['profiling_info']['memory_profile_mode']==memory_args['profile_mode']
     if memory_args['profile_mode']=='sequence':
      assert 0<memory_args['profile_min_seq_length']<=memory_args['profile_max_seq_length']<=c['seq_length']
     else:assert memory_args['profile_mode']=='static' and memory_args['profile_fixed_seq_length_list']==[c['seq_length']]
     for kind in ('computation','memory'):
      jobs=json.loads((run/(kind+'_jobs.json')).read_text())
      assert len(jobs)==len(list((run/'stages').glob(kind+'_*.json')))
     record['memory_profile_mode']=memory_args['profile_mode']
     record['checks'].append('Native memory profiling mode matches search; computation and search retain actual target sequence; every generated profiling job recorded')
    for f in (run/'stages').glob('*.json'):
     if f.stem.startswith(('memory_','computation_')):
      s=json.loads(f.read_text());assert s['status']=='completed' and sha(f.with_suffix('.log'))==s['log_sha256']
      profile_stage_seconds+=s['seconds']
    assert elapsed+0.05>=stage['seconds']+profile_stage_seconds,'End-to-end search omits profile/search stage time'
    record['search_e2e_seconds']=elapsed;record['search_timing_method']=timing['method']
    record['checks'].append('End-to-end search timing present and covers serial profile/search stages; reconstruction explicitly labelled')
    training=[p for p in [run/'training_summary.json']+sorted(run.glob('training_retries/*/training_summary.json')) if p.exists()]
    if not training:
     raise AssertionError('No successful training evidence; inspect actual failure before classifying terminal result')
    assert len(training)==1,'More than one completed independent training; explicit protocol resolution required'
    t=training[0].parent;record['training']=str(t);summary=json.loads(training[0].read_text());assert summary['iterations']==10 and summary['measured_iterations']==[6,7,8,9,10]
    ranks=[json.loads((t/f'rank{i}.json').read_text()) for i in range(8)]
    if system=='devastator':
     selected=json.loads((run/'selected_candidate.json').read_text())
     dp,pp,cp,up,tp,sp,ep,mbs,chunks=selected['parallel'];strategy=selected['strategy']
     expected_parallel=dict(data_parallel_size=dp,pipeline_model_parallel_size=pp,
      context_parallel_size=cp*up,tensor_model_parallel_size=tp,expert_model_parallel_size=ep,
      micro_batch_size=mbs,sequence_parallel=sp>1,
      num_query_groups=strategy['Hybrid_MHA_MQA'][1],
      use_distributed_optimizer=strategy['DistributedOptimizer'],
      num_layers_per_virtual_pipeline_stage=strategy['VirtualPipe'],
      virtual_pipeline_model_parallel_size=None if strategy['VirtualPipe'] is None else c['num_layers']//(pp*strategy['VirtualPipe']),
      recompute_granularity=strategy['ReCompute'][0],recompute_modules=strategy['ReCompute'][1],
      cp_comm_type=['a2a+p2p'] if cp>1 and up>1 else ['a2a'] if up>1 else ['p2p'],
      hierarchical_context_parallel_sizes=[up,cp] if cp>1 and up>1 else None)
     assert dp*pp*cp*up*tp==8 and dp*mbs*chunks==c['global_batch_size']
     for rank in ranks:
      for key,value in expected_parallel.items():
       assert rank['effective'][key]==value,('Selected/runtime mismatch',rank['rank'],key,rank['effective'][key],value)
     record['checks'].append('Every rank effective DP/PP/TP/CP-UP communication, microbatch, KV, recompute, optimizer sharding and VPP matches selected candidate')
    assert {r['rank'] for r in ranks}==set(range(8)) and {r['device'] for r in ranks}==set(range(8))
    for rank in ranks:
     assert rank['changed_sample_elements']>0 and rank['optimizer_parameter_dtypes']==['torch.float32']
     assert [x['iteration'] for x in rank['iterations']]==list(range(1,11))
     assert all(x['skipped']==0 and math.isfinite(x['seconds']) and x['seconds']>0 and (x['loss'] is None or math.isfinite(x['loss'])) for x in rank['iterations'])
     effective=rank['effective'];model=effective if system=='devastator' else effective['model'];train=effective if system=='devastator' else effective['train'];data=effective if system=='devastator' else effective['data']
     for key in ('num_layers','hidden_size','ffn_hidden_size','num_attention_heads'):
      assert model[key]==c[key],(key,model[key],c[key])
     assert train['seq_length']==c['seq_length'] and train['global_batch_size']==c['global_batch_size'] and train['train_iters']==10
     kv=model['num_query_groups'];assert kv==c['native_kv_heads'] if space=='common' or system=='galvatron' else min(8,c['native_kv_heads'])<=kv<=c['num_attention_heads'] and c['num_attention_heads']%kv==0
     assert model['padded_vocab_size']==c['declared_vocab_size'] and model['rotary_base']==c['rotary_base'] and model['norm_epsilon']==c['norm_epsilon']
     assert model['qk_layernorm']==c['qk_layernorm'] and model['add_qkv_bias']==c['add_qkv_bias'] and model['untie_embeddings_and_output_weights']==c['untie_embeddings_and_output_weights']
     if system=='galvatron':
      assert not data['use_random_dataset']
      assert effective['parallel']['mixed_precision']==c['dtype']
      assert effective['parallel']['reduce_in_fp32'] and effective['parallel']['entropy_in_fp32']
     else:
      assert bool(effective['fp16'])==(c['dtype']=='fp16') and bool(effective['bf16'])==(c['dtype']=='bf16')
      assert effective['accumulate_allreduce_grads_in_fp32']
     assert str(MEG/c['inputs']['data_prefix']) in list(map(str,data['data_path']))
     assert data['tokenizer_model']==str(MEG/c['inputs']['galvatron_tokenizer'])
     for key,value in [('lr',1e-6),('min_lr',1e-7),('weight_decay',.1),('adam_beta1',.9),('adam_beta2',.95),('adam_eps',1e-8),('clip_grad',1.0)]:
      assert math.isclose(train[key],value,rel_tol=1e-10), (key,train[key],value)
    times=[max(r['iterations'][i]['seconds'] for r in ranks) for i in range(10)]
    assert times==summary['iteration_s'] and math.isclose(sum(times[5:])/5,summary['mean_iteration_s'],rel_tol=1e-12)
    assert all(any(r['iterations'][i]['loss'] is not None for r in ranks) for i in range(10))
    s=json.loads((t/'stages/selected_training.json').read_text());assert s['status']=='completed' and sha(t/'stages/selected_training.log')==s['log_sha256']
    record['checks'].append('Eight ranks, ten finite steps, nonzero parameter updates, FP32 optimizer parameters, full workload dimensions, real dataset, recomputed last-five timing, log hashes')
    record['verified']=True
   except Exception as e:record['issues'].append(type(e).__name__+': '+str(e))
   rows.append(record)
save(root/'completion_audit.json',dict(all_32_successful=all(r['verified'] for r in rows),verified_rows=sum(r['verified'] for r in rows),rows=rows,note='Failed/no-feasible/unsupported rows require explicit evidence-based classification before final completion; this audit does not waive failures or mark the goal complete.'))
print('Verified successful rows:',sum(r['verified'] for r in rows),'/32')
for row in rows:
 if row['issues'] and row.get('training'):print(row['case'],row['system'],row['space'],row['issues'])
