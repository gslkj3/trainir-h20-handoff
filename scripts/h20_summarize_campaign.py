"""Plot-ready campaign and per-step tables, retaining every failed attempt."""
import csv
from h20_campaign import *
root=DEFAULT_ROOT
import subprocess,sys
subprocess.run([sys.executable,str(REPO/'scripts/h20_reconstruct_search_timing.py')],check=True,capture_output=True)
rows=[];steps=[];attempts=[];rank_steps=[]
for case in json.loads((root/'cases8.json').read_text())['cases']:
 for system in ('devastator','galvatron'):
  for space in ('common','full'):
   row=dict(case=case['id'],system=system,space=space,status='pending',gbs=case['global_batch_size'],seq=case['seq_length'],dtype=case['dtype'])
   for run in sorted((root/'runs'/case['id']/system/space).glob('*')):
    if not run.is_dir():continue
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'running'}
    stages={p.stem:json.loads(p.read_text()) for p in (run/'stages').glob('*.json')}
    profile_s=sum(v.get('seconds',0) for k,v in stages.items() if k.startswith(('computation_','memory_')))+sum(json.loads(p.read_text())['seconds'] for p in run.glob('*_processing.json'))
    summary=dict(case=case['id'],system=system,space=space,path=str(run),**status,profile_seconds=profile_s,search_stage_seconds=stages.get('search',{}).get('seconds'))
    summary['profile_and_search_s']=profile_s+summary.get('search_stage_seconds',0) if summary.get('search_stage_seconds') is not None else None
    timing=run/'search_timing.json'
    summary['measured_stage_sum_s']=summary.pop('profile_and_search_s')
    if timing.exists():
     e2e=json.loads(timing.read_text());summary['search_e2e_seconds']=e2e['search_e2e_seconds'];summary['search_timing_method']=e2e['method']
     summary['search_preparation_and_other_s']=e2e['search_e2e_seconds']-summary['measured_stage_sum_s'] if summary['measured_stage_sum_s'] is not None else None
    if system=='galvatron':
     args_file=run/'memory_profiler_args.json'
     profile_args=json.loads(args_file.read_text()) if args_file.exists() else {'profile_mode':'static'}
     summary['memory_profile_mode']=profile_args['profile_mode']
     summary['memory_profile_min_sequence']=profile_args.get('profile_min_seq_length') or case['seq_length']
     summary['memory_profile_max_sequence']=profile_args.get('profile_max_seq_length') or case['seq_length']
    attempts.append(summary)
    # A new attempt must not inherit strategy, errors or measurements from its predecessor.
    row=dict(case=case['id'],system=system,space=space,gbs=case['global_batch_size'],seq=case['seq_length'],dtype=case['dtype'])
    row.update(summary)
    classification_file=run/'failure_classification.json'
    if classification_file.exists():
     classification=json.loads(classification_file.read_text())
     row.update(failure_classification=classification['classification'],failure_phase=classification['phase'],failure_evidence=classification['diagnostic_log'])
     for key,value in classification.get('oom_memory_report',{}).items():
      if isinstance(value,(int,float)):
       row['oom_'+key+('' if key=='device' else '_gib')]=value
    for retry in sorted(run.glob('training_retries/*')):
     if retry.is_dir():
      rs=json.loads((retry/'status.json').read_text()) if (retry/'status.json').exists() else {'status':'running'}
      attempts.append(dict(case=case['id'],system=system,space=space,path=str(retry),phase='training_retry',**rs))
      row['status']=rs['status']
    if system=='devastator' and (run/'selected_candidate.json').exists():
     selected=json.loads((run/'selected_candidate.json').read_text());s=selected['parallel']
     row.update(dict(zip(('dp','pp','cp','up','tp','sp','ep','mbs','chunks'),s)))
     row['kv_heads']=selected['strategy']['Hybrid_MHA_MQA'][1];row['selected_strategy']=json.dumps(selected['strategy'],sort_keys=True)
     row['predicted_cost_native_units']=selected['predicted_cost']
    elif system=='galvatron':
     selected_files=list((run/'selected').glob('galvatron_config_*.json'))
     if len(selected_files)==1:
      selected=json.loads(selected_files[0].read_text());row.update(pp=selected['pp_deg'],chunks=selected['chunks'],kv_heads=case['native_kv_heads'],selected_strategy=json.dumps(selected,sort_keys=True))
     candidate_file=run/'selected_candidate.json'
     if space=='common' and candidate_file.exists():
      candidate=json.loads(candidate_file.read_text())['candidate']
      if candidate is not None:
       dp,pp,tp,mbs=candidate
       row.update(dp=dp,pp=pp,tp=tp,mbs=mbs,cp=1,up=1,sp=tp,ep=1)
    trained=[run/'training_summary.json']+sorted(run.glob('training_retries/*/training_summary.json'))
    passed=[p for p in trained if p.exists()]
    if passed:
     p=passed[-1];data=json.loads(p.read_text());row.update({k:v for k,v in data.items() if not isinstance(v,(dict,list))});row['training_evidence']=str(p.parent)
     if row['status']=='completed' and row.get('error'):
      row['prior_attempt_error']=row.pop('error')
     for rankfile in sorted(p.parent.glob('rank[0-7].json')):
      rank=json.loads(rankfile.read_text())
      for step in rank['iterations']:rank_steps.append(dict(case=case['id'],system=system,space=space,run=str(p.parent),rank=rank['rank'],**step))
     gpu_csv=p.parent/'stages/selected_training.gpu.csv'
     if gpu_csv.exists():
      values=[]
      for fields in csv.reader(gpu_csv.open()):
       try:values.append(float(fields[3].strip().split()[0]))
       except (ValueError,IndexError):pass
      if values:row['peak_nvml_mib']=max(values)
     row['allocated_within_85gib']=data['peak_allocated_bytes']<=85*2**30
     for i,seconds in enumerate(data['iteration_s'],1):steps.append(dict(case=case['id'],system=system,space=space,run=str(p.parent),iteration=i,seconds=seconds,measured=i>=6))
   workflow_attempts=[x for x in attempts if x['case']==case['id'] and x['system']==system and x['space']==space and x.get('phase')!='training_retry']
   timed_attempts=[x for x in workflow_attempts if x.get('search_e2e_seconds') is not None]
   row['search_attempt_count']=len(workflow_attempts)
   row['search_attempts_recorded_seconds']=sum(x['search_e2e_seconds'] for x in timed_attempts)
   row['search_attempts_timing_complete']=len(timed_attempts)==len(workflow_attempts) and bool(workflow_attempts)
   row['search_cumulative_e2e_seconds']=row['search_attempts_recorded_seconds'] if row['search_attempts_timing_complete'] else None
   rows.append(row)
allocator_audit=root/'allocator_diagnostic_audit.json'
if allocator_audit.exists():
 for diagnostic in json.loads(allocator_audit.read_text())['records']:
  if not (diagnostic.get('verified') and diagnostic.get('status')=='passed'):continue
  evidence=Path(diagnostic['evidence'])
  provenance=json.loads((evidence/'provenance.json').read_text())
  if diagnostic['diagnostic']!='megatron_selected_training':continue
  assert sha(evidence/'stages/selected_training.log')==diagnostic['log_sha256']
  for row in rows:
   if row.get('path')==provenance['original_run']:
    row.update(oom_attribution='memory_management',allocator_resolution_verified=True,
     allocator_resolution_setting='expandable_segments:True',allocator_diagnostic_evidence=str(evidence),
     oom_attribution_note='Identical selected workload passed ten steps with allocator setting only; original default-allocator OOM and performance fields retained. Not evidence of intrinsic capacity prediction failure.')
save(root/'results.json',rows);save(root/'attempts.json',attempts)
for name,data in [('results',rows),('steps',steps),('rank_steps',rank_steps),('attempts',attempts)]:
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (root/(name+'.csv')).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(data)
from h20_pair_results import export_pairs
export_pairs(root)
for audit_script in ('h20_audit_campaign.py','h20_audit_terminal_results.py','h20_audit_common_phase.py','h20_audit_full_phase.py','h20_export_search_tables.py','h20_export_layer_strategies.py','h20_export_replay_index.py','h20_audit_allocator_diagnostics.py','h20_write_campaign_report.py'):
 subprocess.run([sys.executable,str(REPO/'scripts'/audit_script)],check=True)
print({s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))})
