"""Replay two historical OOMs with expandable segments, outside formal result rows."""
import argparse
import math
import shutil
import subprocess
import time
from h20_campaign import DEFAULT_ROOT, REPO, environment, save, sha, stage
from pathlib import Path
import json

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
p.add_argument('--wait-pid',type=int)
a=p.parse_args()
root=a.root.resolve()
out=root/'diagnostics/allocator-expandable-01'
out.mkdir(parents=True,exist_ok=False)
setting='expandable_segments:True'
save(out/'policy.json',dict(allocator_setting=setting,
    requested_by_user=True,changed_execution_setting_only=True,
    rule='If identical workload completes with allocator setting only, classify this OOM as memory-management sensitive, not demonstrated intrinsic capacity/prediction failure. Do not infer general predictor accuracy.',
    formal_results_unchanged=True))
if a.wait_pid:
    proc=Path(f'/proc/{a.wait_pid}/cmdline')
    identity=proc.read_bytes() if proc.exists() else None
    save(out/'status.json',dict(status='waiting_for_formal_queue',pid=a.wait_pid))
    while identity is not None:
        try:
            if proc.read_bytes()!=identity:break
        except FileNotFoundError:break
        time.sleep(10)
save(out/'status.json',dict(status='waiting_for_idle_gpus'))
while True:
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if not r.stdout.strip():break
    time.sleep(10)
jobs=[('qwen3_14b_4k/devastator/full/01','selected_training','megatron'),
      ('qwen2_7b_32k/galvatron/common/01','memory_001','galvatron')]
results=[]
for relative,name,system in jobs:
    original=root/'runs'/relative
    target=out/(system+'_'+name)
    target.mkdir()
    old=json.loads((original/'stages'/f'{name}.json').read_text())
    cmd=list(old['command'])
    env=environment(system,target)
    env.update(old['environment'])
    env.pop('PYTORCH_ALLOC_CONF',None)
    env['PYTORCH_CUDA_ALLOC_CONF']=setting
    # Execute the exact Python sources snapshotted with the failed attempt.
    sources=target/'original_sources'
    shutil.copytree(old['code_snapshot'],sources)
    for i,arg in enumerate(cmd):
        if arg.startswith(str(REPO/'scripts')+'/'):
            replacement=sources/Path(arg).name
            assert replacement.exists()
            cmd[i]=str(replacement)
    cwd=old['cwd']
    if system=='megatron':
        for key in ('H20_RUN_OUT','DTSIR_IR_OUT'):
            env[key]=str(target)
        env['DTSIR_LOG_DIR']=str(target/'native_logs')
        for f in ('selected_candidate.json',):
            shutil.copy2(original/f,target/f)
        assert env['DTSIR_MEASURE_CANDIDATE_JSON']==old['environment']['DTSIR_MEASURE_CANDIDATE_JSON']
    else:
        # Native profiler writes alongside train_dist.py: isolate that tree.
        shutil.copytree(original/'gpt',target/'gpt')
        for f in ('memory_runtime.yaml','model_template.yaml'):
            (target/f).write_text((original/f).read_text().replace(str(original),str(target)))
        cmd=[arg.replace(str(original),str(target)) for arg in cmd]
        cwd=str(target/'gpt')
    save(target/'provenance.json',dict(original_run=str(original),original_stage=old,
        original_log_sha256=sha(original/'stages'/f'{name}.log'),
        allocator=setting,command=cmd,cwd=cwd,
        environment_changes={k:v for k,v in env.items() if k in old['environment'] and old['environment'][k]!=v},
        original_source_hashes={f.name:sha(f) for f in sources.glob('*.py')},
        experimental_change='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; workload unchanged; output paths isolated'))
    save(out/'status.json',dict(status='running',job=relative,stage=name))
    try:
        stage(target,name,cmd,cwd,env,7200 if system=='megatron' else 1800)
        if system=='megatron':
            summary=json.loads((target/'training_summary.json').read_text())
            assert summary['iterations']==10
            assert all(math.isfinite(x) and x>0 for x in summary['iteration_s'])
            for rank in range(8):
                rows=json.loads((target/f'rank{rank}.json').read_text())['iterations']
                assert [x['iteration'] for x in rows]==list(range(1,11))
                assert all(x['skipped']==0 for x in rows)
        result=dict(job=relative,status='passed',classification='memory_management_sensitive_oom',
            interpretation='Original workload completes with expandable segments; original OOM retained. Exact fragmentation mechanism not independently proven.',
            evidence=str(target))
    except Exception as e:
        result=dict(job=relative,status='failed',classification='allocator_change_did_not_resolve',error=repr(e),evidence=str(target))
    save(target/'resolution.json',result)
    results.append(result)
    save(out/'results.json',results)
save(out/'status.json',dict(status='completed',results=results))
