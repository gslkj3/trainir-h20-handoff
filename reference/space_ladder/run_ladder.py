"""Four-GPU pilot. Run from the user's Megatron root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p,obj):
    p.write_text(json.dumps(obj,indent=2))

def key(r):
    return json.dumps({'parallel':r['parallel'],'strategy':r['strategy']},sort_keys=True)

def check_nested(previous,current):
    old={key(r) for r in previous['screened']}; new={key(r) for r in current['screened']}
    old_eval={key(r):r for r in previous['evaluated']}; new_eval={key(r):r for r in current['evaluated']}
    if not old <= new or not old_eval.keys() <= new_eval.keys():
        raise RuntimeError('Candidate-space or feasibility nesting failed; preserve results for inspection.')
    old_best=previous['best']; new_best=current['best']
    # Independent warm starts may measure missing keys differently: report, do not fabricate monotonicity.
    return {'screened_nested':True,'feasible_nested':True,
            'best_prediction_nonincreasing': None if not old_best else bool(new_best and new_best['predicted_cost']<=old_best['predicted_cost']),
            'common_cost_changes':sum(abs(float(v['predicted_cost'])-float(new_eval[k]['predicted_cost'])) > 1e-6 for k,v in old_eval.items())}

def job(path,env,launcher,timeout,search=False):
    status=path/'status.json'
    if status.exists():
        old=json.loads(status.read_text())
        if old.get('ok') or old.get('outcome')=='cuda_oom':
            print('skip',path,flush=True); return old
        raise RuntimeError(f'Previous failure: {path}; do not delete evidence or blindly retry.')
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'Unfinished job without status: {path}; inspect before retrying.')
    path.mkdir(parents=True,exist_ok=True)
    env=dict(env,DTSIR_IR_OUT=str(path))
    print('run',path,flush=True)
    start=time.perf_counter()
    with (path/'console.log').open('w') as log:
        proc=subprocess.Popen(['bash','-o','pipefail',str(launcher)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            code=-1
    text=(path/'console.log').read_text(errors='replace')
    oom=not search and code not in (0,-1) and 'torch.OutOfMemoryError: CUDA out of memory' in text
    times=[float(x) for x in re.findall(r'elapsed time per iteration\s*\(ms\)\s*:\s*([\d.]+)',text)]
    if search:
        valid=(path/'search.json').exists()
    else:
        counts=re.findall(r'number of (?:skipped|nan) iterations:\s*(\d+)',text)
        valid=len(times)==10 and bool(counts) and all(int(x)==0 for x in counts)
    result={'ok':code==0 and valid,'exit_code':code,'outcome':'cuda_oom' if oom else ('completed' if code==0 and valid else 'failed'),
            'wall_seconds':time.perf_counter()-start,'iteration_ms':times}
    dump(status,result)
    if not result['ok'] and not oom: raise RuntimeError(f'Job failed: {path}')
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launcher',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed-evidence',default='mm_logs')
    p.add_argument('--mbs',type=int,nargs='+',default=[1,2])
    p.add_argument('--kv-heads',type=int,default=32)
    p.add_argument('--vpp-sizes',type=int,nargs='+',default=[1,2])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--through',type=int,choices=range(4),default=3)
    p.add_argument('--timeout',type=int,default=7200)
    ns=p.parse_args()
    if min(ns.mbs+ns.vpp_sizes+[ns.kv_heads,ns.repeats])<1: p.error('All counts must be positive')
    base=Path.cwd(); here=Path(__file__).resolve().parent
    launcher=Path(ns.launcher).resolve(); out=Path(ns.out).resolve(); source=Path(ns.seed_evidence).resolve()
    raw=launcher.read_text()
    if len(re.findall(r'(?<![\w/])pretrain_gpt\.py(?!\w)',raw))!=1:
        raise RuntimeError('Expected exactly one plain pretrain_gpt.py token in working launcher; send launcher for review.')
    for name in ('test_parallel_model.py','dtsir_collect.py','megatron/training/initialize.py'):
        if not (base/name).exists(): raise RuntimeError(f'Missing {name}')
    if 'DTSIR_IR_FIXED' not in (base/'megatron/training/initialize.py').read_text(): raise RuntimeError('Need the already-working IR initialize hook')
    if 'DTSIR_IR_OBSERVE' not in (base/'megatron/training/initialize.py').read_text(): raise RuntimeError('Need observer hook')
    evidence_files=[]
    for directory in ('calc_data','comm_data'):
        evidence_files+=list((source/directory).rglob('*.json'))
    if not any('comm_data' in f.parts for f in evidence_files): raise RuntimeError('Missing calibrated comm_data JSON files')
    contract={'launcher_sha256':sha(launcher),'code':{name:sha(base/name) for name in ('test_parallel_model.py','dtsir_collect.py','megatron/training/initialize.py')},
              'kit':{name:sha(here/name) for name in ('run_ladder.py','space_entry.py')},
              'mbs':sorted(set(ns.mbs)),'vpp_sizes':sorted(set(ns.vpp_sizes)),'kv_heads':ns.kv_heads,'repeats':ns.repeats,
              'evidence':{str(f.relative_to(source)):sha(f) for f in sorted(evidence_files)},'policy':'independent-identical-warm-seed; SP off; selective MLP recompute; four-GPU pilot'}
    out.mkdir(parents=True,exist_ok=True)
    cp=out/'run_contract.json'
    if cp.exists():
        if json.loads(cp.read_text())!=contract: raise RuntimeError('Run contract changed; use a new output directory, do not overwrite old results.')
    else:
        if any(out.iterdir()): raise RuntimeError('Output must be empty for a new run')
        dump(cp,contract)
        for f in evidence_files:
            target=out/'seed'/f.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
    for rel,digest in contract['evidence'].items():
        if sha(out/'seed'/rel)!=digest: raise RuntimeError('Seed snapshot changed')
    generated=out/'launcher.sh'
    # Preserve all user data paths/backend flags, replace only the entry point and log destination.
    transformed=re.sub(r'(?<![\w/])pretrain_gpt\.py(?!\w)', '"'+str(here/'space_entry.py')+'"',raw)
    transformed=transformed.replace('logs/train_llama_7b.log','"${DTSIR_IR_OUT}/launcher_train.log"')
    generated.write_text(transformed)
    base_env=os.environ.copy()
    for k in list(base_env):
        if k.startswith('DTSIR_') or k=='AUTOMM': base_env.pop(k)
    base_env.update({'DTSIR_EXPERIMENT':'measure','GPUS_PER_NODE':'4','NNODES':'1','NODE_RANK':'0','PYTHONUNBUFFERED':'1','DTSIR_PROFILE_WARMUP':'5','DTSIR_PROFILE_ITERS':'5'})
    summary=[]; previous=None
    for level in range(ns.through+1):
        stage=out/f'P{level}'; stage.mkdir(exist_ok=True)
        evidence=stage/'evidence'
        if not evidence.exists(): shutil.copytree(out/'seed',evidence)
        plan={'level':level,'mbs':contract['mbs'],'kv_heads':ns.kv_heads,'vpp_sizes':contract['vpp_sizes']}
        boot={'candidate_id':'search_boot','parallel':[4,1,1,1,1,1,1,2,32],
              'strategy':{'VirtualPipe':None,'DistributedOptimizer':False,'ReCompute':[None,None],'Hybrid_MHA_MQA':[ns.kv_heads!=32,ns.kv_heads]}}
        env=dict(base_env,DTSIR_MML_LOGS=str(evidence),DTSIR_SPACE_PLAN=json.dumps(plan),DTSIR_MEASURE_CANDIDATE_JSON=json.dumps(boot))
        job(stage/'search',dict(env,DTSIR_SPACE_SEARCH='1',DTSIR_IR_FIXED='1'),generated,ns.timeout,search=True)
        data=json.loads((stage/'search/search.json').read_text())
        nesting=check_nested(previous,data) if previous else None; previous=data
        record={'stage':f'P{level}','screened':len({key(r) for r in data['screened']}),'feasible':len({key(r) for r in data['evaluated']}),'search_seconds':data['search_seconds'],
                'stats':data['stats'],'nesting':nesting,'best':data['best']}
        summary.append(record); dump(out/'ladder_summary.json',summary)
        if data['best']:
            env.update(DTSIR_MEASURE_CANDIDATE_JSON=json.dumps(data['best']))
            obs=job(stage/'observe_0',dict(env,DTSIR_IR_OBSERVE='1'),generated,ns.timeout)
            record['observe_outcome']=obs['outcome']
            if obs['ok']:
                starts=list((stage/'observe_0').glob('rank*_started.json'))
                if len(starts)!=4: raise RuntimeError('Missing per-rank effective configuration')
                for f in starts:
                    a=json.loads(f.read_text())['effective_args']; best=data['best']; s=best['parallel']; st=best['strategy']
                    expect={'tensor_model_parallel_size':s[4],'pipeline_model_parallel_size':s[1],'micro_batch_size':s[7],'sequence_parallel':False,'use_distributed_optimizer':st['DistributedOptimizer'],'num_layers_per_virtual_pipeline_stage':st['VirtualPipe'],'num_query_groups':st['Hybrid_MHA_MQA'][1],'recompute_granularity':st['ReCompute'][0],'recompute_modules':st['ReCompute'][1]}
                    if any(a.get(k)!=v for k,v in expect.items()): raise RuntimeError(f'Effective configuration mismatch: {f}')
            # Clean runs are still attempted after observer OOM to distinguish instrumentation effects.
            means=[]; outcomes=[]
            for rep in range(ns.repeats):
                result=job(stage/f'clean_{rep}',env,generated,ns.timeout)
                outcomes.append(result['outcome'])
                if result['ok']: means.append(statistics.mean(result['iteration_ms'][1:])/1000)
                else: break
            record.update(clean_outcomes=outcomes,completed_repeats=len(means),mean_iteration_s=statistics.mean(means) if len(means)==ns.repeats else None,repeat_means_s=means)
        dump(out/'ladder_summary.json',summary)
        lines=['# Space-ladder pilot','','Warm-start search: each tier begins from the same frozen evidence snapshot.','','| Tier | Screened | Feasible | Search s | Clean iteration s |','|---|---:|---:|---:|---:|']
        for r in summary:
            val=r.get('mean_iteration_s'); lines.append(f"|{r['stage']}|{r['screened']}|{r['feasible']}|{r['search_seconds']:.3f}|{val if val is not None else 'N/A'}|")
        (out/'ladder_summary.md').write_text('\n'.join(lines))
    print('DONE:',out/'ladder_summary.md',flush=True)

if __name__=='__main__': main()
