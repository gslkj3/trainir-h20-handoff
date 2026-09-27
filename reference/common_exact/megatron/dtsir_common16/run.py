import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import re

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2))
def specs():
    data=json.loads((HERE/'cases.json').read_text())
    for spec in data.values():
        spec['repeats']=1
    data.pop('qwen2_1p5b_32k',None)
    return data

def measured_iteration_s(result):
    values=result['iteration_ms']
    if len(values)!=10 or any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Expected exactly ten finite positive iteration times')
    return statistics.mean(values[5:10])/1000

def validate_local(name,spec):
    required=['test_parallel_model.py','dtsir_collect.py','megatron/training/initialize.py','pretrain_gpt.py']
    for f in required:
        if not Path(f).exists():raise RuntimeError('Missing '+f+'; run from Megatron root')
    if 'DTSIR_IR_FIXED' not in Path(required[2]).read_text():raise RuntimeError('Missing existing fixed-prediction hook')
    if spec.get('existing_launcher'):
        raw=Path(spec['existing_launcher']).read_text()
        if raw.count('pretrain_gpt.py')!=1:raise RuntimeError('Expected exactly one pretrain_gpt.py token in the working four-GPU launcher')
    else:
        prefix=Path(spec['data_path'])
        for suffix in ('.bin','.idx'):
            if not Path(str(prefix)+suffix).exists():raise RuntimeError(f'{name}: missing dataset {prefix}{suffix}; edit cases.json paths')
        if not Path(spec['tokenizer_path']).exists():raise RuntimeError(f'{name}: missing tokenizer {spec["tokenizer_path"]}; edit cases.json')
    comm=Path('mm_logs/comm_data/profile_comm.json')
    if not comm.exists() or not json.loads(comm.read_text()):raise RuntimeError('Missing/nonempty calibrated communication model: '+str(comm))

def validate_effective(root,best,spec):
    files=list(root.glob('rank*_effective.json'))
    world=spec.get('nodes',2)*spec.get('gpus_per_node',8)
    if len(files)!=world:raise RuntimeError(f'Expected {world} effective-argument records, found {len(files)}')
    p=best['parallel'];st=best['strategy']
    expected={'tensor_model_parallel_size':p[4],'pipeline_model_parallel_size':p[1],'micro_batch_size':p[7],
        'sequence_parallel':p[4]>1,'context_parallel_size':1,'expert_model_parallel_size':1,
        'use_distributed_optimizer':False,'recompute_granularity':None,'num_layers_per_virtual_pipeline_stage':None,
        'num_query_groups':spec['kv_heads'],'group_query_attention':spec['kv_heads']!=spec['num_attention_heads']}
    expected.update({k:spec[k] for k in ('num_layers','hidden_size','ffn_hidden_size','num_attention_heads','seq_length','global_batch_size')})
    for f in files:
        a=json.loads(f.read_text())
        if any(a.get(k)!=v for k,v in expected.items()):raise RuntimeError('Requested/effective mismatch: '+str(f))

def execute(root,env,timeout,is_search):
    status=root/'status.json'
    if status.exists():
        old=json.loads(status.read_text())
        if old.get('outcome') in ('completed','oom'):
            print('skip',root,flush=True);return old
        raise RuntimeError(f'Previous failure {root}; inspect it, do not overwrite')
    if root.exists() and any(root.iterdir()):raise RuntimeError(f'Unfinished job {root}; preserve and inspect')
    root.mkdir(parents=True,exist_ok=True)
    env=dict(env,DTSIR_IR_OUT=str(root))
    nodes=env.get('DTSIR_COMMON_NNODES','2');gpus=env.get('DTSIR_COMMON_GPUS_PER_NODE','8')
    cmd=['srun',f'--nodes={nodes}',f'--ntasks={nodes}','--ntasks-per-node=1','--cpus-per-task=8',f'--gres=gpu:{gpus}','--kill-on-bad-exit=1','--unbuffered','bash','dtsir_common16/worker.sh']
    t=time.perf_counter()
    with (root/'console.log').open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=-1
    text=(root/'console.log').read_text(errors='replace')
    values=[float(v) for v in re.findall(r'elapsed time per iteration\s*\(ms\)\s*:\s*([\d.]+)',text)]
    counts=re.findall(r'number of (?:skipped|nan) iterations:\s*(\d+)',text)
    loss=re.findall(r'lm loss:\s*([^ |]+)',text)
    invalid_comm=any(x in text for x in ('无效操作类型','Invalid communication operation','Unsupported communication operation'))
    good=((root/'search.json').exists() and not invalid_comm) if is_search else len(values)==10 and bool(counts) and all(int(v)==0 for v in counts) and bool(loss) and all(math.isfinite(float(v)) for v in loss)
    outcome='completed' if code==0 and good else ('oom' if not is_search and code not in (0,-1) and 'torch.OutOfMemoryError: CUDA out of memory' in text else 'failed')
    result={'outcome':outcome,'exit_code':code,'wall_seconds':time.perf_counter()-t,'iteration_ms':values,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    write(status,result)
    if outcome=='failed':raise RuntimeError(f'Failed: {root}/console.log')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=specs(),required=True)
    parser.add_argument('--check-only',action='store_true')
    ns=parser.parse_args();spec=specs()[ns.case]
    validate_local(ns.case,spec)
    if ns.check_only:print('LOCAL INPUTS OK:',ns.case);return
    nodes=spec.get('nodes',2);gpus=spec.get('gpus_per_node',8);world=nodes*gpus;repeats=1
    if int(os.environ.get('SLURM_NNODES','0'))!=nodes:raise RuntimeError('Submit the matching sbatch file')
    base=Path.cwd()
    out=base/'mm_logs'/os.environ.get('PAIR16_MEG_TAG','common5_exact_v1_megatron')/ns.case
    codefiles=[base/p for p in ('test_parallel_model.py','dtsir_collect.py','megatron/training/initialize.py')]+[HERE/p for p in ('run.py','entry.py','common5_space.py','worker.sh','env.sh','cases.json')]
    codefiles.append(base/spec['existing_launcher'] if spec.get('existing_launcher') else HERE/'launchers'/f'{ns.case}.sh')
    codefiles.append(base/'paired16/network.py')
    seeds=sorted(list(Path('mm_logs/calc_data').glob('*.json'))+list(Path('mm_logs/comm_data').glob('*.json')))
    contract={'case':ns.case,'spec':spec,'world_size':world,'gpus_per_node':gpus,'code':{str(p.relative_to(base)):sha(p) for p in codefiles},
        'evidence':{str(p.relative_to('mm_logs')):sha(p) for p in seeds},'policy':f'Shared DP/TP/PP/MBS contract, PP<=8, TP<=8, MBS {spec["mbs"]}, 28-GiB cap, SP follows TP, fixed native GQA; cold operator Profile, calibrated communication model excluded; 1x10 training iterations; measure iterations 6-10'}
    out.mkdir(parents=True,exist_ok=True);cp=out/'run_contract.json'
    if cp.exists():
        if json.loads(cp.read_text())!=contract:raise RuntimeError('Inputs changed since original run; preserve old results and use a separately versioned protocol')
    else:
        if any(out.iterdir()):raise RuntimeError('Unexpected files in new output directory')
        write(cp,contract)
        for p in seeds:
            target=out/'evidence'/p.relative_to('mm_logs');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    # One active driver per case; stale lock requires checking Slurm state, never automatic deletion.
    lock=out/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Active or interrupted driver: {lock}; inspect before removing lock')
    os.write(fd,str(os.environ.get('SLURM_JOB_ID')).encode());os.close(fd)
    try:
        env=os.environ.copy()
        for k in list(env):
            if k.startswith('DTSIR_') or k=='AUTOMM':env.pop(k)
        env.update(DTSIR_COMMON_CASE=ns.case,DTSIR_COMMON_SPEC=json.dumps(spec),DTSIR_EXPERIMENT='measure',
            DTSIR_COMMON_NNODES=str(nodes),DTSIR_COMMON_GPUS_PER_NODE=str(gpus),
            DTSIR_MML_LOGS=str(out/'evidence'),DTSIR_PROFILE_WARMUP='5',DTSIR_PROFILE_ITERS='5',
            PYTHONUNBUFFERED='1')
        if spec.get('existing_launcher'):
            raw=(base/spec['existing_launcher']).read_text()
            generated=out/'launcher_smoke4.sh'
            raw=raw.replace('pretrain_gpt.py','dtsir_common16/entry.py')
            raw=raw.replace('logs/train_llama_7b.log','"${DTSIR_IR_OUT}/node${NODE_RANK}_launcher.log"')
            generated.write_text(raw)
            env['DTSIR_COMMON_LAUNCHER']=str(generated)
        else:
            env.update(DTSIR_DATA_PATH=str(Path(spec['data_path']).resolve()),DTSIR_TOKENIZER_PATH=str(Path(spec['tokenizer_path']).resolve()))
        from entry import strategy
        boot={'candidate_id':'search_boot','parallel':[world,1,1,1,1,1,1,1,spec['global_batch_size']//world],'strategy':strategy(spec)}
        searchenv=dict(env,DTSIR_COMMON_PHASE='search',DTSIR_IR_FIXED='1',DTSIR_MEASURE_CANDIDATE_JSON=json.dumps(boot))
        execute(out/'search',searchenv,10800,True)
        data=json.loads((out/'search/search.json').read_text());best=data['best']
        search_status=json.loads((out/'search/status.json').read_text())
        summary={'case':ns.case,'search_seconds':data['search_seconds'],
            'search_stage_wall_s':search_status['wall_seconds'],
            'timing_note':'search_seconds is evaluator-internal; search_stage_wall_s includes launch/import/init and cold operator Profile, excluding hardware communication calibration and selected training.',
            'screened':len(data['screened']),'feasible':len(data['evaluated']),'best':best,'clean':[],'scope':contract['policy']}
        write(out/'summary.json',summary)
        if best:
            for rep in range(repeats):
                result=execute(out/f'clean_{rep}',dict(env,DTSIR_COMMON_PHASE='train',DTSIR_MEASURE_CANDIDATE_JSON=json.dumps(best)),7200,False)
                if result['outcome']=='completed':
                    validate_effective(out/f'clean_{rep}',best,spec)
                summary['clean'].append(result);write(out/'summary.json',summary)
                if result['outcome']=='oom':break
        good=[measured_iteration_s(r) for r in summary['clean'] if r['outcome']=='completed']
        summary['measurement_protocol']={'independent_runs':1,'total_iterations':10,'discard_first':5,'measured_iterations':[6,7,8,9,10]}
        summary['mean_iteration_s']=statistics.mean(good) if len(good)==repeats else None
        summary['samples_per_second']=spec['global_batch_size']/summary['mean_iteration_s'] if summary['mean_iteration_s'] else None
        summary['repeat_stdev_s']=None
        write(out/'summary.json',summary);print('DONE',out/'summary.json',flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
