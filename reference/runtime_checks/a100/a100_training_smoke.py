"""Tiny one-GPU training integration check; not a comparison or paper measurement."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

def write(path, value):
    path.write_text(json.dumps(value,indent=2,default=str))

def inside(path, root):
    return Path(path).resolve().is_relative_to(root.resolve())

def gal_worker(config, out):
    import torch
    import torch.distributed as dist
    from galvatron.core.arguments import load_with_hydra
    source=Path(os.environ['GALV_REPO'])/'galvatron/models/gpt/train_dist.py'
    module=runpy.run_path(str(source),run_name='a100_smoke_native_train')
    args=load_with_hydra(str(config),overrides=[],mode='train_dist')
    from galvatron.utils.hf_config_adapter import resolve_model_config
    resolve_model_config(args)
    rows=[]
    class Timer:
        def profile_memory(self,*a,**kw): pass
        def post_profile_memory(self,*a,**kw): pass
        def profile_time_start(self,iteration):
            torch.cuda.synchronize()
            self.start=time.perf_counter()
        def profile_time_end(self,iteration,loss=None,learning_rate=None,grad_norm=None):
            torch.cuda.synchronize()
            def scalar(value):
                return None if value is None else float(value.item() if hasattr(value,'item') else value)
            row=dict(iteration=iteration+1,seconds=time.perf_counter()-self.start,
                     loss=scalar(loss),grad_norm=scalar(grad_norm))
            rows.append(row)
            print('A100_SMOKE_STEP',json.dumps(row),flush=True)
    # Observe the native loop, including its optimizer step; do not replace training.
    module['train'].__globals__['get_runtime_profiler']=lambda *a,**kw:Timer()
    module['initialize_galvatron'](args)
    try:
        if dist.get_world_size()!=1: raise RuntimeError('This smoke requires world size 1')
        module['train'](args)
        write(out/'effective.json',args.model_dump())
        write(out/'iterations.json',rows)
        if [r['iteration'] for r in rows]!=list(range(1,11)):
            raise RuntimeError('Expected ten completed training iterations')
        for row in rows:
            if row['loss'] is None or not math.isfinite(row['loss']):
                raise RuntimeError('Missing/non-finite training loss')
            if not math.isfinite(row['seconds']) or row['seconds']<=0:
                raise RuntimeError('Invalid iteration duration')
            if row['grad_norm'] is not None and not math.isfinite(row['grad_norm']):
                raise RuntimeError('Non-finite gradient norm')
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('system',choices=('megatron','galvatron'))
    p.add_argument('--out',required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--config')
    a=p.parse_args()
    out=Path(a.out).resolve()
    if a.worker:
        gal_worker(Path(a.config),out)
        return
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Run on an allocated A100 compute node, not the login node')
    expected=Path.home()/'.conda/envs'/('dtsir-a100' if a.system=='megatron' else 'galvatron-a100')
    if Path(sys.prefix).resolve()!=expected.resolve(): raise RuntimeError('Wrong Python environment')
    meg=Path(os.environ['MEGATRON_ROOT']).resolve()
    gal=Path(os.environ['GALV_REPO']).resolve()
    repo=meg if a.system=='megatron' else gal
    package=__import__('megatron' if a.system=='megatron' else 'galvatron')
    if not inside(package.__file__,repo): raise RuntimeError('Wrong source import: '+str(package.__file__))
    if a.system=='galvatron':
        import galvatron_dp_core
        if not inside(galvatron_dp_core.__file__,gal):
            raise RuntimeError('Rebuild galvatron_dp_core in the migrated repository first; loaded '+galvatron_dp_core.__file__)
        print('NEW SEARCH EXTENSION:',galvatron_dp_core.__file__,flush=True)
    if out.exists(): raise RuntimeError('Refusing to overwrite existing smoke output: '+str(out))
    out.mkdir(parents=True)
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('DTSIR_') or key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','MASTER_ADDR','MASTER_PORT'):
            env.pop(key)
    env.update(AUTOMM='0',DTSIR_EXPERIMENT='off',CUDA_DEVICE_MAX_CONNECTIONS='1',
               NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',
               TORCHINDUCTOR_COMPILE_THREADS='1',OMP_NUM_THREADS='1')
    # Single process launches one GPU worker; retain the scheduler's visibility mask.
    for key,subdir in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
        (out/subdir).mkdir()
        env[key]=str(out/subdir)
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1']
    if a.system=='megatron':
        cmd += [str(meg/'pretrain_gpt.py'),
            '--use-mcore-models','--transformer-impl','local',
            '--tensor-model-parallel-size','1','--pipeline-model-parallel-size','1',
            '--context-parallel-size','1','--num-layers','2','--hidden-size','512',
            '--ffn-hidden-size','1376','--num-attention-heads','8','--kv-channels','64',
            '--tokenizer-type','Llama2Tokenizer','--tokenizer-model',str(meg/'model_from_hf/llama2-hf/tokenizer.model'),
            '--make-vocab-size-divisible-by','1','--seq-length','512','--max-position-embeddings','512',
            '--micro-batch-size','1','--global-batch-size','4','--train-iters','10',
            '--lr','1e-6','--min-lr','1e-7','--lr-decay-style','cosine','--lr-warmup-fraction','0.01',
            '--untie-embeddings-and-output-weights','--disable-bias-linear',
            '--attention-dropout','0.0','--hidden-dropout','0.0','--init-method-std','0.01',
            '--position-embedding-type','rope','--rotary-base','10000',
            '--normalization','RMSNorm','--norm-epsilon','1e-5','--swiglu',
            '--no-persist-layer-norm',
            '--use-flash-attn','--no-masked-softmax-fusion','--attention-softmax-in-fp32',
            '--weight-decay','0.1','--clip-grad','1.0','--adam-beta1','0.9','--adam-beta2','0.95',
            '--no-gradient-accumulation-fusion','--bf16','--seed','42',
            '--data-path',str(meg/'dataset/llama/enwiki_text_document'),'--split','10,0,0',
            '--log-interval','1','--eval-iters','0','--eval-interval','1000',
            '--no-load-optim','--no-load-rng','--distributed-backend','nccl','--distributed-timeout-minutes','15']
    else:
        harness=gal/'dtsir_galvatron6/run_six.py'
        spec=importlib.util.spec_from_file_location('native_harness',harness)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        c=dict(id='llama_a100_smoke',hidden=512,ffn=1376,layers=2,heads=8,kv=8,
            seq=512,gbs=4,dtype='bf16',vocab=32000,eps=1e-5,rope=10000,
            qkv_bias=False,qk_norm=False,untied=True,
            tokenizer='model_from_hf/llama2-hf',data='dataset/llama/enwiki_text_document')
        r=module.runtime_config(c,meg)
        write(out/'model_template.yaml',{})
        r['model'].update(model_config_path=str(out/'model_template.yaml'),
                          set_layernum_manually=1,set_seqlen_manually=1)
        # Native Galvatron Attention requires this API flag even for TP=1.
        # With world size 1, its SP group is [0]; there is no inter-rank sharding.
        r['train'].update(sequence_parallel=True,chunks=4)
        r['parallel'].update(async_grad_reduce=False)
        write(out/'runtime.yaml',{'runtime':r})
        # Check schema on CPU before launching the worker.
        from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
        GalvatronRuntimeArgs.model_validate(r)
        cmd += [str(Path(__file__).resolve()),'galvatron','--worker','--out',str(out),
                '--config',str(out/'runtime.yaml')]
    write(out/'launch.json',dict(command=cmd,python=sys.executable,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        shape=dict(layers=2,hidden=512,ffn=1376,heads=8,sequence=512,global_batch=4),
        scope='One-GPU environment/training integration smoke; not a performance comparison or search test.'))
    print('START',a.system,'tiny training: TP1 PP1 DP1, 10 iterations',flush=True)
    # Full log goes to disk; phase and final status remain visible in driver output.
    with (out/'train.log').open('w') as stream:
        process=subprocess.run(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    log=(out/'train.log').read_text(errors='replace')
    errors=[]
    if process.returncode: errors.append(f'Training exit code {process.returncode}')
    if a.system=='megatron':
        losses=[float(v) for v in re.findall(r'lm loss:\s*([^ |]+)',log)]
        times=[float(v) for v in re.findall(r'elapsed time per iteration\s*\(ms\)\s*:\s*([\d.]+)',log)]
        if len(losses)!=10 or not all(math.isfinite(x) for x in losses): errors.append('Expected 10 finite losses')
        if len(times)!=10 or not all(math.isfinite(x) and x>0 for x in times): errors.append('Expected 10 valid iteration times')
        for kind in ('skipped','nan'):
            counts=re.findall(r'number of '+kind+r' iterations:\s*(\d+)',log)
            if len(counts)!=10 or any(int(x) for x in counts): errors.append('Invalid '+kind+' iteration counts')
    elif not (out/'iterations.json').is_file():
        errors.append('Missing native-loop iteration records')
    write(out/'status.json',dict(passed=not errors,exit_code=process.returncode,errors=errors,
        scope='Training environment smoke only; no cross-system performance conclusion.'))
    if errors:
        print('\n'.join(log.splitlines()[-90:]),flush=True)
        raise RuntimeError(f'{errors}; full log: {out}/train.log')
    print('TRAINING SMOKE PASS:',a.system,out/'status.json',flush=True)

if __name__=='__main__':
    main()
