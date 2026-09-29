import argparse
import json
import runpy
import time
import faulthandler
import signal
import os
from pathlib import Path

# Opt-in snapshots on demand, without ptrace privileges or changes to native kernels.
# Send SIGUSR1 only to workers launched with this entry revision.
faulthandler.register(signal.SIGUSR1, all_threads=True)
if os.environ.get('H20_DIAGNOSTIC_STACK_DIR'):
    stack_dir=Path(os.environ['H20_DIAGNOSTIC_STACK_DIR']);stack_dir.mkdir(parents=True,exist_ok=True)
    stack_file=(stack_dir/('rank'+os.environ.get('LOCAL_RANK','unknown')+'.txt')).open('a')
    faulthandler.dump_traceback_later(60,repeat=True,file=stack_file)
import torch
import torch.distributed as dist
from h20_campaign import *
from h20_workloads import galvatron_qk_adapter
from h20_training_evidence import sample,finish,model_evidence
from h20_galvatron_rope import needs_native_layer_rope,use_native_layer_rope

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--config',required=True);p.add_argument('--out',type=Path)
p.add_argument('--mode',choices=['profile','train'],required=True);p.add_argument('overrides',nargs=argparse.REMAINDER)
a=p.parse_args()
galvatron_qk_adapter()
from galvatron.core.arguments import load_with_hydra
from galvatron.utils.hf_config_adapter import resolve_model_config
m=runpy.run_path(str(a.source),run_name='h20_galvatron_native')
overrides=a.overrides[1:] if a.overrides[:1]==['--'] else a.overrides
args=load_with_hydra(a.config,overrides=overrides,mode='train_dist');resolve_model_config(args)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
m['initialize_galvatron'](args)
rows=[];state={}
if a.mode=='train':
    native_optimizer=m['get_optimizer_and_param_scheduler']
    def optimizer(model,arguments):
        opt,scheduler=native_optimizer(model,arguments)
        model_evidence(a.out,model)
        state.update(optimizer=opt,parameters=list(model.parameters()))
        state['before']=sample(state['parameters'])
        return opt,scheduler
    native_batch=m['get_batch']
    layer_rope=needs_native_layer_rope(args)
    if dist.get_rank()==0:save(a.out/'rope_dispatch.json',dict(native_per_layer_rope=layer_rope))
    def batch(*aa,**kw):
        torch.cuda.synchronize();state['start']=time.perf_counter()
        result=native_batch(*aa,**kw)
        return use_native_layer_rope(result) if layer_rope else result
    native_profiler=m['get_runtime_profiler']
    def profiler(*aa,**kw):
        prof=native_profiler(*aa,**kw);end=prof.profile_time_end
        def observed_end(iteration,loss=None,learning_rate=None,grad_norm=None):
            torch.cuda.synchronize();seconds=time.perf_counter()-state['start']
            end(iteration,loss,learning_rate,grad_norm)
            row=dict(iteration=iteration+1,seconds=seconds,loss=None if loss is None else float(loss),
                     grad_norm=None if grad_norm is None else float(grad_norm),lr=learning_rate,skipped=0,
                     memory_allocated_bytes=torch.cuda.memory_allocated(),memory_reserved_bytes=torch.cuda.memory_reserved())
            rows.append(row)
            with (a.out/f'rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        prof.profile_time_end=observed_end
        return prof
    m['train'].__globals__.update(get_optimizer_and_param_scheduler=optimizer,get_batch=batch,get_runtime_profiler=profiler)
try:
    m['train'](args)
    if a.mode=='train':
        finish(a.out,rows,state['before'],state['parameters'],args.model_dump(),state['optimizer'],args.train.global_batch_size,args.train.seq_length)
except Exception:
    # A peer may be blocked in pipeline communication. Collective teardown here
    # can hide the original error forever; let torchrun terminate the other ranks.
    import traceback,sys
    traceback.print_exc();sys.stderr.flush()
    os._exit(1)
finally:
    if dist.is_initialized():dist.destroy_process_group()
