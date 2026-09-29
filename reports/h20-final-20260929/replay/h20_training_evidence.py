"""Low-overhead native-loop evidence; parameter samples are taken outside timed steps."""
import json
import math
import statistics
import time
import torch
import torch.distributed as dist
from h20_campaign import save

def sample(parameters):
    values=[]
    for p in parameters:
        if p.numel():
            flat=p.detach().reshape(-1)
            count=min(64,flat.numel())
            idx=torch.arange(count,device=flat.device,dtype=torch.int64)*(flat.numel()-1)//max(1,count-1)
            values.append(flat[idx].float().cpu())
    return torch.cat(values) if values else torch.empty(0)

def model_evidence(out,models):
    if not isinstance(models,(tuple,list)):models=[models]
    modules=[]
    for model in models:
        for name,module in model.named_modules():
            if any(key in name.lower() for key in ('q_layernorm','k_layernorm','word_embeddings','output_layer','embed_tokens','lm_head')):
                modules.append(dict(name=name,type=type(module).__module__+'.'+type(module).__name__,parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype)) for n,p in module.named_parameters(recurse=False)}))
    save(out/f'model_rank{dist.get_rank()}.json',dict(local_parameter_elements=sum(p.numel() for m in models for p in m.parameters()),modules=modules))

def finish(out, rows, before, parameters, effective, optimizer, gbs, seq, optimizer_before=None):
    rank=dist.get_rank();world=dist.get_world_size()
    after=sample(parameters)
    changed=int(torch.count_nonzero(before!=after))
    optimizer_after=sample([p for g in optimizer.param_groups for p in g['params']]) if optimizer_before is not None else None
    changed_optimizer=int(torch.count_nonzero(optimizer_before!=optimizer_after)) if optimizer_before is not None else None
    assert before.shape==after.shape and (changed>0 or (changed_optimizer or 0)>0),(rank,'no sampled model or optimizer parameter update')
    assert [r['iteration'] for r in rows]==list(range(1,11))
    assert all(math.isfinite(r['seconds']) and r['seconds']>0 and r.get('skipped',0)==0 for r in rows)
    assert all(r.get('loss') is None or math.isfinite(r['loss']) for r in rows)
    core_optimizer=getattr(optimizer,'optimizer',optimizer)
    state_dtypes=sorted({str(v.dtype) for state in core_optimizer.state.values() for v in state.values() if torch.is_tensor(v)})
    record=dict(rank=rank,device=torch.cuda.current_device(),iterations=rows,
        parameter_sample_elements=len(before),changed_sample_elements=changed,optimizer_parameter_sample_elements=None if optimizer_before is None else len(optimizer_before),changed_optimizer_sample_elements=changed_optimizer,parameter_update_evidence='model and FP32 optimizer parameter samples' if optimizer_before is not None else 'model parameter samples',effective=effective,
        optimizer_state_tensor_dtypes=state_dtypes,optimizer_type=type(optimizer).__module__+'.'+type(optimizer).__name__,
        optimizer_parameter_dtypes=sorted({str(p.dtype) for g in optimizer.param_groups for p in g['params']}),
        memory_allocated_peak=torch.cuda.max_memory_allocated(),memory_reserved_peak=torch.cuda.max_memory_reserved())
    save(out/f'rank{rank}.json',record)
    allrows=[None]*world;dist.all_gather_object(allrows,record)
    assert world==8 and {r['device'] for r in allrows}==set(range(8))
    assert all(any(r['iterations'][i].get('loss') is not None for r in allrows) for i in range(10))
    times=[max(r['iterations'][i]['seconds'] for r in allrows) for i in range(10)]
    mean=statistics.mean(times[5:])
    if rank==0:
        save(out/'training_summary.json',dict(status='completed',iterations=10,independent_runs=1,
            measured_iterations=[6,7,8,9,10],iteration_s=times,mean_iteration_s=mean,
            samples_per_second=gbs/mean,tokens_per_second=gbs*seq/mean,gbs=gbs,seq=seq,
            peak_allocated_bytes=max(r['memory_allocated_peak'] for r in allrows),
            peak_reserved_bytes=max(r['memory_reserved_peak'] for r in allrows),
            timing_scope='CUDA-synchronized native whole step including batch preparation, forward, backward, optimizer and zero_grad; max over ranks; excludes parameter evidence sampling and post-step logging/barrier.'))
    dist.barrier()

