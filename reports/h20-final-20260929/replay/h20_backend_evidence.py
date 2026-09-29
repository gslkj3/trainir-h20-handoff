"""Observe real first-step dispatch; remove instrumentation before measured steps."""
import collections
import atexit
import functools
import importlib
import importlib.metadata
import inspect
import sys
import time
import torch
import torch.distributed as dist
from h20_campaign import save


class BackendEvidence:
    def __init__(self, system, out, models, optimizer):
        self.system, self.out = system, out
        self.rank=dist.get_rank()
        self.finished=False
        atexit.register(self.flush_on_exit)
        self.restore = []
        self.handles = []
        self.calls = {}
        self.symbols = {}
        self.collectives = []
        self.started = time.perf_counter()
        if not isinstance(models, (tuple,list)):
            models = [models]
        self.modules = []
        for model in models:
            for name,module in model.named_modules():
                kind = type(module).__module__+'.'+type(module).__name__
                row = dict(name=name,type=kind)
                for attr in ('normalization','sequence_parallel','gradient_accumulation_fusion',
                        'gather_output','sharding_strategy','mixed_precision'):
                    if hasattr(module,attr):
                        row[attr]=str(getattr(module,attr))
                groups={}
                for attr in ('dp_group','tp_group','pp_group','process_group'):
                    group=getattr(module,attr,None)
                    group=getattr(group,'group',group)
                    if isinstance(group,dist.ProcessGroup):
                        groups[attr]=dist.get_process_group_ranks(group)
                if groups:row['groups']=groups
                self.modules.append(row)
                if any(s in type(module).__name__ for s in ('RMSNorm','LayerNormLinear','LayerNormColumnParallel','FlashAttention','FusedAttention','UnfusedDotProductAttention')) or getattr(module,'normalization',None)=='RMSNorm':
                    def hook(mod,args,kwargs,kind=kind):
                        self.observe('module:'+kind,args,kwargs)
                    self.handles.append(module.register_forward_pre_hook(hook,with_kwargs=True))
        self.optimizers=[]
        current=optimizer
        while current is not None:
            self.optimizers.append(dict(type=type(current).__module__+'.'+type(current).__name__,
                defaults={k:str(v) for k,v in getattr(current,'defaults',{}).items()}))
            following=getattr(current,'optimizer',None)
            if following is current:break
            current=following
        self.packages={}
        for name in ('torch','flash-attn','flash-attn-3','transformer-engine','apex','triton'):
            try:self.packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:self.packages[name]=None
        self.import_locations={}
        for name in ('torch','flash_attn','flash_attn_2_cuda','apex','amp_C','transformer_engine'):
            module=sys.modules.get(name)
            if module is not None:self.import_locations[name]=getattr(module,'__file__',None)
        self.missing=[]
        if system=='megatron':
            targets={
                'transformer_engine.pytorch.attention.dot_product_attention.backends':
                    ['flash_attn_func','flash_attn_varlen_func','flash_attn_func_v3','flash_attn_varlen_func_v3',
                     '_flash_attn_fwd','_flash_attn_bwd','_flash_attn_varlen_fwd','_flash_attn_varlen_bwd'],
                'megatron.core.models.common.language_module.language_module':['fused_vocab_parallel_cross_entropy'],
                'megatron.core.tensor_parallel':['vocab_parallel_cross_entropy'],
                'flash_attn.flash_attn_interface':['flash_attn_varlen_func'],
                'megatron.core.transformer.mlp':['bias_swiglu_impl'],
                'megatron.core.optimizer.clip_grads':['l2_norm_impl','multi_tensor_scale_impl'],
                'megatron.core.models.common.embeddings.rope_utils':['fused_apply_rotary_pos_emb','fused_apply_rotary_pos_emb_thd']}
        else:
            targets={
                'galvatron.core.runtime.transformer.attention_impl':['flash_attn_unpadded_func'],
                'galvatron.core.runtime.models.modules':['fused_vocab_parallel_cross_entropy'],
                'galvatron.core.runtime.transformer.mlp':['bias_swiglu_impl'],
                'galvatron.core.runtime.optimizer.clip_grads':['l2_norm_impl','multi_tensor_scale_impl'],
                'galvatron.core.runtime.transformer.rope_utils':['fused_apply_rotary_pos_emb','fused_apply_rotary_pos_emb_thd']}
        for module,names in targets.items():
            owner=importlib.import_module(module)
            for name in names:
                native=getattr(owner,name,None)
                if not callable(native):
                    self.missing.append(module+'.'+name)
                    continue
                self.wrap(owner,name,native,module+'.'+name)
        for name in ('broadcast','all_reduce','reduce_scatter_tensor','all_gather_into_tensor','all_gather',
                     'reduce_scatter','all_to_all_single','batch_isend_irecv'):
            native=getattr(dist,name,None)
            if native is not None:self.wrap(dist,name,native,'collective:'+name,collective=True)

    @staticmethod
    def describe(value):
        if torch.is_tensor(value):
            return dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
        if isinstance(value,(tuple,list)):
            return [BackendEvidence.describe(x) for x in value[:4]]
        if value is None or isinstance(value,(str,bool,int,float)):return value
        return type(value).__module__+'.'+type(value).__name__

    def observe(self,key,args,kwargs):
        if key not in self.calls:
            self.calls[key]=dict(count=0,args=[self.describe(x) for x in args[:4]],
                kwargs={k:self.describe(v) for k,v in kwargs.items()})
        self.calls[key]['count']+=1

    def wrap(self,owner,name,native,key,collective=False):
        self.symbols[key]=dict(module=getattr(native,'__module__',None),name=getattr(native,'__qualname__',type(native).__qualname__))
        try:signature=inspect.signature(native)
        except (ValueError,TypeError):signature=None
        @functools.wraps(native)
        def observed(*args,**kwargs):
            self.observe(key,args,kwargs)
            if collective:
                try:bound=signature.bind_partial(*args,**kwargs).arguments if signature else kwargs
                except TypeError:bound=kwargs
                group=bound.get('group')
                try:ranks=dist.get_process_group_ranks(group or dist.group.WORLD)
                except Exception:ranks=None
                frames=inspect.stack(context=0)[1:6]
                self.collectives.append(dict(name=name,t=time.perf_counter()-self.started,
                    graph_task_id=torch._C._current_graph_task_id(),group_ranks=ranks,
                    async_op=bound.get('async_op',False),
                    args=[self.describe(x) for x in args[:2]],
                    callers=[f'{f.filename}:{f.lineno}:{f.function}' for f in frames]))
            result=native(*args,**kwargs)
            if 'first_result' not in self.calls[key]:
                self.calls[key]['first_result']=self.describe(result)
            return result
        setattr(owner,name,observed)
        self.restore.append((owner,name,native))

    def flush_on_exit(self):
        if not self.finished:
            self.finish(completed_step=False)

    def finish(self,completed_step=True):
        if self.finished:return
        self.finished=True
        for owner,name,native in reversed(self.restore):setattr(owner,name,native)
        for handle in self.handles:handle.remove()
        save(self.out/f'backend_rank{self.rank}.json',dict(system=self.system,rank=self.rank,
            first_step_completed=completed_step,
            packages=self.packages,modules=self.modules,optimizers=self.optimizers,
            import_locations=self.import_locations,
            resolved_symbols=self.symbols,
            calls=self.calls,collectives=self.collectives,unavailable_symbols=self.missing,
            scope='Actual first-step Python dispatch and collective call sites; instrumentation removed after step1, before measured steps6–10. Dispatch evidence establishes implementation entry points, not identical assembly or exact overlap duration.'))
