"""One visible GPU, small kernel tests only; no distributed training or profiling."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

def main():
    p=argparse.ArgumentParser()
    p.add_argument('system',choices=('megatron','galvatron'))
    args=p.parse_args()
    prefix=Path.home()/'.conda/envs'/('dtsir-a100' if args.system=='megatron' else 'galvatron-a100')
    if Path(sys.prefix).resolve()!=prefix.resolve():
        raise SystemExit('Wrong environment: '+sys.prefix)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No allocated CUDA GPU is visible. Run on an allocated A100 compute node.')
    torch.cuda.set_device(0)
    name=torch.cuda.get_device_name(0)
    if 'A100' not in name:
        raise SystemExit('Expected A100, found '+name)
    torch.manual_seed(20260924)
    dtype=torch.bfloat16
    def finish(y,inputs,params=()):
        if not torch.isfinite(y).all().item(): raise RuntimeError('Non-finite forward output')
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        for x in list(inputs)+list(params):
            if x.requires_grad and (x.grad is None or not torch.isfinite(x.grad).all().item()):
                raise RuntimeError('Missing/non-finite gradient')
    def matmul():
        x=torch.randn(64,128,device='cuda',dtype=dtype,requires_grad=True)
        w=torch.randn(128,128,device='cuda',dtype=dtype,requires_grad=True)
        finish(x@w,[x,w])
    def attention(kv):
        from flash_attn import flash_attn_func
        q=torch.randn(2,128,8,64,device='cuda',dtype=dtype,requires_grad=True)
        k=torch.randn(2,128,kv,64,device='cuda',dtype=dtype,requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        finish(flash_attn_func(q,k,v,dropout_p=0.0,causal=True),[q,k,v])
    def norm(kind):
        if kind=='flash':
            from flash_attn.ops.rms_norm import RMSNorm
            layer=RMSNorm(128)
        else:
            from apex.normalization import FusedLayerNorm
            layer=FusedLayerNorm(128)
        layer=layer.to(device='cuda',dtype=dtype)
        x=torch.randn(16,128,device='cuda',dtype=dtype,requires_grad=True)
        finish(layer(x),[x],layer.parameters())
    def te_linear():
        import transformer_engine.pytorch as te
        layer=te.Linear(128,128,params_dtype=dtype).cuda()
        x=torch.randn(16,128,device='cuda',dtype=dtype,requires_grad=True)
        finish(layer(x),[x],layer.parameters())
    result=dict(system=args.system,python=sys.executable,arch=platform.machine(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=name,
        visible_devices=torch.cuda.device_count(),tests=[],
        scope='Small kernel execution checks on one GPU; not full training or benchmark validation.')
    print(json.dumps(result,indent=2),flush=True)
    tests=[('Torch BF16 matmul',matmul),('FlashAttention MHA',lambda:attention(8)),
        ('FlashAttention GQA',lambda:attention(2)),('FlashAttention RMSNorm',lambda:norm('flash')),
        ('Apex LayerNorm',lambda:norm('apex')),('Transformer Engine Linear',te_linear)]
    for label,fn in tests:
        print('START',label,flush=True)
        try:
            fn()
            row=dict(test=label,passed=True)
            print('PASS',label,flush=True)
        except Exception:
            row=dict(test=label,passed=False,error=traceback.format_exc())
            print(row['error'],flush=True)
        result['tests'].append(row)
    dest=Path(tempfile.mkdtemp(prefix='a100_gpu_'+args.system+'_',dir=os.environ['A100_WORK']))
    (dest/'gpu_checks.json').write_text(json.dumps(result,indent=2))
    print('RESULT:',dest/'gpu_checks.json',flush=True)
    return 0 if all(t['passed'] for t in result['tests']) else 1

if __name__=='__main__':
    raise SystemExit(main())
