"""Exercise installed CUDA kernels against PyTorch references on an H20."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import flash_attn
import apex
import amp_C
import fused_layer_norm_cuda
import transformer_engine
import transformer_engine.pytorch as te
from apex.normalization import FusedRMSNorm, FusedLayerNorm
from apex.optimizers import FusedAdam

torch.manual_seed(42)
rows = []
for dtype in (torch.float16, torch.bfloat16):
    for cls, kind in ((FusedRMSNorm, 'apex_rms'), (FusedLayerNorm, 'apex_layer'),
                      (te.RMSNorm, 'te_rms'), (te.LayerNorm, 'te_layer')):
        layer = cls(512, eps=1e-5).cuda().to(dtype)
        x = torch.randn(8, 128, 512, device='cuda', dtype=dtype, requires_grad=True)
        refx = x.detach().float().requires_grad_()
        y = layer(x)
        ref = (refx * torch.rsqrt(refx.square().mean(-1, keepdim=True)+1e-5)
               if kind.endswith('rms') else F.layer_norm(refx, (512,), eps=1e-5))
        torch.testing.assert_close(y.float(), ref, rtol=0.03, atol=0.03)
        dy = torch.randn_like(y)
        y.backward(dy)
        ref.backward(dy.float())
        torch.testing.assert_close(x.grad.float(), refx.grad, rtol=0.05, atol=0.05)
        assert all(torch.isfinite(p.grad).all() for p in layer.parameters())
        rows.append(dict(kernel=kind, dtype=str(dtype), passed=True))
    q = torch.randn(2, 128, 8, 64, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn(2, 128, 4, 64, device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = flash_attn.flash_attn_func(q, k, v, causal=True)
    qr, kr, vr = [x.detach().float().requires_grad_() for x in (q, k, v)]
    ref = F.scaled_dot_product_attention(qr.transpose(1, 2),
        kr.repeat_interleave(2, dim=2).transpose(1, 2),
        vr.repeat_interleave(2, dim=2).transpose(1, 2), is_causal=True).transpose(1, 2)
    torch.testing.assert_close(output.float(), ref, rtol=0.03, atol=0.03)
    dy = torch.randn_like(output)
    output.backward(dy)
    ref.backward(dy.float())
    for x, r in zip((q, k, v), (qr, kr, vr)):
        torch.testing.assert_close(x.grad.float(), r.grad, rtol=0.06, atol=0.06)
    rows.append(dict(kernel='flash_attention_gqa', dtype=str(dtype), passed=True))

p = torch.nn.Parameter(torch.randn(8192, device='cuda'))
r = torch.nn.Parameter(p.detach().clone())
fused = FusedAdam([p], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, adam_w_mode=True)
reference = torch.optim.AdamW([r], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
for _ in range(10):
    grad = torch.randn_like(p)
    p.grad, r.grad = grad, grad.clone()
    fused.step()
    reference.step()
torch.testing.assert_close(p, r, rtol=1e-5, atol=1e-6)
rows.append(dict(kernel='apex_fused_adam_10_updates', dtype='float32', passed=True))
torch.cuda.synchronize()
report = dict(passed=True, torch=torch.__version__, flash_attention=flash_attn.__version__,
              transformer_engine=transformer_engine.__version__, apex_path=apex.__file__,
              gpu=str(torch.cuda.get_device_properties(0)), checks=rows)
Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
