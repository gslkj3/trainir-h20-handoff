"""Observe the delivered Megatron native loop without replacing its optimizer."""
import json
import math
import os
from pathlib import Path
import runpy
import sys

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import training

out = Path(os.environ['H20_VALIDATION_OUT'])
root = Path(os.environ['MEGATRON_ROOT'])
native_setup = training.setup_model_and_optimizer
updates = []
module_evidence = []


def setup(*a, **kw):
    models, optimizer, scheduler = native_setup(*a, **kw)
    if get_args().qk_layernorm:
        for model in models:
            for name, module in model.named_modules():
                if name.endswith(('q_layernorm', 'k_layernorm')):
                    assert type(module).__name__ == 'RMSNorm', (name, type(module))
                    module_evidence.append(dict(name=name, implementation=type(module).__module__+'.'+type(module).__name__))
        assert module_evidence, 'No actual Q/K RMSNorm modules found'
    native_step = optimizer.step

    def step(*a, **kw):
        params = [p for model in models for p in model.parameters() if p.requires_grad]
        grads = [getattr(p, 'main_grad', p.grad) for p in params]
        grads = [g for g in grads if g is not None]
        assert grads and all(torch.isfinite(g).all().item() for g in grads)
        norm_sq = sum(g.float().square().sum().item() for g in grads)
        assert math.isfinite(norm_sq) and norm_sq > 0
        group = parallel_state.get_data_parallel_group()
        dp = dist.get_world_size(group)
        checks = 0
        if dp > 1:
            for grad in grads:
                sample = grad.detach().flatten()[:4096].float().contiguous()
                replicas = [torch.empty_like(sample) for _ in range(dp)]
                dist.all_gather(replicas, sample, group=group)
                for other in replicas:
                    torch.testing.assert_close(sample, other, rtol=1e-5, atol=1e-6)
                checks += 1
        before = [p.detach().clone() for p in params]
        result = native_step(*a, **kw)
        assert result[0], 'Native optimizer skipped update'
        changed = sum(torch.count_nonzero(p.detach() != old).item() for p, old in zip(params, before))
        assert changed > 0, 'No actual parameter changes'
        row = dict(iteration=len(updates)+1, update_successful=bool(result[0]),
                   grad_norm=float(result[1]), changed_elements=changed,
                   dp_gradient_replica_checks=checks, pre_step_grad_sq=norm_sq)
        assert math.isfinite(row['grad_norm'])
        updates.append(row)
        with (out/f'rank{dist.get_rank()}.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        return result

    optimizer.step = step
    return models, optimizer, scheduler


training.setup_model_and_optimizer = setup
sys.argv[0] = str(root/'pretrain_gpt.py')
runpy.run_path(str(root/'pretrain_gpt.py'), run_name='__main__')
assert dist.is_initialized() and dist.get_world_size() == 8
assert len(updates) == 10
args = get_args()
rank = dist.get_rank()
(out/f'rank{rank}.json').write_text(json.dumps(dict(passed=True, rank=rank,
    local_rank=int(os.environ['LOCAL_RANK']), device=torch.cuda.current_device(),
    gpu=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
    updates=updates, effective=vars(args), source=str(root),
    qk_norm_modules=module_evidence,
    scope='Native small-model real-input integration check; instrumented timings are not benchmark data.'),
    indent=2,default=str))
dist.barrier()
dist.destroy_process_group()
