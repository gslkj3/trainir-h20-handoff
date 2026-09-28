"""Observe upstream Galvatron's real train loop; small-model input integration checks.

torchrun --standalone --nproc-per-node=8 h20_galvatron_smoke.py CONFIG OUT
Keeps the native profiler and optimizer; adds evidence around their real calls.
"""
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    import torch
    import torch.distributed as dist
    from galvatron.core.arguments import load_with_hydra
    from galvatron.models.gpt import train_dist as entry
    from galvatron.utils.hf_config_adapter import resolve_model_config

    config, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    args = load_with_hydra(str(config), overrides=[], mode="train_dist")
    resolve_model_config(args)
    expected = json.loads(config.read_text())['runtime']['model']
    for key in ('rotary_base', 'qk_layernorm', 'norm_epsilon', 'padded_vocab_size'):
        if key in expected:
            assert getattr(args.model, key) == expected[key], (key, getattr(args.model, key), expected[key])
    if args.model.qk_layernorm:
        # Bind the backend's RMSNorm implementation for Q/K, as in the handoff.
        import copy
        from galvatron.core.runtime.models import modules
        from galvatron.core.runtime.transformer.norm import GalvatronNorm
        native_attention = modules.SelfAttention
        class QKNormAttention(native_attention):
            def __init__(self, config, submodules, *a, **kw):
                submodules = copy.copy(submodules)
                submodules.q_layernorm = GalvatronNorm
                submodules.k_layernorm = GalvatronNorm
                super().__init__(config, submodules, *a, **kw)
        modules.SelfAttention = QKNormAttention
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    entry.initialize_galvatron(args)
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    out.mkdir(parents=True, exist_ok=True)
    from h20_galvatron_rope import needs_native_layer_rope,use_native_layer_rope
    layer_rope=needs_native_layer_rope(args)
    if layer_rope:
        native_batch=entry.get_batch
        def batch(*a,**kw):return use_native_layer_rope(native_batch(*a,**kw))
        entry.get_batch=batch
    if rank==0:
        (out/'rope_dispatch.json').write_text(json.dumps(dict(native_per_layer_rope=layer_rope)))
    rows, updates = [], []
    original_optimizer = entry.get_optimizer_and_param_scheduler
    original_profiler = entry.get_runtime_profiler
    # Give DP replicas different synthetic examples, while TP/PP peers retain
    # the same native dataset stream. No training computation is substituted.
    from galvatron.core.runtime import dataloader, parallel_state
    import numpy as np
    native_random_iterator = dataloader._build_random_data_iterator

    def random_iterator():
        np.random.seed(42 + parallel_state.get_vocab_dp_rank())
        return native_random_iterator()

    dataloader._build_random_data_iterator = random_iterator

    def observed_optimizer(model, arguments):
        optimizer, scheduler = original_optimizer(model, arguments)
        original_step = optimizer.step
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        layout = [dict(name=name, sharding_strategy=str(module.sharding_strategy),
                       group_size=dist.get_world_size(module.process_group))
                  for name, module in model.named_modules()
                  if isinstance(module, FSDP) and module._handle is not None]
        (out/f'fsdp_rank{rank}.json').write_text(json.dumps(layout, indent=2))

        def observed_step(*a, **kw):
            params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            assert params, 'No gradients before optimizer.step'
            assert all(torch.isfinite(p.grad).all().item() for p in params), 'Nonfinite gradients'
            grad_sq = sum(p.grad.float().square().sum().item() for p in params)
            assert grad_sq > 0, 'All gradients zero'
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
            replica_checks = 0
            replicated_modules = 0
            for module in model.modules():
                if not isinstance(module, FSDP) or module._handle is None:
                    continue
                if module.sharding_strategy != ShardingStrategy.NO_SHARD:
                    continue
                group = module.process_group
                if dist.get_world_size(group) <= 1:
                    continue
                replicated_modules += 1
                grad = module._handle.flat_param.grad
                if grad is None:
                    continue
                sample = grad.detach().flatten()[:4096].float().contiguous()
                replicas = [torch.empty_like(sample) for _ in range(dist.get_world_size(group))]
                dist.all_gather(replicas, sample, group=group)
                for other in replicas:
                    torch.testing.assert_close(sample, other, rtol=1e-5, atol=1e-6)
                replica_checks += 1
            if replicated_modules:
                assert replica_checks > 0, 'No replicated DP gradients checked'
            # Clone actual parameters to prove that a real update happened on each rank.
            before = [p.detach().clone() for p in params]
            result = original_step(*a, **kw)
            changed = sum(torch.count_nonzero(p.detach() != old).item() for p, old in zip(params, before))
            assert changed > 0, 'Optimizer did not change parameters'
            updates.append(dict(step=len(updates)+1, grad_sq=grad_sq, changed_elements=changed,
                                dp_gradient_replica_checks=replica_checks,
                                replicated_modules=replicated_modules))
            return result

        optimizer.step = observed_step
        return optimizer, scheduler

    def observed_profiler(*a, **kw):
        profiler = original_profiler(*a, **kw)
        original_start, original_end = profiler.profile_time_start, profiler.profile_time_end
        started = None

        def start(iteration):
            nonlocal started
            original_start(iteration)
            torch.cuda.synchronize()
            started = time.perf_counter()

        def end(iteration, loss=None, learning_rate=None, grad_norm=None):
            original_end(iteration, loss, learning_rate, grad_norm)
            torch.cuda.synchronize()
            scalar = lambda x: None if x is None else float(x.item() if hasattr(x, 'item') else x)
            row = dict(iteration=iteration+1, seconds=time.perf_counter()-started,
                       loss=scalar(loss), grad_norm=scalar(grad_norm), optimizer_steps=len(updates))
            assert row['optimizer_steps'] == iteration+1
            for key in ('seconds', 'loss', 'grad_norm'):
                assert row[key] is None or math.isfinite(row[key]), row
            rows.append(row)
            with (out/f'rank{rank}.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            if rank == 7:
                print('H20_NATIVE_STEP', json.dumps(row), flush=True)

        profiler.profile_time_start, profiler.profile_time_end = start, end
        return profiler

    entry.get_optimizer_and_param_scheduler = observed_optimizer
    entry.get_runtime_profiler = observed_profiler
    try:
        entry.train(args)
        assert [r['iteration'] for r in rows] == list(range(1, 11)), rows
        assert len(updates) == 10
        if rank == 7:
            assert all(r['loss'] is not None and r['loss'] > 0 for r in rows)
        (out/f'rank{rank}.json').write_text(json.dumps(dict(
            passed=True, rank=rank, local_rank=int(os.environ['LOCAL_RANK']),
            device=torch.cuda.current_device(), gpu=str(torch.cuda.get_device_properties(rank)),
            updates=updates, iterations=rows,
            effective=args.model_dump(),
            scope='Small-model input integration validation; timing includes evidence instrumentation.'
        ), indent=2, default=str))
        dist.barrier()
    except Exception:
        # A peer can still be inside a pipeline collective. Do not hide the
        # original validation error behind collective process-group teardown.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
