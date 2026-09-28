"""Eight-rank NCCL correctness, rank identity, and FP16/BF16 matmul backward."""
import json
import os
from pathlib import Path
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=120))
assert dist.get_world_size() == 8
rows = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    x = torch.full((1024 * 1024,), rank + 1, device='cuda', dtype=dtype)
    dist.all_reduce(x)
    assert torch.all(x == 36).item()
    rows.append(str(dtype))
for dtype in (torch.float16, torch.bfloat16):
    x = torch.randn(256, 256, device='cuda', dtype=dtype, requires_grad=True)
    y = (x @ x.T).float().square().mean()
    y.backward()
    assert torch.isfinite(y).item() and torch.isfinite(x.grad).all().item()
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out/f'rank{rank}.json').write_text(json.dumps(dict(
    passed=True, rank=dist.get_rank(), local_rank=rank,
    gpu=str(torch.cuda.get_device_properties(rank)),
    torch=torch.__version__, cuda=torch.version.cuda,
    nccl=torch.cuda.nccl.version(), allreduce_dtypes=rows), indent=2))
dist.barrier()
dist.destroy_process_group()
