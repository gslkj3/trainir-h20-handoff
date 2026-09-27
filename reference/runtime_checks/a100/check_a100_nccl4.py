"""Four local A100 ranks; correctness checks only, not communication calibration."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import torch
import torch.distributed as dist

local=int(os.environ['LOCAL_RANK'])
if torch.cuda.device_count()!=4:
    raise RuntimeError('Expected exactly four allocated visible GPUs; do not override the scheduler visibility mask')
torch.cuda.set_device(local)
if 'A100' not in torch.cuda.get_device_name(local):
    raise RuntimeError('Expected A100')
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
groups=[]
try:
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=4: raise RuntimeError('Expected world size 4')
    passed=[]
    for dtype in (torch.float32,torch.bfloat16):
        for count in (16,262144):
            x=torch.full((count,),rank+1,device='cuda',dtype=dtype)
            dist.all_reduce(x)
            torch.cuda.synchronize()
            if not torch.all(x==10).item(): raise RuntimeError('all_reduce values differ')
            x.fill_(rank+1)
            gathered=[torch.empty_like(x) for _ in range(world)]
            dist.all_gather(gathered,x)
            if not all(torch.all(t==i+1).item() for i,t in enumerate(gathered)):
                raise RuntimeError('all_gather values/order differ')
            inputs=torch.cat([torch.full_like(x,rank+1+i) for i in range(world)])
            target=torch.empty_like(x)
            dist.reduce_scatter_tensor(target,inputs)
            if not torch.all(target==10+world*rank).item(): raise RuntimeError('reduce_scatter values differ')
            recv=torch.empty_like(x)
            handles=dist.batch_isend_irecv([
                dist.P2POp(dist.irecv,recv,(rank-1)%world),
                dist.P2POp(dist.isend,x,(rank+1)%world)])
            for h in handles: h.wait()
            if not torch.all(recv==((rank-1)%world)+1).item(): raise RuntimeError('P2P ring values differ')
            passed.append(dict(dtype=str(dtype),elements=count,
                operations=['all_reduce','all_gather','reduce_scatter','p2p_ring']))
            if rank==0: print('PASS world collectives/P2P',dtype,count,flush=True)
    for members in ([0,1],[2,3]):
        group=dist.new_group(members,timeout=timedelta(seconds=180))
        if rank in members:
            groups.append(group)
            x=torch.tensor([rank+1.0],device='cuda')
            dist.all_reduce(x,group=group)
            if x.item()!=sum(i+1 for i in members): raise RuntimeError('Pair subgroup reduction differs')
    dist.barrier()
    info=dict(rank=rank,local_rank=local,host=socket.gethostname(),gpu=torch.cuda.get_device_name(local),
        device_uuid=str(getattr(torch.cuda.get_device_properties(local),'uuid','unavailable')),
        torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),passed=passed)
    records=[None]*world
    dist.all_gather_object(records,info)
    if len({r['host'] for r in records})!=1: raise RuntimeError('Expected one physical host')
    if rank==0:
        output=Path(os.environ['A100_NCCL_OUT'])
        topo=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True)
        (output/'topology.txt').write_text(topo.stdout+topo.stderr)
        (output/'nccl_checks.json').write_text(json.dumps(dict(passed=True,ranks=records,
            pair_subgroups=[[0,1],[2,3]],scope='One-node four-GPU communication correctness smoke, not bandwidth calibration or distributed training.'),indent=2))
        print('A100 NCCL4 PASS:',output/'nccl_checks.json',flush=True)
    dist.barrier()
finally:
    for group in reversed(groups): dist.destroy_process_group(group)
    dist.destroy_process_group()
