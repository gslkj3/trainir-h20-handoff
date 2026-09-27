"""Small live collective check inside the production allocation; not a benchmark."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import torch
import torch.distributed as dist

local=int(os.environ['LOCAL_RANK'])
assert torch.cuda.device_count()==8
torch.cuda.set_device(local)
assert '5090' in torch.cuda.get_device_name(local)
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
try:
    rank=dist.get_rank();world=dist.get_world_size()
    assert world==16
    for size in (1,262144):
        x=torch.full((size,),float(rank+1),device='cuda')
        dist.all_reduce(x)
        assert torch.all(x==world*(world+1)/2).item()
    x=torch.tensor([37.0],device='cuda')
    if rank==0: dist.send(x,dst=8)
    elif rank==8:
        x.zero_();dist.recv(x,src=0);assert x.item()==37.0
    dist.barrier()
    row=dict(rank=rank,host=socket.gethostname(),torch=torch.__version__,
             cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
             net={k:os.environ.get(k) for k in ('NCCL_NET','NCCL_IB_HCA','NCCL_SOCKET_IFNAME')})
    gathered=[None]*world
    dist.all_gather_object(gathered,row)
    if rank==0:
        path=Path(os.environ['PAIR16_PROBE_OUT'])
        path.write_text(json.dumps(dict(passed=True,ranks=gathered),indent=2))
        print('PAIRED16 NETWORK PROBE PASSED',flush=True)
finally:
    dist.destroy_process_group()
