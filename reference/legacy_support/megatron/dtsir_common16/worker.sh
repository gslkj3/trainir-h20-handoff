#!/bin/bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source dtsir_common16/env.sh
export NODE_RANK="${SLURM_PROCID:?}"
export NNODES=${DTSIR_COMMON_NNODES:-2} GPUS_PER_NODE=${DTSIR_COMMON_GPUS_PER_NODE:-8}
export PAIR16_HOME="${PAIR16_HOME:-$PWD/paired16}"
network_exports=$(python "$PAIR16_HOME/network.py" --nodes "$NNODES")
eval "$network_exports"
python - <<'PY'
import torch, os, json, subprocess
from pathlib import Path
n=int(os.environ['GPUS_PER_NODE'])
assert torch.cuda.device_count()==n, f'Expected {n} allocated visible GPUs'
names=[torch.cuda.get_device_name(i) for i in range(n)]
assert all('5090' in n for n in names), names
record={'node':os.uname().nodename,'torch':torch.__version__,'cuda':torch.version.cuda,
        'nccl':torch.cuda.nccl.version(),'devices':names,
        'network':{k:os.environ.get(k) for k in ('NCCL_IB_HCA','NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','NCCL_NET')},
        'memory_bytes':[torch.cuda.get_device_properties(i).total_memory for i in range(n)],
        'gpu_topology':subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True).stdout}
Path(os.environ['DTSIR_IR_OUT'],f"node{os.environ['NODE_RANK']}_environment.json").write_text(json.dumps(record,indent=2))
PY
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/dtsir-common16-inductor-XXXXXX)
export TRITON_CACHE_DIR=$(mktemp -d /tmp/dtsir-common16-triton-XXXXXX)
export TORCHINDUCTOR_COMPILE_THREADS=1
bash "${DTSIR_COMMON_LAUNCHER:-dtsir_common16/launchers/${DTSIR_COMMON_CASE:?}.sh}"
