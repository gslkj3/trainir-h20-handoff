#!/bin/bash
# Matches the environment supplied by the user; no package installation.
module purge
module load miniforge3/26.3.2-3
module load cuda/12.8
module load nccl/2.28_cuda12.8_4090_5090
module load cmake/4.2.0
module load openssl/3.0.2
module load zlib/1.3.1
# Initialize conda for non-interactive Slurm shells using the loaded module.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py312-t29
export GPUS_PER_NODE=${DTSIR_COMMON_GPUS_PER_NODE:-8}
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2:1
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
