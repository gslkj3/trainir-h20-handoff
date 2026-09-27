#!/bin/bash
# Source this file; no installation and no network access.
module purge
module load miniforge3/26.3.2-3 cuda/12.8 nccl/2.28_cuda12.8_4090_5090
module load cmake/4.2.0 openssl/3.0.2 zlib/1.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GALV_ENV:-/data/home/LEGACY_USER/run/conda-envs/galvatron}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${GALV_REPO:?}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_2:1
export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_TIMEOUT=22
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
