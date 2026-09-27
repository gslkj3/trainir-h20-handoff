#!/bin/bash
# Source inside a subshell: source ./a100_env.sh megatron|galvatron
dtsir_a100_activate() {
    local system="${1:-}"
    local env_name
    case "$system" in
        megatron) env_name=dtsir-a100 ;;
        galvatron) env_name=galvatron-a100 ;;
        *) echo 'Usage: source a100_env.sh megatron|galvatron' >&2; return 2 ;;
    esac
    local target_env="/home/bingxing2/home/LEGACY_USER/.conda/envs/$env_name"
    test -x "$target_env/bin/python" || {
        echo "Missing target interpreter: $target_env/bin/python" >&2
        return 2
    }
    export A100_WORK=/home/bingxing2/home/LEGACY_USER/dtsir_a100_work_JSVXCTf7
    export MEGATRON_ROOT="$A100_WORK/Megatron-LM"
    export GALV_REPO="$A100_WORK/Hetu-Galvatron-dtsir"
    test -d "$MEGATRON_ROOT" && test -d "$GALV_REPO" || return 2
    # Never source the old project launchers or activate their original environments.
    module purge || return
    module load miniforge3/24.1 compilers/cuda/12.1 compilers/gcc/11.3.0 \
        cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1 || return
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
    source /home/bingxing2/apps/miniforge3/24.1.2/etc/profile.d/conda.sh || return
    conda activate "$target_env" || return
    if [ "${CONDA_PREFIX:-}" != "$target_env" ]; then
        echo "Conda did not activate $target_env; CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
        return 2
    fi
    # A separately activated virtualenv can remain ahead of Conda in PATH.
    # Removing VIRTUAL_ENV alone does not repair PATH. All changes are local
    # to the caller's subshell; no original environment is modified.
    export PATH="$target_env/bin:$PATH"
    unalias python python3 pip pip3 torchrun 2>/dev/null || true
    unset -f python python3 pip pip3 torchrun 2>/dev/null || true
    export A100_PYTHON="$target_env/bin/python"
    hash -r
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    local gomp_files=("$CONDA_PREFIX"/lib/python3.10/site-packages/scikit_learn.libs/libgomp-*.so*)
    if [ "${#gomp_files[@]}" -ne 1 ] || [ ! -f "${gomp_files[0]}" ]; then
        echo "Expected one sklearn libgomp in $CONDA_PREFIX; inspect before continuing." >&2
        return 2
    fi
    # Use this environment's copy, not the old environment's TLS preload.
    export LD_PRELOAD="${gomp_files[0]}"
    "$A100_PYTHON" - "$target_env" <<'PY' || return
import os
import sys
from pathlib import Path
expected = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected:
    raise SystemExit(f"Wrong interpreter prefix: {sys.prefix}; expected {expected}")
if Path(os.environ['CONDA_PREFIX']).resolve() != expected:
    raise SystemExit('Conda prefix mismatch')
print('VERIFIED Python:', sys.executable, flush=True)
PY
    if [ "$system" = megatron ]; then
        export PYTHONPATH="$MEGATRON_ROOT"
        cd "$MEGATRON_ROOT" || return
    else
        export PYTHONPATH="$GALV_REPO"
        cd "$GALV_REPO" || return
    fi
    echo "A100 system=$system Python=$A100_PYTHON Source=$PWD"
}
dtsir_a100_activate "$@"
