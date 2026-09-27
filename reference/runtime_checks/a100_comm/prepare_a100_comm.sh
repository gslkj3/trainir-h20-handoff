#!/bin/bash
# Run on the A100 login node. No GPU allocation, pip install, or repo edits.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/a100_env.sh" megatron
test "$(uname -m)" = aarch64 || { echo 'Use the A100 ARM cluster, not the 5090 cluster.' >&2; exit 2; }
if [ -n "${A100_NCCL_SOURCE:-}" ]; then
    SRC=$A100_NCCL_SOURCE
else
    SRC="$HERE/nccl-tests-2.13.8"
    if [ ! -f "$SRC/src/common.cu" ]; then
        DL=$(mktemp -d "$HERE/nccl_download_XXXXXXXX")
        echo "Downloading official nccl-tests v2.13.8 into $DL"
        curl --noproxy '*' -fL --connect-timeout 20 --max-time 180 --retry 2 \
          https://codeload.github.com/NVIDIA/nccl-tests/tar.gz/refs/tags/v2.13.8 \
          -o "$DL/source.tar.gz"
        tar -xzf "$DL/source.tar.gz" -C "$DL"
        SRC="$DL/nccl-tests-2.13.8"
    fi
fi
"$A100_PYTHON" -u "$HERE/a100_comm.py" prepare --source "$SRC"
source "$HERE/a100_env.sh" galvatron
"$A100_PYTHON" -u "$HERE/a100_comm.py" preflight
echo 'PREPARE PASS. Now run: bash ~/a100_transfer/submit_a100_comm.sh'
