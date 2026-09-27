#!/bin/bash
set -eo pipefail
source "${GALV6_DIR:?}/env.sh"
nodes=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["nodes"])' "$1")
network_exports=$(python "${PAIR16_HOME:?}/network.py" --nodes "$nodes")
eval "$network_exports"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/galv6-${USER}-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
mkdir -p "$TRITON_CACHE_DIR"
exec python -u "$GALV6_DIR/run_six.py" worker "$1"
    