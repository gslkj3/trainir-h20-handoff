#!/bin/bash
set -Eeo pipefail
trap 'echo "PROBE FAILED host=$(hostname) line=$LINENO status=$?" >&2' ERR
if [ "$1" = megatron ]; then
  cd "$MEGATRON_ROOT"
  source dtsir_common16/env.sh
else
  cd "$GALV_REPO"
  source "$GALV6_DIR/env.sh"
fi
network_exports=$(python "$PAIR16_HOME/network.py" --nodes 2)
eval "$network_exports"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$PAIR16_HOME/probe.py"
