#!/bin/bash
set -euo pipefail
export MEGATRON_ROOT="${MEGATRON_ROOT:-/data/run01/LEGACY_USER/wjy/Megatron-LM}"
export GALV_REPO="${GALV_REPO:-/data/run01/LEGACY_USER/wjy/dependencies/Hetu-Galvatron-dtsir}"
export GALV6_DIR="$GALV_REPO/dtsir_galvatron6"
export PAIR16_HOME="$MEGATRON_ROOT/paired16"
export PAIR16_TAG="${PAIR16_TAG:-common5_exact_$(date +%Y%m%d_%H%M%S)_$$}"
export PAIR16_SYSTEMS="${PAIR16_SYSTEMS:-both}"
case "$PAIR16_SYSTEMS" in both|megatron|galvatron) ;; *) echo 'Invalid PAIR16_SYSTEMS'; exit 2;; esac
case "${1:-}" in ''|--check-only) ;; *) echo 'Usage: bash paired16/submit_exact.sh [--check-only]'; exit 2;; esac
cd "$MEGATRON_ROOT"
test -f "$PAIR16_HOME/network.py"
test -f "$GALV6_DIR/run_six.py"
for f in "$PAIR16_HOME/all5_exact.sbatch" "$PAIR16_HOME/probe_worker.sh" \
         "$MEGATRON_ROOT/dtsir_common16/worker.sh" "$GALV6_DIR/worker.sh"; do
  bash -n "$f"
done
python -m py_compile dtsir_common16/run.py dtsir_common16/entry.py \
    dtsir_common16/common5_space.py "$GALV6_DIR/run_six.py" \
    "$GALV6_DIR/common5_space.py"
python "$PAIR16_HOME/preflight_exact.py"
(
  set +u  # Site module/conda initialization may reference unset shell variables.
  cd "$GALV_REPO"
  source "$GALV6_DIR/env.sh"
  python -u "$GALV6_DIR/run_six.py" space-check
)
if [ "${1:-}" = --check-only ]; then
  echo 'CHECKS PASSED; no Slurm job submitted.'
  exit 0
fi
for out in "$MEGATRON_ROOT/mm_logs/$PAIR16_TAG" \
           "$MEGATRON_ROOT/mm_logs/${PAIR16_TAG}_megatron" \
           "$GALV_REPO/mm_logs/${PAIR16_TAG}_galvatron"; do
  if [ -e "$out" ]; then echo "Output tag already exists: $out; use a new PAIR16_TAG"; exit 2; fi
done
mkdir -p logs
echo "Submitting one 16-GPU allocation; systems=$PAIR16_SYSTEMS; tag=$PAIR16_TAG"
sbatch --export=ALL "$PAIR16_HOME/all5_exact.sbatch"
