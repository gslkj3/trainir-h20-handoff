#!/bin/bash
#SBATCH -J common5-exact
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH -p hp_5090
#SBATCH --qos=gpugpu
#SBATCH -o logs/paired16-%j.log
#SBATCH -e logs/paired16-%j.log
# No invented time limit: use this partition/QOS default (as original Megatron jobs).
set -Eeo pipefail
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NUM_NODES=2 NUM_GPUS_PER_NODE=8 NNODES=2 GPUS_PER_NODE=8
export PAIR16_MEG_TAG="${PAIR16_TAG}_megatron"
export GALV6_TAG="${PAIR16_TAG}_galvatron"
export PAIR16_LOGS="$MEGATRON_ROOT/mm_logs/$PAIR16_TAG"
mkdir -p "$PAIR16_LOGS"
printf 'system\tcase\texit_code\n' > "$PAIR16_LOGS/driver_status.tsv"
scontrol show job "$SLURM_JOB_ID" > "$PAIR16_LOGS/slurm_allocation.txt"
names=(llama7b_2k llama2_7b_4k llama2_13b_4k llama3_8b_8k qwen3_14b_4k)
galv_indices=(0 1 2 3 5)
failed=0
for system in megatron galvatron; do
  if [ "$PAIR16_SYSTEMS" != both ] && [ "$PAIR16_SYSTEMS" != "$system" ]; then continue; fi
  # Isolate conda activation between systems rather than mutate a running Python.
  (
    set -eo pipefail
    if [ "$system" = megatron ]; then
      cd "$MEGATRON_ROOT"; source dtsir_common16/env.sh
    else
      cd "$GALV_REPO"; source "$GALV6_DIR/env.sh"
    fi
    export PAIR16_PROBE_OUT="$PAIR16_LOGS/${system}_network.json"
    echo "START integrated network check: $system"
    srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=8 \
      --gres=gpu:8 --kill-on-bad-exit=1 --time=00:30:00 \
      bash "$PAIR16_HOME/probe_worker.sh" "$system" \
      > "$PAIR16_LOGS/${system}_network.log" 2>&1
    test -f "$PAIR16_PROBE_OUT"
    for i in 0 1 2 3 4; do
      name="${names[$i]}"
      echo "START $system $name $(date -Is)"
      code=0
      if [ "$system" = megatron ]; then
        python -u dtsir_common16/run.py --case "$name" \
          > "$PAIR16_LOGS/${system}_${name}.log" 2>&1 || code=$?
      else
        python -u "$GALV6_DIR/run_six.py" run --index "${galv_indices[$i]}" \
          > "$PAIR16_LOGS/${system}_${name}.log" 2>&1 || code=$?
      fi
      printf '%s\t%s\t%s\n' "$system" "$name" "$code" >> "$PAIR16_LOGS/driver_status.tsv"
      echo "FINISH $system $name exit=$code $(date -Is)"
    done
  )
done
if awk -F '\t' 'NR>1 && $3!=0 {bad=1} END {exit !bad}' "$PAIR16_LOGS/driver_status.tsv"; then failed=1; fi
echo "FINISHED. Exit codes are driver statuses, not proof of successful training. Results: $PAIR16_LOGS"
exit "$failed"
