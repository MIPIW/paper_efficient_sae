#!/bin/bash
# Flow 7+8, Part A: alpha sweep (linear -> conjunctive/XOR) x combine_rule (mand, concat).
# 5 alphas x 2 rules x {trained, control} = 20 cells, run with bounded parallelism.
set -uo pipefail
ROOT=/home/hyohyeongjang/paper_efficient_sae
cd "$ROOT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

ALPHAS=(0.0 0.25 0.5 0.75 1.0)
RULES=(mand concat)
TOTAL_STEPS=2000
PARALLEL=5

run_cell () {
  local alpha="$1" rule="$2" mode="$3"
  local tag="${alpha//./p}"
  local outdir="$ROOT/runs/conj_alpha_${rule}/${mode}_alpha_${tag}"
  mkdir -p "$outdir"
  local extra=()
  if [ "$mode" == "control" ]; then
    extra=(--synthetic_skip_training)
  fi
  echo "[$(date +%H:%M:%S)] START rule=$rule mode=$mode alpha=$alpha"
  python3 src/train.py --synthetic_diagnostic --joint --joint_mode dpo_cross \
    --synthetic_conjunctive --conjunctive_alpha "$alpha" --conjunctive_balance on \
    --combine_rule "$rule" --k 24 --flat_dict_size 16384 \
    --joint_h 128 --joint_m 8 --joint_n 16 --total_steps "$TOTAL_STEPS" --seed 42 \
    "${extra[@]}" --output_dir "$outdir" > "$outdir/log.txt" 2>&1
  status=$?
  echo "[$(date +%H:%M:%S)] DONE(status=$status) rule=$rule mode=$mode alpha=$alpha"
}
export -f run_cell
export ROOT TOTAL_STEPS

JOBS_FILE=$(mktemp)
for rule in "${RULES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    echo "$alpha $rule control" >> "$JOBS_FILE"
    echo "$alpha $rule trained" >> "$JOBS_FILE"
  done
done

echo "Total cells: $(wc -l < "$JOBS_FILE"), parallelism=$PARALLEL, total_steps=$TOTAL_STEPS"
xargs -a "$JOBS_FILE" -n3 -P"$PARALLEL" bash -c 'run_cell "$@"' _
rm -f "$JOBS_FILE"

echo "=== all cells done, aggregating ==="
for rule in "${RULES[@]}"; do
  python3 src/aggregate_conjunctive_alpha_sweep.py --sweep_dir "runs/conj_alpha_${rule}" \
    --output_json "runs/conj_alpha_${rule}/conjunctive_alpha_sweep_results.json"
done
echo "=== sweep complete ==="
