#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/u50/capstone/cs4zp6g17/miniconda/envs/wispr/bin/python"
TRAIN_PY="$ROOT/src/wsts/src/train.py"
CFG="$ROOT/src/wsts/cfgs/UTAE_Res18/paper_utae_res18_t5_all.yaml"
TRAINER_CFG="$ROOT/src/wsts/cfgs/trainer_single_gpu_paper_10000.yaml"
DATA_CFG="$ROOT/src/wsts/cfgs/data_multitemporal_full_features_paper.yaml"
DATA_DIR="/u50/capstone/cs4zp6g17/data/hdf5"
RUN_DIR="${RUN_DIR:-$ROOT/cv_runs/paper_utae_res18_t5_all_wsts12}"

GPUS_STRING="${GPUS:-7 8 9}"
IFS=' ' read -r -a GPUS <<< "$GPUS_STRING"

FOLDS_STRING="${FOLDS:-0 1 2 3 4 5 6 7 8 9 10 11}"
IFS=' ' read -r -a FOLDS <<< "$FOLDS_STRING"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metrics" "$RUN_DIR/pr_curves"

LAUNCHED_PID=""

launch_job() {
  local gpu="$1"
  local fold="$2"
  local run_name="paper_utae_res18_t5_all_fold${fold}"
  local metrics_file="$RUN_DIR/metrics/fold_${fold}.json"
  local pr_curve_file="$RUN_DIR/pr_curves/fold_${fold}.json"
  local log_file="$RUN_DIR/logs/fold_${fold}.log"

  if [[ -f "$metrics_file" ]]; then
    echo "Skipping fold ${fold}: metrics already exist at $metrics_file"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  WANDB_PROJECT=wildfire_progression \
  WANDB_NAME="$run_name" \
  "$PYTHON_BIN" "$TRAIN_PY" \
    --config="$CFG" \
    --trainer="$TRAINER_CFG" \
    --data="$DATA_CFG" \
    --data.data_dir="$DATA_DIR" \
    --data.data_fold_id="$fold" \
    --do_train=True \
    --do_validate=False \
    --do_test=True \
    --trainer.devices=1 \
    --metrics_output_path="$metrics_file" \
    --pr_curve_output_path="$pr_curve_file" \
    > "$log_file" 2>&1 &

  LAUNCHED_PID="$!"
  return 0
}

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs provided in GPUS." >&2
  exit 1
fi

declare -a PIDS=()
declare -a ACTIVE_FOLDS=()
declare -a ACTIVE_GPUS=()

for fold in "${FOLDS[@]}"; do
  slot=$(( ${#PIDS[@]} % ${#GPUS[@]} ))

  if [[ "${#PIDS[@]}" -eq "${#GPUS[@]}" ]]; then
    echo "Waiting for current batch to finish: folds ${ACTIVE_FOLDS[*]}"
    for pid in "${PIDS[@]}"; do
      wait "$pid"
    done
    PIDS=()
    ACTIVE_FOLDS=()
    ACTIVE_GPUS=()
  fi

  gpu="${GPUS[$slot]}"
  if launch_job "$gpu" "$fold"; then
    pid="$LAUNCHED_PID"
    PIDS+=("$pid")
    ACTIVE_FOLDS+=("$fold")
    ACTIVE_GPUS+=("$gpu")
    echo "Launched fold ${fold} on GPU ${gpu} (pid ${pid})"
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  echo "Waiting for final batch to finish: folds ${ACTIVE_FOLDS[*]}"
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
fi

echo "All requested folds finished."
