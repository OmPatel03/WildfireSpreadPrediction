#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/u50/capstone/cs4zp6g17/miniconda/envs/wispr/bin/python"
TRAIN_PY="$ROOT/src/wsts/src/train.py"
TRAINER_CFG="$ROOT/src/wsts/cfgs/trainer_single_gpu_patience20.yaml"
CFG="$ROOT/src/wsts/cfgs/UTAE_Res18/domain_adversarial_reconstruction_fire_aware_fixedmask_t5_bs2x.yaml"
DATA_DIR="/u50/capstone/cs4zp6g17/data/hdf5"
RUN_DIR="$ROOT/cv_runs/recon_fire_aware_fixedmask_wsts12_t5_bs2x"

GPUS_STRING="${GPUS:-4 5 8}"
IFS=' ' read -r -a GPUS <<< "$GPUS_STRING"
FOLDS=(0 1 2 3 4 5 6 7 8 9 10 11)

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metrics"

LAUNCHED_PID=""

launch_job() {
  local gpu="$1"
  local fold="$2"
  local run_name="recon_fire_fixedmask_wsts12_t5_bs2x_fold${fold}"
  local metrics_file="$RUN_DIR/metrics/fold_${fold}.json"
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
    --do_train=True \
    --do_validate=False \
    --do_test=True \
    --trainer.devices=1 \
    --data.data_dir="$DATA_DIR" \
    --data.data_fold_id="$fold" \
    --metrics_output_path="$metrics_file" \
    > "$log_file" 2>&1 &

  LAUNCHED_PID="$!"
  return 0
}

if [[ "${#GPUS[@]}" -ne 3 ]]; then
  echo "Expected exactly 3 GPUs in GPUS, got: $GPUS_STRING" >&2
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
