#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/u50/capstone/cs4zp6g17/miniconda/envs/wispr/bin/python"
TRAIN_PY="$ROOT/src/wsts/src/train.py"
TRAINER_CFG="$ROOT/src/wsts/cfgs/trainer_single_gpu_patience20.yaml"
CFG="$ROOT/src/wsts/cfgs/UTAE_Res18/domain_adversarial_reconstruction_fire_aware_decay_fixedmask_t5_bs2x.yaml"
DATA_DIR="/u50/capstone/cs4zp6g17/data/hdf5"
RUN_DIR="$ROOT/cv_runs/recon_fire_aware_decay_fixedmask_wsts12_t5_bs2x_badfolds"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metrics"

launch_job() {
  local gpu="$1"
  local fold="$2"
  local run_name="recon_fire_decay_fixedmask_t5_bs2x_fold${fold}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  WANDB_PROJECT=wildfire_progression \
  WANDB_NAME="$run_name" \
  nohup "$PYTHON_BIN" "$TRAIN_PY" \
    --config="$CFG" \
    --trainer="$TRAINER_CFG" \
    --do_train=True \
    --do_validate=False \
    --do_test=True \
    --trainer.devices=1 \
    --data.data_dir="$DATA_DIR" \
    --data.data_fold_id="$fold" \
    --metrics_output_path="$RUN_DIR/metrics/fold_${fold}.json" \
    > "$RUN_DIR/logs/fold_${fold}.log" 2>&1 &
}

launch_job 4 3
launch_job 5 5
launch_job 8 10

echo "Launched combined 5-day reconstruction bad-fold ablation:"
echo "  folds 3, 5, 10 on GPUs 4, 5, 8"
