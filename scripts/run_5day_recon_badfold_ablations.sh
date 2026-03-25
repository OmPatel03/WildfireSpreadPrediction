#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/u50/capstone/cs4zp6g17/miniconda/envs/wispr/bin/python"
TRAIN_PY="$ROOT/src/wsts/src/train.py"
TRAINER_CFG="$ROOT/src/wsts/cfgs/trainer_single_gpu_patience20.yaml"
DATA_DIR="/u50/capstone/cs4zp6g17/data/hdf5"

DECAY_CFG="$ROOT/src/wsts/cfgs/UTAE_Res18/domain_adversarial_reconstruction_fire_aware_decay_t5_bs2x.yaml"
FIXED_CFG="$ROOT/src/wsts/cfgs/UTAE_Res18/domain_adversarial_reconstruction_fire_aware_fixedmask_t5_bs2x.yaml"

DECAY_RUN_DIR="$ROOT/cv_runs/recon_fire_aware_decay_wsts12_t5_bs2x_badfolds"
FIXED_RUN_DIR="$ROOT/cv_runs/recon_fire_aware_fixedmask_wsts12_t5_bs2x_badfolds"

mkdir -p "$DECAY_RUN_DIR/logs" "$DECAY_RUN_DIR/metrics"
mkdir -p "$FIXED_RUN_DIR/logs" "$FIXED_RUN_DIR/metrics"

launch_job() {
  local gpu="$1"
  local fold="$2"
  local cfg="$3"
  local run_dir="$4"
  local run_name="$5"

  CUDA_VISIBLE_DEVICES="$gpu" \
  WANDB_PROJECT=wildfire_progression \
  WANDB_NAME="$run_name" \
  nohup "$PYTHON_BIN" "$TRAIN_PY" \
    --config="$cfg" \
    --trainer="$TRAINER_CFG" \
    --do_train=True \
    --do_validate=False \
    --do_test=True \
    --trainer.devices=1 \
    --data.data_dir="$DATA_DIR" \
    --data.data_fold_id="$fold" \
    --metrics_output_path="$run_dir/metrics/fold_${fold}.json" \
    > "$run_dir/logs/fold_${fold}.log" 2>&1 &
}

launch_job 4 3 "$DECAY_CFG" "$DECAY_RUN_DIR" "recon_fire_decay_t5_bs2x_fold3"
launch_job 5 5 "$DECAY_CFG" "$DECAY_RUN_DIR" "recon_fire_decay_t5_bs2x_fold5"
launch_job 8 10 "$DECAY_CFG" "$DECAY_RUN_DIR" "recon_fire_decay_t5_bs2x_fold10"

launch_job 9 3 "$FIXED_CFG" "$FIXED_RUN_DIR" "recon_fire_fixedmask_t5_bs2x_fold3"
launch_job 10 5 "$FIXED_CFG" "$FIXED_RUN_DIR" "recon_fire_fixedmask_t5_bs2x_fold5"
launch_job 11 10 "$FIXED_CFG" "$FIXED_RUN_DIR" "recon_fire_fixedmask_t5_bs2x_fold10"

echo "Launched 6 jobs:"
echo "  decay:      folds 3, 5, 10 on GPUs 4, 5, 8"
echo "  fixed-mask: folds 3, 5, 10 on GPUs 9, 10, 11"
