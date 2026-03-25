#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_PYTHON = "/u50/capstone/cs4zp6g17/miniconda/envs/wispr/bin/python"
DEFAULT_TRAIN_SCRIPT = "/u50/rajguruu/WildfireSpreadPrediction/src/wsts/src/train.py"
DEFAULT_TRAINER = "/u50/rajguruu/WildfireSpreadPrediction/src/wsts/cfgs/trainer_single_gpu_patience20.yaml"
DEFAULT_DATA_DIR = "/u50/capstone/cs4zp6g17/data/hdf5"
DEFAULT_WANDB_PROJECT = "wildfire_progression"


def parse_csv_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def fold_iter(folds: str) -> list[int]:
    if folds.strip().lower() == "all":
        return list(range(12))
    return parse_csv_ints(folds)


def build_command(
    python_bin: str,
    train_script: str,
    config: str,
    trainer: str,
    data_dir: str,
    fold_id: int,
    metrics_path: Path,
    extra_args: Iterable[str],
) -> list[str]:
    cmd = [
        python_bin,
        train_script,
        f"--config={config}",
        f"--trainer={trainer}",
        "--do_train=True",
        "--do_validate=False",
        "--do_test=True",
        "--trainer.devices=1",
        f"--data.data_dir={data_dir}",
        f"--data.data_fold_id={fold_id}",
        f"--metrics_output_path={metrics_path}",
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch 12-fold WSTS runs for a given config and collect per-fold test metrics."
    )
    parser.add_argument("--config", required=True, help="Experiment config YAML to evaluate.")
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Short name used for output folders and W&B run names.",
    )
    parser.add_argument(
        "--output-dir",
        default="/u50/rajguruu/WildfireSpreadPrediction/cv_runs",
        help="Root directory for per-fold logs and metrics.",
    )
    parser.add_argument(
        "--folds",
        default="all",
        help='Comma-separated fold ids or "all" for 0..11.',
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids used round-robin across folds.",
    )
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional CLI arg to pass through to train.py. Repeat for multiple values.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually launch the generated commands. Without this flag, the script only prints them.",
    )
    args = parser.parse_args()

    folds = fold_iter(args.folds)
    if not folds:
        raise SystemExit("No folds selected.")
    gpu_ids = parse_csv_ints(args.gpus)
    if not gpu_ids:
        raise SystemExit("At least one GPU id is required.")

    run_root = Path(args.output_dir) / args.experiment_name
    log_dir = run_root / "logs"
    metrics_dir = run_root / "metrics"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    launched: list[tuple[int, int, Path, Path, str]] = []
    for idx, fold_id in enumerate(folds):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        log_path = log_dir / f"fold_{fold_id}.log"
        metrics_path = metrics_dir / f"fold_{fold_id}.json"
        wandb_name = f"{args.experiment_name}_fold{fold_id}"
        cmd = build_command(
            python_bin=args.python_bin,
            train_script=args.train_script,
            config=args.config,
            trainer=args.trainer,
            data_dir=args.data_dir,
            fold_id=fold_id,
            metrics_path=metrics_path,
            extra_args=args.extra_arg,
        )
        env_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} WANDB_PROJECT={shlex.quote(args.wandb_project)} WANDB_NAME={shlex.quote(wandb_name)}"
        shell_cmd = f"cd /u50/rajguruu/WildfireSpreadPrediction && {env_prefix} nohup {' '.join(shlex.quote(part) for part in cmd)} > {shlex.quote(str(log_path))} 2>&1 &"
        launched.append((fold_id, gpu_id, log_path, metrics_path, shell_cmd))

    for fold_id, gpu_id, log_path, metrics_path, shell_cmd in launched:
        print(f"fold={fold_id} gpu={gpu_id}")
        print(f"  log: {log_path}")
        print(f"  metrics: {metrics_path}")
        print(f"  cmd: {shell_cmd}")
        if args.launch:
            subprocess.run(["bash", "-lc", shell_cmd], check=True)

    print("\nSummary")
    print(f"  run_root: {run_root}")
    print(f"  logs: {log_dir}")
    print(f"  metrics: {metrics_dir}")
    if args.launch:
        print("  next: run summarize_wsts_cv.py on the metrics directory after folds finish")
    else:
        print("  note: this was a preview only; add --launch to actually start jobs")


if __name__ == "__main__":
    main()
