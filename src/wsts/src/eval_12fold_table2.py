import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, pstdev


def load_fold_metrics(metrics_path: Path, fold_id: int) -> dict:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if "test_AP" not in metrics:
        raise RuntimeError(f"Fold {fold_id} metrics missing 'test_AP': {metrics}")

    return metrics


def run_fold(config_path: Path, fold_id: int, output_dir: Path, extra_args: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"fold_{fold_id}_metrics.json"
    log_path = output_dir / f"fold_{fold_id}.log"

    cmd = [
        sys.executable,
        "src/train.py",
        "--config",
        str(config_path),
        f"--data.data_fold_id={fold_id}",
        "--do_train=true",
        "--do_validate=true",
        "--do_test=true",
        f"--metrics_output_path={metrics_path}",
    ] + extra_args

    print(f"[Fold {fold_id}] Running: {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    if not metrics_path.exists():
        raise RuntimeError(f"Fold {fold_id} did not produce metrics file: {metrics_path}")

    return load_fold_metrics(metrics_path, fold_id)


def main():
    parser = argparse.ArgumentParser(description="Run 12-fold evaluation and compute mean test AP.")
    parser.add_argument("--config", type=str, required=True, help="Path to Lightning CLI config YAML")
    parser.add_argument("--num_folds", type=int, default=12, help="Number of folds (default: 12)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./table2_eval_outputs",
        help="Directory for fold logs and metrics",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=20,
        help="Max retries per fold before giving up (default: 20)",
    )
    parser.add_argument(
        "--retry_delay_seconds",
        type=int,
        default=30,
        help="Delay between retries in seconds (default: 30)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode and re-run folds even if metrics already exist",
    )
    args, unknown = parser.parse_known_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    resume = not args.no_resume
    fold_results = []
    for fold_id in range(args.num_folds):
        metrics_path = output_dir / f"fold_{fold_id}_metrics.json"

        # Check if metrics exist before each attempt
        if resume and metrics_path.exists():
            print(f"[Fold {fold_id}] Found existing metrics. Skipping training and loading from disk.")
            metrics = load_fold_metrics(metrics_path, fold_id)
        else:
            last_error = None
            for attempt in range(1, args.max_retries + 1):
                # Re-check for metrics at start of each retry (in case previous attempt wrote it before crashing)
                if resume and metrics_path.exists():
                    print(f"[Fold {fold_id}] Metrics found during retry attempt {attempt}. Loading from disk.")
                    metrics = load_fold_metrics(metrics_path, fold_id)
                    last_error = None
                    break
                
                try:
                    print(f"[Fold {fold_id}] Attempt {attempt}/{args.max_retries}")
                    metrics = run_fold(config_path, fold_id, output_dir, unknown)
                    last_error = None
                    break
                except Exception as err:
                    last_error = err
                    print(f"[Fold {fold_id}] Attempt {attempt} failed: {err}")
                    if attempt < args.max_retries:
                        print(
                            f"[Fold {fold_id}] Retrying in {args.retry_delay_seconds}s..."
                        )
                        time.sleep(args.retry_delay_seconds)

            if last_error is not None:
                raise RuntimeError(
                    f"Fold {fold_id} failed after {args.max_retries} attempts"
                ) from last_error

        fold_results.append(
            {
                "fold": fold_id,
                "test_AP": float(metrics["test_AP"]),
                "test_f1": float(metrics.get("test_f1", 0.0)),
                "test_loss": float(metrics.get("test_loss", 0.0)),
            }
        )
        print(f"[Fold {fold_id}] test_AP={metrics['test_AP']:.6f}")

    ap_values = [r["test_AP"] for r in fold_results]
    summary = {
        "num_folds": args.num_folds,
        "mean_test_AP": mean(ap_values),
        "std_test_AP": pstdev(ap_values),
        "fold_results": fold_results,
    }

    summary_path = output_dir / "table2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Table 2 Comparison Summary ===")
    print(f"Mean test AP over {args.num_folds} folds: {summary['mean_test_AP']:.6f}")
    print(f"Std  test AP over {args.num_folds} folds: {summary['std_test_AP']:.6f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
