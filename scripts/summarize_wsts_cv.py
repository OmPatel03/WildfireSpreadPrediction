#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize per-fold WSTS metrics JSON files into mean/std results."
    )
    parser.add_argument(
        "metrics_dir",
        help="Directory containing fold_*.json files produced by train.py --metrics_output_path.",
    )
    parser.add_argument(
        "--metric",
        default="test_AP",
        help="Primary metric to summarize first. Default: test_AP",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    files = sorted(metrics_dir.glob("fold_*.json"))
    if not files:
        raise SystemExit(f"No fold_*.json files found in {metrics_dir}")

    rows: list[tuple[str, dict]] = []
    for path in files:
        with open(path) as f:
            rows.append((path.stem, json.load(f)))

    keys = sorted({key for _, row in rows for key in row.keys()})
    primary_values = []
    print("Per-fold metrics")
    for fold_name, row in rows:
        primary = row.get(args.metric)
        primary_values.append(float(primary))
        print(f"  {fold_name}: {args.metric}={primary:.6f}")

    print("\nSummary")
    print(
        f"  {args.metric}: mean={mean(primary_values):.6f}, std={pstdev(primary_values):.6f}, "
        f"min={min(primary_values):.6f}, max={max(primary_values):.6f}, n={len(primary_values)}"
    )

    print("\nAll metrics")
    for key in keys:
        values = [float(row[key]) for _, row in rows if key in row]
        if not values:
            continue
        std = pstdev(values) if len(values) > 1 else 0.0
        print(
            f"  {key}: mean={mean(values):.6f}, std={std:.6f}, min={min(values):.6f}, max={max(values):.6f}"
        )


if __name__ == "__main__":
    main()
