from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "good_fire_predictions_2021.csv"
CSV_COLUMNS = [
    "fire_id",
    "year",
    "sample_index",
    "target_date",
    "threshold",
    "ground_truth_pixels",
    "positive_pixels",
    "precision",
    "recall",
    "f1",
    "accuracy",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan wildfire predictions for a year and export fires whose latest-sample "
            "prediction F1 exceeds a cutoff and whose ground truth has at least a "
            "minimum number of positive pixels."
        )
    )
    parser.add_argument("--year", type=int, default=2021, help="Year to evaluate.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction probability threshold used to create the binary mask.",
    )
    parser.add_argument(
        "--f1-cutoff",
        type=float,
        default=0.55,
        help="Keep only fires whose latest-sample F1 is strictly greater than this value.",
    )
    parser.add_argument(
        "--min-ground-truth-pixels",
        type=int,
        default=15,
        help="Minimum latest-sample ground-truth positive pixels required.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path.",
    )
    return parser.parse_args(argv)


def iter_good_prediction_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, int]]:
    from wildfire_api import WildfireService, get_settings
    from wildfire_api.geojson import build_prediction_summary
    from wildfire_api.repository import WildfireRepository

    settings = get_settings()
    repository = WildfireRepository(settings)
    service = WildfireService(settings)

    metadata_rows = repository.list_year(args.year)
    eligible_rows = [
        metadata
        for metadata in metadata_rows
        if metadata.latest_target_positive_pixels >= args.min_ground_truth_pixels
    ]

    rows: list[dict[str, object]] = []
    successful_inferences = 0
    failures = 0

    for metadata in eligible_rows:
        try:
            prediction, _ = service.find_spread(
                fire_id=metadata.fire_id,
                year=args.year,
                sample_offset=-1,
                threshold=args.threshold,
            )
            successful_inferences += 1
            summary = build_prediction_summary(prediction)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(
                f"[warn] failed to evaluate {metadata.fire_id}: {exc}",
                file=sys.stderr,
            )
            continue

        if summary["groundTruthPixels"] < args.min_ground_truth_pixels:
            continue
        if summary["f1"] <= args.f1_cutoff:
            continue

        rows.append(
            {
                "fire_id": metadata.fire_id,
                "year": metadata.year,
                "sample_index": prediction.sample_index,
                "target_date": prediction.target_date or "",
                "threshold": prediction.threshold,
                "ground_truth_pixels": summary["groundTruthPixels"],
                "positive_pixels": summary["positivePixels"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "accuracy": summary["accuracy"],
            }
        )

    counts = {
        "total_fires": len(metadata_rows),
        "eligible_fires": len(eligible_rows),
        "successful_inferences": successful_inferences,
        "good_predictions": len(rows),
        "failures": failures,
    }
    return rows, counts


def write_rows(output_path: Path, rows: Iterable[dict[str, object]]) -> None:
    resolved = output_path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, counts = iter_good_prediction_rows(args)
    write_rows(args.output, rows)

    print(f"Output: {args.output.expanduser()}")
    print(f"Total fires seen: {counts['total_fires']}")
    print(f"Eligible after ground-truth filter: {counts['eligible_fires']}")
    print(f"Successful inferences: {counts['successful_inferences']}")
    print(f"Good predictions written: {counts['good_predictions']}")
    print(f"Failures skipped: {counts['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
