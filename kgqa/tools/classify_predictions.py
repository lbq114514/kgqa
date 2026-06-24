"""Classify sample prediction JSON files by exact-match and recall buckets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Classify sample prediction JSON files and move them into bucket directories."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing per-sample prediction JSON files",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write classification outputs. Defaults to <input_dir>/classified",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern used to collect prediction files",
    )
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, object]:
    """Load one prediction JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def classify_bucket(metrics: dict[str, object]) -> str:
    """Map metrics to one of the three requested buckets."""
    exact_match = float(metrics.get("exact_match", 0.0))
    recall = float(metrics.get("recall", 0.0))
    if exact_match == 1.0:
        return "exact_match"
    if recall > 0.0:
        return "recall_nonzero"
    return "recall_zero"


def build_record(path: Path, payload: dict[str, object], bucket: str) -> dict[str, object]:
    """Build a compact record for output manifests."""
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "bucket": bucket,
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "sample_id": payload.get("sample_id"),
        "question": payload.get("question"),
        "exact_match": metrics.get("exact_match"),
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "predicted_answers": payload.get("predicted_answers"),
        "gold_answers": payload.get("gold_answers"),
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_name_list(path: Path, records: list[dict[str, object]]) -> None:
    """Write one filename per line for quick inspection."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record['file_name']}\n")


def main() -> None:
    """Classify prediction files, move them into bucket dirs, and write manifests."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "classified"
    output_dir.mkdir(parents=True, exist_ok=True)

    bucketed: dict[str, list[tuple[Path, dict[str, object]]]] = {
        "exact_match": [],
        "recall_nonzero": [],
        "recall_zero": [],
    }
    skipped: list[dict[str, object]] = []

    for path in sorted(input_dir.glob(args.glob)):
        try:
            payload = load_prediction(path)
            metrics = payload.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError("metrics is missing or not an object")
            bucket = classify_bucket(metrics)
            bucketed[bucket].append((path, payload))
        except Exception as exc:  # pragma: no cover - best effort CLI handling
            skipped.append({"file_name": path.name, "file_path": str(path.resolve()), "error": str(exc)})

    moved_records: dict[str, list[dict[str, object]]] = {bucket: [] for bucket in bucketed}
    for bucket, entries in bucketed.items():
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for source_path, payload in entries:
            destination_path = bucket_dir / source_path.name
            if destination_path.exists():
                destination_path.unlink()
            shutil.move(str(source_path), str(destination_path))
            moved_records[bucket].append(build_record(destination_path, payload, bucket))

    for bucket, records in moved_records.items():
        write_jsonl(output_dir / f"{bucket}.jsonl", records)
        write_name_list(output_dir / f"{bucket}.txt", records)

    summary = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_files": sum(len(records) for records in moved_records.values()) + len(skipped),
        "classified_files": sum(len(records) for records in moved_records.values()),
        "skipped_files": len(skipped),
        "counts": {bucket: len(records) for bucket, records in moved_records.items()},
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
