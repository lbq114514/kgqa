"""Analyze gold-answer coverage in WebQSP sample graphs and candidate paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.datasets.webqsp import load_webqsp_samples
from kgqa.utils.text import normalize_text
from kgqa.utils.types import WebQSPSample


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Check how many gold answers appear in the provided graph and in candidate paths."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--predictions-dir",
        help="Directory containing per-sample prediction JSON files. Defaults to evaluation.output_dir/sample_output_dir.",
    )
    parser.add_argument("--limit", type=int, help="Only analyze the first N prediction files after sorting by sample_id.")
    parser.add_argument("--sample-id", help="Analyze a single sample_id only.")
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Defaults to outputs/webqsp_validation/gold_answer_coverage.json.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_base_dir(config_path: str) -> Path:
    """Return the directory relative config paths should resolve from."""
    return Path(config_path).resolve().parent


def resolve_predictions_dir(config: dict[str, Any], config_path: str, override: str | None) -> Path:
    """Resolve the directory containing per-sample prediction JSON files."""
    if override:
        path = Path(override)
        return path if path.is_absolute() else config_base_dir(config_path) / path
    output_dir = config.get("evaluation", {}).get("output_dir", "outputs/webqsp_validation")
    sample_dir = config.get("evaluation", {}).get("sample_output_dir", "sample_predictions")
    return config_base_dir(config_path) / output_dir / sample_dir


def resolve_output_path(config: dict[str, Any], config_path: str, override: str | None) -> Path:
    """Resolve the analysis JSON output path."""
    if override:
        path = Path(override)
        return path if path.is_absolute() else config_base_dir(config_path) / path
    output_dir = config.get("evaluation", {}).get("output_dir", "outputs/webqsp_validation")
    return config_base_dir(config_path) / output_dir / "gold_answer_coverage.json"


def resolve_validation_path(config: dict[str, Any], config_path: str) -> Path:
    """Resolve the configured WebQSP validation path."""
    dataset_path = Path(config["datasets"]["webqsp"]["validation_path"])
    if dataset_path.is_absolute():
        return dataset_path
    return config_base_dir(config_path) / dataset_path


def load_prediction(path: Path) -> dict[str, Any]:
    """Load one per-sample prediction JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_inline_triple(value: Any) -> bool:
    """Return True if the value is a simple triple dict that should stay on one line."""
    return (
        isinstance(value, dict)
        and tuple(value.keys()) == ("head", "relation", "tail")
        and all(isinstance(value[key], str) for key in ("head", "relation", "tail"))
    )


def format_pretty_value(value: Any, indent: int = 0) -> str:
    """Format JSON-like data with readable indentation and inline triple objects."""
    current_indent = " " * indent
    next_indent = " " * (indent + 2)

    if is_inline_triple(value):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f'{next_indent}{json.dumps(key, ensure_ascii=False)}: {format_pretty_value(item, indent + 2)}'
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{current_indent}" + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [f"{next_indent}{format_pretty_value(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{current_indent}" + "]"
    return json.dumps(value, ensure_ascii=False)


def collect_graph_values(sample: WebQSPSample) -> set[str]:
    """Collect normalized head/tail values from the sample-local graph."""
    values: set[str] = set()
    for triple in sample.graph_triples:
        values.add(normalize_text(triple.head))
        values.add(normalize_text(triple.tail))
    return values


def collect_candidate_values(prediction: dict[str, Any]) -> set[str]:
    """Collect normalized head/tail values from candidate path triples."""
    values: set[str] = set()
    for path in prediction.get("candidate_paths", []):
        for triple in path.get("triples", []):
            head = triple.get("head")
            tail = triple.get("tail")
            if isinstance(head, str):
                values.add(normalize_text(head))
            if isinstance(tail, str):
                values.add(normalize_text(tail))
    return values


def collect_candidate_paths_with_gold_hits(
    prediction: dict[str, Any],
    gold_answers: list[str],
) -> list[dict[str, Any]]:
    """Collect candidate-path snippets that contain one or more gold answers."""
    normalized_gold = {normalize_text(answer): answer for answer in gold_answers}
    hit_paths: list[dict[str, Any]] = []
    for index, path in enumerate(prediction.get("candidate_paths", [])):
        hit_answers: list[str] = []
        for triple in path.get("triples", []):
            for field in ("head", "tail"):
                value = triple.get(field)
                if not isinstance(value, str):
                    continue
                normalized_value = normalize_text(value)
                if normalized_value in normalized_gold and normalized_gold[normalized_value] not in hit_answers:
                    hit_answers.append(normalized_gold[normalized_value])
        if hit_answers:
            hit_paths.append(
                {
                    "candidate_index": index,
                    "matched_gold_answers": hit_answers,
                    "text": path.get("text", ""),
                    "source_stage": path.get("source_stage", ""),
                    "pruning_status": path.get("pruning_status", ""),
                }
            )
    return hit_paths


def collect_graph_hit_one_hop_subgraph(
    sample: WebQSPSample,
    graph_hits: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Collect 1-hop subgraphs around each graph-hit answer."""
    subgraphs: dict[str, list[dict[str, str]]] = {}
    for answer in graph_hits:
        normalized_answer = normalize_text(answer)
        triples: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for triple in sample.graph_triples:
            if normalize_text(triple.head) != normalized_answer and normalize_text(triple.tail) != normalized_answer:
                continue
            triple_key = (triple.head, triple.relation, triple.tail)
            if triple_key in seen:
                continue
            seen.add(triple_key)
            triples.append(
                {
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                }
            )
        subgraphs[answer] = triples
    return subgraphs


def analyze_sample(sample: WebQSPSample, prediction: dict[str, Any]) -> dict[str, Any]:
    """Analyze graph and candidate-path gold-answer coverage for one sample."""
    normalized_gold = {normalize_text(answer): answer for answer in sample.answers if normalize_text(answer)}
    graph_values = collect_graph_values(sample)
    candidate_values = collect_candidate_values(prediction)

    graph_hits = [answer for key, answer in normalized_gold.items() if key in graph_values]
    candidate_hits = [answer for key, answer in normalized_gold.items() if key in candidate_values]
    candidate_hit_paths = collect_candidate_paths_with_gold_hits(prediction, sample.answers)
    graph_hit_one_hop_subgraph = collect_graph_hit_one_hop_subgraph(sample, graph_hits)

    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "gold_answers": sample.answers,
        "gold_answer_count": len(sample.answers),
        "graph_hit_count": len(graph_hits),
        "graph_hit_rate": 0.0 if not sample.answers else len(graph_hits) / len(sample.answers),
        "graph_hits": graph_hits,
        "graph_misses": [answer for answer in sample.answers if answer not in graph_hits],
        "graph_hit_one_hop_subgraph": graph_hit_one_hop_subgraph,
        "candidate_hit_count": len(candidate_hits),
        "candidate_hit_rate": 0.0 if not sample.answers else len(candidate_hits) / len(sample.answers),
        "candidate_hits": candidate_hits,
        "candidate_misses": [answer for answer in sample.answers if answer not in candidate_hits],
        "candidate_hit_path_count": len(candidate_hit_paths),
        "candidate_hit_paths": candidate_hit_paths,
    }


def aggregate_results(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate coverage statistics across all analyzed samples."""
    total_samples = len(samples)
    total_gold_answers = sum(item["gold_answer_count"] for item in samples)
    total_graph_hits = sum(item["graph_hit_count"] for item in samples)
    total_candidate_hits = sum(item["candidate_hit_count"] for item in samples)
    samples_with_full_graph_hit = sum(1 for item in samples if item["graph_hit_count"] == item["gold_answer_count"])
    samples_with_full_candidate_hit = sum(
        1 for item in samples if item["candidate_hit_count"] == item["gold_answer_count"]
    )
    samples_with_zero_graph_hit = sum(1 for item in samples if item["graph_hit_count"] == 0)
    samples_with_zero_candidate_hit = sum(1 for item in samples if item["candidate_hit_count"] == 0)

    return {
        "num_samples": total_samples,
        "total_gold_answers": total_gold_answers,
        "graph_hit_count": total_graph_hits,
        "graph_hit_rate": 0.0 if total_gold_answers == 0 else total_graph_hits / total_gold_answers,
        "candidate_hit_count": total_candidate_hits,
        "candidate_hit_rate": 0.0 if total_gold_answers == 0 else total_candidate_hits / total_gold_answers,
        "samples_with_full_graph_hit": samples_with_full_graph_hit,
        "samples_with_full_candidate_hit": samples_with_full_candidate_hit,
        "samples_with_zero_graph_hit": samples_with_zero_graph_hit,
        "samples_with_zero_candidate_hit": samples_with_zero_candidate_hit,
    }


def main() -> None:
    """Run coverage analysis and write a JSON report."""
    args = parse_args()
    config = load_config(args.config)
    validation_path = resolve_validation_path(config, args.config)
    predictions_dir = resolve_predictions_dir(config, args.config, args.predictions_dir)
    output_path = resolve_output_path(config, args.config, args.output)

    samples = {sample.sample_id: sample for sample in load_webqsp_samples(validation_path)}
    prediction_files = sorted(predictions_dir.glob("*.json"))
    if args.sample_id:
        prediction_files = [path for path in prediction_files if path.stem == args.sample_id]
    if args.limit is not None:
        prediction_files = prediction_files[: args.limit]

    analyses: list[dict[str, Any]] = []
    for path in prediction_files:
        prediction = load_prediction(path)
        sample_id = str(prediction.get("sample_id", path.stem))
        sample = samples.get(sample_id)
        if sample is None:
            continue
        analyses.append(analyze_sample(sample, prediction))

    report = {
        "validation_path": str(validation_path),
        "predictions_dir": str(predictions_dir),
        "summary": aggregate_results(analyses),
        "samples": analyses,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(format_pretty_value(report))
        handle.write("\n")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved detailed report to {output_path}")


if __name__ == "__main__":
    main()
