"""Command-line entrypoint for the KGQA demo pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import yaml

from kgqa.datasets.cwq import load_cwq_samples
from kgqa.datasets.webqsp import build_sample_kg, load_webqsp_samples
from kgqa.evaluation.webqsp import aggregate_validation_metrics, build_validation_prediction
from kgqa.kg.graph import KnowledgeGraph
from kgqa.kg.loader import load_knowledge_graph
from kgqa.llm.vllm_client import VLLMLLM
from kgqa.reasoning.pipeline import KGQAPipeline
from kgqa.utils.logging import configure_logging, get_logger
from kgqa.utils.types import CWQSample, ValidationPrediction, WebQSPSample


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_llm(config: dict[str, Any]) -> VLLMLLM:
    """Build the configured local vLLM client."""
    llm_config = config["llm"]
    provider = llm_config.get("provider", "vllm")
    if provider != "vllm":
        raise ValueError(f"Unsupported llm.provider={provider!r}. Only 'vllm' is supported.")
    return VLLMLLM(
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        api_key=llm_config.get("api_key", "EMPTY"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 1024),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Training-free PoG-style KGQA")
    parser.add_argument("--question", help="Natural-language question to answer.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--dataset", choices=["webqsp", "cwq"], help="Dataset integration to run.")
    parser.add_argument("--split", default=None, help="Dataset split name.")
    parser.add_argument("--mode", choices=["single", "validate"], help="Dataset execution mode.")
    parser.add_argument("--index", type=int, help="Sample index for single-sample dataset mode.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of samples for batch validation.")
    parser.add_argument("--qa-concurrency", type=int, help="Number of QA samples to run concurrently in validation mode.")
    parser.add_argument("--resume", action="store_true", help="Resume validation from existing per-sample outputs.")
    parser.add_argument("--graphapi-enabled", action="store_true", help="Enable optional external graphapi augmentation.")
    parser.add_argument("--use-q-entities", action="store_true", help="Use dataset q_entity values as topic entity seeds.")
    parser.add_argument(
        "--compact-precise-path-prompt",
        action="store_true",
        help="Use compact path strings in PRECISE_PATH_SELECTION_PROMPT for ablation.",
    )
    return parser.parse_args()


def _config_base_dir(config_path: str) -> Path:
    """Return the directory that relative config paths should resolve from."""
    return Path(config_path).resolve().parent


def _resolve_output_dir(config: dict[str, Any], config_path: str) -> Path:
    """Resolve the configured evaluation output directory."""
    output_dir = config.get("evaluation", {}).get("output_dir", "outputs/webqsp_validation")
    return _config_base_dir(config_path) / output_dir


def _resolve_sample_output_dir(config: dict[str, Any], config_path: str) -> Path:
    """Resolve the directory for per-sample pretty prediction files."""
    output_dir = _resolve_output_dir(config, config_path)
    sample_dir_name = config.get("evaluation", {}).get("sample_output_dir", "sample_predictions")
    return output_dir / sample_dir_name


def _resolve_qa_concurrency(config: dict[str, Any], cli_override: int | None = None) -> int:
    """Resolve the configured QA concurrency for validation runs."""
    if cli_override is not None:
        return max(1, cli_override)
    return max(1, int(config.get("evaluation", {}).get("qa_concurrency", 1)))


def _resolve_graphapi_db_path(config: dict[str, Any], config_path: str) -> None:
    """Resolve the configured graphapi SQLite database path in-place."""
    graphapi_cfg = config.get("graphapi")
    if not isinstance(graphapi_cfg, dict):
        graphapi_cfg = {}
    for key in ("db_path", "fb2m_db_path"):
        raw_db_path = str(graphapi_cfg.get(key, "")).strip()
        if not raw_db_path:
            continue
        db_path = Path(raw_db_path)
        if not db_path.is_absolute():
            graphapi_cfg[key] = str((_config_base_dir(config_path) / db_path).resolve())
    retrieval_cfg = config.get("retrieval")
    if not isinstance(retrieval_cfg, dict):
        return
    raw_index_dir = str(retrieval_cfg.get("index_dir", "")).strip()
    if raw_index_dir:
        index_dir = Path(raw_index_dir)
        if not index_dir.is_absolute():
            retrieval_cfg["index_dir"] = str((_config_base_dir(config_path) / index_dir).resolve())


def _webqsp_uses_external_graph(config: dict[str, Any]) -> bool:
    """Return whether WebQSP dataset execution should use the external SQLite graph only."""
    dataset_cfg = config.get("datasets", {}).get("webqsp", {})
    return _graphapi_is_enabled(config) and bool(dataset_cfg.get("external_graph_only", True))


def _apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply optional runtime overrides without changing default behavior."""
    if args.graphapi_enabled:
        config.setdefault("graphapi", {})
        config["graphapi"]["enabled"] = True
    if args.use_q_entities:
        config.setdefault("datasets", {}).setdefault("webqsp", {})
        config["datasets"]["webqsp"]["use_q_entities_for_topic_entities"] = True
    if args.compact_precise_path_prompt:
        config.setdefault("pruning", {})
        config["pruning"]["compact_precise_path_prompt"] = True


def _is_inline_triple(value: Any) -> bool:
    """Return True if the value is a simple triple dict that should stay on one line."""
    return (
        isinstance(value, dict)
        and tuple(value.keys()) == ("head", "relation", "tail")
        and all(isinstance(value[key], str) for key in ("head", "relation", "tail"))
    )


def _format_pretty_value(value: Any, indent: int = 0) -> str:
    """Format JSON-like data with readable indentation and inline triple objects."""
    current_indent = " " * indent
    next_indent = " " * (indent + 2)

    if _is_inline_triple(value):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f'{next_indent}{json.dumps(key, ensure_ascii=False)}: {_format_pretty_value(item, indent + 2)}'
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{current_indent}" + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [f"{next_indent}{_format_pretty_value(item, indent + 2)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{current_indent}" + "]"
    return json.dumps(value, ensure_ascii=False)


def _ordered_prediction_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Reorder prediction fields so the most important information appears first."""
    preferred_order = [
        "sample_id",
        "question",
        "predicted_answers",
        "gold_answers",
        "answer",
        "metrics",
        "sufficient",
        "confidence",
        "topic_entities",
        "question_analysis",
        "exploration_hints",
        "supplement_hints",
        "summarized_paths",
        "pruned_paths",
        "candidate_paths",
        "search_trace",
    ]
    ordered: dict[str, Any] = {}
    for key in preferred_order:
        if key in row:
            ordered[key] = row[key]
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _write_sample_prediction(path: Path, row: dict[str, Any]) -> None:
    """Write one pretty per-sample prediction file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_row = _ordered_prediction_payload(row)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(_format_pretty_value(ordered_row))
        handle.write("\n")


def _load_sample_prediction(path: Path) -> ValidationPrediction:
    """Load one per-sample prediction JSON file into a ValidationPrediction object."""
    with path.open("r", encoding="utf-8") as handle:
        row = json.load(handle)
    return ValidationPrediction(
        sample_id=str(row.get("sample_id", "")),
        question=str(row.get("question", "")),
        gold_answers=[str(item) for item in row.get("gold_answers", [])],
        predicted_answers=[str(item) for item in row.get("predicted_answers", [])],
        answer=str(row.get("answer", "")),
        sufficient=bool(row.get("sufficient", False)),
        confidence=str(row.get("confidence", "")),
        topic_entities=[str(item) for item in row.get("topic_entities", [])],
        question_analysis=row.get("question_analysis", {}) if isinstance(row.get("question_analysis", {}), dict) else {},
        exploration_hints=row.get("exploration_hints", {}) if isinstance(row.get("exploration_hints", {}), dict) else {},
        supplement_hints=row.get("supplement_hints", {}) if isinstance(row.get("supplement_hints", {}), dict) else {},
        candidate_paths=row.get("candidate_paths", []) if isinstance(row.get("candidate_paths", []), list) else [],
        pruned_paths=row.get("pruned_paths", []) if isinstance(row.get("pruned_paths", []), list) else [],
        summarized_paths=row.get("summarized_paths", []) if isinstance(row.get("summarized_paths", []), list) else [],
        search_trace=row.get("search_trace", []) if isinstance(row.get("search_trace", []), list) else [],
        metrics=row.get("metrics", {}) if isinstance(row.get("metrics", {}), dict) else {},
    )


def _run_single_question(config: dict[str, Any], config_path: str, question: str) -> None:
    """Run the original sample-KG single-question pipeline."""
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    logger = get_logger(__name__)
    kg_path = _config_base_dir(config_path) / config["kg"]["path"]
    logger.info("Loading knowledge graph from %s", kg_path)
    kg = load_knowledge_graph(kg_path)
    llm = build_llm(config)
    pipeline = KGQAPipeline(kg=kg, llm=llm, config=config)
    result = pipeline.run(question)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def _load_webqsp_config(
    config: dict[str, Any],
    config_path: str,
    split: str | None,
) -> tuple[list[WebQSPSample], str]:
    """Load configured WebQSP samples for the selected split."""
    dataset_cfg = config["datasets"]["webqsp"]
    selected_split = split or dataset_cfg.get("default_split", "validation")
    if selected_split != "validation":
        raise ValueError(f"Unsupported WebQSP split {selected_split!r}. Only 'validation' is implemented.")
    dataset_path = Path(dataset_cfg["validation_path"])
    if not dataset_path.is_absolute():
        dataset_path = _config_base_dir(config_path) / dataset_path
    samples = load_webqsp_samples(dataset_path)
    return samples, selected_split


def _graphapi_is_enabled(config: dict[str, Any]) -> bool:
    """Return whether graphapi is enabled in the runtime config."""
    return bool(config.get("graphapi", {}).get("enabled", False))


def _validate_cwq_runtime_options(config: dict[str, Any], use_q_entities: bool) -> None:
    """Validate CWQ external-only runtime constraints."""
    if use_q_entities:
        raise ValueError("CWQ external-only mode does not support --use-q-entities.")
    if not _graphapi_is_enabled(config):
        raise ValueError("CWQ external-only mode requires graphapi.enabled=true.")


def _load_cwq_config(
    config: dict[str, Any],
    config_path: str,
    split: str | None,
) -> tuple[list[CWQSample], str]:
    """Load configured CWQ samples for the selected split."""
    dataset_cfg = config["datasets"]["cwq"]
    selected_split = split or dataset_cfg.get("default_split", "test")
    if selected_split != "test":
        raise ValueError(f"Unsupported CWQ split {selected_split!r}. Only 'test' is implemented.")
    dataset_path = Path(dataset_cfg["test_path"])
    if not dataset_path.is_absolute():
        dataset_path = _config_base_dir(config_path) / dataset_path
    samples = load_cwq_samples(dataset_path)
    return samples, selected_split


def _run_webqsp_single(config: dict[str, Any], config_path: str, split: str | None, index: int) -> None:
    """Run the pipeline for one WebQSP validation sample."""
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    samples, selected_split = _load_webqsp_config(config, config_path, split)
    if index < 0 or index >= len(samples):
        raise IndexError(f"Sample index {index} is out of range for WebQSP {selected_split} ({len(samples)} samples).")
    sample = samples[index]
    llm = build_llm(config)
    kg = KnowledgeGraph([]) if _webqsp_uses_external_graph(config) else build_sample_kg(sample)
    pipeline = KGQAPipeline(kg=kg, llm=llm, config=config)
    use_q_entities = bool(config.get("datasets", {}).get("webqsp", {}).get("use_q_entities_for_topic_entities", False))
    result = pipeline.run_with_metadata(
        question=sample.question,
        sample_id=sample.sample_id,
        gold_answers=sample.answers,
        dataset_name="webqsp",
        entity_source="provided" if use_q_entities else "llm",
        topic_entities_override=sample.q_entities if use_q_entities else None,
    )
    prediction = build_validation_prediction(result)
    print(_format_pretty_value(_ordered_prediction_payload(prediction.to_dict())))


def _run_webqsp_sample(config: dict[str, Any], sample: WebQSPSample) -> Any:
    """Run one WebQSP sample through the pipeline and return its validation prediction."""
    llm = build_llm(config)
    kg = KnowledgeGraph([]) if _webqsp_uses_external_graph(config) else build_sample_kg(sample)
    pipeline = KGQAPipeline(kg=kg, llm=llm, config=config)
    use_q_entities = bool(config.get("datasets", {}).get("webqsp", {}).get("use_q_entities_for_topic_entities", False))
    result = pipeline.run_with_metadata(
        question=sample.question,
        sample_id=sample.sample_id,
        gold_answers=sample.answers,
        dataset_name="webqsp",
        entity_source="provided" if use_q_entities else "llm",
        topic_entities_override=sample.q_entities if use_q_entities else None,
    )
    return build_validation_prediction(result)


def _run_webqsp_validation(
    config: dict[str, Any],
    config_path: str,
    split: str | None,
    limit: int | None,
    qa_concurrency: int | None = None,
    resume: bool = False,
) -> None:
    """Run batch validation over WebQSP and save predictions plus metrics."""
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    logger = get_logger(__name__)
    samples, selected_split = _load_webqsp_config(config, config_path, split)
    if limit is not None:
        samples = samples[:limit]

    concurrency = _resolve_qa_concurrency(config, qa_concurrency)
    output_dir = _resolve_output_dir(config, config_path)
    sample_output_dir = _resolve_sample_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for existing_file in sample_output_dir.glob("*.json"):
            existing_file.unlink()

    predictions: list[Any] = []
    samples_to_run: list[WebQSPSample] = []
    if resume:
        for sample in samples:
            sample_path = sample_output_dir / f"{sample.sample_id}.json"
            if not sample_path.exists():
                samples_to_run.append(sample)
                continue
            try:
                predictions.append(_load_sample_prediction(sample_path))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to load existing prediction for %s: %s. Re-running sample.", sample.sample_id, exc)
                samples_to_run.append(sample)
        logger.info(
            "Resuming WebQSP validation on %d samples with qa_concurrency=%d (%d loaded, %d remaining)",
            len(samples),
            concurrency,
            len(predictions),
            len(samples_to_run),
        )
    else:
        samples_to_run = list(samples)
        logger.info(
            "Running WebQSP validation on %d samples with qa_concurrency=%d",
            len(samples_to_run),
            concurrency,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(_run_webqsp_sample, config, sample): index
            for index, sample in enumerate(samples_to_run)
        }
        for future in as_completed(future_to_index):
            prediction = future.result()
            predictions.append(prediction)
            prediction_row = prediction.to_dict()
            _write_sample_prediction(sample_output_dir / f"{prediction.sample_id}.json", prediction_row)
            logger.info(
                "Completed and saved WebQSP sample %s (%d/%d)",
                prediction.sample_id,
                len(predictions),
                len(samples),
            )

    metrics = aggregate_validation_metrics(predictions, dataset="webqsp", split=selected_split)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, ensure_ascii=False, indent=2)

    logger.info("Saved per-sample WebQSP predictions to %s", sample_output_dir)
    logger.info("Saved validation metrics to %s", metrics_path)
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))


def _run_cwq_single(config: dict[str, Any], config_path: str, split: str | None, index: int) -> None:
    """Run the pipeline for one CWQ sample in external-only mode."""
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    _validate_cwq_runtime_options(
        config,
        use_q_entities=bool(config.get("datasets", {}).get("webqsp", {}).get("use_q_entities_for_topic_entities", False)),
    )
    samples, selected_split = _load_cwq_config(config, config_path, split)
    if index < 0 or index >= len(samples):
        raise IndexError(f"Sample index {index} is out of range for CWQ {selected_split} ({len(samples)} samples).")
    sample = samples[index]
    llm = build_llm(config)
    pipeline = KGQAPipeline(kg=KnowledgeGraph([]), llm=llm, config=config)
    result = pipeline.run_with_metadata(
        question=sample.question,
        sample_id=sample.sample_id,
        gold_answers=sample.answers,
        dataset_name="cwq",
    )
    prediction = build_validation_prediction(result)
    print(_format_pretty_value(_ordered_prediction_payload(prediction.to_dict())))


def _run_cwq_sample(config: dict[str, Any], sample: CWQSample) -> Any:
    """Run one CWQ sample through the pipeline and return its validation prediction."""
    llm = build_llm(config)
    pipeline = KGQAPipeline(kg=KnowledgeGraph([]), llm=llm, config=config)
    result = pipeline.run_with_metadata(
        question=sample.question,
        sample_id=sample.sample_id,
        gold_answers=sample.answers,
        dataset_name="cwq",
    )
    return build_validation_prediction(result)


def _run_cwq_validation(
    config: dict[str, Any],
    config_path: str,
    split: str | None,
    limit: int | None,
    qa_concurrency: int | None = None,
    resume: bool = False,
) -> None:
    """Run batch validation over CWQ in external-only mode and save predictions plus metrics."""
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    logger = get_logger(__name__)
    _validate_cwq_runtime_options(
        config,
        use_q_entities=bool(config.get("datasets", {}).get("webqsp", {}).get("use_q_entities_for_topic_entities", False)),
    )
    samples, selected_split = _load_cwq_config(config, config_path, split)
    if limit is not None:
        samples = samples[:limit]

    concurrency = _resolve_qa_concurrency(config, qa_concurrency)
    output_dir = _resolve_output_dir(config, config_path)
    sample_output_dir = _resolve_sample_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for existing_file in sample_output_dir.glob("*.json"):
            existing_file.unlink()

    predictions: list[Any] = []
    samples_to_run: list[CWQSample] = []
    if resume:
        for sample in samples:
            sample_path = sample_output_dir / f"{sample.sample_id}.json"
            if not sample_path.exists():
                samples_to_run.append(sample)
                continue
            try:
                predictions.append(_load_sample_prediction(sample_path))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to load existing prediction for %s: %s. Re-running sample.", sample.sample_id, exc)
                samples_to_run.append(sample)
        logger.info(
            "Resuming CWQ validation on %d samples with qa_concurrency=%d (%d loaded, %d remaining)",
            len(samples),
            concurrency,
            len(predictions),
            len(samples_to_run),
        )
    else:
        samples_to_run = list(samples)
        logger.info(
            "Running CWQ validation on %d samples with qa_concurrency=%d",
            len(samples_to_run),
            concurrency,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(_run_cwq_sample, config, sample): index
            for index, sample in enumerate(samples_to_run)
        }
        for future in as_completed(future_to_index):
            prediction = future.result()
            predictions.append(prediction)
            prediction_row = prediction.to_dict()
            _write_sample_prediction(sample_output_dir / f"{prediction.sample_id}.json", prediction_row)
            logger.info(
                "Completed and saved CWQ sample %s (%d/%d)",
                prediction.sample_id,
                len(predictions),
                len(samples),
            )

    metrics = aggregate_validation_metrics(predictions, dataset="cwq", split=selected_split)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, ensure_ascii=False, indent=2)

    logger.info("Saved per-sample CWQ predictions to %s", sample_output_dir)
    logger.info("Saved validation metrics to %s", metrics_path)
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    """Run the end-to-end pipeline from the command line."""
    args = parse_args()
    config = load_config(args.config)
    _apply_runtime_overrides(config, args)
    _resolve_graphapi_db_path(config, args.config)

    if args.dataset == "webqsp":
        mode = args.mode or ("single" if args.index is not None else "validate")
        if mode == "single":
            _run_webqsp_single(config, args.config, args.split, args.index or 0)
            return
        _run_webqsp_validation(config, args.config, args.split, args.limit, args.qa_concurrency, args.resume)
        return

    if args.dataset == "cwq":
        mode = args.mode or ("single" if args.index is not None else "validate")
        if mode == "single":
            _run_cwq_single(config, args.config, args.split, args.index or 0)
            return
        _run_cwq_validation(config, args.config, args.split, args.limit, args.qa_concurrency, args.resume)
        return

    if not args.question:
        raise ValueError("--question is required when --dataset is not specified.")
    _run_single_question(config, args.config, args.question)


if __name__ == "__main__":
    main()
