"""WebQSP answer normalization and validation metrics."""

from __future__ import annotations

from typing import Any

from kgqa.utils.text import normalize_text
from kgqa.utils.types import PipelineResult, ValidationMetrics, ValidationPrediction


def normalize_answer_list(values: list[str]) -> list[str]:
    """Normalize and deduplicate a list of answers."""
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = normalize_text(str(value))
        if not text or text == "insufficient" or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def answers_from_llm_output(answer_field: Any) -> list[str]:
    """Convert LLM answer JSON into a list of predicted answers."""
    if isinstance(answer_field, list):
        return normalize_answer_list([str(item) for item in answer_field])
    if isinstance(answer_field, str):
        return normalize_answer_list([answer_field])
    return []


def score_prediction(gold_answers: list[str], predicted_answers: list[str]) -> dict[str, float]:
    """Compute exact-match and set-level precision/recall/F1."""
    gold = set(normalize_answer_list(gold_answers))
    normalized_predicted_answers = normalize_answer_list(predicted_answers)
    predicted = set(normalized_predicted_answers)

    exact_match = float(gold == predicted)
    if normalized_predicted_answers:
        hit_at_1 = float(normalized_predicted_answers[0] in gold)
    else:
        hit_at_1 = 0.0
    if not predicted:
        precision = 0.0
    else:
        precision = len(gold & predicted) / len(predicted)

    if not gold:
        recall = 1.0 if not predicted else 0.0
    else:
        recall = len(gold & predicted) / len(gold)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "exact_match": exact_match,
        "hit_at_1": hit_at_1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_validation_prediction(result: PipelineResult) -> ValidationPrediction:
    """Convert a pipeline result into a per-sample validation record."""
    metrics = score_prediction(result.gold_answers, result.predicted_answers)
    return ValidationPrediction(
        sample_id=result.sample_id or "",
        question=result.question,
        gold_answers=result.gold_answers,
        predicted_answers=result.predicted_answers,
        answer=result.answer,
        sufficient=result.sufficient,
        confidence=result.confidence,
        topic_entities=result.topic_entities,
        question_analysis=result.question_analysis.to_dict(),
        exploration_hints=result.exploration_hints.to_dict(),
        supplement_hints=result.supplement_hints.to_dict(),
        candidate_paths=result.candidate_paths,
        pruned_paths=result.pruned_paths,
        summarized_paths=result.summarized_paths,
        search_trace=result.search_trace,
        metrics=metrics,
        alignment_debug=result.alignment_debug,
    )


def aggregate_validation_metrics(
    predictions: list[ValidationPrediction],
    dataset: str,
    split: str,
) -> ValidationMetrics:
    """Aggregate per-sample validation metrics."""
    count = len(predictions)
    if count == 0:
        return ValidationMetrics(
            dataset=dataset,
            split=split,
            num_samples=0,
            exact_match=0.0,
            hit_at_1=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            empty_prediction_rate=0.0,
            avg_candidate_paths=0.0,
        )

    empty_predictions = sum(1 for prediction in predictions if not prediction.predicted_answers)
    total_candidate_paths = sum(len(prediction.candidate_paths) for prediction in predictions)
    return ValidationMetrics(
        dataset=dataset,
        split=split,
        num_samples=count,
        exact_match=sum(item.metrics["exact_match"] for item in predictions) / count,
        hit_at_1=sum(item.metrics["hit_at_1"] for item in predictions) / count,
        precision=sum(item.metrics["precision"] for item in predictions) / count,
        recall=sum(item.metrics["recall"] for item in predictions) / count,
        f1=sum(item.metrics["f1"] for item in predictions) / count,
        empty_prediction_rate=empty_predictions / count,
        avg_candidate_paths=total_candidate_paths / count,
    )
