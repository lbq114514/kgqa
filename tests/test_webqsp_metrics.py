from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.evaluation.webqsp import aggregate_validation_metrics, score_prediction
from kgqa.utils.types import ValidationPrediction


def test_score_prediction_computes_hit_at_1_from_first_prediction() -> None:
    metrics = score_prediction(
        gold_answers=["Dallas"],
        predicted_answers=["Dealey Plaza", "Dallas"],
    )

    assert metrics["hit_at_1"] == 0.0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_score_prediction_hit_at_1_uses_normalized_first_prediction() -> None:
    metrics = score_prediction(
        gold_answers=["Dallas"],
        predicted_answers=["  dallas  ", "dealey plaza"],
    )

    assert metrics["hit_at_1"] == 1.0


def test_aggregate_validation_metrics_averages_hit_at_1() -> None:
    predictions = [
        ValidationPrediction(
            sample_id="a",
            question="q1",
            gold_answers=["a"],
            predicted_answers=["a"],
            answer="a",
            sufficient=True,
            confidence="high",
            topic_entities=[],
            question_analysis={},
            supplement_hints={},
            candidate_paths=[],
            pruned_paths=[],
            summarized_paths=[],
            search_trace=[],
            metrics={"exact_match": 1.0, "hit_at_1": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        ),
        ValidationPrediction(
            sample_id="b",
            question="q2",
            gold_answers=["b"],
            predicted_answers=["c", "b"],
            answer="c",
            sufficient=True,
            confidence="high",
            topic_entities=[],
            question_analysis={},
            supplement_hints={},
            candidate_paths=[],
            pruned_paths=[],
            summarized_paths=[],
            search_trace=[],
            metrics={"exact_match": 0.0, "hit_at_1": 0.0, "precision": 0.5, "recall": 1.0, "f1": 2 / 3},
        ),
    ]

    aggregated = aggregate_validation_metrics(predictions, dataset="webqsp", split="validation")

    assert aggregated.hit_at_1 == 0.5
