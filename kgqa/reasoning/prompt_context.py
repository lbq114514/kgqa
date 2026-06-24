"""Compact prompt-context builders for token-sensitive KGQA stages."""

from __future__ import annotations

from typing import Any

from kgqa.utils.types import AgenticRunState, QuestionAnalysisResult, ReasoningPath


def compact_question_analysis(
    question_analysis: QuestionAnalysisResult,
    max_sub_questions: int = 3,
) -> dict[str, Any]:
    """Return a compact question-analysis payload for prompt construction."""
    return {
        "reasoning_indicator": question_analysis.reasoning_indicator,
        "ordered_topic_entities": list(question_analysis.ordered_topic_entities[:5]),
        "predicted_depth": int(question_analysis.predicted_depth),
        "sub_questions": [
            {
                "id": sub_question.id,
                "question": sub_question.question,
                "topic_entities": [item.name for item in sub_question.topic_entities[:3] if item.name],
                "local_topic_entities": [item.name for item in sub_question.local_topic_entities[:3] if item.name],
                "interested_nodes": [item.name for item in sub_question.interested_nodes[:3] if item.name],
                "interested_relations": [item.name for item in sub_question.interested_relations[:3] if item.name],
                "expected_answer_type": sub_question.expected_answer_type,
                "depends_on": list(sub_question.depends_on),
            }
            for sub_question in question_analysis.sub_questions[:max_sub_questions]
        ],
    }


def compact_reasoning_paths(
    paths: list[ReasoningPath],
    max_paths: int = 4,
    max_triples_per_path: int = 3,
) -> list[dict[str, Any]]:
    """Serialize a bounded number of paths into a prompt-friendly shape."""
    compact_paths: list[dict[str, Any]] = []
    for path in paths[:max_paths]:
        triple_limit = max_triples_per_path
        matched_relation_limit = 3
        if path.source_stage == "relation_beam_cvt_bundle":
            # CVT bundle paths carry answer and constraint slots on later edges.
            triple_limit = max(max_triples_per_path, 6)
            matched_relation_limit = 6
        compact_paths.append(
            {
                "text": path.text,
                "triples": [
                    {
                        "head": triple.head,
                        "relation": triple.relation,
                        "tail": triple.tail,
                    }
                    for triple in path.triples[:triple_limit]
                ],
                "matched_relations": list(path.matched_relations[:matched_relation_limit]),
                "source_stage": path.source_stage,
                "terminal_node_id": path.terminal_node_id,
                "terminal_node_kind": path.terminal_node_kind,
            }
        )
    return compact_paths


def compact_evidence_bank(
    evidence_bank: list[Any],
    max_items: int = 4,
    max_triples: int = 4,
    max_evidence_lines: int = 4,
) -> list[dict[str, Any]]:
    """Keep only the most recent high-signal summaries for later prompts."""
    compact_items: list[dict[str, Any]] = []
    for item in evidence_bank[-max_items:]:
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                "summary_type": item.get("summary_type", ""),
                "sub_question_id": item.get("sub_question_id", ""),
                "question_focus": item.get("question_focus", ""),
                "key_triples": list(item.get("key_triples", []))[:max_triples],
                "evidence": list(item.get("evidence", []))[:max_evidence_lines],
            }
        )
    return compact_items


def compact_agentic_state(
    agentic_state: AgenticRunState | None,
    max_steps: int = 2,
    max_evidence_items: int = 3,
) -> dict[str, Any]:
    """Return a compact loop state without replaying every retrieved path."""
    if agentic_state is None:
        return {}
    return {
        "loop_index": int(agentic_state.loop_index),
        "current_frontier_entities": list(agentic_state.current_frontier_entities[:5]),
        "current_text_constraints": list(agentic_state.current_text_constraints[:5]),
        "current_sub_question_id": agentic_state.current_sub_question_id,
        "pending_sub_question_ids": list(agentic_state.pending_sub_question_ids[:5]),
        "failed_sub_question_ids": list(agentic_state.failed_sub_question_ids[:5]),
        "resolved_sub_answers": {
            key: value
            for key, value in list(agentic_state.resolved_sub_answers.items())[:max_evidence_items]
        },
        "recent_step_history": [
            {
                "step_id": step.step_id,
                "sub_question_id": step.sub_question_id,
                "sub_question_text": step.sub_question_text,
                "resolved_entity_ids": list(step.resolved_entity_ids[:5]),
                "resolved_relation_ids": list(step.resolved_relation_ids[:5]),
                "candidate_answers": list(step.candidate_answers[:5]),
                "sub_answer": step.sub_answer,
                "sub_answer_entities": list(step.sub_answer_entities[:5]),
                "sub_answer_literals": list(step.sub_answer_literals[:5]),
                "status": step.status,
                "attempt_count": int(step.attempt_count),
                "depends_on_step_ids": list(step.depends_on_step_ids[:5]),
                "failure_reason": step.failure_reason,
                "carryover_entities": list(step.carryover_entities[:5]),
                "carryover_text_values": list(step.carryover_text_values[:5]),
                "is_step_resolved": bool(step.is_step_resolved),
                "summarized_evidence": compact_evidence_bank(
                    step.summarized_evidence,
                    max_items=2,
                    max_triples=2,
                    max_evidence_lines=2,
                ),
            }
            for step in agentic_state.step_history[-max_steps:]
        ],
        "recent_evidence_bank": compact_evidence_bank(
            agentic_state.evidence_bank,
            max_items=max_evidence_items,
            max_triples=2,
            max_evidence_lines=2,
        ),
    }


def compact_summarized_paths(
    summarized_paths: list[Any],
    max_items: int = 5,
    max_triples: int = 4,
    max_evidence_lines: int = 4,
) -> list[dict[str, Any]]:
    """Bound the evidence set passed into sufficiency and answering prompts."""
    return compact_evidence_bank(
        list(summarized_paths),
        max_items=max_items,
        max_triples=max_triples,
        max_evidence_lines=max_evidence_lines,
    )
