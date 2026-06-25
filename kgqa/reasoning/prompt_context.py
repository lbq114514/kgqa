"""Compact prompt-context builders for token-sensitive KGQA stages."""

from __future__ import annotations

import re
from typing import Any

from kgqa.utils.types import AgenticRunState, QuestionAnalysisResult, ReasoningPath


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _shorten_text(text: Any, max_chars: int = 120) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    head_budget = max(16, (max_chars - 3) // 2)
    tail_budget = max(12, max_chars - 3 - head_budget)
    return f"{normalized[:head_budget]}...{normalized[-tail_budget:]}"


def _compact_relation_name(relation: Any, max_segments: int = 3) -> str:
    text = _normalize_text(relation)
    if not text or "." not in text:
        return text
    segments = [segment for segment in text.split(".") if segment]
    if len(segments) == 1:
        return segments[0]

    prefix = segments[0]
    if len(prefix) > 4:
        prefix = prefix[:4]

    if len(segments) <= max_segments:
        suffix = segments[1:]
    else:
        suffix = segments[-(max_segments - 1) :]
    return ".".join([prefix, "..", *suffix])


def _compact_evidence_line(line: Any, max_chars: int = 220) -> str:
    text = _normalize_text(line)
    if not text:
        return ""
    replacements = (
        (" -> ", " -> "),
        (" ; ", " | "),
        (" (reverse) ", " (rev) "),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return _shorten_text(text, max_chars=max_chars)


def _compact_triple_dict(triple: Any) -> dict[str, str]:
    if not isinstance(triple, dict):
        return {}
    return {
        "head": _shorten_text(triple.get("head", ""), max_chars=48),
        "relation": _compact_relation_name(triple.get("relation", "")),
        "tail": _shorten_text(triple.get("tail", ""), max_chars=72),
    }


def _compact_grounding(grounding: Any) -> dict[str, Any]:
    if not isinstance(grounding, dict):
        return {}
    answer_texts = [_shorten_text(item, max_chars=72) for item in grounding.get("answer_texts", [])[:3]]
    entity_labels = [_shorten_text(item, max_chars=72) for item in grounding.get("entity_labels", [])[:3]]
    entity_ids = [str(item).strip() for item in grounding.get("entity_ids", [])[:3] if str(item).strip()]
    literal_values = [_shorten_text(item, max_chars=72) for item in grounding.get("literal_values", [])[:3]]
    compacted = {
        "primary_answer_text": _shorten_text(grounding.get("primary_answer_text", ""), max_chars=96),
        "primary_entity_id": str(grounding.get("primary_entity_id", "")).strip(),
        "answer_texts": answer_texts,
        "entity_labels": entity_labels,
        "entity_ids": entity_ids,
        "literal_values": literal_values,
        "source_mode": _normalize_text(grounding.get("source_mode", "")),
    }
    if grounding.get("supporting_relation_ids"):
        compacted["supporting_relation_ids"] = [
            _compact_relation_name(item) for item in grounding.get("supporting_relation_ids", [])[:6]
        ]
    return {key: value for key, value in compacted.items() if value}


def _compact_resolved_sub_answer(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compacted = {
        "answer": _shorten_text(payload.get("answer", ""), max_chars=96),
        "predicted_answers": [_shorten_text(item, max_chars=72) for item in payload.get("predicted_answers", [])[:3]],
        "entity_ids": [str(item).strip() for item in payload.get("entity_ids", [])[:3] if str(item).strip()],
        "evidence_relations": [_compact_relation_name(item) for item in payload.get("evidence_relations", [])[:5]],
        "grounding": _compact_grounding(payload.get("grounding", {})),
    }
    return {key: value for key, value in compacted.items() if value}


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
                "matched_relations": [
                    _compact_relation_name(relation)
                    for relation in list(path.matched_relations[:matched_relation_limit])
                ],
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
                "question_focus": _shorten_text(item.get("question_focus", ""), max_chars=140),
                "key_triples": [
                    _compact_triple_dict(triple)
                    for triple in list(item.get("answer_view_triples", item.get("key_triples", [])))[:max_triples]
                    if isinstance(triple, dict)
                ],
                "evidence": [
                    compacted
                    for compacted in (
                        _compact_evidence_line(line, max_chars=220)
                        for line in list(item.get("evidence", []))[:max_evidence_lines]
                    )
                    if compacted
                ],
                "grounding": _compact_grounding(item.get("grounding", {})),
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
            key: _compact_resolved_sub_answer(value)
            for key, value in list(agentic_state.resolved_sub_answers.items())[:max_evidence_items]
        },
        "recent_step_history": [
            {
                "step_id": step.step_id,
                "sub_question_id": step.sub_question_id,
                "sub_question_text": step.sub_question_text,
                "resolved_entity_ids": list(step.resolved_entity_ids[:5]),
                "resolved_relation_ids": [
                    _compact_relation_name(item) for item in step.resolved_relation_ids[:5]
                ],
                "candidate_answers": list(step.candidate_answers[:5]),
                "sub_answer": _shorten_text(step.sub_answer, max_chars=96),
                "sub_answer_entities": [_shorten_text(item, max_chars=72) for item in step.sub_answer_entities[:5]],
                "sub_answer_literals": [_shorten_text(item, max_chars=72) for item in step.sub_answer_literals[:5]],
                "sub_answer_grounding": _compact_grounding(step.sub_answer_grounding),
                "status": step.status,
                "attempt_count": int(step.attempt_count),
                "depends_on_step_ids": list(step.depends_on_step_ids[:5]),
                "failure_reason": _shorten_text(step.failure_reason, max_chars=160),
                "carryover_entities": list(step.carryover_entities[:5]),
                "carryover_text_values": [_shorten_text(item, max_chars=72) for item in step.carryover_text_values[:5]],
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
