"""Reasoning path summarization with the local LLM."""

from __future__ import annotations

import json
from typing import Any

from kgqa.llm.base import BaseLLM
from kgqa.llm.prompts import PATH_SUMMARIZATION_PROMPT
from kgqa.reasoning.prompt_context import compact_evidence_bank, compact_question_analysis, compact_reasoning_paths
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.types import QuestionAnalysisResult, ReasoningPath, SubQuestionSpec

LOGGER = get_logger(__name__)


def _summary_debug_preview(items: list[Any], max_chars: int = 1200) -> str:
    """Return a truncated JSON preview of summarized evidence for logs."""
    try:
        text = json.dumps(items, ensure_ascii=False)
    except TypeError:
        text = repr(items)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...<truncated {len(text) - max_chars} chars>"


def _fallback_summary(
    question: str,
    pruned_paths: list[ReasoningPath],
    sub_question: SubQuestionSpec | None = None,
) -> list[dict[str, Any]]:
    """Build a lossless fallback summary from the preserved paths."""
    return [
        {
            "question": question,
            "question_focus": question,
            "summary_type": "sub_question" if sub_question is not None else "question",
            "sub_question_id": sub_question.id if sub_question is not None else "",
            "key_triples": [triple.__dict__ for path in pruned_paths for triple in path.triples],
            "evidence": [path.text for path in pruned_paths],
        }
    ]


def _has_fact_evidence(items: list[Any]) -> bool:
    """Return True when summarized items already contain concrete triples or evidence text."""
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("key_triples") or item.get("evidence"):
            return True
    return False

def summarize_paths(
    question: str,
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    pruned_paths: list[ReasoningPath],
    llm: BaseLLM,
    sub_question: SubQuestionSpec | None = None,
    evidence_bank: list[Any] | None = None,
    plan_step_context: dict[str, Any] | None = None,
) -> list[Any]:
    """Summarize candidate reasoning paths into concise evidence JSON."""
    if not pruned_paths:
        return []

    prompt = PATH_SUMMARIZATION_PROMPT.format(
        question=question,
        topic_entities=json.dumps(topic_entities, ensure_ascii=False),
        question_analysis=json.dumps(compact_question_analysis(question_analysis), ensure_ascii=False, indent=2),
        evidence_bank=json.dumps(compact_evidence_bank(evidence_bank or []), ensure_ascii=False, indent=2),
        plan_step_context=json.dumps(plan_step_context, ensure_ascii=False, indent=2),
        sub_question_context=json.dumps(
            sub_question.to_dict() if sub_question is not None else None,
            ensure_ascii=False,
            indent=2,
        ),
        pruned_paths=json.dumps(compact_reasoning_paths(pruned_paths), ensure_ascii=False, indent=2),
    )
    raw = llm.generate(prompt, max_tokens=512)
    parsed = robust_json_parse(raw, fallback={})
    if isinstance(parsed, dict):
        parsed.setdefault("question", question)
        parsed.setdefault("question_focus", question)
        parsed.setdefault("summary_type", "sub_question" if sub_question is not None else "question")
        if sub_question is not None:
            parsed.setdefault("sub_question_id", sub_question.id)
        normalized = [parsed]
        if not _has_fact_evidence(normalized):
            fallback = _fallback_summary(question, pruned_paths, sub_question)
            LOGGER.info(
                "Summarized %d pruned paths produced empty dict summary; using fallback sub_question=%s summary=%s",
                len(pruned_paths),
                sub_question.id if sub_question is not None else "",
                _summary_debug_preview(fallback),
            )
            return fallback
        LOGGER.info(
            "Summarized %d pruned paths sub_question=%s summary=%s",
            len(pruned_paths),
            sub_question.id if sub_question is not None else "",
            _summary_debug_preview(normalized),
        )
        return normalized
    if isinstance(parsed, list):
        normalized_items: list[Any] = []
        for item in parsed:
            if isinstance(item, dict):
                item.setdefault("question", question)
                item.setdefault("question_focus", question)
                item.setdefault("summary_type", "sub_question" if sub_question is not None else "question")
                if sub_question is not None:
                    item.setdefault("sub_question_id", sub_question.id)
            normalized_items.append(item)
        if not _has_fact_evidence(normalized_items):
            fallback = _fallback_summary(question, pruned_paths, sub_question)
            LOGGER.info(
                "Summarized %d pruned paths produced empty list summary; using fallback sub_question=%s summary=%s",
                len(pruned_paths),
                sub_question.id if sub_question is not None else "",
                _summary_debug_preview(fallback),
            )
            return fallback
        LOGGER.info(
            "Summarized %d pruned paths sub_question=%s summary=%s",
            len(pruned_paths),
            sub_question.id if sub_question is not None else "",
            _summary_debug_preview(normalized_items),
        )
        return normalized_items
    fallback = _fallback_summary(question, pruned_paths, sub_question)
    LOGGER.info(
        "Summarized %d pruned paths via fallback sub_question=%s summary=%s",
        len(pruned_paths),
        sub_question.id if sub_question is not None else "",
        _summary_debug_preview(fallback),
    )
    return fallback
