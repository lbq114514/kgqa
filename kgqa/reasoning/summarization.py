"""Reasoning path summarization with the local LLM."""

from __future__ import annotations

import json
from typing import Any

from kgqa.llm.base import BaseLLM
from kgqa.kg.relation_beam import CVT_BUNDLE_SOURCE_STAGE
from kgqa.llm.prompts import PATH_SUMMARIZATION_PROMPT
from kgqa.reasoning.prompt_context import compact_evidence_bank, compact_question_analysis, compact_reasoning_paths
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.types import QuestionAnalysisResult, ReasoningPath, SubQuestionSpec, TripleFact

LOGGER = get_logger(__name__)


def _looks_like_mid_or_cvt_label(value: str) -> bool:
    normalized = str(value or "").strip()
    return normalized.startswith("m.") or normalized.startswith("g.") or normalized == "[CVT]"


def _build_answer_view_triples(pruned_paths: list[ReasoningPath]) -> list[dict[str, str]]:
    """Derive answer-facing triples from raw paths, collapsing CVT bridge nodes when possible."""
    answer_view: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in pruned_paths:
        if path.source_stage != CVT_BUNDLE_SOURCE_STAGE or not path.triples:
            for triple in path.triples:
                triple_key = (str(triple.head), str(triple.relation), str(triple.tail))
                if triple_key in seen:
                    continue
                seen.add(triple_key)
                answer_view.append(
                    {"head": triple_key[0], "relation": triple_key[1], "tail": triple_key[2]}
                )
            continue

        triples = path.triples
        entry = triples[0]
        anchor = str(entry.head)
        cvt_labels: set[str] = set()
        label_counts: dict[str, int] = {}
        for triple in triples:
            for value in (triple.head, triple.tail):
                normalized = str(value or "").strip()
                if not normalized:
                    continue
                label_counts[normalized] = label_counts.get(normalized, 0) + 1
        for label, count in label_counts.items():
            if count >= 2 and _looks_like_mid_or_cvt_label(label):
                cvt_labels.add(label)
        if not cvt_labels and _looks_like_mid_or_cvt_label(entry.tail):
            cvt_labels.add(str(entry.tail))

        synthesized_any = False
        for triple in triples[1:]:
            head = str(triple.head)
            tail = str(triple.tail)
            if head in cvt_labels and tail not in cvt_labels and tail != anchor:
                triple_key = (anchor, str(triple.relation), tail)
            elif tail in cvt_labels and head not in cvt_labels and head != anchor:
                triple_key = (anchor, str(triple.relation), head)
            else:
                continue
            if triple_key in seen:
                continue
            synthesized_any = True
            seen.add(triple_key)
            answer_view.append(
                {"head": triple_key[0], "relation": triple_key[1], "tail": triple_key[2]}
            )
        if not synthesized_any:
            for triple in triples:
                triple_key = (str(triple.head), str(triple.relation), str(triple.tail))
                if triple_key in seen:
                    continue
                seen.add(triple_key)
                answer_view.append(
                    {"head": triple_key[0], "relation": triple_key[1], "tail": triple_key[2]}
                )
    return answer_view


def _triple_fact_to_dict(fact: TripleFact) -> dict[str, str]:
    return {
        "head_id": str(fact.head_id or "").strip(),
        "head_label": str(fact.head_label or "").strip(),
        "relation_id": str(fact.relation_id or "").strip(),
        "tail_id": str(fact.tail_id or "").strip(),
        "tail_label": str(fact.tail_label or "").strip(),
        "tail_kind": str(fact.tail_kind or "id").strip(),
    }


def _path_fact_dicts(path: ReasoningPath) -> list[dict[str, str]]:
    """Return explicit path facts or derive a best-effort id view from edge_ids."""
    explicit = [_triple_fact_to_dict(fact) for fact in getattr(path, "triple_facts", [])]
    if explicit:
        return explicit
    if not path.edge_ids:
        return []
    facts: list[dict[str, str]] = []
    for index, edge_id in enumerate(path.edge_ids):
        parts = str(edge_id or "").split(":", 2)
        if len(parts) != 3:
            continue
        head_id, relation_id, tail_id = (part.strip() for part in parts)
        head_label = str(path.nodes[index] if index < len(path.nodes) else "").strip()
        tail_label = str(path.nodes[index + 1] if index + 1 < len(path.nodes) else "").strip()
        tail_kind = "id" if tail_id.startswith(("m.", "g.")) else "literal"
        facts.append(
            {
                "head_id": head_id if head_id.startswith(("m.", "g.")) else "",
                "head_label": head_label,
                "relation_id": relation_id,
                "tail_id": tail_id if tail_kind == "id" else "",
                "tail_label": tail_label,
                "tail_kind": tail_kind,
            }
        )
    return facts


def _build_key_triple_facts(pruned_paths: list[ReasoningPath]) -> list[dict[str, str]]:
    """Flatten raw path evidence facts without dropping ids."""
    facts: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in pruned_paths:
        for fact_dict in _path_fact_dicts(path):
            key = (
                fact_dict["head_id"],
                fact_dict["relation_id"],
                fact_dict["tail_id"],
                fact_dict["tail_label"],
            )
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact_dict)
    return facts


def _build_answer_view_facts(pruned_paths: list[ReasoningPath]) -> list[dict[str, str]]:
    """Derive answer-facing facts from raw paths while preserving ids."""
    answer_view: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in pruned_paths:
        path_facts = _path_fact_dicts(path)
        if path.source_stage != CVT_BUNDLE_SOURCE_STAGE or not path_facts:
            for fact_dict in path_facts:
                key = (
                    fact_dict["head_id"],
                    fact_dict["relation_id"],
                    fact_dict["tail_id"],
                    fact_dict["tail_label"],
                )
                if key in seen:
                    continue
                seen.add(key)
                answer_view.append(fact_dict)
            continue

        facts = list(path_facts)
        if not facts:
            continue
        entry_fact = facts[0]
        anchor_id = str(entry_fact.get("head_id") or "").strip()
        anchor_label = str(entry_fact.get("head_label") or "").strip()
        cvt_id = str(entry_fact.get("tail_id") or "").strip()
        cvt_label = str(entry_fact.get("tail_label") or "").strip()
        cvt_ids = {value for value in (cvt_id,) if value}
        cvt_labels = {value for value in (cvt_label,) if value and _looks_like_mid_or_cvt_label(value)}
        for fact in facts[1:]:
            head_id = str(fact.get("head_id") or "").strip()
            tail_id = str(fact.get("tail_id") or "").strip()
            head_label = str(fact.get("head_label") or "").strip()
            tail_label = str(fact.get("tail_label") or "").strip()
            relation_id = str(fact.get("relation_id") or "").strip()
            tail_kind = str(fact.get("tail_kind") or "id").strip()
            if not relation_id:
                continue
            if head_id in cvt_ids and tail_label and tail_label != anchor_label:
                fact_dict = {
                    "head_id": anchor_id,
                    "head_label": anchor_label,
                    "relation_id": relation_id,
                    "tail_id": tail_id if tail_kind == "id" else "",
                    "tail_label": tail_label,
                    "tail_kind": tail_kind,
                }
            elif tail_id in cvt_ids and head_label and head_label != anchor_label:
                fact_dict = {
                    "head_id": anchor_id,
                    "head_label": anchor_label,
                    "relation_id": relation_id,
                    "tail_id": head_id if head_id.startswith(("m.", "g.")) else "",
                    "tail_label": head_label,
                    "tail_kind": "id" if head_id.startswith(("m.", "g.")) else "literal",
                }
            elif head_label in cvt_labels or tail_label in cvt_labels:
                continue
            else:
                fact_dict = dict(fact)
            key = (
                fact_dict["head_id"],
                fact_dict["relation_id"],
                fact_dict["tail_id"],
                fact_dict["tail_label"],
            )
            if key in seen:
                continue
            seen.add(key)
            answer_view.append(fact_dict)
    return answer_view


def _attach_answer_view_triples(
    items: list[dict[str, Any]],
    pruned_paths: list[ReasoningPath],
) -> list[dict[str, Any]]:
    """Augment summaries with answer-facing triples for downstream consumers."""
    answer_view_triples = _build_answer_view_triples(pruned_paths)
    answer_view_facts = _build_answer_view_facts(pruned_paths)
    key_triple_facts = _build_key_triple_facts(pruned_paths)
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        copied = dict(item)
        copied.setdefault("answer_view_triples", answer_view_triples)
        copied.setdefault("answer_view_facts", answer_view_facts)
        copied.setdefault("key_triple_facts", key_triple_facts)
        normalized_items.append(copied)
    return normalized_items


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
    evidence = _compress_fallback_evidence(pruned_paths)
    return _attach_answer_view_triples([
        {
            "question": question,
            "question_focus": question,
            "summary_type": "sub_question" if sub_question is not None else "question",
            "sub_question_id": sub_question.id if sub_question is not None else "",
            "key_triples": [triple.__dict__ for path in pruned_paths for triple in path.triples],
            "evidence": evidence,
        }
    ], pruned_paths)


def _compress_fallback_evidence(pruned_paths: list[ReasoningPath]) -> list[str]:
    """Prefer one whole CVT bundle evidence line over redundant atomic fragments."""
    bundle_lines: list[str] = []
    bundle_heads: set[str] = set()
    atomic_lines: list[str] = []
    seen: set[str] = set()

    for path in pruned_paths:
        line = str(path.text or "").strip()
        if not line or line in seen:
            continue
        seen.add(line)
        if path.source_stage == CVT_BUNDLE_SOURCE_STAGE and "{" in line and ";" in line:
            bundle_lines.append(line)
            prefix = line.split("{", 1)[0].strip()
            if ";" in prefix:
                bundle_head = prefix.rsplit(";", 1)[-1].strip()
                if bundle_head:
                    bundle_heads.add(bundle_head)
            continue
        atomic_lines.append(line)

    compressed = list(bundle_lines)
    for line in atomic_lines:
        line_head = line.split("->", 1)[0].strip() if "->" in line else ""
        if line_head and line_head in bundle_heads:
            continue
        compressed.append(line)
    return compressed


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
        normalized = _attach_answer_view_triples([parsed], pruned_paths)
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
        normalized_items: list[dict[str, Any]] = []
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
        normalized_items = _attach_answer_view_triples(normalized_items, pruned_paths)
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
