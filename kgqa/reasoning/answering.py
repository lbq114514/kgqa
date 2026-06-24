"""Answer sufficiency checks and final answer generation."""

from __future__ import annotations

import json
from typing import Any

from kgqa.llm.base import BaseLLM
from kgqa.llm.prompts import ANSWERING_PROMPT, SUFFICIENCY_PROMPT
from kgqa.evaluation.webqsp import answers_from_llm_output
from kgqa.reasoning.prompt_context import (
    compact_agentic_state,
    compact_question_analysis,
    compact_summarized_paths,
)
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.types import AgenticRunState, AnswerResult, ExplorationHints, QuestionAnalysisResult, SubQuestionSpec

LOGGER = get_logger(__name__)


def _is_failure_like_answer(value: str) -> bool:
    """Return True when an answer string is just a failure placeholder."""
    normalized = str(value).strip().lower()
    if not normalized:
        return True
    blocked_markers = (
        "insufficient evidence",
        "insufficient",
        "unknown",
        "none",
        "null",
        "not enough evidence",
        "no evidence",
    )
    return any(marker in normalized for marker in blocked_markers)


def _build_answering_context(
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    dmax: int,
    dpredict: int,
    exploration_hints: ExplorationHints | None = None,
    agentic_state: AgenticRunState | None = None,
) -> dict[str, Any]:
    """Build compact reasoning context for sufficiency and answer prompts."""
    search_context = {
        "dmax": int(dmax),
        "current_evaluation_depth": int(dpredict),
    }
    if exploration_hints is not None:
        if exploration_hints.answer_type_hints:
            search_context["answer_type_hints"] = list(exploration_hints.answer_type_hints)
        if exploration_hints.reasoning_focus:
            search_context["reasoning_focus"] = exploration_hints.reasoning_focus
    return {
        "topic_entities": list(topic_entities),
        "task_analysis": {
            **compact_question_analysis(question_analysis),
        },
        "agentic_context": compact_agentic_state(agentic_state),
        "search_context": search_context,
    }


def _normalize_answer_result(parsed: dict[str, Any]) -> AnswerResult:
    """Normalize parsed answering JSON into a stable AnswerResult."""
    raw_predicted_answers = parsed.get("predicted_answers", [])
    predicted_answers = answers_from_llm_output(raw_predicted_answers)

    raw_answer = parsed.get("answer", "insufficient")
    if isinstance(raw_answer, list):
        answer = str(raw_answer[0]) if raw_answer else "insufficient"
    else:
        answer = str(raw_answer)

    if not predicted_answers:
        if isinstance(raw_answer, list):
            predicted_answers = answers_from_llm_output(raw_answer)
        else:
            predicted_answers = answers_from_llm_output(answer)

    predicted_answers = [item for item in predicted_answers if not _is_failure_like_answer(item)]

    if predicted_answers and _is_failure_like_answer(answer):
        answer = ", ".join(predicted_answers)

    raw_entities = parsed.get("resolved_entity_mentions", [])
    resolved_entity_mentions = [
        str(item).strip()
        for item in (raw_entities if isinstance(raw_entities, list) else [raw_entities])
        if str(item).strip()
    ]
    raw_literals = parsed.get("resolved_literals", [])
    resolved_literals = [
        str(item).strip()
        for item in (raw_literals if isinstance(raw_literals, list) else [raw_literals])
        if str(item).strip()
    ]
    if not resolved_literals and predicted_answers:
        resolved_literals = list(predicted_answers)

    supporting_paths = parsed.get("supporting_paths", [])
    return AnswerResult(
        sufficient=bool(predicted_answers) and not _is_failure_like_answer(answer),
        answer=answer,
        predicted_answers=predicted_answers,
        resolved_entity_mentions=resolved_entity_mentions,
        resolved_literals=resolved_literals,
        supporting_paths=supporting_paths if isinstance(supporting_paths, list) else [],
        reason=str(parsed.get("reason") or ""),
    )


def _normalize_sufficiency_result(parsed: dict[str, Any]) -> dict[str, Any]:
    """Normalize sufficiency JSON and preserve optional answer hints."""
    sufficient = bool(parsed.get("sufficient", False))
    reason = str(parsed.get("reason") or "")
    raw_candidates = parsed.get("answer_candidates", [])
    answer_candidates = answers_from_llm_output(raw_candidates)
    raw_primary = parsed.get("primary_answer", "")
    if isinstance(raw_primary, list):
        primary_answer = str(raw_primary[0]).strip() if raw_primary else ""
    else:
        primary_answer = str(raw_primary or "").strip()
    if not answer_candidates and primary_answer:
        answer_candidates = answers_from_llm_output(primary_answer)
    answer_candidates = [
        item for item in answer_candidates if not _is_failure_like_answer(item)
    ]
    if _is_failure_like_answer(primary_answer):
        primary_answer = ""
    if not primary_answer and answer_candidates:
        primary_answer = str(answer_candidates[0]).strip()
    if not sufficient:
        primary_answer = ""
        answer_candidates = []
    return {
        "sufficient": sufficient,
        "reason": reason,
        "primary_answer": primary_answer,
        "answer_candidates": answer_candidates,
    }


def check_sufficiency(
    question: str,
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    dmax: int,
    dpredict: int,
    split_questions: list[str],
    summarized_paths: list[Any],
    llm: BaseLLM,
    exploration_hints: ExplorationHints | None = None,
    agentic_state: AgenticRunState | None = None,
) -> dict[str, Any]:
    """Check whether the summarized evidence is sufficient to answer the question."""
    context = _build_answering_context(
        topic_entities=topic_entities,
        question_analysis=question_analysis,
        dmax=dmax,
        dpredict=dpredict,
        exploration_hints=exploration_hints,
        agentic_state=agentic_state,
    )
    compact_paths = compact_summarized_paths(summarized_paths)
    prompt = SUFFICIENCY_PROMPT.format(
        question=question,
        topic_entities=json.dumps(context["topic_entities"], ensure_ascii=False),
        task_analysis=json.dumps(context["task_analysis"], ensure_ascii=False, indent=2),
        agentic_context=json.dumps(context["agentic_context"], ensure_ascii=False, indent=2),
        search_context=json.dumps(context["search_context"], ensure_ascii=False, indent=2),
        split_questions=json.dumps(split_questions, ensure_ascii=False),
        summarized_paths=json.dumps(compact_paths, ensure_ascii=False, indent=2),
    )
    raw = llm.generate(prompt, max_tokens=256)
    parsed = robust_json_parse(
        raw,
        fallback={"sufficient": False, "reason": "LLM JSON parse failed.", "primary_answer": "", "answer_candidates": []},
    )
    if isinstance(parsed, dict) and "sufficient" in parsed:
        normalized = _normalize_sufficiency_result(parsed)
        LOGGER.info("Sufficiency result: %s", normalized)
        return normalized
    return {
        "sufficient": False,
        "reason": "LLM JSON parse failed.",
        "primary_answer": "",
        "answer_candidates": [],
    }


def generate_answer(
    question: str,
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    dmax: int,
    dpredict: int,
    split_questions: list[str],
    summarized_paths: list[Any],
    llm: BaseLLM,
    exploration_hints: ExplorationHints | None = None,
    agentic_state: AgenticRunState | None = None,
) -> AnswerResult:
    """Generate the final answer using only the summarized evidence."""
    context = _build_answering_context(
        topic_entities=topic_entities,
        question_analysis=question_analysis,
        dmax=dmax,
        dpredict=dpredict,
        exploration_hints=exploration_hints,
        agentic_state=agentic_state,
    )
    compact_paths = compact_summarized_paths(summarized_paths)
    prompt = ANSWERING_PROMPT.format(
        question=question,
        topic_entities=json.dumps(context["topic_entities"], ensure_ascii=False),
        task_analysis=json.dumps(context["task_analysis"], ensure_ascii=False, indent=2),
        agentic_context=json.dumps(context["agentic_context"], ensure_ascii=False, indent=2),
        search_context=json.dumps(context["search_context"], ensure_ascii=False, indent=2),
        split_questions=json.dumps(split_questions, ensure_ascii=False),
        summarized_paths=json.dumps(compact_paths, ensure_ascii=False, indent=2),
    )
    raw = llm.generate(prompt, max_tokens=256)
    parsed = robust_json_parse(
        raw,
        fallback={"predicted_answers": [], "answer": "insufficient", "supporting_paths": []},
    )
    if isinstance(parsed, dict):
        return _normalize_answer_result(parsed)
    LOGGER.warning("Answer parsing failed. Returning insufficient answer.")
    return AnswerResult(
        sufficient=False,
        answer="insufficient",
        predicted_answers=[],
        resolved_entity_mentions=[],
        resolved_literals=[],
        supporting_paths=[],
    )


def check_subquestion_sufficiency(
    sub_question: SubQuestionSpec,
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    dmax: int,
    dpredict: int,
    summarized_paths: list[Any],
    llm: BaseLLM,
    exploration_hints: ExplorationHints | None = None,
    agentic_state: AgenticRunState | None = None,
) -> dict[str, Any]:
    """Check whether the evidence is sufficient for the current sub-question only."""
    return check_sufficiency(
        question=sub_question.question,
        topic_entities=topic_entities,
        question_analysis=question_analysis,
        dmax=dmax,
        dpredict=dpredict,
        split_questions=[sub_question.question],
        summarized_paths=summarized_paths,
        llm=llm,
        exploration_hints=exploration_hints,
        agentic_state=agentic_state,
    )


def generate_subquestion_answer(
    sub_question: SubQuestionSpec,
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    dmax: int,
    dpredict: int,
    summarized_paths: list[Any],
    llm: BaseLLM,
    exploration_hints: ExplorationHints | None = None,
    agentic_state: AgenticRunState | None = None,
) -> AnswerResult:
    """Generate an answer for the current sub-question only."""
    return generate_answer(
        question=sub_question.question,
        topic_entities=topic_entities,
        question_analysis=question_analysis,
        dmax=dmax,
        dpredict=dpredict,
        split_questions=[sub_question.question],
        summarized_paths=summarized_paths,
        llm=llm,
        exploration_hints=exploration_hints,
        agentic_state=agentic_state,
    )
