"""Question analysis using local vLLM prompts."""

from __future__ import annotations

import json
import re

from kgqa.llm.base import BaseLLM
from kgqa.llm.prompts import AGENTIC_STEP_PLANNING_PROMPT, QUESTION_ANALYSIS_PROMPT
from kgqa.reasoning.prompt_context import compact_agentic_state, compact_question_analysis
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.types import (
    AgenticPlanStep,
    AgenticRunState,
    EntityMentionSpec,
    QuestionAnalysisResult,
    RelationHintSpec,
    SubQuestionSpec,
)

LOGGER = get_logger(__name__)


def _string_list(value: object) -> list[str]:
    """Normalize a JSON field into a list of non-empty strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _json_list(value: object) -> list[object]:
    """Normalize a JSON field into a list-like container."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _safe_int(value: object, default: int) -> int:
    """Parse an integer field defensively."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_entity_spec(value: object, default_role: str) -> EntityMentionSpec | None:
    """Parse one entity-like spec from LLM JSON."""
    if isinstance(value, str):
        name = value.strip()
        if not name:
            return None
        return EntityMentionSpec(name=name, role=default_role)
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or value.get("mention") or "").strip()
    if not name:
        return None
    return EntityMentionSpec(
        name=name,
        aliases=_string_list(value.get("aliases")),
        expected_type=str(value.get("expected_type") or "").strip(),
        role=str(value.get("role") or default_role).strip() or default_role,
    )


def _parse_relation_spec(value: object) -> RelationHintSpec | None:
    """Parse one relation-hint spec from LLM JSON."""
    if isinstance(value, str):
        name = value.strip()
        if not name:
            return None
        return RelationHintSpec(name=name)
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or value.get("hint") or "").strip()
    if not name:
        return None
    return RelationHintSpec(
        name=name,
        aliases=_string_list(value.get("aliases")),
        freebase_like_ids=_string_list(value.get("freebase_like_ids")),
        direction=str(value.get("direction") or "").strip(),
        description=str(value.get("description") or "").strip(),
    )


def _default_agentic_step(
    question: str,
    loop_index: int,
    topic_entities: list[str],
    current_constraints: list[str],
) -> AgenticPlanStep:
    """Create a safe fallback step when the planner output is invalid."""
    mentions = [
        EntityMentionSpec(name=name, role="topic_entity")
        for name in topic_entities
        if name
    ]
    return AgenticPlanStep(
        step_id=f"step_{loop_index + 1}",
        question=question,
        goal=question,
        topic_entity_mentions=mentions,
        carryover_constraints=[value for value in current_constraints if value],
        stop_if_answered=(loop_index > 0),
    )


def _default_sub_question(question: str, topic_entities: list[str]) -> SubQuestionSpec:
    """Create a single compatible sub-question when the LLM returns only the old schema."""
    return SubQuestionSpec(
        id="sq1",
        question=question,
        topic_entities=[
            EntityMentionSpec(name=str(entity).strip(), role="topic_entity")
            for entity in topic_entities
            if str(entity).strip()
        ],
        local_topic_entities=[
            EntityMentionSpec(name=str(entity).strip(), role="topic_entity")
            for entity in topic_entities
            if str(entity).strip()
        ],
        expected_hop=1,
        solver_type=_infer_solver_type(question=question, expected_answer_type="", interested_relations=[]),
        solver_reason="fallback single sub-question",
    )


def _infer_solver_type(
    question: str,
    expected_answer_type: str,
    interested_relations: list[RelationHintSpec],
) -> str:
    """Infer one coarse solver type only as a weak fallback."""
    normalized = " ".join(
        [
            str(question or "").strip().lower(),
            str(expected_answer_type or "").strip().lower(),
            " ".join(
                [
                    str(item.name or "").strip().lower()
                    for item in interested_relations
                    if str(item.name or "").strip()
                ]
            ),
        ]
    )
    tokens = re.findall(r"[a-z0-9]+", normalized)
    token_set = set(tokens)
    aggregate_markers = (
        "how many",
        "number of",
        "which of those",
        "among them",
        "among these",
        "among those",
    )
    if (
        any(marker in normalized for marker in aggregate_markers)
        or "most" in token_set
        or "least" in token_set
        or "largest" in token_set
        or "smallest" in token_set
        or "highest" in token_set
        or "lowest" in token_set
        or "latest" in token_set
        or "earliest" in token_set
        or "population" in token_set
        or "area" in token_set
        or "gdp" in token_set
        or "rate" in token_set
        or "count" in token_set
        or "same" in token_set
        or "intersection" in token_set
        or "both" in token_set
        or "also" in token_set
    ):
        return "aggregate"
    if any(marker in normalized for marker in ("same as", "belong to", "satisfy", "whether", "is the same", "is that", "did that", "does that")):
        return "verify"
    return "explore"


def _is_soft_semantic_verify(
    question: str,
    interested_relations: list[RelationHintSpec],
) -> bool:
    """Detect vague verification requests that rarely map to explicit KG edges."""
    normalized_question = str(question or "").strip().lower()
    relation_text = " ".join(
        filter(
            None,
            [
                str(item.name or "").strip().lower()
                for item in interested_relations
            ]
            + [
                str(alias).strip().lower()
                for item in interested_relations
                for alias in item.aliases
                if str(alias).strip()
            ]
            + [
                str(item.description or "").strip().lower()
                for item in interested_relations
                if str(item.description or "").strip()
            ],
        )
    )
    combined = f"{normalized_question} {relation_text}".strip()
    if not combined:
        return False
    soft_markers = (
        "popular",
        "famous",
        "well known",
        "well-known",
        "common",
        "notable",
        "important",
        "major",
        "traditional",
        "typical",
        "associated with",
        "known for",
    )
    return any(marker in combined for marker in soft_markers)


def _is_boolean_like_answer_type(expected_answer_type: str) -> bool:
    normalized = str(expected_answer_type or "").strip().lower()
    return normalized in {"boolean", "bool", "yes/no", "yesno"}


def _is_wh_question(question: str) -> bool:
    normalized = str(question or "").strip().lower()
    return normalized.startswith(("what ", "which ", "who ", "where ", "when "))


def _drop_redundant_tail_verify_subquestions(
    question: str,
    sub_questions: list[SubQuestionSpec],
) -> list[SubQuestionSpec]:
    """Remove trailing verify steps that are soft semantic checks for WH questions."""
    if len(sub_questions) <= 1 or not _is_wh_question(question):
        return sub_questions

    cleaned = list(sub_questions)
    while len(cleaned) > 1:
        tail = cleaned[-1]
        tail_solver_reason = str(tail.solver_reason or "").strip().lower()
        if tail.solver_type != "verify" and "verify constraint" not in tail_solver_reason:
            break
        if not _is_boolean_like_answer_type(tail.expected_answer_type):
            break
        if not tail.depends_on:
            break
        if tail.interested_nodes:
            break
        if any(spec.freebase_like_ids for spec in tail.interested_relations):
            break
        if not _is_soft_semantic_verify(tail.question, tail.interested_relations):
            break
        cleaned.pop()
    return cleaned


def _normalize_solver_type(
    raw_solver_type: object,
    *,
    question: str,
    expected_answer_type: str,
    topic_specs: list[EntityMentionSpec],
    local_topic_specs: list[EntityMentionSpec],
    interested_nodes: list[EntityMentionSpec],
    interested_relations: list[RelationHintSpec],
    depends_on: list[str],
) -> str:
    """Validate and lightly refine one solver type against the sub-question structure."""
    allowed = {"explore", "verify", "aggregate", "constrained_collect"}
    solver_type = str(raw_solver_type or "").strip().lower()
    if solver_type not in allowed:
        solver_type = _infer_solver_type(
            question=question,
            expected_answer_type=expected_answer_type,
            interested_relations=interested_relations,
        )

    has_local_anchor = bool(interested_nodes)
    normalized_question = str(question or "").strip().lower()

    if solver_type == "verify" and not depends_on and not has_local_anchor:
        return "explore"
    if solver_type == "verify" and any(
        marker in normalized_question for marker in ("what is", "who is", "where is", "when is")
    ) and not any(marker in normalized_question for marker in ("whether", "is that", "does that", "did that", "is the same")):
        return "explore"
    if solver_type == "verify" and _is_soft_semantic_verify(question, interested_relations):
        if not any(item.freebase_like_ids for item in interested_relations):
            return "explore"
    return solver_type


def _fallback_topic_entities(question: str, parsed_topic_entities: list[str], ordered_topic_entities: list[str]) -> list[str]:
    """Return the best available question-level topic entities."""
    if parsed_topic_entities:
        return parsed_topic_entities
    if ordered_topic_entities:
        return ordered_topic_entities
    return []


def _parse_sub_questions(
    parsed: dict[str, object],
    question: str,
    topic_entities: list[str],
    split_questions: list[str],
    dmax: int,
) -> list[SubQuestionSpec]:
    """Parse structured sub-questions while preserving compatibility with the old schema."""
    raw_sub_questions = parsed.get("sub_questions")
    if not isinstance(raw_sub_questions, list) or not raw_sub_questions:
        return [_default_sub_question(question, topic_entities)]

    sub_questions: list[SubQuestionSpec] = []
    for index, item in enumerate(raw_sub_questions, start=1):
        if not isinstance(item, dict):
            continue
        sub_question_text = str(
            item.get("question")
            or (split_questions[index - 1] if index - 1 < len(split_questions) else "")
            or question
        ).strip()
        topic_specs = [
            spec
            for spec in (
                _parse_entity_spec(raw_item, "topic_entity")
                for raw_item in _json_list(item.get("topic_entities"))
            )
            if spec is not None
        ]
        local_topic_specs = [
            spec
            for spec in (
                _parse_entity_spec(raw_item, "topic_entity")
                for raw_item in _json_list(item.get("local_topic_entities"))
            )
            if spec is not None
        ]
        explicit_local_topic_specs = list(local_topic_specs)
        interested_nodes = [
            spec
            for spec in (
                _parse_entity_spec(raw_item, "constraint_or_anchor")
                for raw_item in _json_list(item.get("interested_nodes"))
            )
            if spec is not None
        ]
        interested_relations = [
            spec
            for spec in (
                _parse_relation_spec(raw_item) for raw_item in _json_list(item.get("interested_relations"))
            )
            if spec is not None
        ]
        normalized_solver_type = _normalize_solver_type(
            item.get("solver_type"),
            question=sub_question_text or question,
            expected_answer_type=str(item.get("expected_answer_type") or "").strip(),
            topic_specs=topic_specs,
            local_topic_specs=explicit_local_topic_specs,
            interested_nodes=interested_nodes,
            interested_relations=interested_relations,
            depends_on=_string_list(item.get("depends_on")),
        )
        if not local_topic_specs and not topic_specs and interested_nodes:
            local_topic_specs = [
                EntityMentionSpec(
                    name=spec.name,
                    aliases=list(spec.aliases),
                    expected_type=spec.expected_type,
                    role="topic_entity",
                )
                for spec in interested_nodes
                if spec.name
            ]
        if not topic_specs and not local_topic_specs and not interested_nodes:
            topic_specs = [
                EntityMentionSpec(name=entity, role="topic_entity")
                for entity in topic_entities
                if entity
            ]
        sub_questions.append(
            SubQuestionSpec(
                id=str(item.get("id") or f"sq{index}").strip() or f"sq{index}",
                question=sub_question_text or question,
                topic_entities=topic_specs,
                local_topic_entities=local_topic_specs,
                interested_nodes=interested_nodes,
                interested_relations=interested_relations,
                expected_answer_type=str(item.get("expected_answer_type") or "").strip(),
                expected_hop=max(1, min(_safe_int(item.get("expected_hop"), 1), dmax)),
                depends_on=_string_list(item.get("depends_on")),
                solver_type=normalized_solver_type,
                solver_reason=str(item.get("solver_reason") or "").strip(),
                downstream_filters=_string_list(item.get("downstream_filters")),
                execution_hints=dict(item.get("execution_hints", {}))
                if isinstance(item.get("execution_hints"), dict)
                else {},
            )
        )

    normalized_sub_questions = _drop_redundant_tail_verify_subquestions(question, sub_questions)
    return normalized_sub_questions or [_default_sub_question(question, topic_entities)]


def _parse_agentic_step(
    parsed: dict[str, object],
    question: str,
    loop_index: int,
    topic_entities: list[str],
    current_constraints: list[str],
) -> AgenticPlanStep:
    """Parse one planner-produced agentic step."""
    raw_topics = parsed.get("topic_entity_mentions")
    if raw_topics is None:
        raw_topics = parsed.get("topic_entities")
    topic_entity_mentions = [
        spec
        for spec in (
            _parse_entity_spec(item, "topic_entity")
            for item in _json_list(raw_topics)
        )
        if spec is not None
    ]
    relation_hints = [
        spec
        for spec in (
            _parse_relation_spec(item)
            for item in _json_list(parsed.get("relation_hints"))
        )
        if spec is not None
    ]
    if not topic_entity_mentions and topic_entities:
        topic_entity_mentions = [
            EntityMentionSpec(name=name, role="topic_entity")
            for name in topic_entities
            if name
        ]
    step_question = str(parsed.get("question") or question).strip() or question
    step_id = str(parsed.get("step_id") or f"step_{loop_index + 1}").strip() or f"step_{loop_index + 1}"
    carryover_constraints = _string_list(parsed.get("carryover_constraints")) or list(current_constraints)
    return AgenticPlanStep(
        step_id=step_id,
        question=step_question,
        goal=str(parsed.get("goal") or step_question).strip() or step_question,
        depends_on_step_ids=_string_list(parsed.get("depends_on_step_ids") or parsed.get("depends_on")),
        topic_entity_mentions=topic_entity_mentions,
        relation_hints=relation_hints,
        expected_answer_type=str(parsed.get("expected_answer_type") or "").strip(),
        carryover_constraints=carryover_constraints,
        downstream_filters=_string_list(parsed.get("downstream_filters")),
        stop_if_answered=bool(parsed.get("stop_if_answered", loop_index > 0)),
        strategy=str(parsed.get("strategy") or "auto").strip() or "auto",
    )


def analyze_question(
    question: str,
    llm: BaseLLM,
    dmax: int = 3,
    topic_entities_override: list[str] | None = None,
) -> QuestionAnalysisResult:
    """Analyze the question, extract topic entities, and predict a starting reasoning depth."""
    prompt = QUESTION_ANALYSIS_PROMPT.format(
        question=question,
        dmax=dmax,
        topic_entities_override=json.dumps(topic_entities_override or [], ensure_ascii=False),
    )
    raw = llm.generate(prompt)
    parsed = robust_json_parse(raw, fallback={})
    if isinstance(parsed, dict):
        override_topic_entities = [str(item).strip() for item in (topic_entities_override or []) if str(item).strip()]
        parsed_topic_entities = _string_list(parsed.get("topic_entities"))
        split_questions = _string_list(parsed.get("split_questions")) or [question]
        reasoning_indicator = str(parsed.get("reasoning_indicator") or question)
        topic_entities = override_topic_entities or _fallback_topic_entities(
            question=question,
            parsed_topic_entities=parsed_topic_entities,
            ordered_topic_entities=_string_list(parsed.get("ordered_topic_entities")),
        )
        ordered_topic_entities = _string_list(parsed.get("ordered_topic_entities")) or topic_entities
        predicted_depth = int(parsed.get("predicted_depth") or min(max(1, len(topic_entities)), dmax))
        sub_questions = _parse_sub_questions(
            parsed=parsed,
            question=question,
            topic_entities=[str(item) for item in ordered_topic_entities or topic_entities],
            split_questions=[str(item) for item in split_questions],
            dmax=dmax,
        )
    else:
        topic_entities = [str(item).strip() for item in (topic_entities_override or []) if str(item).strip()]
        split_questions = [question]
        reasoning_indicator = question
        ordered_topic_entities = topic_entities
        predicted_depth = min(max(1, len(topic_entities)), dmax)
        sub_questions = [_default_sub_question(question, topic_entities)]

    result = QuestionAnalysisResult(
        split_questions=[str(item) for item in split_questions],
        reasoning_indicator=reasoning_indicator,
        ordered_topic_entities=[str(item) for item in ordered_topic_entities],
        predicted_depth=max(1, min(predicted_depth, len(sub_questions), dmax)),
        topic_entities=[str(item) for item in topic_entities],
        sub_questions=sub_questions,
    )
    LOGGER.info("Question analysis result: %s", result)
    return result


def plan_agentic_step(
    question: str,
    llm: BaseLLM,
    question_analysis: QuestionAnalysisResult,
    run_state: AgenticRunState,
    dmax: int,
    max_loops: int | None = None,
) -> AgenticPlanStep:
    """Plan the next single reasoning step for the agent loop."""
    prompt = AGENTIC_STEP_PLANNING_PROMPT.format(
        question=question,
        dmax=dmax,
        loop_index=run_state.loop_index,
        remaining_loops=max(0, (max_loops if max_loops is not None else dmax) - run_state.loop_index),
        question_analysis=json.dumps(compact_question_analysis(question_analysis), ensure_ascii=False, indent=2),
        agent_state=json.dumps(compact_agentic_state(run_state), ensure_ascii=False, indent=2),
    )
    raw = llm.generate(prompt, max_tokens=384)
    parsed = robust_json_parse(raw, fallback={})
    if isinstance(parsed, dict):
        step = _parse_agentic_step(
            parsed=parsed,
            question=question,
            loop_index=run_state.loop_index,
            topic_entities=list(question_analysis.topic_entities or question_analysis.ordered_topic_entities),
            current_constraints=list(run_state.current_text_constraints),
        )
        LOGGER.info("Agentic plan step: %s", step)
        return step
    fallback = _default_agentic_step(
        question=question,
        loop_index=run_state.loop_index,
        topic_entities=list(question_analysis.topic_entities or question_analysis.ordered_topic_entities),
        current_constraints=list(run_state.current_text_constraints),
    )
    LOGGER.warning("Agentic planner parse failed; using fallback step %s", fallback.step_id)
    return fallback
