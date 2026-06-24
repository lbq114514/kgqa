"""Relation-type beam search for sub-question exploration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from statistics import mean
from time import perf_counter
from typing import Any, Literal, Protocol

from kgqa.kg.graph_api import BaseGraphAPI, ExternalGraphNeighbor
from kgqa.llm.base import BaseLLM
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.types import ReasoningPath, Triple, TripleFact

LOGGER = get_logger(__name__)

Direction = Literal["forward", "reverse"]
RelationRole = Literal["answer", "intermediate", "constraint", "alternative"]
RelationLabel = Literal["supportive", "weak", "off_topic"]


@dataclass(frozen=True)
class RelationCandidate:
    """One relation candidate aggregated over a node or frontier."""

    relation: str
    relation_name: str
    direction: Direction
    supporting_node_count: int
    total_neighbor_count: int
    support_ratio: float
    global_frequency: int
    sample_target_types: tuple[str, ...] = ()
    sample_target_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationAction:
    """One LLM-selected relation expansion action."""

    relation: str
    relation_name: str
    direction: Direction
    role: RelationRole
    confidence: float
    expected_next_type: str | None
    reason: str
    protect_for_next_hop: bool


@dataclass(frozen=True)
class RelationPathStep:
    """One step in a relation path."""

    relation: str
    relation_name: str
    direction: Direction
    role: str
    confidence: float


@dataclass(frozen=True)
class EvidenceEdge:
    """One edge preserved as evidence during frontier expansion."""

    source_id: str
    relation: str
    direction: Direction
    target_id: str


@dataclass(frozen=True)
class NodeMetadata:
    """Lightweight node metadata for frontier summarization and scoring."""

    node_id: str
    name: str
    types: tuple[str, ...]
    degree: int
    is_literal: bool
    is_probable_cvt: bool


@dataclass(frozen=True)
class FrontierExpansion:
    """The result of expanding a frontier by one chosen relation."""

    next_nodes: tuple[str, ...]
    evidence_paths: dict[str, tuple[EvidenceEdge, ...]]
    source_node_count: int
    edge_count: int
    truncated: bool


@dataclass(frozen=True)
class RelationBeamState:
    """One beam-search state keyed by a relation path and frontier."""

    state_id: str
    frontier_nodes: tuple[str, ...]
    relation_path: tuple[RelationPathStep, ...]
    evidence_paths: dict[str, tuple[EvidenceEdge, ...]]
    source_nodes: tuple[str, ...]
    hop: int
    semantic_score: float
    type_score: float
    answer_score: float
    diversity_score: float
    cost_penalty: float
    protected_until_hop: int
    contains_answer_candidates: bool
    path_reason: str
    visited_relation_signatures: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class RelationBeamConfig:
    """Search configuration for relation-type beam exploration."""

    max_hops: int = 3
    beam_width: int = 6
    relations_per_state: int = 3
    relation_retrieval_top_k: int = 20
    neighbors_per_source: int = 20
    max_nodes_per_path: int = 50
    llm_path_rerank_top_k: int = 20
    answer_threshold: float = 0.82
    continue_threshold: float = 0.45
    llm_relation_batch_size: int = 100


CVT_BUNDLE_SOURCE_STAGE = "relation_beam_cvt_bundle"
CVT_DISPLAY_PLACEHOLDER = "[CVT]"
_CVT_CONSTRAINT_KEYWORDS = (
    "date",
    "year",
    "time",
    "start",
    "end",
    "from",
    "to",
    "title",
    "office",
    "holder",
    "spouse",
    "nominee",
    "winner",
    "employee",
    "employment",
    "role",
    "team",
    "performance",
    "jurisdiction",
    "district",
    "governmental body",
)
_CVT_SCHEMA_RELATION_PRIORS: dict[str, tuple[str, ...]] = {
    "government.government_position_held": (
        "office_holder",
        "basic_title",
        "office_position_or_title",
        "jurisdiction_of_office",
        "district_represented",
        "governmental_body",
        "from",
        "to",
    ),
    "people.marriage": (
        "spouse",
        "from",
        "to",
        "date",
        "location",
    ),
    "award.award_nomination": (
        "award",
        "award_nominee",
        "award_winner",
        "year",
        "category",
        "ceremony",
    ),
    "business.employment_tenure": (
        "person",
        "company",
        "title",
        "role",
        "from",
        "to",
    ),
    "film.performance": (
        "actor",
        "character",
        "film",
        "special_performance_type",
    ),
    "sports.sports_team_roster": (
        "player",
        "position",
        "number",
        "from",
        "to",
        "team",
    ),
}


@dataclass(frozen=True)
class SubquestionExplorationResult:
    """Thin adapter result for DAG integration."""

    answer_entities: list[str]
    answer_states: list[RelationBeamState]
    searched_hops: int
    complete: bool
    best_score: float
    evidence_paths: dict[str, tuple[EvidenceEdge, ...]]
    candidate_paths: list[ReasoningPath]


class RelationBeamLLM(Protocol):
    """LLM contract for relation beam planning and reranking."""

    def select_relation_actions(
        self,
        *,
        original_question: str,
        subquestion: str,
        expected_answer_type: str | None,
        state: RelationBeamState,
        frontier_summary: dict[str, Any],
        candidates: list[RelationCandidate],
        anchor_mentions: list[str],
        relation_hint_names: list[str],
        resolved_dependencies: list[dict[str, Any]],
        remaining_hops: int,
        top_k: int,
    ) -> list[RelationAction]:
        """Select relation actions for one frontier state."""

    def rerank_paths(
        self,
        *,
        original_question: str,
        subquestion: str,
        expected_answer_type: str | None,
        states: list[RelationBeamState],
        anchor_mentions: list[str],
        relation_hint_names: list[str],
        resolved_dependencies: list[dict[str, Any]],
        remaining_hops: int,
    ) -> dict[str, dict[str, Any]]:
        """Return path-level rerank annotations keyed by state id."""


class DeterministicMockRelationBeamLLM:
    """Deterministic LLM mock used by unit tests and fallback wiring."""

    def select_relation_actions(
        self,
        *,
        original_question: str,
        subquestion: str,
        expected_answer_type: str | None,
        state: RelationBeamState,
        frontier_summary: dict[str, Any],
        candidates: list[RelationCandidate],
        anchor_mentions: list[str],
        relation_hint_names: list[str],
        resolved_dependencies: list[dict[str, Any]],
        remaining_hops: int,
        top_k: int,
    ) -> list[RelationAction]:
        actions: list[RelationAction] = []
        for candidate in candidates[:top_k]:
            role: RelationRole = "intermediate"
            type_hint = normalize_text(expected_answer_type or "")
            names_blob = normalize_text(" ".join(candidate.sample_target_names))
            types_blob = normalize_text(" ".join(candidate.sample_target_types))
            if type_hint and (type_hint in names_blob or type_hint in types_blob):
                role = "answer"
            elif candidate.support_ratio >= 0.5:
                role = "constraint"
            actions.append(
                RelationAction(
                    relation=candidate.relation,
                    relation_name=candidate.relation_name,
                    direction=candidate.direction,
                    role=role,
                    confidence=min(1.0, 0.45 + (candidate.support_ratio * 0.4)),
                    expected_next_type=expected_answer_type,
                    reason="deterministic mock selection",
                    protect_for_next_hop=role == "intermediate",
                )
            )
        return actions

    def rerank_paths(
        self,
        *,
        original_question: str,
        subquestion: str,
        expected_answer_type: str | None,
        states: list[RelationBeamState],
        anchor_mentions: list[str],
        relation_hint_names: list[str],
        resolved_dependencies: list[dict[str, Any]],
        remaining_hops: int,
    ) -> dict[str, dict[str, Any]]:
        verdicts: dict[str, dict[str, Any]] = {}
        for state in states:
            verdicts[state.state_id] = {
                "semantic_relevance": min(1.0, max(0.0, state.semantic_score)),
                "answer_likelihood": min(1.0, max(0.0, state.answer_score)),
                "continue_likelihood": min(1.0, max(0.0, state.type_score)),
                "verdict": "answer" if state.answer_score >= 0.7 else "continue",
            }
        return verdicts


RELATION_ACTION_PROMPT = """You are a KG relation-path planner, not an answer generator.
SQ: {subquestion}
Expected type: {expected_answer_type}
Anchors: {anchor_mentions}
Relation hints: {relation_hints}
Resolved deps: {resolved_dependencies}
Current path: {relation_path}
Remaining hops: {remaining_hops}
Frontier: {frontier_summary}
Candidates: {candidates}

Return JSON:
{{
  "actions": [
    {{
      "relation": "...",
      "direction": "forward",
      "role": "answer",
      "confidence": 0.0,
      "expected_next_type": "...",
      "reason": "...",
      "protect_for_next_hop": false
    }}
  ]
}}

Rules:
- Select only from the provided candidate relations.
- Keep the provided direction.
- You may choose direct-answer relations or intermediate relations.
- CVT or mediator transitions are valid and must not be rejected only because names are sparse.
- Consider the whole current relation path and remaining hops.
- Use the anchor mentions and resolved dependencies to preserve the current semantic focus.
- Prefer relations that help satisfy the current sub-question constraints, not just surface-name overlap.
- Output JSON only.
"""

PATH_RERANK_PROMPT = """You rerank relation paths for sub-question search.
SQ: {subquestion}
Expected type: {expected_answer_type}
Anchors: {anchor_mentions}
Relation hints: {relation_hints}
Resolved deps: {resolved_dependencies}
Remaining hops: {remaining_hops}
States: {states}

Return JSON:
{{
  "states": [
    {{
      "state_id": "...",
      "semantic_relevance": 0.0,
      "answer_likelihood": 0.0,
      "continue_likelihood": 0.0,
      "verdict": "answer"
    }}
  ]
}}

Verdict must be one of: answer, continue, discard.
Output JSON only.
"""


class BaseLLMRelationBeamAdapter:
    """Adapter from project BaseLLM to the RelationBeamLLM protocol."""

    def __init__(self, llm: BaseLLM, relation_batch_size: int = 100) -> None:
        self.llm = llm
        self.relation_batch_size = max(1, int(relation_batch_size))

    @staticmethod
    def _json_compact(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _compact_list(items: list[str], limit: int, max_chars: int = 48) -> list[str]:
        compact: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            normalized = normalize_text(text)
            if normalized in seen:
                continue
            seen.add(normalized)
            compact.append(text[:max_chars])
            if len(compact) >= limit:
                break
        return compact

    @classmethod
    def _compact_resolved_dependencies(cls, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_items: list[dict[str, Any]] = []
        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            compact_items.append(
                {
                    "sub_question_id": str(item.get("sub_question_id") or "")[:32],
                    "answer": str(item.get("answer") or "")[:64],
                    "entity_ids": cls._compact_list([str(value) for value in item.get("entity_ids", [])], limit=1, max_chars=32),
                }
            )
        return compact_items

    @classmethod
    def _compact_relation_candidates(cls, candidates: list[RelationCandidate]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for candidate in candidates:
            item: dict[str, Any] = {
                "r": candidate.relation,
                "d": candidate.direction,
                "s": round(candidate.support_ratio, 3),
            }
            relation_name = candidate.relation_name[:36].strip()
            if relation_name and normalize_text(relation_name) != normalize_text(candidate.relation):
                item["n"] = relation_name
            target_types = cls._compact_list(list(candidate.sample_target_types), limit=1, max_chars=20)
            target_names = cls._compact_list(list(candidate.sample_target_names), limit=1, max_chars=24)
            if target_types:
                item["tt"] = target_types
            if target_names:
                item["tn"] = target_names
            compact.append(item)
        return compact

    @classmethod
    def _compact_frontier_summary(cls, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "n": int(summary.get("node_count", 0) or 0),
            "e": int(summary.get("entity_count", 0) or 0),
            "l": int(summary.get("literal_count", 0) or 0),
            "c": round(float(summary.get("probable_cvt_ratio", 0.0) or 0.0), 3),
            "names": cls._compact_list([str(item) for item in summary.get("sample_names", [])], limit=2, max_chars=24),
            "types": cls._compact_list([str(item) for item in summary.get("sample_types", [])], limit=2, max_chars=18),
        }

    def select_relation_actions(
        self,
        *,
        original_question: str,
        subquestion: str,
        expected_answer_type: str | None,
        state: RelationBeamState,
        frontier_summary: dict[str, Any],
        candidates: list[RelationCandidate],
        anchor_mentions: list[str],
        relation_hint_names: list[str],
        resolved_dependencies: list[dict[str, Any]],
        remaining_hops: int,
        top_k: int,
    ) -> list[RelationAction]:
        compact_anchor_mentions = self._compact_list(anchor_mentions, limit=4, max_chars=40)
        compact_relation_hints = self._compact_list(relation_hint_names, limit=4, max_chars=32)
        compact_dependencies = self._compact_resolved_dependencies(resolved_dependencies)
        compact_relation_path = [
            {
                "relation": step.relation,
                "direction": step.direction,
                "role": step.role,
            }
            for step in state.relation_path[-3:]
        ]
        candidate_index = {(item.relation, item.direction): item for item in candidates}
        actions_by_signature: dict[tuple[str, str], RelationAction] = {}
        frontier_summary_json = self._json_compact(self._compact_frontier_summary(frontier_summary))
        for batch_index, batch_start in enumerate(range(0, len(candidates), self.relation_batch_size), start=1):
            batch = candidates[batch_start : batch_start + self.relation_batch_size]
            prompt = RELATION_ACTION_PROMPT.format(
                subquestion=subquestion,
                expected_answer_type=expected_answer_type or "",
                anchor_mentions=self._json_compact(compact_anchor_mentions),
                relation_hints=self._json_compact(compact_relation_hints),
                resolved_dependencies=self._json_compact(compact_dependencies),
                relation_path=self._json_compact(compact_relation_path),
                remaining_hops=remaining_hops,
                frontier_summary=frontier_summary_json,
                candidates=self._json_compact(self._compact_relation_candidates(batch)),
            )
            LOGGER.info(
                "Relation beam llm_prompt kind=select_actions state_id=%s hop=%d batch=%d candidate_count=%d prompt_chars=%d frontier_nodes=%d remaining_hops=%d",
                state.state_id,
                state.hop,
                batch_index,
                len(batch),
                len(prompt),
                len(state.frontier_nodes),
                remaining_hops,
            )
            raw = self.llm.generate(prompt, max_tokens=512)
            parsed = robust_json_parse(raw, fallback={})
            if not isinstance(parsed, dict):
                continue
            for item in parsed.get("actions", []):
                if not isinstance(item, dict):
                    continue
                relation = str(item.get("relation") or "").strip()
                direction = str(item.get("direction") or "").strip().lower()
                if (relation, direction) not in candidate_index:
                    continue
                role = str(item.get("role") or "intermediate").strip().lower()
                if role not in {"answer", "intermediate", "constraint", "alternative"}:
                    role = "intermediate"
                confidence = float(item.get("confidence", 0.5) or 0.5)
                confidence = min(1.0, max(0.0, confidence))
                chosen = candidate_index[(relation, direction)]
                action = RelationAction(
                    relation=relation,
                    relation_name=chosen.relation_name,
                    direction=chosen.direction,
                    role=role,  # type: ignore[arg-type]
                    confidence=confidence,
                    expected_next_type=str(item.get("expected_next_type") or "").strip() or None,
                    reason=str(item.get("reason") or "").strip(),
                    protect_for_next_hop=bool(item.get("protect_for_next_hop", False)),
                )
                previous = actions_by_signature.get((relation, direction))
                if previous is None or action.confidence > previous.confidence:
                    actions_by_signature[(relation, direction)] = action
        ranked_actions = sorted(
            actions_by_signature.values(),
            key=lambda action: (
                0 if action.role == "answer" else 1,
                -action.confidence,
                action.relation,
                action.direction,
            ),
        )
        return ranked_actions[:top_k]

    def rerank_paths(
        self,
        *,
        original_question: str,
        subquestion: str,
        expected_answer_type: str | None,
        states: list[RelationBeamState],
        anchor_mentions: list[str],
        relation_hint_names: list[str],
        resolved_dependencies: list[dict[str, Any]],
        remaining_hops: int,
    ) -> dict[str, dict[str, Any]]:
        compact_anchor_mentions = self._compact_list(anchor_mentions, limit=4, max_chars=40)
        compact_relation_hints = self._compact_list(relation_hint_names, limit=4, max_chars=32)
        compact_dependencies = self._compact_resolved_dependencies(resolved_dependencies)
        prompt = PATH_RERANK_PROMPT.format(
            subquestion=subquestion,
            expected_answer_type=expected_answer_type or "",
            anchor_mentions=self._json_compact(compact_anchor_mentions),
            relation_hints=self._json_compact(compact_relation_hints),
            resolved_dependencies=self._json_compact(compact_dependencies),
            remaining_hops=remaining_hops,
            states=self._json_compact(
                [
                    {
                        "id": state.state_id,
                        "path": [
                            {"r": step.relation, "d": step.direction, "role": step.role}
                            for step in state.relation_path[-3:]
                        ],
                        "f": len(state.frontier_nodes),
                        "sem": round(state.semantic_score, 4),
                        "typ": round(state.type_score, 4),
                        "ans": round(state.answer_score, 4),
                        "div": round(state.diversity_score, 4),
                        "cost": round(state.cost_penalty, 4),
                    }
                    for state in states
                ]
            ),
        )
        LOGGER.info(
            "Relation beam llm_prompt kind=rerank_paths state_count=%d prompt_chars=%d remaining_hops=%d",
            len(states),
            len(prompt),
            remaining_hops,
        )
        raw = self.llm.generate(prompt, max_tokens=512)
        parsed = robust_json_parse(raw, fallback={})
        verdicts: dict[str, dict[str, Any]] = {}
        valid_ids = {state.state_id for state in states}
        if isinstance(parsed, dict):
            for item in parsed.get("states", []):
                if not isinstance(item, dict):
                    continue
                state_id = str(item.get("state_id") or "").strip()
                if state_id not in valid_ids:
                    continue
                verdict = str(item.get("verdict") or "continue").strip().lower()
                if verdict not in {"answer", "continue", "discard"}:
                    verdict = "continue"
                verdicts[state_id] = {
                    "semantic_relevance": _clamp_score(item.get("semantic_relevance", 0.5)),
                    "answer_likelihood": _clamp_score(item.get("answer_likelihood", 0.5)),
                    "continue_likelihood": _clamp_score(item.get("continue_likelihood", 0.5)),
                    "verdict": verdict,
                }
        return verdicts


def _clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.5
    return min(1.0, max(0.0, score))


def normalize_text(text: str | None) -> str:
    """Normalize text for heuristic scoring."""
    normalized = str(text or "").strip().lower()
    for separator in (".", "_", "/", "-", "(", ")", "|", ",", ":"):
        normalized = normalized.replace(separator, " ")
    return " ".join(normalized.split())


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in normalize_text(left).split() if token}
    right_tokens = {token for token in normalize_text(right).split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _is_entity_id(value: str | None) -> bool:
    stripped = str(value or "").strip()
    return stripped.startswith(("m.", "g."))


def _infer_cvt_schema_hint(node_metadata: NodeMetadata | None, evidence: tuple[EvidenceEdge, ...]) -> str:
    schema_candidates: list[str] = []
    if node_metadata is not None:
        schema_candidates.extend(node_metadata.types)
    schema_candidates.extend(edge.relation for edge in evidence if edge.relation)
    for candidate in schema_candidates:
        normalized = normalize_text(candidate)
        for schema_hint in _CVT_SCHEMA_RELATION_PRIORS:
            if normalize_text(schema_hint) in normalized:
                return schema_hint
    return ""


def _relation_text_matches_keywords(relation_text: str, keywords: list[str]) -> bool:
    normalized = normalize_text(relation_text)
    return any(normalize_text(keyword) in normalized for keyword in keywords if keyword)


def _cvt_bundle_relation_keywords(
    *,
    subquestion: str,
    expected_answer_type: str | None,
    relation_hint_names: list[str] | None,
    anchor_mentions: list[str] | None,
    schema_hint: str,
) -> list[str]:
    keywords = [
        *(_CVT_SCHEMA_RELATION_PRIORS.get(schema_hint, ()) if schema_hint else ()),
        *(_CVT_CONSTRAINT_KEYWORDS),
        expected_answer_type or "",
        subquestion,
        *(relation_hint_names or []),
        *(anchor_mentions or []),
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = normalize_text(keyword)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(keyword)
    return deduped


def _cvt_bundle_edge_score(
    edge: ExternalGraphNeighbor,
    *,
    keyword_terms: list[str],
    schema_hint: str,
    expected_answer_type: str | None,
    target_metadata: NodeMetadata | None,
) -> tuple[float, str]:
    relation_text = normalize_text(f"{edge.triple.relation} {edge.relation_name}")
    target_label = edge.triple.head if edge.reversed else edge.triple.tail
    target_text = normalize_text(target_label)
    target_types = normalize_text(" ".join(target_metadata.types if target_metadata is not None else ()))
    score = 0.0
    score += 12.0 * max((_token_overlap(term, relation_text) for term in keyword_terms), default=0.0)
    score += 4.0 * max((_token_overlap(term, target_text) for term in keyword_terms), default=0.0)
    if expected_answer_type:
        expected_score = _type_match_score(
            expected_answer_type,
            node_name=target_label,
            node_types=target_metadata.types if target_metadata is not None else (),
        )
        score += 6.0 * expected_score
    if schema_hint:
        score += 8.0 * max((_token_overlap(term, relation_text) for term in _CVT_SCHEMA_RELATION_PRIORS.get(schema_hint, ())), default=0.0)
    if _relation_text_matches_keywords(relation_text, ["from", "to", "date", "year", "time"]):
        score += 3.0
        return score, "constraint"
    if _relation_text_matches_keywords(relation_text, ["title", "office", "holder", "spouse", "role", "team", "performance"]):
        score += 3.0
    if expected_answer_type and target_metadata is not None and not target_metadata.is_probable_cvt:
        type_match = _type_match_score(expected_answer_type, node_name=target_label, node_types=target_metadata.types)
        if type_match >= 0.75:
            score += 8.0
            return score, "answer"
    if target_metadata is not None and target_metadata.is_literal:
        return score, "constraint"
    if target_metadata is not None and target_metadata.is_probable_cvt:
        score -= 2.0
        return score, "intermediate"
    if any(token in relation_text for token in ("title", "office", "holder", "spouse", "award", "role", "position")):
        return score, "answer"
    return score, "constraint"


def _edge_to_triple(edge: ExternalGraphNeighbor, graph: BaseGraphAPI) -> Triple:
    source_label = graph.get_entity_display_name(edge.source_id)
    target_is_entity = _is_entity_id(edge.neighbor_id)
    target_label = graph.get_entity_display_name(edge.neighbor_id) if target_is_entity else edge.neighbor_id
    if edge.reversed:
        return Triple(head=target_label, relation=edge.triple.relation, tail=source_label)
    return Triple(head=source_label, relation=edge.triple.relation, tail=target_label)


def _edge_to_fact(edge: ExternalGraphNeighbor, graph: BaseGraphAPI) -> TripleFact:
    """Convert one backend edge to an id-preserving fact."""
    source_label = graph.get_entity_display_name(edge.source_id)
    target_is_entity = _is_entity_id(edge.neighbor_id)
    target_label = graph.get_entity_display_name(edge.neighbor_id) if target_is_entity else edge.neighbor_id
    if edge.reversed:
        return TripleFact(
            head_id=edge.neighbor_id if target_is_entity else "",
            head_label=target_label,
            relation_id=edge.triple.relation,
            tail_id=edge.source_id,
            tail_label=source_label,
            tail_kind="id",
        )
    return TripleFact(
        head_id=edge.source_id,
        head_label=source_label,
        relation_id=edge.triple.relation,
        tail_id=edge.neighbor_id if target_is_entity else "",
        tail_label=target_label,
        tail_kind="id" if target_is_entity else "literal",
    )


def _evidence_edge_to_fact(edge: EvidenceEdge, graph: BaseGraphAPI) -> TripleFact:
    """Convert one frontier evidence edge to an id-preserving fact."""
    source_label = graph.get_entity_display_name(edge.source_id)
    target_is_entity = _is_entity_id(edge.target_id)
    target_label = graph.get_entity_display_name(edge.target_id) if target_is_entity else edge.target_id
    if edge.direction == "forward":
        return TripleFact(
            head_id=edge.source_id,
            head_label=source_label,
            relation_id=edge.relation,
            tail_id=edge.target_id if target_is_entity else "",
            tail_label=target_label,
            tail_kind="id" if target_is_entity else "literal",
        )
    return TripleFact(
        head_id=edge.target_id if target_is_entity else "",
        head_label=target_label,
        relation_id=edge.relation,
        tail_id=edge.source_id,
        tail_label=source_label,
        tail_kind="id",
    )


def _append_unique_node(nodes: list[str], value: str) -> None:
    """Keep bundle nodes readable by avoiding repeated labels."""
    normalized = str(value or "").strip()
    if not normalized or normalized in nodes:
        return
    nodes.append(normalized)


def _cvt_display_placeholder(cvt_label: str) -> str:
    """Render one stable but anonymized placeholder for a specific CVT node."""
    normalized = str(cvt_label or "").strip()
    if not normalized:
        return CVT_DISPLAY_PLACEHOLDER
    suffix = normalized.split(".")[-1][-4:]
    if not suffix:
        return CVT_DISPLAY_PLACEHOLDER
    return f"[CVT:{suffix}]"


def _format_bundle_text(
    *,
    entry_path: ReasoningPath,
    cvt_label: str,
    bundle_triples: list[Triple],
    graph: BaseGraphAPI,
) -> str:
    """Render a CVT bundle as one grouped record instead of repeated arrow segments."""
    placeholder = _cvt_display_placeholder(cvt_label)
    grouped_values: dict[str, list[str]] = {}
    for triple in bundle_triples:
        relation_name = graph.get_relation_display_name(triple.relation)
        value = triple.tail if triple.head == cvt_label else triple.head
        values = grouped_values.setdefault(relation_name, [])
        if value not in values:
            values.append(value)
    grouped_parts = []
    for relation_name, values in grouped_values.items():
        if len(values) == 1:
            grouped_parts.append(f"{relation_name}: {values[0]}")
        else:
            grouped_parts.append(f"{relation_name}: [{', '.join(values)}]")
    if not grouped_parts:
        return entry_path.text.replace(cvt_label, placeholder)
    entry_text = entry_path.text.replace(cvt_label, placeholder)
    return f"{entry_text} ; {placeholder} {{{'; '.join(grouped_parts)}}}"


def _build_plain_reasoning_path_from_evidence(
    graph: BaseGraphAPI,
    target_node_id: str,
    evidence: tuple[EvidenceEdge, ...],
    *,
    path_score: float,
    score_breakdown: dict[str, float],
    matched_answer_type_hints: list[str],
) -> ReasoningPath:
    if not evidence:
        label = graph.get_entity_display_name(target_node_id)
        return ReasoningPath(
            triples=[],
            nodes=[label],
            text=label,
            source_stage="relation_beam",
            pruning_status="preserved",
            path_score=path_score,
            search_score_breakdown=score_breakdown,
            terminal_node_id=target_node_id,
            terminal_node_kind="literal" if not _is_entity_id(target_node_id) else "id",
            matched_answer_type_hints=matched_answer_type_hints,
            search_strategy="relation_beam",
        )
    nodes: list[str] = []
    triples: list[Triple] = []
    triple_facts: list[TripleFact] = []
    for index, edge in enumerate(evidence):
        source_label = graph.get_entity_display_name(edge.source_id)
        target_is_entity = _is_entity_id(edge.target_id)
        target_label = graph.get_entity_display_name(edge.target_id) if target_is_entity else edge.target_id
        if index == 0:
            nodes.append(source_label)
        if edge.direction == "forward":
            triples.append(Triple(head=source_label, relation=edge.relation, tail=target_label))
            nodes.append(target_label)
        else:
            triples.append(Triple(head=target_label, relation=edge.relation, tail=source_label))
            nodes.append(target_label)
        triple_facts.append(_evidence_edge_to_fact(edge, graph))
    text_parts = [
        f"{triple.head} -> {graph.get_relation_display_name(triple.relation)} -> {triple.tail}"
        for triple in triples
    ]
    return ReasoningPath(
        triples=triples,
        nodes=nodes,
        text=" ; ".join(text_parts),
        source_stage="relation_beam",
        triple_facts=triple_facts,
        pruning_status="preserved",
        matched_relations=[triple.relation for triple in triples],
        path_score=path_score,
        search_score_breakdown=score_breakdown,
        edge_ids=[f"{edge.source_id}:{edge.relation}:{edge.target_id}" for edge in evidence],
        terminal_node_id=target_node_id,
        terminal_node_kind="literal" if not _is_entity_id(target_node_id) else "id",
        matched_answer_type_hints=matched_answer_type_hints,
        search_strategy="relation_beam",
    )


def _expand_cvt_bundle_path(
    graph: BaseGraphAPI,
    cvt_node_id: str,
    evidence: tuple[EvidenceEdge, ...],
    *,
    subquestion: str,
    expected_answer_type: str | None,
    relation_hint_names: list[str] | None,
    anchor_mentions: list[str] | None,
    path_score: float,
    score_breakdown: dict[str, float],
    node_metadata: NodeMetadata | None = None,
) -> ReasoningPath | None:
    if not hasattr(graph, "get_node_edges"):
        return None
    try:
        cvt_edges = getattr(graph, "get_node_edges")(cvt_node_id, include_reverse=False, include_literals=True, limit=64)
    except Exception:
        return None
    if not cvt_edges:
        return None

    schema_hint = _infer_cvt_schema_hint(node_metadata, evidence)
    keyword_terms = _cvt_bundle_relation_keywords(
        subquestion=subquestion,
        expected_answer_type=expected_answer_type,
        relation_hint_names=relation_hint_names,
        anchor_mentions=anchor_mentions,
        schema_hint=schema_hint,
    )
    target_entity_ids = [edge.neighbor_id for edge in cvt_edges if _is_entity_id(edge.neighbor_id)]
    target_metadata = {}
    if hasattr(graph, "get_nodes_metadata_batched") and target_entity_ids:
        try:
            target_metadata = getattr(graph, "get_nodes_metadata_batched")(target_entity_ids)
        except Exception:
            target_metadata = {}

    scored_edges: list[tuple[float, str, ExternalGraphNeighbor]] = []
    for edge in cvt_edges:
        neighbor_metadata = target_metadata.get(edge.neighbor_id)
        score, role = _cvt_bundle_edge_score(
            edge,
            keyword_terms=keyword_terms,
            schema_hint=schema_hint,
            expected_answer_type=expected_answer_type,
            target_metadata=neighbor_metadata,
        )
        scored_edges.append((score, role, edge))
    scored_edges.sort(key=lambda item: (-item[0], item[2].triple.relation, item[2].neighbor_id))

    selected_edges: list[tuple[float, str, ExternalGraphNeighbor]] = []
    seen_relation_counts: dict[str, int] = {}
    max_bundle_edges = 6
    entry_source_ids = {edge.source_id for edge in evidence if edge.source_id}

    def _can_keep_bundle_edge(item: tuple[float, str, ExternalGraphNeighbor]) -> bool:
        _, role, edge = item
        relation_id = edge.triple.relation
        current_count = seen_relation_counts.get(relation_id, 0)
        per_relation_cap = 2 if role in {"answer", "constraint"} else 1
        if current_count >= per_relation_cap:
            return False
        if edge.neighbor_id in entry_source_ids and role != "answer":
            return False
        return True

    def _remember_bundle_edge(item: tuple[float, str, ExternalGraphNeighbor]) -> None:
        relation_id = item[2].triple.relation
        seen_relation_counts[relation_id] = seen_relation_counts.get(relation_id, 0) + 1

    answer_edges = [item for item in scored_edges if item[1] == "answer"][:4]
    constraint_edges = [item for item in scored_edges if item[1] == "constraint"][:4]
    for item in answer_edges:
        if len(selected_edges) >= max_bundle_edges or not _can_keep_bundle_edge(item):
            continue
        selected_edges.append(item)
        _remember_bundle_edge(item)
    for item in constraint_edges:
        if len(selected_edges) >= max_bundle_edges or item in selected_edges or not _can_keep_bundle_edge(item):
            continue
        selected_edges.append(item)
        _remember_bundle_edge(item)
    for item in scored_edges:
        if len(selected_edges) >= max_bundle_edges:
            break
        if item in selected_edges or not _can_keep_bundle_edge(item):
            continue
        selected_edges.append(item)
        _remember_bundle_edge(item)
    expandable_answer_edges = [
        item
        for item in answer_edges
        if _can_keep_bundle_edge(item) or item in selected_edges
    ]
    allow_single_answer_bundle = len(expandable_answer_edges) >= 1
    if len(selected_edges) < 2 and not allow_single_answer_bundle:
        return None
    if not answer_edges and not constraint_edges:
        return None

    entry_path = _build_plain_reasoning_path_from_evidence(
        graph,
        cvt_node_id,
        evidence,
        path_score=path_score,
        score_breakdown=score_breakdown,
        matched_answer_type_hints=[],
    )
    cvt_label = graph.get_entity_display_name(cvt_node_id)
    triples = list(entry_path.triples)
    triple_facts = list(entry_path.triple_facts)
    nodes = list(entry_path.nodes)
    matched_relations = [triple.relation for triple in triples]
    matched_answer_type_hints: list[str] = []
    answer_terminal_id = cvt_node_id
    answer_terminal_kind = "id"
    best_answer_score = -1.0
    bundle_triples: list[Triple] = []
    for score, role, edge in selected_edges:
        triple = _edge_to_triple(edge, graph)
        fact = _edge_to_fact(edge, graph)
        triples.append(triple)
        triple_facts.append(fact)
        bundle_triples.append(triple)
        _append_unique_node(nodes, cvt_label)
        tail_value = triple.tail if triple.head == cvt_label else triple.head
        _append_unique_node(nodes, tail_value)
        matched_relations.append(triple.relation)
        if role == "answer":
            candidate_penalty = 0.1 if edge.neighbor_id in entry_source_ids else 0.0
            adjusted_score = score - candidate_penalty
            if adjusted_score > best_answer_score:
                best_answer_score = adjusted_score
                answer_terminal_id = edge.neighbor_id
                answer_terminal_kind = "id" if _is_entity_id(edge.neighbor_id) else "literal"
                if expected_answer_type:
                    matched_answer_type_hints = [expected_answer_type]
    text = _format_bundle_text(
        entry_path=entry_path,
        cvt_label=cvt_label,
        bundle_triples=bundle_triples,
        graph=graph,
    )
    bundle_breakdown = dict(score_breakdown)
    bundle_breakdown.update(
        {
            "bundle_expanded": 1.0,
            "bundle_relation_count": float(len(selected_edges)),
            "bundle_answer_entity_count": float(len(answer_edges)),
            "cvt_frontier_detected": 1.0,
        }
    )
    if schema_hint:
        bundle_breakdown["bundle_schema_hint"] = 1.0
    LOGGER.info(
        "Relation beam CVT bundle expanded cvt_node_id=%s schema_hint=%s bundle_relation_count=%d bundle_answer_entity_count=%d answer_terminal_id=%s constraint_relation_ids=%s single_answer_bundle=%s",
        cvt_node_id,
        schema_hint,
        len(selected_edges),
        len(answer_edges),
        answer_terminal_id,
        [edge.triple.relation for _, role, edge in selected_edges if role == "constraint"][:6],
        allow_single_answer_bundle,
    )
    return ReasoningPath(
        triples=triples,
        nodes=nodes,
        text=text,
        source_stage=CVT_BUNDLE_SOURCE_STAGE,
        triple_facts=triple_facts,
        pruning_status="preserved",
        matched_relations=matched_relations,
        path_score=path_score + 0.05,
        search_score_breakdown=bundle_breakdown,
        edge_ids=[*entry_path.edge_ids, *[f"{edge.source_id}:{edge.triple.relation}:{edge.neighbor_id}" for _, _, edge in selected_edges]],
        terminal_node_id=answer_terminal_id,
        terminal_node_kind=answer_terminal_kind,
        matched_answer_type_hints=matched_answer_type_hints,
        search_strategy="relation_beam",
    )


def _reasoning_paths_from_evidence(
    graph: BaseGraphAPI,
    target_node_id: str,
    evidence: tuple[EvidenceEdge, ...],
    *,
    subquestion: str,
    expected_answer_type: str | None,
    relation_hint_names: list[str] | None,
    anchor_mentions: list[str] | None,
    supporting_state: RelationBeamState | None = None,
    node_metadata: NodeMetadata | None = None,
) -> list[ReasoningPath]:
    score_breakdown: dict[str, float] = {}
    path_score = 0.0
    matched_answer_type_hints: list[str] = []
    if supporting_state is not None:
        path_score = state_score(supporting_state)
        score_breakdown = {
            "semantic_score": supporting_state.semantic_score,
            "type_score": supporting_state.type_score,
            "answer_score": supporting_state.answer_score,
        }
    if expected_answer_type and node_metadata is not None:
        match_score = _type_match_score(
            expected_answer_type,
            node_name=node_metadata.name,
            node_types=node_metadata.types,
        )
        if match_score >= 0.6:
            matched_answer_type_hints = [expected_answer_type]
    if node_metadata is not None and node_metadata.is_probable_cvt:
        bundle_path = _expand_cvt_bundle_path(
            graph,
            target_node_id,
            evidence,
            subquestion=subquestion,
            expected_answer_type=expected_answer_type,
            relation_hint_names=relation_hint_names,
            anchor_mentions=anchor_mentions,
            path_score=path_score,
            score_breakdown=score_breakdown,
            node_metadata=node_metadata,
        )
        if bundle_path is not None:
            return [bundle_path]
    return [
        _build_plain_reasoning_path_from_evidence(
            graph,
            target_node_id,
            evidence,
            path_score=path_score,
            score_breakdown=score_breakdown,
            matched_answer_type_hints=matched_answer_type_hints,
        )
    ]


def expand_cvt_bundle_paths(
    graph: BaseGraphAPI,
    candidate_paths: list[ReasoningPath] | None = None,
    *,
    paths: list[ReasoningPath] | None = None,
    subquestion: str,
    expected_answer_type: str | None,
    relation_hint_names: list[str] | None = None,
    anchor_mentions: list[str] | None = None,
) -> list[ReasoningPath]:
    """Post-process ordinary candidate paths into CVT bundle paths when they terminate on one CVT node."""
    if candidate_paths is None:
        candidate_paths = paths or []
    if not candidate_paths or not hasattr(graph, "get_nodes_metadata_batched"):
        return candidate_paths
    terminal_ids = [path.terminal_node_id for path in candidate_paths if _is_entity_id(path.terminal_node_id)]
    if not terminal_ids:
        return candidate_paths
    try:
        metadata_by_id = getattr(graph, "get_nodes_metadata_batched")(terminal_ids)
    except Exception:
        return candidate_paths
    expanded: list[ReasoningPath] = []
    for path in candidate_paths:
        metadata = metadata_by_id.get(path.terminal_node_id)
        if metadata is None or not metadata.is_probable_cvt or not path.triples:
            expanded.append(path)
            continue
        evidence: tuple[EvidenceEdge, ...] = ()
        if path.edge_ids:
            raw = str(path.edge_ids[-1])
            parts = raw.split(":", 2)
            if len(parts) == 3:
                evidence = (
                    EvidenceEdge(
                        source_id=parts[0],
                        relation=parts[1],
                        direction="forward",
                        target_id=parts[2],
                    ),
                )
        if not evidence:
            continue
        bundle_path = _expand_cvt_bundle_path(
            graph,
            path.terminal_node_id,
            evidence,
            subquestion=subquestion,
            expected_answer_type=expected_answer_type,
            relation_hint_names=relation_hint_names,
            anchor_mentions=anchor_mentions,
            path_score=path.path_score,
            score_breakdown=path.search_score_breakdown,
            node_metadata=metadata,
        )
        expanded.append(bundle_path or path)
    return expanded


def filter_relation_candidates(
    state: RelationBeamState,
    candidates: list[RelationCandidate],
    expected_answer_type: str | None = None,
    frontier_summary: dict[str, Any] | None = None,
) -> list[RelationCandidate]:
    """Filter and de-prioritize unusable relation candidates."""
    filtered: list[RelationCandidate] = []
    blocked_prefixes = ("common.", "freebase.", "user.", "base.")
    blocked_exact = {"type.object.name", "type.object.key", "type.object.guid", "type.object.type"}
    last_signature = state.relation_path[-1] if state.relation_path else None
    cvt_heavy_frontier = float((frontier_summary or {}).get("probable_cvt_ratio", 0.0) or 0.0) >= 0.5
    for candidate in candidates:
        relation = candidate.relation
        if relation in blocked_exact or relation.startswith(blocked_prefixes):
            continue
        if last_signature is not None:
            if (
                last_signature.relation == candidate.relation
                and last_signature.direction != candidate.direction
            ):
                continue
        if cvt_heavy_frontier and relation.endswith((".name", ".type", ".guid")):
            continue
        filtered.append(candidate)
    return filtered


def retrieve_relation_candidates(
    query: str,
    relation_path: tuple[RelationPathStep, ...],
    candidates: list[RelationCandidate],
    relation_hint_names: list[str] | None,
    anchor_mentions: list[str] | None,
    resolved_dependencies: list[dict[str, Any]] | None,
    frontier_summary: dict[str, Any] | None,
    top_k: int,
) -> list[RelationCandidate]:
    """Lightweight relation retrieval without an external vector store."""
    path_context = " ".join(step.relation_name for step in relation_path)
    relation_hint_text = " ".join(relation_hint_names or [])
    anchor_text = " ".join(anchor_mentions or [])
    frontier_is_cvt = float((frontier_summary or {}).get("probable_cvt_ratio", 0.0) or 0.0) >= 0.5
    dependency_text = " ".join(
        str(item.get("answer") or "")
        for item in (resolved_dependencies or [])
        if isinstance(item, dict)
    )
    scored: list[tuple[float, RelationCandidate]] = []
    for candidate in candidates:
        relation_text = f"{candidate.relation} {candidate.relation_name}"
        sample_text = " ".join(candidate.sample_target_names)
        score = 0.0
        score += 0.45 * _token_overlap(query, relation_text)
        score += 0.10 * _token_overlap(query, sample_text)
        score += 0.10 * _token_overlap(path_context, relation_text)
        score += 0.15 * _token_overlap(relation_hint_text, relation_text)
        score += 0.10 * _token_overlap(anchor_text, relation_text)
        score += 0.05 * _token_overlap(anchor_text, sample_text)
        score += 0.05 * _token_overlap(dependency_text, sample_text)
        score += 0.15 * min(1.0, candidate.support_ratio)
        score -= 0.05 * min(1.0, candidate.total_neighbor_count / 100.0)
        score -= 0.05 * min(1.0, candidate.global_frequency / 5000.0)
        if frontier_is_cvt:
            cvt_bonus_terms = (
                "from",
                "to",
                "date",
                "year",
                "title",
                "office",
                "holder",
                "spouse",
                "nominee",
                "winner",
                "employee",
                "role",
                "team",
                "performance",
                "jurisdiction",
                "district",
                "governmental_body",
            )
            score += 0.30 * max((_token_overlap(term, relation_text) for term in cvt_bonus_terms), default=0.0)
            score += 0.10 * max((_token_overlap(term, sample_text) for term in cvt_bonus_terms), default=0.0)
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], -item[1].supporting_node_count, item[1].relation))
    return [candidate for _, candidate in scored[: max(1, top_k)]]


def summarize_frontier(
    metadata: dict[str, NodeMetadata],
    max_examples: int = 5,
) -> dict[str, Any]:
    """Build a compact frontier summary for LLM prompts and logs."""
    if not metadata:
        return {
            "node_count": 0,
            "entity_count": 0,
            "literal_count": 0,
            "probable_cvt_ratio": 0.0,
            "sample_names": [],
            "sample_types": [],
            "degree_summary": {"min": 0, "max": 0, "average": 0.0},
        }
    nodes = list(metadata.values())
    entity_nodes = [node for node in nodes if not node.is_literal]
    literal_nodes = [node for node in nodes if node.is_literal]
    cvt_nodes = [node for node in entity_nodes if node.is_probable_cvt]
    degrees = [node.degree for node in nodes]
    sample_names = [node.name for node in nodes if node.name][:max_examples]
    sample_types = sorted(
        {
            item
            for node in nodes[: max_examples * 2]
            for item in node.types
            if item
        }
    )[:max_examples]
    return {
        "node_count": len(nodes),
        "entity_count": len(entity_nodes),
        "literal_count": len(literal_nodes),
        "probable_cvt_ratio": round(len(cvt_nodes) / max(len(entity_nodes), 1), 4) if entity_nodes else 0.0,
        "sample_names": sample_names,
        "sample_types": sample_types,
        "degree_summary": {
            "min": min(degrees),
            "max": max(degrees),
            "average": round(mean(degrees), 4),
        },
    }


def relation_path_signature(state: RelationBeamState) -> tuple[tuple[str, str], ...]:
    """Return a stable signature for a relation path."""
    return tuple((step.relation, step.direction) for step in state.relation_path)


def merge_same_relation_paths(
    states: list[RelationBeamState],
    max_nodes_per_path: int,
) -> list[RelationBeamState]:
    """Merge states that share the same relation path to preserve frontier diversity."""
    grouped: dict[tuple[tuple[str, str], ...], list[RelationBeamState]] = {}
    for state in states:
        grouped.setdefault(relation_path_signature(state), []).append(state)
    merged: list[RelationBeamState] = []
    for signature, group in grouped.items():
        best = max(group, key=state_score)
        frontier_nodes: list[str] = []
        evidence_paths: dict[str, tuple[EvidenceEdge, ...]] = {}
        for state in group:
            for node_id in state.frontier_nodes:
                if node_id not in frontier_nodes:
                    frontier_nodes.append(node_id)
                existing = evidence_paths.get(node_id)
                candidate = state.evidence_paths.get(node_id, ())
                if existing is None or (candidate and len(candidate) < len(existing)):
                    evidence_paths[node_id] = candidate
        merged.append(
            replace(
                best,
                frontier_nodes=tuple(frontier_nodes[:max_nodes_per_path]),
                evidence_paths={node: evidence_paths[node] for node in frontier_nodes[:max_nodes_per_path]},
                visited_relation_signatures=frozenset(signature),
            )
        )
    return merged


def state_score(state: RelationBeamState) -> float:
    """Compute the composite state score used for ranking."""
    return (
        0.45 * state.semantic_score
        + 0.25 * state.type_score
        + 0.20 * state.answer_score
        + 0.10 * state.diversity_score
        - state.cost_penalty
        - 0.04 * state.hop
    )


def prune_relation_beam(
    states: list[RelationBeamState],
    current_hop: int,
    beam_width: int,
) -> list[RelationBeamState]:
    """Keep a diverse relation beam instead of plain global top-k states."""
    if not states:
        return []
    unique_states: dict[tuple[tuple[str, str], ...], RelationBeamState] = {}
    for state in sorted(states, key=state_score, reverse=True):
        unique_states.setdefault(relation_path_signature(state), state)
    candidates = list(unique_states.values())
    protected = sorted(
        [state for state in candidates if state.protected_until_hop >= current_hop],
        key=state_score,
        reverse=True,
    )
    if len(protected) >= beam_width:
        return protected[:beam_width]

    selected: list[RelationBeamState] = list(protected)
    selected_signatures = {relation_path_signature(state) for state in selected}

    answer_like = [
        state
        for state in sorted(candidates, key=state_score, reverse=True)
        if state.contains_answer_candidates and relation_path_signature(state) not in selected_signatures
    ][:2]
    for state in answer_like:
        selected.append(state)
        selected_signatures.add(relation_path_signature(state))
        if len(selected) >= beam_width:
            return selected[:beam_width]

    by_first_hop: dict[tuple[str, str], RelationBeamState] = {}
    for state in sorted(candidates, key=state_score, reverse=True):
        signature = relation_path_signature(state)
        if not signature or signature in selected_signatures:
            continue
        by_first_hop.setdefault(signature[0], state)
    for state in by_first_hop.values():
        selected.append(state)
        selected_signatures.add(relation_path_signature(state))
        if len(selected) >= beam_width:
            return selected[:beam_width]

    intermediate = [
        state
        for state in sorted(candidates, key=state_score, reverse=True)
        if any(step.role == "intermediate" for step in state.relation_path)
        and relation_path_signature(state) not in selected_signatures
    ]
    for state in intermediate:
        selected.append(state)
        selected_signatures.add(relation_path_signature(state))
        if len(selected) >= beam_width:
            return selected[:beam_width]

    for state in sorted(candidates, key=state_score, reverse=True):
        signature = relation_path_signature(state)
        if signature in selected_signatures:
            continue
        selected.append(state)
        selected_signatures.add(signature)
        if len(selected) >= beam_width:
            break
    return selected[:beam_width]


def _infer_literal_answer_match(node_id: str, expected_answer_type: str | None) -> float:
    expected = normalize_text(expected_answer_type or "")
    value = str(node_id).strip()
    if not expected:
        return 0.5
    if expected in {"date", "year", "datetime", "time"}:
        if any(char.isdigit() for char in value):
            return 0.8
    return 0.3


def _expected_type_aliases(expected_answer_type: str | None) -> tuple[str, ...]:
    expected = normalize_text(expected_answer_type or "")
    if not expected:
        return ()
    aliases = {expected}
    alias_map = {
        "movie": {"film", "motion picture", "feature film"},
        "film": {"movie", "motion picture", "feature film"},
        "actor": {"performer", "cast", "person", "film actor"},
        "actress": {"performer", "cast", "person", "film actor"},
        "person": {"people", "human"},
        "city": {"town", "location", "citytown"},
        "country": {"nation", "sovereign state"},
        "language": {"spoken language"},
        "religion": {"faith", "belief system"},
        "airport": {"airfield", "aviation facility"},
        "school": {"university", "college", "educational institution"},
        "date": {"year", "datetime", "time"},
        "year": {"date", "datetime", "time"},
    }
    for token in list(aliases):
        aliases.update(alias_map.get(token, set()))
    return tuple(sorted(aliases))


def _type_match_score(
    expected_answer_type: str | None,
    *,
    node_name: str,
    node_types: tuple[str, ...],
) -> float:
    aliases = _expected_type_aliases(expected_answer_type)
    if not aliases:
        return 0.5
    haystack = normalize_text(" ".join([node_name, *node_types]))
    if not haystack:
        return 0.0
    best = 0.0
    for alias in aliases:
        if alias and alias in haystack:
            best = max(best, 1.0)
        best = max(best, _token_overlap(alias, haystack))
    return best


def build_child_state(
    *,
    parent: RelationBeamState,
    action: RelationAction,
    expansion: FrontierExpansion,
    metadata: dict[str, NodeMetadata],
    max_nodes_per_path: int,
    state_index: int,
    expected_answer_type: str | None,
) -> RelationBeamState:
    """Build one child beam state after a frontier expansion."""
    next_frontier = tuple(expansion.next_nodes[:max_nodes_per_path])
    next_relation_path = parent.relation_path + (
        RelationPathStep(
            relation=action.relation,
            relation_name=action.relation_name,
            direction=action.direction,
            role=action.role,
            confidence=action.confidence,
        ),
    )
    semantic_score = (0.6 * parent.semantic_score) + (0.4 * action.confidence)
    if expected_answer_type:
        type_scores: list[float] = []
        for node_id in next_frontier:
            node = metadata.get(node_id)
            if node is None:
                continue
            type_scores.append(_type_match_score(expected_answer_type, node_name=node.name, node_types=node.types))
        type_score = sum(type_scores) / max(len(type_scores), 1)
    else:
        type_score = 0.5
    answer_score = 0.2 if action.role != "answer" else 0.55
    cvt_node_count = 0
    for node_id in next_frontier:
        node = metadata.get(node_id)
        if node is None:
            continue
        if node.is_probable_cvt:
            cvt_node_count += 1
        if node.is_literal:
            answer_score = max(answer_score, _infer_literal_answer_match(node.node_id, expected_answer_type))
        elif not node.is_probable_cvt and type_score > 0.0:
            answer_score = max(answer_score, 0.55 + (0.35 * type_score))
            if action.role == "intermediate" and _type_match_score(expected_answer_type, node_name=node.name, node_types=node.types) >= 0.95:
                answer_score = max(answer_score, 0.84)
    cvt_ratio = cvt_node_count / max(len(next_frontier), 1)
    diversity_score = 1.0 / max(len(next_relation_path), 1)
    avg_degree = mean([metadata[node_id].degree for node_id in next_frontier if node_id in metadata] or [0.0])
    cost_penalty = min(
        0.4,
        (len(next_frontier) / max(max_nodes_per_path, 1)) * 0.15
        + (expansion.edge_count / max(max_nodes_per_path, 1)) * 0.1
        + (0.1 if expansion.truncated else 0.0)
        + min(0.15, avg_degree / 5000.0),
    )
    if cvt_ratio >= 0.5:
        cost_penalty = max(0.0, cost_penalty - 0.08)
    contains_answer_candidates = answer_score >= 0.6 or type_score >= 0.6
    protect_for_next_hop = action.protect_for_next_hop or cvt_ratio >= 0.5
    protected_until_hop = parent.hop + 1 if protect_for_next_hop else parent.protected_until_hop
    return RelationBeamState(
        state_id=f"rb_{state_index}",
        frontier_nodes=next_frontier,
        relation_path=next_relation_path,
        evidence_paths={node_id: expansion.evidence_paths.get(node_id, ()) for node_id in next_frontier},
        source_nodes=parent.source_nodes,
        hop=parent.hop + 1,
        semantic_score=min(1.0, max(0.0, semantic_score)),
        type_score=min(1.0, max(0.0, type_score)),
        answer_score=min(1.0, max(0.0, answer_score)),
        diversity_score=min(1.0, max(0.0, diversity_score)),
        cost_penalty=min(1.0, max(0.0, cost_penalty)),
        protected_until_hop=protected_until_hop,
        contains_answer_candidates=contains_answer_candidates,
        path_reason=action.reason,
        visited_relation_signatures=frozenset(
            set(parent.visited_relation_signatures) | {(action.relation, action.direction)}
        ),
    )


class RelationTypeBeamSearcher:
    """LLM-guided, relation-oriented beam search over frontier node sets."""

    def __init__(
        self,
        graph: BaseGraphAPI,
        llm: RelationBeamLLM,
        config: RelationBeamConfig | None = None,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.config = config or RelationBeamConfig()

    def search(
        self,
        *,
        original_question: str,
        subquestion: str,
        input_nodes: list[str],
        expected_answer_type: str | None = None,
        anchor_mentions: list[str] | None = None,
        relation_hint_names: list[str] | None = None,
        resolved_dependencies: list[dict[str, Any]] | None = None,
    ) -> list[RelationBeamState]:
        """Search up to max_hops with relation-level actions over frontier node sets."""
        source_nodes = tuple(dict.fromkeys(input_nodes))
        if not source_nodes:
            return []
        anchor_mentions = [item for item in (anchor_mentions or []) if item]
        relation_hint_names = [item for item in (relation_hint_names or []) if item]
        resolved_dependencies = [
            item for item in (resolved_dependencies or []) if isinstance(item, dict)
        ]
        initial_state = RelationBeamState(
            state_id="rb_0",
            frontier_nodes=source_nodes,
            relation_path=(),
            evidence_paths={node_id: () for node_id in source_nodes},
            source_nodes=source_nodes,
            hop=0,
            semantic_score=0.5,
            type_score=0.5 if expected_answer_type is None else 0.0,
            answer_score=0.0,
            diversity_score=1.0,
            cost_penalty=0.0,
            protected_until_hop=0,
            contains_answer_candidates=False,
            path_reason="initial",
            visited_relation_signatures=frozenset(),
        )
        beam = [initial_state]
        all_states: list[RelationBeamState] = [initial_state]
        answer_states: list[RelationBeamState] = []
        next_state_index = 1

        for hop in range(1, self.config.max_hops + 1):
            child_states: list[RelationBeamState] = []
            total_candidates = 0
            selected_actions = 0
            expanded_edges = 0
            for state in beam:
                state_start = perf_counter()
                child_count_before_state = len(child_states)
                try:
                    candidates = self.graph.get_frontier_relations(
                        list(state.frontier_nodes),
                        include_reverse=True,
                        include_literals=True,
                    )
                except AttributeError:
                    continue
                total_candidates += len(candidates)
                frontier_meta = self.graph.get_nodes_metadata_batched(list(state.frontier_nodes))
                frontier_summary = summarize_frontier(frontier_meta)
                filtered = filter_relation_candidates(
                    state,
                    candidates,
                    expected_answer_type=expected_answer_type,
                    frontier_summary=frontier_summary,
                )
                retrieved = retrieve_relation_candidates(
                    query=" ".join(
                        part
                        for part in [
                            subquestion,
                            expected_answer_type or "",
                            " ".join(anchor_mentions),
                            " ".join(relation_hint_names),
                        ]
                        if part
                    ),
                    relation_path=state.relation_path,
                    candidates=filtered,
                    relation_hint_names=relation_hint_names,
                    anchor_mentions=anchor_mentions,
                    resolved_dependencies=resolved_dependencies,
                    frontier_summary=frontier_summary,
                    top_k=self.config.relation_retrieval_top_k,
                )
                ranked_candidates = retrieve_relation_candidates(
                    query=" ".join(
                        part
                        for part in [
                            subquestion,
                            expected_answer_type or "",
                            " ".join(anchor_mentions),
                            " ".join(relation_hint_names),
                        ]
                        if part
                    ),
                    relation_path=state.relation_path,
                    candidates=filtered,
                    relation_hint_names=relation_hint_names,
                    anchor_mentions=anchor_mentions,
                    resolved_dependencies=resolved_dependencies,
                    frontier_summary=frontier_summary,
                    top_k=max(1, len(filtered)),
                )
                remaining_hops = self.config.max_hops - state.hop
                LOGGER.info(
                    "Relation beam state state_id=%s hop=%d frontier=%d raw_candidates=%d filtered=%d ranked_for_llm=%d fallback_ranked=%d remaining_hops=%d",
                    state.state_id,
                    hop,
                    len(state.frontier_nodes),
                    len(candidates),
                    len(filtered),
                    len(ranked_candidates),
                    len(retrieved),
                    remaining_hops,
                )
                llm_select_start = perf_counter()
                try:
                    actions = self.llm.select_relation_actions(
                        original_question=original_question,
                        subquestion=subquestion,
                        expected_answer_type=expected_answer_type,
                        state=state,
                        frontier_summary=frontier_summary,
                        candidates=ranked_candidates,
                        anchor_mentions=anchor_mentions,
                        relation_hint_names=relation_hint_names,
                        resolved_dependencies=resolved_dependencies,
                        remaining_hops=remaining_hops,
                        top_k=self.config.relations_per_state,
                    )
                except Exception:
                    actions = []
                llm_select_elapsed = perf_counter() - llm_select_start
                LOGGER.info(
                    "Relation beam llm_select state_id=%s hop=%d elapsed_sec=%.3f returned_actions=%d action_preview=%s",
                    state.state_id,
                    hop,
                    llm_select_elapsed,
                    len(actions),
                    [
                        {
                            "relation": action.relation,
                            "direction": action.direction,
                            "role": action.role,
                            "confidence": round(action.confidence, 3),
                        }
                        for action in actions[: min(3, len(actions))]
                    ],
                )
                if not actions:
                    actions = [
                        RelationAction(
                            relation=item.relation,
                            relation_name=item.relation_name,
                            direction=item.direction,
                            role="intermediate",
                            confidence=min(1.0, max(0.1, item.support_ratio)),
                            expected_next_type=expected_answer_type,
                            reason="fallback retrieval candidate",
                            protect_for_next_hop=True,
                        )
                        for item in retrieved[: self.config.relations_per_state]
                    ]
                selected_actions += len(actions)
                for action in actions[: self.config.relations_per_state]:
                    expand_start = perf_counter()
                    try:
                        expansion = self.graph.expand_frontier_by_relation(
                            list(state.frontier_nodes),
                            relation=action.relation,
                            direction=action.direction,
                            parent_evidence_paths=state.evidence_paths,
                            limit_per_source=self.config.neighbors_per_source,
                            total_limit=self.config.max_nodes_per_path,
                        )
                    except AttributeError:
                        continue
                    expand_elapsed = perf_counter() - expand_start
                    if not expansion.next_nodes:
                        LOGGER.info(
                            "Relation beam expand state_id=%s hop=%d relation=%s direction=%s elapsed_sec=%.3f next_nodes=0 edge_count=0 truncated=%s",
                            state.state_id,
                            hop,
                            action.relation,
                            action.direction,
                            expand_elapsed,
                            False,
                        )
                        continue
                    LOGGER.info(
                        "Relation beam expand state_id=%s hop=%d relation=%s direction=%s elapsed_sec=%.3f next_nodes=%d edge_count=%d truncated=%s",
                        state.state_id,
                        hop,
                        action.relation,
                        action.direction,
                        expand_elapsed,
                        len(expansion.next_nodes),
                        expansion.edge_count,
                        expansion.truncated,
                    )
                    expanded_edges += expansion.edge_count
                    metadata = self.graph.get_nodes_metadata_batched(list(expansion.next_nodes))
                    child = build_child_state(
                        parent=state,
                        action=action,
                        expansion=expansion,
                        metadata=metadata,
                        max_nodes_per_path=self.config.max_nodes_per_path,
                        state_index=next_state_index,
                        expected_answer_type=expected_answer_type,
                    )
                    next_state_index += 1
                    child_states.append(child)
                LOGGER.info(
                    "Relation beam state_complete state_id=%s hop=%d elapsed_sec=%.3f child_states=%d",
                    state.state_id,
                    hop,
                    perf_counter() - state_start,
                    len(child_states) - child_count_before_state,
                )

            merged = merge_same_relation_paths(child_states, self.config.max_nodes_per_path)
            rerank_slice = sorted(merged, key=state_score, reverse=True)[: self.config.llm_path_rerank_top_k]
            remaining = self.config.max_hops - hop
            rerank_start = perf_counter()
            try:
                reranked = self.llm.rerank_paths(
                    original_question=original_question,
                    subquestion=subquestion,
                    expected_answer_type=expected_answer_type,
                    states=rerank_slice,
                    anchor_mentions=anchor_mentions,
                    relation_hint_names=relation_hint_names,
                    resolved_dependencies=resolved_dependencies,
                    remaining_hops=remaining,
                )
            except Exception:
                reranked = {}
            LOGGER.info(
                "Relation beam rerank hop=%d slice=%d elapsed_sec=%.3f returned=%d",
                hop,
                len(rerank_slice),
                perf_counter() - rerank_start,
                len(reranked),
            )
            reranked_ids = {state.state_id for state in rerank_slice}
            final_candidates: list[RelationBeamState] = []
            for state in merged:
                update = reranked.get(state.state_id)
                if update:
                    updated_state = replace(
                        state,
                        semantic_score=max(state.semantic_score, _clamp_score(update.get("semantic_relevance", state.semantic_score))),
                        answer_score=max(state.answer_score, _clamp_score(update.get("answer_likelihood", state.answer_score))),
                        type_score=max(state.type_score, _clamp_score(update.get("continue_likelihood", state.type_score))),
                    )
                    verdict = str(update.get("verdict") or "continue").strip().lower()
                    if verdict == "discard" and updated_state.protected_until_hop < hop:
                        continue
                    final_candidates.append(updated_state)
                else:
                    final_candidates.append(state)
            answer_states.extend(
                state
                for state in final_candidates
                if state.answer_score >= self.config.answer_threshold
            )
            next_beam = prune_relation_beam(final_candidates, hop, self.config.beam_width)
            all_states.extend(final_candidates)
            LOGGER.info(
                "Relation beam hop=%d beam_states=%d relation_candidates=%d selected_actions=%d expanded_edges=%d child_states=%d merged_states=%d answer_states=%d next_beam=%d",
                hop,
                len(beam),
                total_candidates,
                selected_actions,
                expanded_edges,
                len(child_states),
                len(merged),
                len(answer_states),
                len(next_beam),
            )
            if answer_states:
                break
            if not next_beam:
                break
            beam = next_beam

        final_states = answer_states or beam or all_states
        return sorted(final_states, key=state_score, reverse=True)


class SubquestionRelationExplorer:
    """Thin adapter that runs relation beam search for one DAG sub-question."""

    def __init__(
        self,
        graph: BaseGraphAPI,
        llm: RelationBeamLLM,
        config: RelationBeamConfig | None = None,
    ) -> None:
        self.graph = graph
        self.searcher = RelationTypeBeamSearcher(graph=graph, llm=llm, config=config)
        self.config = config or RelationBeamConfig()

    def explore(
        self,
        *,
        original_question: str,
        subquestion_id: str,
        subquestion_text: str,
        input_entities: list[str],
        expected_answer_type: str | None,
        anchor_mentions: list[str] | None = None,
        relation_hint_names: list[str] | None = None,
        resolved_dependencies: list[dict[str, Any]] | None = None,
    ) -> SubquestionExplorationResult:
        """Explore one sub-question and return answer entities plus evidence paths."""
        states = self.searcher.search(
            original_question=original_question,
            subquestion=subquestion_text,
            input_nodes=input_entities,
            expected_answer_type=expected_answer_type,
            anchor_mentions=anchor_mentions,
            relation_hint_names=relation_hint_names,
            resolved_dependencies=resolved_dependencies,
        )
        best_score = state_score(states[0]) if states else 0.0
        answer_states = [state for state in states if state.contains_answer_candidates] or states[: self.config.beam_width]
        answer_entities: list[str] = []
        merged_paths: dict[str, tuple[EvidenceEdge, ...]] = {}
        best_states: dict[str, RelationBeamState] = {}
        candidate_paths: list[ReasoningPath] = []
        node_metadata: dict[str, NodeMetadata] = {}
        if hasattr(self.graph, "get_nodes_metadata_batched"):
            try:
                node_metadata = getattr(self.graph, "get_nodes_metadata_batched")(list({node for state in answer_states for node in state.frontier_nodes}))
            except Exception:
                node_metadata = {}
        for state in answer_states:
            for node_id in state.frontier_nodes:
                if node_id not in answer_entities:
                    answer_entities.append(node_id)
                evidence = state.evidence_paths.get(node_id, ())
                existing = merged_paths.get(node_id, ())
                existing_state = best_states.get(node_id)
                replace_current = node_id not in merged_paths
                if not replace_current and evidence:
                    if not existing or len(evidence) < len(existing):
                        replace_current = True
                    elif len(evidence) == len(existing) and existing_state is not None and state_score(state) > state_score(existing_state):
                        replace_current = True
                if replace_current:
                    merged_paths[node_id] = evidence
                    best_states[node_id] = state
        for node_id, evidence in merged_paths.items():
            candidate_paths.extend(
                _reasoning_paths_from_evidence(
                    self.graph,
                    node_id,
                    evidence,
                    subquestion=subquestion_text,
                    expected_answer_type=expected_answer_type,
                    relation_hint_names=relation_hint_names,
                    anchor_mentions=anchor_mentions,
                    supporting_state=best_states.get(node_id),
                    node_metadata=node_metadata.get(node_id),
                )
            )
        return SubquestionExplorationResult(
            answer_entities=answer_entities,
            answer_states=answer_states,
            searched_hops=max((state.hop for state in states), default=0),
            complete=bool(answer_states and answer_states[0].answer_score >= self.config.answer_threshold),
            best_score=best_score,
            evidence_paths=merged_paths,
            candidate_paths=candidate_paths,
        )


def _reasoning_path_from_evidence(
    graph: BaseGraphAPI,
    target_node_id: str,
    evidence: tuple[EvidenceEdge, ...],
    *,
    supporting_state: RelationBeamState | None = None,
    expected_answer_type: str | None = None,
    node_metadata: NodeMetadata | None = None,
) -> ReasoningPath:
    """Backward-compatible wrapper over the multi-path evidence conversion."""
    return _reasoning_paths_from_evidence(
        graph,
        target_node_id,
        evidence,
        subquestion="",
        expected_answer_type=expected_answer_type,
        relation_hint_names=[],
        anchor_mentions=[],
        supporting_state=supporting_state,
        node_metadata=node_metadata,
    )[0]
