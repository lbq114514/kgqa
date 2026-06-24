"""Dataclasses shared across the KGQA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Triple:
    """A single knowledge graph triple."""

    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class TripleFact:
    """One id-preserving evidence triple used for program-side grounding."""

    head_id: str = ""
    head_label: str = ""
    relation_id: str = ""
    tail_id: str = ""
    tail_label: str = ""
    tail_kind: str = "id"


@dataclass
class QuestionAnalysisResult:
    """Structured question analysis produced by the LLM."""

    split_questions: list[str]
    reasoning_indicator: str
    ordered_topic_entities: list[str]
    predicted_depth: int
    topic_entities: list[str] = field(default_factory=list)
    sub_questions: list["SubQuestionSpec"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class EntityMentionSpec:
    """One entity-like mention extracted from a question or sub-question."""

    name: str
    aliases: list[str] = field(default_factory=list)
    expected_type: str = ""
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class RelationHintSpec:
    """One relation hint extracted from a question or sub-question."""

    name: str
    aliases: list[str] = field(default_factory=list)
    freebase_like_ids: list[str] = field(default_factory=list)
    direction: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class SubQuestionSpec:
    """A structured sub-question within a multi-step CWQ reasoning chain."""

    id: str
    question: str
    topic_entities: list[EntityMentionSpec] = field(default_factory=list)
    local_topic_entities: list[EntityMentionSpec] = field(default_factory=list)
    interested_nodes: list[EntityMentionSpec] = field(default_factory=list)
    interested_relations: list[RelationHintSpec] = field(default_factory=list)
    expected_answer_type: str = ""
    expected_hop: int = 0
    depends_on: list[str] = field(default_factory=list)
    solver_type: str = "explore"
    solver_reason: str = ""
    downstream_filters: list[str] = field(default_factory=list)
    execution_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class AgenticPlanStep:
    """One planner-produced reasoning step inside the agent loop."""

    step_id: str
    question: str
    goal: str = ""
    depends_on_step_ids: list[str] = field(default_factory=list)
    topic_entity_mentions: list[EntityMentionSpec] = field(default_factory=list)
    relation_hints: list[RelationHintSpec] = field(default_factory=list)
    expected_answer_type: str = ""
    carryover_constraints: list[str] = field(default_factory=list)
    downstream_filters: list[str] = field(default_factory=list)
    stop_if_answered: bool = False
    strategy: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ResolvedCandidate:
    """A resolved graph candidate retained for debugging and downstream routing."""

    id_or_mid: str
    label: str
    mention_or_hint: str
    match_type: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ResolvedSubQuestion:
    """A sub-question after entity and relation resolution against SQLite Freebase."""

    id: str
    question: str
    topic_entity_candidates: list[ResolvedCandidate] = field(default_factory=list)
    interested_node_candidates: list[ResolvedCandidate] = field(default_factory=list)
    relation_candidates: list[ResolvedCandidate] = field(default_factory=list)
    seed_node_ids: list[str] = field(default_factory=list)
    constraint_target_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    selected_relation_ids: list[str] = field(default_factory=list)
    propagated_answer_node_ids: list[str] = field(default_factory=list)
    expected_answer_type: str = ""
    depends_on: list[str] = field(default_factory=list)
    resolution_debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class AgenticStepResult:
    """One executed agent step with retrieved evidence and carryover state."""

    step_id: str
    sub_question_id: str = ""
    sub_question_text: str = ""
    resolved_entity_ids: list[str] = field(default_factory=list)
    resolved_relation_ids: list[str] = field(default_factory=list)
    retrieved_paths: list[ReasoningPath] = field(default_factory=list)
    summarized_evidence: list[Any] = field(default_factory=list)
    candidate_answers: list[str] = field(default_factory=list)
    sub_answer: str = ""
    sub_answer_entities: list[str] = field(default_factory=list)
    sub_answer_literals: list[str] = field(default_factory=list)
    sub_answer_grounding: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    attempt_count: int = 0
    depends_on_step_ids: list[str] = field(default_factory=list)
    failure_reason: str = ""
    carryover_entities: list[str] = field(default_factory=list)
    carryover_text_values: list[str] = field(default_factory=list)
    is_step_resolved: bool = False
    solver_type: str = ""
    solver_debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "step_id": self.step_id,
            "sub_question_id": self.sub_question_id,
            "sub_question_text": self.sub_question_text,
            "resolved_entity_ids": list(self.resolved_entity_ids),
            "resolved_relation_ids": list(self.resolved_relation_ids),
            "retrieved_paths": [path.to_dict() for path in self.retrieved_paths],
            "summarized_evidence": list(self.summarized_evidence),
            "candidate_answers": list(self.candidate_answers),
            "sub_answer": self.sub_answer,
            "sub_answer_entities": list(self.sub_answer_entities),
            "sub_answer_literals": list(self.sub_answer_literals),
            "sub_answer_grounding": dict(self.sub_answer_grounding),
            "status": self.status,
            "attempt_count": int(self.attempt_count),
            "depends_on_step_ids": list(self.depends_on_step_ids),
            "failure_reason": self.failure_reason,
            "carryover_entities": list(self.carryover_entities),
            "carryover_text_values": list(self.carryover_text_values),
            "is_step_resolved": self.is_step_resolved,
            "solver_type": self.solver_type,
            "solver_debug": dict(self.solver_debug),
        }


@dataclass
class SupplementHints:
    """LLM-suggested entity and relation hints for supplementary search."""

    entity_candidates: list[str] = field(default_factory=list)
    linked_entities: list[str] = field(default_factory=list)
    relation_candidates: list[str] = field(default_factory=list)
    linked_relations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class AgenticRunState:
    """Persistent state threaded through the step-by-step agent loop."""

    original_question: str
    global_goal: str
    step_history: list[AgenticStepResult] = field(default_factory=list)
    evidence_bank: list[Any] = field(default_factory=list)
    current_frontier_entities: list[str] = field(default_factory=list)
    current_text_constraints: list[str] = field(default_factory=list)
    resolved_sub_answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_sub_question_ids: list[str] = field(default_factory=list)
    failed_sub_question_ids: list[str] = field(default_factory=list)
    current_sub_question_id: str = ""
    loop_index: int = 0
    sufficiency_history: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "original_question": self.original_question,
            "global_goal": self.global_goal,
            "step_history": [step.to_dict() for step in self.step_history],
            "evidence_bank": list(self.evidence_bank),
            "current_frontier_entities": list(self.current_frontier_entities),
            "current_text_constraints": list(self.current_text_constraints),
            "resolved_sub_answers": dict(self.resolved_sub_answers),
            "pending_sub_question_ids": list(self.pending_sub_question_ids),
            "failed_sub_question_ids": list(self.failed_sub_question_ids),
            "current_sub_question_id": self.current_sub_question_id,
            "loop_index": self.loop_index,
            "sufficiency_history": list(self.sufficiency_history),
            "final_answer": self.final_answer,
        }


@dataclass
class ExplorationHints:
    """Structured graph exploration hints, primarily for CWQ external-only mode."""

    focus_entities: list[str] = field(default_factory=list)
    answer_type_hints: list[str] = field(default_factory=list)
    relation_name_hints: list[str] = field(default_factory=list)
    aligned_entities: list[str] = field(default_factory=list)
    aligned_relations: list[str] = field(default_factory=list)
    reasoning_focus: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class ReasoningPath:
    """A candidate reasoning path through the knowledge graph."""

    triples: list[Triple]
    nodes: list[str]
    text: str
    source_stage: str
    triple_facts: list[TripleFact] = field(default_factory=list)
    pruning_status: str = "unknown"
    matched_relations: list[str] = field(default_factory=list)
    path_score: float = 0.0
    search_score_breakdown: dict[str, float] = field(default_factory=dict)
    edge_ids: list[str] = field(default_factory=list)
    terminal_node_id: str = ""
    terminal_node_kind: str = "id"
    matched_answer_type_hints: list[str] = field(default_factory=list)
    search_strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "triples": [asdict(triple) for triple in self.triples],
            "triple_facts": [asdict(fact) for fact in self.triple_facts],
            "nodes": list(self.nodes),
            "text": self.text,
            "source_stage": self.source_stage,
            "pruning_status": self.pruning_status,
            "matched_relations": list(self.matched_relations),
            "path_score": self.path_score,
            "search_score_breakdown": dict(self.search_score_breakdown),
            "edge_ids": list(self.edge_ids),
            "terminal_node_id": self.terminal_node_id,
            "terminal_node_kind": self.terminal_node_kind,
            "matched_answer_type_hints": list(self.matched_answer_type_hints),
            "search_strategy": self.search_strategy,
        }


@dataclass
class AnswerGrounding:
    """Canonical answer representation shared by sub-question and final-answer stages."""

    answer_texts: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    entity_labels: list[str] = field(default_factory=list)
    literal_values: list[str] = field(default_factory=list)
    primary_answer_text: str = ""
    primary_entity_id: str = ""
    answer_view_triples: list[dict[str, str]] = field(default_factory=list)
    answer_view_facts: list[dict[str, str]] = field(default_factory=list)
    supporting_relation_ids: list[str] = field(default_factory=list)
    source_mode: str = ""
    confidence: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class AnswerResult:
    """Answering stage output."""

    sufficient: bool
    answer: str
    predicted_answers: list[str] = field(default_factory=list)
    resolved_entity_mentions: list[str] = field(default_factory=list)
    resolved_literals: list[str] = field(default_factory=list)
    supporting_paths: list[Any] = field(default_factory=list)
    reason: str = ""
    grounding: AnswerGrounding | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class SearchTraceStep:
    """A compact record of one search or pruning step."""

    stage: str
    depth: int
    candidate_count: int
    pruned_count: int
    sufficient: bool | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class PipelineResult:
    """Final end-to-end KGQA output."""

    question: str
    topic_entities: list[str]
    question_analysis: QuestionAnalysisResult
    dmax: int
    dpredict: int
    candidate_paths: list[ReasoningPath]
    pruned_paths: list[ReasoningPath]
    summarized_paths: list[Any]
    answer: str
    sufficient: bool
    confidence: str
    supplement_hints: SupplementHints = field(default_factory=SupplementHints)
    exploration_hints: ExplorationHints = field(default_factory=ExplorationHints)
    sample_id: str | None = None
    gold_answers: list[str] = field(default_factory=list)
    predicted_answers: list[str] = field(default_factory=list)
    final_grounding: dict[str, Any] = field(default_factory=dict)
    search_trace: list[SearchTraceStep] = field(default_factory=list)
    alignment_debug: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "topic_entities": self.topic_entities,
            "question_analysis": self.question_analysis.to_dict(),
            "exploration_hints": self.exploration_hints.to_dict(),
            "supplement_hints": self.supplement_hints.to_dict(),
            "dmax": self.dmax,
            "dpredict": self.dpredict,
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "pruned_paths": [path.to_dict() for path in self.pruned_paths],
            "summarized_paths": self.summarized_paths,
            "gold_answers": self.gold_answers,
            "predicted_answers": self.predicted_answers,
            "final_grounding": self.final_grounding,
            "answer": self.answer,
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "search_trace": [step.to_dict() for step in self.search_trace],
            "alignment_debug": self.alignment_debug,
        }


@dataclass
class WebQSPSample:
    """A single WebQSP JSONL sample."""

    sample_id: str
    question: str
    answers: list[str]
    q_entities: list[str]
    a_entities: list[str]
    graph_triples: list[Triple]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "answers": self.answers,
            "q_entities": self.q_entities,
            "a_entities": self.a_entities,
            "graph_triples": [asdict(triple) for triple in self.graph_triples],
        }


@dataclass
class CWQSample:
    """A single CWQ JSONL sample used in external-only mode."""

    sample_id: str
    question: str
    answers: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "answers": self.answers,
        }


@dataclass
class ValidationPrediction:
    """Per-sample validation output."""

    sample_id: str
    question: str
    gold_answers: list[str]
    predicted_answers: list[str]
    answer: str
    sufficient: bool
    confidence: str
    topic_entities: list[str]
    question_analysis: dict[str, Any]
    supplement_hints: dict[str, Any]
    candidate_paths: list[ReasoningPath]
    pruned_paths: list[ReasoningPath]
    summarized_paths: list[Any]
    search_trace: list[SearchTraceStep]
    metrics: dict[str, float]
    exploration_hints: dict[str, Any] = field(default_factory=dict)
    alignment_debug: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "gold_answers": self.gold_answers,
            "predicted_answers": self.predicted_answers,
            "answer": self.answer,
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "topic_entities": self.topic_entities,
            "question_analysis": self.question_analysis,
            "exploration_hints": self.exploration_hints,
            "supplement_hints": self.supplement_hints,
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "pruned_paths": [path.to_dict() for path in self.pruned_paths],
            "summarized_paths": self.summarized_paths,
            "search_trace": [step.to_dict() for step in self.search_trace],
            "alignment_debug": self.alignment_debug,
            "metrics": self.metrics,
        }


@dataclass
class ValidationMetrics:
    """Aggregate metrics for a validation run."""

    dataset: str
    split: str
    num_samples: int
    exact_match: float
    hit_at_1: float
    precision: float
    recall: float
    f1: float
    empty_prediction_rate: float
    avg_candidate_paths: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)
