"""Typed sub-question solvers and routing for agentic KGQA."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sqlite3
from typing import Any, Callable

from kgqa.utils.logging import get_logger
from kgqa.utils.text import normalize_text, path_to_text
from kgqa.utils.types import ReasoningPath, SubQuestionSpec, Triple

LOGGER = get_logger(__name__)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in normalize_text(left).split() if token}
    right_tokens = {token for token in normalize_text(right).split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def _unique_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _question_implies_operation(question: str) -> str:
    normalized = normalize_text(question)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    token_set = set(tokens)
    if "how many" in normalized or "number of" in normalized or "count" in token_set:
        return "count"
    if (
        "among these" in normalized
        or "intersection" in token_set
        or "overlap" in token_set
        or "same" in token_set
        or "both" in token_set
        or "also" in token_set
    ):
        return "intersect"
    if any(marker in token_set for marker in ("most", "largest", "highest", "biggest", "longest", "latest")):
        return "argmax"
    if any(marker in token_set for marker in ("least", "smallest", "lowest", "earliest", "shortest")):
        return "argmin"
    return "none"


@dataclass
class SubquestionExecutionContext:
    """Runtime context passed into one solver execution."""

    original_question: str
    sub_question: SubQuestionSpec
    graph_api: Any
    step_seed_ids: list[str]
    relation_ids: list[str]
    resolved_sub_answers: dict[str, dict[str, Any]]
    execute_explore: Callable[[], list[ReasoningPath]]


@dataclass
class SubquestionSolverResult:
    """Normalized result from one solver execution."""

    solver_type: str
    candidate_paths: list[ReasoningPath]
    structured_outputs: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    solver_debug: dict[str, Any] = field(default_factory=dict)
    primary_entity_ids: list[str] = field(default_factory=list)
    primary_literals: list[str] = field(default_factory=list)
    allow_local_relation_probe: bool = True


class BaseSubquestionSolver:
    """Base class for one typed sub-question solver."""

    solver_type = "explore"

    def run(self, context: SubquestionExecutionContext) -> SubquestionSolverResult:
        raise NotImplementedError


class ExploreSolver(BaseSubquestionSolver):
    """Thin wrapper around the existing explore pipeline."""

    solver_type = "explore"

    def run(self, context: SubquestionExecutionContext) -> SubquestionSolverResult:
        return SubquestionSolverResult(
            solver_type=self.solver_type,
            candidate_paths=context.execute_explore(),
            structured_outputs={"entity_set_role": "candidate_answers", "set_operation_hint": "none"},
            solver_debug={"mode": "existing_explore_pipeline"},
            allow_local_relation_probe=True,
        )


class ConstrainedCollectSolver(BaseSubquestionSolver):
    """Collect relation targets under active lexical/type constraints before truncation."""

    solver_type = "constrained_collect"

    @staticmethod
    def _constraint_terms(context: SubquestionExecutionContext) -> list[str]:
        dependency_terms: list[str] = []
        for dependency_id in context.sub_question.depends_on:
            payload = context.resolved_sub_answers.get(dependency_id, {})
            dependency_terms.extend(
                [
                    *[str(value).strip() for value in payload.get("predicted_answers", []) if str(value).strip()],
                    *[str(value).strip() for value in payload.get("literals", []) if str(value).strip()],
                    str(payload.get("answer") or "").strip(),
                ]
            )
        return _unique_preserve_order(
            [
                *[node.name for node in context.sub_question.interested_nodes if node.name],
                *[relation.name for relation in context.sub_question.interested_relations if relation.name],
                *[
                    alias
                    for relation in context.sub_question.interested_relations
                    for alias in relation.aliases
                    if alias
                ],
                *[entity.name for entity in context.sub_question.local_topic_entities if entity.name],
                *context.sub_question.downstream_filters,
                *dependency_terms,
            ]
        )

    def run(self, context: SubquestionExecutionContext) -> SubquestionSolverResult:
        graph_api = context.graph_api
        if (
            graph_api is None
            or not context.step_seed_ids
            or not context.relation_ids
            or not hasattr(graph_api, "collect_related_entities_constrained")
        ):
            return SubquestionSolverResult(
                solver_type="explore",
                candidate_paths=context.execute_explore(),
                structured_outputs={"entity_set_role": "candidate_answers", "set_operation_hint": "none"},
                solver_debug={"requested_solver_type": self.solver_type, "fallback_reason": "constrained_collect_unavailable"},
                allow_local_relation_probe=True,
            )

        constraint_terms = self._constraint_terms(context)
        constrained_rows = graph_api.collect_related_entities_constrained(
            source_ids=list(context.step_seed_ids),
            relation_ids=list(context.relation_ids),
            direction="forward",
            expected_answer_type=context.sub_question.expected_answer_type,
            constraint_terms=constraint_terms,
            limit=max(8, min(40, len(context.step_seed_ids) * 6)),
        )
        if not constrained_rows:
            return SubquestionSolverResult(
                solver_type="explore",
                candidate_paths=context.execute_explore(),
                structured_outputs={"entity_set_role": "candidate_answers", "set_operation_hint": "none"},
                solver_debug={"requested_solver_type": self.solver_type, "fallback_reason": "constrained_collect_empty"},
                allow_local_relation_probe=False,
            )

        candidate_paths: list[ReasoningPath] = []
        primary_entity_ids: list[str] = []
        for row in constrained_rows:
            source_id = str(row.get("source_entity_id") or "").strip()
            target_id = str(row.get("target_entity_id") or "").strip()
            relation_id = str(row.get("relation_id") or "").strip()
            if not source_id or not target_id or not relation_id:
                continue
            source_label = graph_api.get_entity_display_name(source_id)
            target_label = str(row.get("target_label") or graph_api.get_entity_display_name(target_id) or target_id)
            triple = Triple(head=source_label, relation=relation_id, tail=target_label)
            path = ReasoningPath(
                triples=[triple],
                nodes=[source_label, target_label],
                text=path_to_text([source_label, target_label], [triple]),
                source_stage="constrained_collect",
                pruning_status="preserved",
                matched_relations=[relation_id],
                path_score=float(row.get("score", 0.0)),
                terminal_node_id=target_id,
                terminal_node_kind="id",
                matched_answer_type_hints=[context.sub_question.expected_answer_type] if context.sub_question.expected_answer_type else [],
                search_strategy="constrained_collect",
            )
            candidate_paths.append(path)
            primary_entity_ids.append(target_id)
        if not candidate_paths:
            return SubquestionSolverResult(
                solver_type="explore",
                candidate_paths=context.execute_explore(),
                structured_outputs={"entity_set_role": "candidate_answers", "set_operation_hint": "none"},
                solver_debug={"requested_solver_type": self.solver_type, "fallback_reason": "constrained_collect_no_paths"},
                allow_local_relation_probe=False,
            )

        return SubquestionSolverResult(
            solver_type=self.solver_type,
            candidate_paths=candidate_paths,
            structured_outputs={
                "entity_set_role": "candidate_answers",
                "set_operation_hint": "collect",
                "aggregate_rows": [
                    {
                        "operation": "collect",
                        "entity_ids": _unique_preserve_order(primary_entity_ids),
                    }
                ],
                "constrained_rows": constrained_rows[:20],
                "solver_execution_mode": "constrained_collect",
            },
            solver_debug={
                "mode": "relation_locked_constrained_collect",
                "constraint_terms": constraint_terms[:8],
                "selected_relation_ids": list(context.relation_ids),
                "row_count": len(constrained_rows),
            },
            primary_entity_ids=_unique_preserve_order(primary_entity_ids),
            allow_local_relation_probe=False,
        )


class VerifySolver(BaseSubquestionSolver):
    """Direct-edge plus very short path verification solver."""

    solver_type = "verify"

    def _collect_target_entity_ids(self, context: SubquestionExecutionContext) -> list[str]:
        graph_api = context.graph_api
        targets: list[str] = []
        if graph_api is not None and hasattr(graph_api, "resolve_entity_mentions"):
            target_mentions = [node.name for node in context.sub_question.interested_nodes if node.name]
            if target_mentions:
                targets.extend(graph_api.resolve_entity_mentions(target_mentions, top_k=1))
        for dependency_id in context.sub_question.depends_on:
            dependency = context.resolved_sub_answers.get(dependency_id, {})
            targets.extend([str(item) for item in dependency.get("entity_ids", []) if str(item).strip()])
        return _unique_preserve_order(targets)

    def _select_relation_ids(self, context: SubquestionExecutionContext) -> list[str]:
        relation_ids = list(context.relation_ids)
        for hint in context.sub_question.interested_relations:
            if hint.freebase_like_ids:
                relation_ids.extend(hint.freebase_like_ids)
        return _unique_preserve_order(relation_ids)

    def _exact_direct_verify(
        self,
        context: SubquestionExecutionContext,
        target_entity_ids: list[str],
        relation_ids: list[str],
    ) -> tuple[list[ReasoningPath], list[dict[str, Any]]]:
        """Check direct source-target edges via exact SQLite pair queries when available."""
        graph_api = context.graph_api
        if hasattr(graph_api, "find_direct_edges_between"):
            candidate_paths: list[ReasoningPath] = []
            verification_rows: list[dict[str, Any]] = []
            for source_ids, target_ids, reverse_output in (
                (context.step_seed_ids, target_entity_ids, False),
                (target_entity_ids, context.step_seed_ids, True),
            ):
                rows = graph_api.find_direct_edges_between(
                    source_ids=source_ids,
                    target_ids=target_ids,
                    relation_ids=relation_ids or None,
                    limit=50,
                )
                for row in rows:
                    source_id = str(row.get("source_id") or "").strip()
                    target_id = str(row.get("target_id") or "").strip()
                    relation_id = str(row.get("relation_id") or "").strip()
                    if not source_id or not target_id or not relation_id or source_id == target_id:
                        continue
                    seed_id = target_id if reverse_output else source_id
                    candidate_id = source_id if reverse_output else target_id
                    source_label = graph_api.get_entity_display_name(seed_id)
                    target_label = graph_api.get_entity_display_name(candidate_id)
                    triple = (
                        Triple(head=target_label, relation=relation_id, tail=source_label)
                        if reverse_output
                        else Triple(head=source_label, relation=relation_id, tail=target_label)
                    )
                    nodes = [source_label, target_label]
                    candidate_paths.append(
                        ReasoningPath(
                            triples=[triple],
                            nodes=nodes,
                            text=path_to_text(nodes, [triple]),
                            source_stage="verify_solver",
                            pruning_status="preserved",
                            matched_relations=[relation_id],
                            terminal_node_id=candidate_id,
                            terminal_node_kind="id",
                            search_strategy="verify_solver",
                        )
                    )
                    verification_rows.append(
                        {
                            "source_entity_id": seed_id,
                            "target_entity_id": candidate_id,
                            "matched": True,
                            "relation_path": [relation_id],
                            "verification_mode": "direct_verify",
                        }
                    )
            if verification_rows:
                return candidate_paths, verification_rows
        connection = getattr(graph_api, "connection", None)
        if connection is None:
            return [], []

        candidate_paths: list[ReasoningPath] = []
        verification_rows: list[dict[str, Any]] = []
        relation_clause = ""
        relation_params: list[str] = []
        if relation_ids:
            placeholders = ", ".join("?" for _ in relation_ids)
            relation_clause = f" AND relation IN ({placeholders})"
            relation_params = list(relation_ids)

        for source_id in context.step_seed_ids:
            for target_id in target_entity_ids:
                if not target_id or target_id == source_id:
                    continue
                forward_rows = connection.execute(
                    (
                        "SELECT relation FROM triples "
                        "WHERE head = ? AND tail = ?"
                        f"{relation_clause} "
                        "LIMIT 20"
                    ),
                    (source_id, target_id, *relation_params),
                ).fetchall()
                reverse_rows = connection.execute(
                    (
                        "SELECT relation FROM triples "
                        "WHERE head = ? AND tail = ?"
                        f"{relation_clause} "
                        "LIMIT 20"
                    ),
                    (target_id, source_id, *relation_params),
                ).fetchall()

                source_label = graph_api.get_entity_display_name(source_id)
                target_label = graph_api.get_entity_display_name(target_id)
                for row in forward_rows:
                    relation_id = str(row["relation"] if hasattr(row, "__getitem__") else row[0])
                    triple = Triple(head=source_label, relation=relation_id, tail=target_label)
                    nodes = [source_label, target_label]
                    candidate_paths.append(
                        ReasoningPath(
                            triples=[triple],
                            nodes=nodes,
                            text=path_to_text(nodes, [triple]),
                            source_stage="verify_solver",
                            pruning_status="preserved",
                            matched_relations=[relation_id],
                            terminal_node_id=target_id,
                            terminal_node_kind="id",
                            search_strategy="verify_solver",
                        )
                    )
                    verification_rows.append(
                        {
                            "source_entity_id": source_id,
                            "target_entity_id": target_id,
                            "matched": True,
                            "relation_path": [relation_id],
                            "verification_mode": "direct_verify",
                        }
                    )
                for row in reverse_rows:
                    relation_id = str(row["relation"] if hasattr(row, "__getitem__") else row[0])
                    triple = Triple(head=target_label, relation=relation_id, tail=source_label)
                    nodes = [source_label, target_label]
                    candidate_paths.append(
                        ReasoningPath(
                            triples=[triple],
                            nodes=nodes,
                            text=path_to_text(nodes, [triple]),
                            source_stage="verify_solver",
                            pruning_status="preserved",
                            matched_relations=[relation_id],
                            terminal_node_id=target_id,
                            terminal_node_kind="id",
                            search_strategy="verify_solver",
                        )
                    )
                    verification_rows.append(
                        {
                            "source_entity_id": source_id,
                            "target_entity_id": target_id,
                            "matched": True,
                            "relation_path": [relation_id],
                            "verification_mode": "direct_verify",
                        }
                    )
        return candidate_paths, verification_rows

    def _direct_verify(
        self,
        context: SubquestionExecutionContext,
        target_entity_ids: list[str],
        relation_ids: list[str],
    ) -> tuple[list[ReasoningPath], list[dict[str, Any]]]:
        graph_api = context.graph_api
        candidate_paths: list[ReasoningPath] = []
        verification_rows: list[dict[str, Any]] = []
        for source_id in context.step_seed_ids:
            neighbors = graph_api.get_neighbors(
                source_id,
                include_reverse=True,
                relation_filter=relation_ids or None,
                limit=50,
                strict_relation_filter=bool(relation_ids),
            )
            for edge in neighbors:
                if edge.neighbor_id == source_id:
                    continue
                if edge.neighbor_id not in target_entity_ids:
                    continue
                source_label = graph_api.get_entity_display_name(source_id)
                target_label = graph_api.get_entity_display_name(edge.neighbor_id)
                if edge.reversed:
                    triple = Triple(head=target_label, relation=edge.triple.relation, tail=source_label)
                    nodes = [source_label, target_label]
                else:
                    triple = Triple(head=source_label, relation=edge.triple.relation, tail=target_label)
                    nodes = [source_label, target_label]
                candidate_paths.append(
                    ReasoningPath(
                        triples=[triple],
                        nodes=nodes,
                        text=path_to_text(nodes, [triple]),
                        source_stage="verify_solver",
                        pruning_status="preserved",
                        matched_relations=[edge.triple.relation],
                        terminal_node_id=edge.neighbor_id,
                        terminal_node_kind="id",
                        search_strategy="verify_solver",
                    )
                )
                verification_rows.append(
                    {
                        "source_entity_id": source_id,
                        "target_entity_id": edge.neighbor_id,
                        "matched": True,
                        "relation_path": [edge.triple.relation],
                        "verification_mode": "direct_verify",
                    }
                )
        return candidate_paths, verification_rows

    def _two_hop_verify(
        self,
        context: SubquestionExecutionContext,
        target_entity_ids: list[str],
        relation_ids: list[str],
    ) -> tuple[list[ReasoningPath], list[dict[str, Any]]]:
        graph_api = context.graph_api
        if hasattr(graph_api, "find_two_hop_paths_between"):
            rows = graph_api.find_two_hop_paths_between(
                source_ids=context.step_seed_ids,
                target_ids=target_entity_ids,
                relation_ids=relation_ids or None,
                limit=48,
            )
            candidate_paths: list[ReasoningPath] = []
            verification_rows: list[dict[str, Any]] = []
            for row in rows:
                source_id = str(row.get("source_id") or "").strip()
                mid_id = str(row.get("mid_id") or "").strip()
                target_id = str(row.get("target_id") or "").strip()
                first_relation_id = str(row.get("first_relation_id") or "").strip()
                second_relation_id = str(row.get("second_relation_id") or "").strip()
                if not source_id or not mid_id or not target_id or target_id == source_id:
                    continue
                source_label = graph_api.get_entity_display_name(source_id)
                mid_label = graph_api.get_entity_display_name(mid_id)
                target_label = graph_api.get_entity_display_name(target_id)
                first_triple = Triple(head=source_label, relation=first_relation_id, tail=mid_label)
                second_triple = Triple(head=mid_label, relation=second_relation_id, tail=target_label)
                nodes = [source_label, mid_label, target_label]
                candidate_paths.append(
                    ReasoningPath(
                        triples=[first_triple, second_triple],
                        nodes=nodes,
                        text=path_to_text(nodes, [first_triple, second_triple]),
                        source_stage="verify_solver",
                        pruning_status="preserved",
                        matched_relations=[first_relation_id, second_relation_id],
                        terminal_node_id=target_id,
                        terminal_node_kind="id",
                        search_strategy="verify_solver",
                    )
                )
                verification_rows.append(
                    {
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                        "matched": True,
                        "relation_path": [first_relation_id, second_relation_id],
                        "verification_mode": "short_path_verify",
                    }
                )
            if verification_rows:
                return candidate_paths, verification_rows
        candidate_paths: list[ReasoningPath] = []
        verification_rows: list[dict[str, Any]] = []
        second_hop_limit = 12
        for source_id in context.step_seed_ids:
            first_hop = graph_api.get_neighbors(
                source_id,
                include_reverse=True,
                relation_filter=relation_ids or None,
                limit=second_hop_limit,
                strict_relation_filter=False,
            )
            for first_edge in first_hop[:second_hop_limit]:
                second_hop = graph_api.get_neighbors(
                    first_edge.neighbor_id,
                    include_reverse=True,
                    relation_filter=relation_ids or None,
                    limit=second_hop_limit,
                    strict_relation_filter=False,
                )
                for second_edge in second_hop[:second_hop_limit]:
                    if second_edge.neighbor_id == source_id:
                        continue
                    if second_edge.neighbor_id not in target_entity_ids:
                        continue
                    source_label = graph_api.get_entity_display_name(source_id)
                    mid_label = graph_api.get_entity_display_name(first_edge.neighbor_id)
                    target_label = graph_api.get_entity_display_name(second_edge.neighbor_id)
                    first_triple = Triple(head=source_label, relation=first_edge.triple.relation, tail=mid_label)
                    second_triple = Triple(head=mid_label, relation=second_edge.triple.relation, tail=target_label)
                    nodes = [source_label, mid_label, target_label]
                    candidate_paths.append(
                        ReasoningPath(
                            triples=[first_triple, second_triple],
                            nodes=nodes,
                            text=path_to_text(nodes, [first_triple, second_triple]),
                            source_stage="verify_solver",
                            pruning_status="preserved",
                            matched_relations=[first_edge.triple.relation, second_edge.triple.relation],
                            terminal_node_id=second_edge.neighbor_id,
                            terminal_node_kind="id",
                            search_strategy="verify_solver",
                        )
                    )
                    verification_rows.append(
                        {
                            "source_entity_id": source_id,
                            "target_entity_id": second_edge.neighbor_id,
                            "matched": True,
                            "relation_path": [first_edge.triple.relation, second_edge.triple.relation],
                            "verification_mode": "short_path_verify",
                        }
                    )
        return candidate_paths, verification_rows

    def run(self, context: SubquestionExecutionContext) -> SubquestionSolverResult:
        graph_api = context.graph_api
        target_entity_ids = self._collect_target_entity_ids(context)
        if graph_api is None or not hasattr(graph_api, "get_neighbors") or not target_entity_ids:
            fallback = context.execute_explore()
            LOGGER.info(
                "Verify solver execution sub_question_id=%s checked_pair_count=0 matched_pair_count=0 verification_mode=fallback selected_relation_ids=[] fallback_used=true",
                context.sub_question.id,
            )
            return SubquestionSolverResult(
                solver_type="explore",
                candidate_paths=fallback,
                structured_outputs={"verification_rows": [], "entity_set_role": "candidate_answers", "set_operation_hint": "none"},
                solver_debug={"requested_solver_type": "verify", "fallback_reason": "verify_no_target_entities", "fallback_used": True},
                allow_local_relation_probe=True,
            )
        relation_ids = self._select_relation_ids(context)
        if not relation_ids:
            fallback = context.execute_explore()
            LOGGER.info(
                "Verify solver execution sub_question_id=%s checked_pair_count=%d matched_pair_count=0 verification_mode=fallback selected_relation_ids=[] fallback_used=true",
                context.sub_question.id,
                len(context.step_seed_ids) * len(target_entity_ids),
            )
            return SubquestionSolverResult(
                solver_type="explore",
                candidate_paths=fallback,
                structured_outputs={"verification_rows": [], "entity_set_role": "candidate_answers", "set_operation_hint": "none"},
                solver_debug={"requested_solver_type": "verify", "fallback_reason": "verify_no_relation_ids", "fallback_used": True},
                allow_local_relation_probe=True,
            )
        candidate_paths, verification_rows = self._exact_direct_verify(context, target_entity_ids, relation_ids)
        helper_backed_verify = hasattr(graph_api, "find_direct_edges_between") or hasattr(graph_api, "find_two_hop_paths_between")
        if not verification_rows:
            if not helper_backed_verify:
                candidate_paths, verification_rows = self._direct_verify(context, target_entity_ids, relation_ids)
        verification_mode = "direct_verify"
        if not verification_rows:
            candidate_paths, verification_rows = self._two_hop_verify(context, target_entity_ids, relation_ids)
            verification_mode = "short_path_verify"
        if not verification_rows and helper_backed_verify:
            candidate_paths, verification_rows = self._direct_verify(context, target_entity_ids, relation_ids)
        if not verification_rows:
            fallback = context.execute_explore()
            LOGGER.info(
                "Verify solver execution sub_question_id=%s checked_pair_count=%d matched_pair_count=0 verification_mode=fallback selected_relation_ids=%s fallback_used=true",
                context.sub_question.id,
                len(context.step_seed_ids) * len(target_entity_ids),
                relation_ids[:5],
            )
            return SubquestionSolverResult(
                solver_type="explore",
                candidate_paths=fallback,
                structured_outputs={"verification_rows": [], "entity_set_role": "candidate_answers", "set_operation_hint": "none"},
                solver_debug={"requested_solver_type": "verify", "fallback_reason": "verify_no_match", "fallback_used": True},
                allow_local_relation_probe=True,
            )
        primary_entity_ids = _unique_preserve_order([row["target_entity_id"] for row in verification_rows if row.get("matched")])
        LOGGER.info(
            "Verify solver execution sub_question_id=%s checked_pair_count=%d matched_pair_count=%d verification_mode=%s selected_relation_ids=%s fallback_used=false",
            context.sub_question.id,
            len(context.step_seed_ids) * len(target_entity_ids),
            len(verification_rows),
            verification_mode,
            relation_ids[:5],
        )
        return SubquestionSolverResult(
            solver_type=self.solver_type,
            candidate_paths=candidate_paths,
            structured_outputs={
                "verification_rows": verification_rows,
                "entity_set_role": "candidate_answers",
                "set_operation_hint": "none",
                "solver_execution_mode": verification_mode,
            },
            solver_debug={
                "verification_mode": verification_mode,
                "checked_pair_count": len(context.step_seed_ids) * len(target_entity_ids),
                "matched_pair_count": len(verification_rows),
                "selected_relation_ids": relation_ids,
                "fallback_used": False,
            },
            primary_entity_ids=primary_entity_ids,
            allow_local_relation_probe=False,
        )


class AggregateSolver(BaseSubquestionSolver):
    """Structured aggregate solver with SQL-backed execution over in-memory rows."""

    solver_type = "aggregate"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def _graph_sql_enabled(self) -> bool:
        aggregate_cfg = self.config.get("aggregate", {})
        graph_sql_cfg = aggregate_cfg.get("graph_sql", {})
        return bool(graph_sql_cfg.get("enabled", True))

    def _aggregate_relation_top_k(self) -> int:
        aggregate_cfg = self.config.get("aggregate", {})
        graph_sql_cfg = aggregate_cfg.get("graph_sql", {})
        return max(1, int(graph_sql_cfg.get("selected_relation_top_k", 3)))

    def _aggregate_collect_limit(self) -> int:
        aggregate_cfg = self.config.get("aggregate", {})
        graph_sql_cfg = aggregate_cfg.get("graph_sql", {})
        return max(1, int(graph_sql_cfg.get("collect_limit", 100)))

    def _aggregate_rank_limit(self) -> int:
        aggregate_cfg = self.config.get("aggregate", {})
        graph_sql_cfg = aggregate_cfg.get("graph_sql", {})
        return max(1, int(graph_sql_cfg.get("rank_limit", 100)))

    def _allow_seed_backed_aggregate(self) -> bool:
        aggregate_cfg = self.config.get("aggregate", {})
        graph_sql_cfg = aggregate_cfg.get("graph_sql", {})
        return bool(graph_sql_cfg.get("allow_seed_backed_aggregate", True))

    def _resolve_relation_ids_from_hints(self, context: SubquestionExecutionContext) -> list[str]:
        relation_ids = list(context.relation_ids)
        for hint in context.sub_question.interested_relations:
            relation_ids.extend(hint.freebase_like_ids)
        graph_api = context.graph_api
        relation_hint_names = [
            str(value).strip()
            for hint in context.sub_question.interested_relations
            for value in [hint.name, *hint.aliases, hint.description]
            if str(value).strip()
        ]
        if graph_api is not None and hasattr(graph_api, "resolve_relation_hints") and relation_hint_names:
            relation_ids.extend(graph_api.resolve_relation_hints(relation_hint_names[:6], top_k=3))
        return _unique_preserve_order(relation_ids)

    def _choose_planned_relations(
        self,
        context: SubquestionExecutionContext,
        seed_ids: list[str],
        operation: str,
    ) -> tuple[list[str], str, str, str]:
        graph_api = context.graph_api
        relation_ids = self._resolve_relation_ids_from_hints(context)
        if relation_ids:
            value_mode = "numeric_attribute" if operation in {"argmax", "argmin"} else "entity"
            return relation_ids[: self._aggregate_relation_top_k()], "forward", value_mode, "hint_only"
        if graph_api is not None and hasattr(graph_api, "expand_relation_candidates_for_aggregate"):
            candidates = graph_api.expand_relation_candidates_for_aggregate(
                seed_ids=seed_ids,
                expected_answer_type=context.sub_question.expected_answer_type,
                include_literals=True,
            )
            if candidates:
                selected = [str(item.get("relation_id") or "").strip() for item in candidates[: self._aggregate_relation_top_k()]]
                selected = [item for item in selected if item]
                if selected:
                    direction = str(candidates[0].get("direction") or "forward").strip() or "forward"
                    relation_text = normalize_text(
                        " ".join(
                            [
                                str(candidates[0].get("relation_name") or ""),
                                *[str(item) for item in candidates[0].get("sample_target_types", [])],
                            ]
                        )
                    )
                    value_mode = "entity"
                    if operation in {"argmax", "argmin"} or any(
                        token in relation_text for token in ("date", "time", "number", "rate", "population", "area", "gdp")
                    ):
                        value_mode = "numeric_attribute"
                    return selected, direction, value_mode, "backend_candidates"
        return [], "forward", "entity", "none"

    def _filter_entity_rows_by_expected_type(
        self,
        graph_api: Any,
        rows: list[dict[str, Any]],
        expected_answer_type: str,
    ) -> tuple[list[dict[str, Any]], int, int]:
        expected_type = normalize_text(expected_answer_type)
        if not expected_type or graph_api is None or not hasattr(graph_api, "get_nodes_metadata_batched"):
            return rows, len(rows), len(rows)
        target_ids = [str(row.get("target_entity_id") or "").strip() for row in rows if str(row.get("target_entity_id") or "").strip()]
        if not target_ids:
            return rows, len(rows), len(rows)
        metadata_by_id = graph_api.get_nodes_metadata_batched(target_ids)
        alias_map = {
            "country": {"location country", "country", "sovereign state", "nation"},
            "nation": {"location country", "country", "sovereign state", "nation"},
            "person": {"people person", "person"},
            "city": {"location citytown", "city", "citytown"},
            "state": {"state", "administrative division"},
            "organization": {"organization"},
            "language": {"language", "human language"},
            "religion": {"religion"},
            "movie": {"film", "movie", "motion picture", "feature film"},
            "film": {"film", "movie", "motion picture", "feature film"},
            "date": {"date", "datetime", "time"},
            "number": {"number", "integer", "float"},
        }
        expected_aliases = {expected_type, *alias_map.get(expected_type, set())}
        filtered: list[dict[str, Any]] = []
        for row in rows:
            target_id = str(row.get("target_entity_id") or "").strip()
            metadata = metadata_by_id.get(target_id)
            if metadata is None:
                continue
            haystack = normalize_text(" ".join([metadata.name, *metadata.types]))
            if any(alias and alias in haystack for alias in expected_aliases):
                filtered.append(row)
        return filtered, len(rows), len(filtered)

    def _graph_sql_aggregate(
        self,
        context: SubquestionExecutionContext,
        operation: str,
        dependency_entity_sets: list[set[str]],
    ) -> SubquestionSolverResult | None:
        graph_api = context.graph_api
        if graph_api is None or not self._graph_sql_enabled():
            return None
        seed_ids: list[str] = []
        target_scope = "dependency_entities"
        if dependency_entity_sets:
            for entity_set in dependency_entity_sets:
                seed_ids.extend(sorted(entity_set))
        elif self._allow_seed_backed_aggregate():
            seed_ids.extend(context.step_seed_ids)
            target_scope = "step_seed_entities"
        seed_ids = _unique_preserve_order(seed_ids)
        if not seed_ids:
            return None
        relation_ids, direction, value_mode, planning_mode = self._choose_planned_relations(context, seed_ids, operation)
        if operation == "intersect" and len(dependency_entity_sets) >= 2:
            entity_ids = self._sql_intersect_entity_sets(dependency_entity_sets)
            return SubquestionSolverResult(
                solver_type=self.solver_type,
                candidate_paths=[],
                structured_outputs={
                    "entity_set_role": "candidate_answers",
                    "set_operation_hint": "intersect",
                    "aggregate_rows": [{"operation": "intersect", "entity_ids": entity_ids}],
                    "solver_execution_mode": "sql_aggregate",
                },
                solver_debug={
                    "operation": "intersect",
                    "selected_relation_ids": relation_ids,
                    "relation_planning_mode": planning_mode,
                    "target_scope": target_scope,
                    "lookup_linked": False,
                    "fallback_used": False,
                    "sql_backed": True,
                    "row_count": len(entity_ids),
                    "value_mode": "entity",
                    "type_filter_before_count": len(entity_ids),
                    "type_filter_after_count": len(entity_ids),
                },
                primary_entity_ids=entity_ids,
            )
        if operation == "count":
            if dependency_entity_sets:
                union_count = self._sql_count_union(dependency_entity_sets)
                return SubquestionSolverResult(
                    solver_type=self.solver_type,
                    candidate_paths=[],
                    structured_outputs={
                        "entity_set_role": "candidate_answers",
                        "set_operation_hint": "count",
                        "aggregate_rows": [{"operation": "count", "value": union_count}],
                        "solver_execution_mode": "sql_aggregate",
                    },
                    solver_debug={
                        "operation": "count",
                        "selected_relation_ids": relation_ids,
                        "relation_planning_mode": planning_mode,
                        "target_scope": target_scope,
                        "lookup_linked": False,
                        "fallback_used": False,
                        "sql_backed": True,
                        "row_count": union_count,
                        "value_mode": "number",
                        "type_filter_before_count": union_count,
                        "type_filter_after_count": union_count,
                    },
                    primary_literals=[str(union_count)],
                )
            if relation_ids and hasattr(graph_api, "collect_related_entities"):
                entity_rows = graph_api.collect_related_entities(
                    source_ids=seed_ids,
                    relation_ids=relation_ids,
                    direction=direction,
                    limit=self._aggregate_collect_limit(),
                    expected_answer_type=context.sub_question.expected_answer_type,
                )
                filtered_rows, before_count, after_count = self._filter_entity_rows_by_expected_type(
                    graph_api,
                    entity_rows,
                    context.sub_question.expected_answer_type,
                )
                unique_entities = _unique_preserve_order(
                    [str(row.get("target_entity_id") or "").strip() for row in filtered_rows if str(row.get("target_entity_id") or "").strip()]
                )
                return SubquestionSolverResult(
                    solver_type=self.solver_type,
                    candidate_paths=[],
                    structured_outputs={
                        "entity_set_role": "candidate_answers",
                        "set_operation_hint": "count",
                        "aggregate_rows": [{"operation": "count", "value": len(unique_entities)}],
                        "solver_execution_mode": "sql_aggregate",
                    },
                    solver_debug={
                        "operation": "count",
                        "selected_relation_ids": relation_ids,
                        "relation_planning_mode": planning_mode,
                        "target_scope": target_scope,
                        "lookup_linked": True,
                        "fallback_used": False,
                        "sql_backed": True,
                        "row_count": len(unique_entities),
                        "value_mode": "entity",
                        "type_filter_before_count": before_count,
                        "type_filter_after_count": after_count,
                    },
                    primary_literals=[str(len(unique_entities))],
                )
        if operation in {"argmax", "argmin"} and relation_ids:
            ranked_rows: list[dict[str, Any]] = []
            if value_mode == "numeric_attribute" and hasattr(graph_api, "rank_entities_by_numeric_attribute"):
                ranked_rows = graph_api.rank_entities_by_numeric_attribute(
                    source_ids=seed_ids,
                    relation_ids=relation_ids,
                    direction=direction,
                    operation=operation,
                    limit=self._aggregate_rank_limit(),
                )
                if ranked_rows:
                    winner = ranked_rows[0]
                    value_id = str(winner.get("source_entity_id") or "").strip()
                    value_label = str(winner.get("source_label") or winner.get("value") or "").strip()
                    score = float(winner.get("numeric_value") or 0.0)
                    return SubquestionSolverResult(
                        solver_type=self.solver_type,
                        candidate_paths=[],
                        structured_outputs={
                            "entity_set_role": "candidate_answers",
                            "set_operation_hint": operation,
                            "aggregate_rows": [{"operation": operation, "winner_entity_id": value_id, "winner_label": value_label, "score": score}],
                            "attribute_rows": ranked_rows,
                            "solver_execution_mode": "sql_aggregate",
                        },
                        solver_debug={
                            "operation": operation,
                            "selected_relation_ids": relation_ids,
                            "relation_planning_mode": planning_mode,
                            "target_scope": target_scope,
                            "lookup_linked": True,
                            "fallback_used": False,
                            "sql_backed": True,
                            "row_count": len(ranked_rows),
                            "value_mode": value_mode,
                            "type_filter_before_count": len(ranked_rows),
                            "type_filter_after_count": len(ranked_rows),
                        },
                        primary_entity_ids=[value_id] if value_id else [],
                        primary_literals=[value_label] if value_label else [],
                    )
        if relation_ids and hasattr(graph_api, "collect_related_entities"):
            entity_rows = graph_api.collect_related_entities(
                source_ids=seed_ids,
                relation_ids=relation_ids,
                direction=direction,
                limit=self._aggregate_collect_limit(),
                expected_answer_type=context.sub_question.expected_answer_type,
            )
            filtered_rows, before_count, after_count = self._filter_entity_rows_by_expected_type(
                graph_api,
                entity_rows,
                context.sub_question.expected_answer_type,
            )
            entity_ids = _unique_preserve_order(
                [str(row.get("target_entity_id") or "").strip() for row in filtered_rows if str(row.get("target_entity_id") or "").strip()]
            )
            if entity_ids:
                return SubquestionSolverResult(
                    solver_type=self.solver_type,
                    candidate_paths=[],
                    structured_outputs={
                        "entity_set_role": "candidate_answers",
                        "set_operation_hint": "collect",
                        "aggregate_rows": [{"operation": "collect", "entity_ids": entity_ids}],
                        "solver_execution_mode": "sql_aggregate",
                    },
                    solver_debug={
                        "operation": "collect",
                        "selected_relation_ids": relation_ids,
                        "relation_planning_mode": planning_mode,
                        "target_scope": target_scope,
                        "lookup_linked": True,
                        "fallback_used": False,
                        "sql_backed": True,
                        "row_count": len(entity_ids),
                        "value_mode": "entity",
                        "type_filter_before_count": before_count,
                        "type_filter_after_count": after_count,
                    },
                    primary_entity_ids=entity_ids,
                )
        return None

    def _sql_intersect_entity_sets(self, dependency_entity_sets: list[set[str]]) -> list[str]:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE TABLE entity_sets (set_idx INTEGER, entity_id TEXT)")
            for set_idx, entity_ids in enumerate(dependency_entity_sets):
                connection.executemany(
                    "INSERT INTO entity_sets(set_idx, entity_id) VALUES (?, ?)",
                    [(set_idx, entity_id) for entity_id in sorted(entity_ids)],
                )
            rows = connection.execute(
                """
                SELECT entity_id
                FROM entity_sets
                GROUP BY entity_id
                HAVING COUNT(DISTINCT set_idx) = ?
                ORDER BY entity_id
                """,
                (len(dependency_entity_sets),),
            ).fetchall()
            return [str(row[0]) for row in rows if str(row[0]).strip()]
        finally:
            connection.close()

    def _sql_count_union(self, dependency_entity_sets: list[set[str]]) -> int:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE TABLE entity_sets (set_idx INTEGER, entity_id TEXT)")
            for set_idx, entity_ids in enumerate(dependency_entity_sets):
                connection.executemany(
                    "INSERT INTO entity_sets(set_idx, entity_id) VALUES (?, ?)",
                    [(set_idx, entity_id) for entity_id in sorted(entity_ids)],
                )
            row = connection.execute("SELECT COUNT(DISTINCT entity_id) FROM entity_sets").fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    def _sql_rank_attribute_rows(
        self,
        dependency_rows: list[dict[str, Any]],
        operation: str,
    ) -> tuple[dict[str, Any] | None, int]:
        connection = sqlite3.connect(":memory:")
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                CREATE TABLE attribute_rows (
                    row_idx INTEGER,
                    source_entity_id TEXT,
                    source_label TEXT,
                    value TEXT,
                    numeric_value REAL
                )
                """
            )
            parsed_rows: list[tuple[int, str, str, str, float]] = []
            for row_idx, row in enumerate(dependency_rows):
                value_text = str(row.get("value") or "").strip()
                try:
                    score = float(value_text.replace(",", ""))
                except ValueError:
                    continue
                parsed_rows.append(
                    (
                        row_idx,
                        str(row.get("source_entity_id") or "").strip(),
                        str(row.get("source_label") or "").strip(),
                        value_text,
                        score,
                    )
                )
            if not parsed_rows:
                return None, 0
            connection.executemany(
                """
                INSERT INTO attribute_rows(row_idx, source_entity_id, source_label, value, numeric_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                parsed_rows,
            )
            order_direction = "DESC" if operation == "argmax" else "ASC"
            best = connection.execute(
                f"""
                SELECT source_entity_id, source_label, value, numeric_value
                FROM attribute_rows
                ORDER BY numeric_value {order_direction}, row_idx ASC
                LIMIT 1
                """
            ).fetchone()
            row_count = connection.execute("SELECT COUNT(*) FROM attribute_rows").fetchone()
            return (dict(best) if best is not None else None), int(row_count[0] if row_count is not None else 0)
        finally:
            connection.close()

    def run(self, context: SubquestionExecutionContext) -> SubquestionSolverResult:
        operation = _question_implies_operation(context.sub_question.question)
        dependency_rows: list[dict[str, Any]] = []
        dependency_entity_sets: list[set[str]] = []
        dependency_entity_ids: list[str] = []
        for value in context.resolved_sub_answers.values():
            rows = value.get("attribute_rows", [])
            if isinstance(rows, list):
                dependency_rows.extend([row for row in rows if isinstance(row, dict)])
            entity_ids = value.get("entity_ids", [])
            if isinstance(entity_ids, list) and entity_ids:
                cleaned = set(str(item) for item in entity_ids if str(item).strip())
                dependency_entity_sets.append(cleaned)
                dependency_entity_ids.extend(sorted(cleaned))
        graph_sql_result = self._graph_sql_aggregate(context, operation, dependency_entity_sets)
        if graph_sql_result is not None:
            LOGGER.info(
                "Aggregate solver execution sub_question_id=%s operation=%s input_set_count=%d attribute_row_count=%d lookup_linked=%s fallback_used=false sql_backed=true",
                context.sub_question.id,
                graph_sql_result.solver_debug.get("operation", operation),
                len(dependency_entity_sets),
                len(dependency_rows),
                bool(graph_sql_result.solver_debug.get("lookup_linked", False)),
            )
            return graph_sql_result
        if operation == "intersect":
            if len(dependency_entity_sets) >= 2:
                entity_ids = self._sql_intersect_entity_sets(dependency_entity_sets)
                LOGGER.info(
                    "Aggregate solver execution sub_question_id=%s operation=intersect input_set_count=%d attribute_row_count=%d lookup_linked=false fallback_used=false sql_backed=true",
                    context.sub_question.id,
                    len(dependency_entity_sets),
                    len(dependency_rows),
                )
                return SubquestionSolverResult(
                    solver_type=self.solver_type,
                    candidate_paths=[],
                    structured_outputs={
                        "entity_set_role": "candidate_answers",
                        "set_operation_hint": "intersect",
                        "aggregate_rows": [{"operation": "intersect", "entity_ids": entity_ids}],
                        "solver_execution_mode": "sql_aggregate",
                    },
                    solver_debug={"operation": "intersect", "input_set_count": len(dependency_entity_sets), "row_count": len(entity_ids), "lookup_linked": False, "fallback_used": False, "sql_backed": True},
                    primary_entity_ids=entity_ids,
                )
        if operation == "count":
            if dependency_entity_sets:
                union_count = self._sql_count_union(dependency_entity_sets)
                LOGGER.info(
                    "Aggregate solver execution sub_question_id=%s operation=count input_set_count=%d attribute_row_count=%d lookup_linked=false fallback_used=false sql_backed=true",
                    context.sub_question.id,
                    len(dependency_entity_sets),
                    len(dependency_rows),
                )
                return SubquestionSolverResult(
                    solver_type=self.solver_type,
                    candidate_paths=[],
                    structured_outputs={
                        "entity_set_role": "candidate_answers",
                        "set_operation_hint": "count",
                        "aggregate_rows": [{"operation": "count", "value": union_count}],
                        "solver_execution_mode": "sql_aggregate",
                    },
                    solver_debug={"operation": "count", "input_set_count": len(dependency_entity_sets), "row_count": union_count, "lookup_linked": False, "fallback_used": False, "sql_backed": True},
                    primary_literals=[str(union_count)],
                )
        if dependency_rows and operation in {"argmax", "argmin"}:
            best, row_count = self._sql_rank_attribute_rows(dependency_rows, operation)
            if best is not None:
                value_id = str(best.get("source_entity_id") or "").strip()
                value_label = str(best.get("source_label") or best.get("value") or "").strip()
                score = float(best.get("numeric_value") or 0.0)
                LOGGER.info(
                    "Aggregate solver execution sub_question_id=%s operation=%s input_set_count=%d attribute_row_count=%d lookup_linked=false fallback_used=false sql_backed=true",
                    context.sub_question.id,
                    operation,
                    len(dependency_entity_sets),
                    len(dependency_rows),
                )
                return SubquestionSolverResult(
                    solver_type=self.solver_type,
                    candidate_paths=[],
                    structured_outputs={
                        "entity_set_role": "candidate_answers",
                        "set_operation_hint": operation,
                        "aggregate_rows": [{"operation": operation, "winner_entity_id": value_id, "winner_label": value_label, "score": score}],
                        "attribute_rows": dependency_rows,
                        "solver_execution_mode": "sql_aggregate",
                    },
                    solver_debug={"operation": operation, "input_set_count": len(dependency_entity_sets), "row_count": row_count, "lookup_linked": False, "fallback_used": False, "sql_backed": True},
                    primary_entity_ids=[value_id] if value_id else [],
                    primary_literals=[value_label] if value_label else [],
                )
        fallback_reason = "aggregate_missing_structured_inputs" if operation != "none" else "aggregate_unknown_operation"
        fallback = context.execute_explore()
        LOGGER.info(
            "Aggregate solver execution sub_question_id=%s operation=%s input_set_count=%d attribute_row_count=%d lookup_linked=false fallback_used=true",
            context.sub_question.id,
            operation,
            len(dependency_entity_sets),
            len(dependency_rows),
        )
        return SubquestionSolverResult(
            solver_type="explore",
            candidate_paths=fallback,
            structured_outputs={"entity_set_role": "candidate_answers", "set_operation_hint": "none"},
            solver_debug={"requested_solver_type": "aggregate", "fallback_reason": fallback_reason, "lookup_linked": False, "fallback_used": True},
            allow_local_relation_probe=True,
        )


class SubquestionSolverRouter:
    """Route one sub-question to a typed solver."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._solvers: dict[str, BaseSubquestionSolver] = {
            "explore": ExploreSolver(),
            "constrained_collect": ConstrainedCollectSolver(),
            "verify": VerifySolver(),
            "aggregate": AggregateSolver(config=config),
        }

    @staticmethod
    def _has_dependency_entities(context: SubquestionExecutionContext) -> bool:
        return any(
            context.resolved_sub_answers.get(dependency_id, {}).get("entity_ids")
            for dependency_id in context.sub_question.depends_on
        )

    @classmethod
    def _effective_solver_type(cls, context: SubquestionExecutionContext) -> str:
        solver_type = str(context.sub_question.solver_type or "").strip().lower() or "explore"
        if solver_type == "lookup":
            return "explore"
        has_constraint_anchor = bool(context.sub_question.interested_nodes)
        has_dependency_entities = cls._has_dependency_entities(context)
        if solver_type == "verify" and not (has_dependency_entities or has_constraint_anchor):
            return "explore"
        return solver_type

    def select_solver(self, sub_question: SubQuestionSpec, context: SubquestionExecutionContext) -> BaseSubquestionSolver:
        solver_type = self._effective_solver_type(context)
        if solver_type == "aggregate":
            return self._solvers["aggregate"]
        if solver_type == "verify":
            return self._solvers["verify"]
        if solver_type == "constrained_collect":
            return self._solvers["constrained_collect"]
        return self._solvers["explore"]

    def run(self, context: SubquestionExecutionContext) -> SubquestionSolverResult:
        solver = self.select_solver(context.sub_question, context)
        LOGGER.info(
            "Sub-question solver selected sub_question_id=%s requested_solver_type=%s effective_solver_type=%s seed_count=%d reason=%s",
            context.sub_question.id,
            str(context.sub_question.solver_type or "").strip().lower() or "explore",
            solver.solver_type,
            len(context.step_seed_ids),
            context.sub_question.solver_reason,
        )
        return solver.run(context)
