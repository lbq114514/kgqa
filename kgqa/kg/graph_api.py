"""Optional external graph API abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kgqa.utils.types import ReasoningPath, Triple


@dataclass(frozen=True)
class ExternalGraphNeighbor:
    """A graph neighbor returned from an external graph backend."""

    source_id: str
    neighbor_id: str
    triple: Triple
    relation_name: str
    reversed: bool


class BaseGraphAPI(ABC):
    """Minimal interface for optional external graph augmentation."""

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        include_reverse: bool = True,
        relation_filter: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ExternalGraphNeighbor]:
        """Return neighbors for one graph node."""

    @abstractmethod
    def resolve_entity_mentions(self, names: list[str], top_k: int = 1) -> list[str]:
        """Resolve surface-form mentions to backend node ids."""

    @abstractmethod
    def get_entity_display_name(self, node_id: str) -> str:
        """Return a readable label for one node id."""

    @abstractmethod
    def resolve_relation_hints(self, names: list[str], top_k: int = 3) -> list[str]:
        """Resolve human-readable relation hints to canonical backend relation ids."""

    @abstractmethod
    def get_relation_display_name(self, relation_id: str) -> str:
        """Return a readable label for one relation id."""

    @abstractmethod
    def find_two_hop_extensions(
        self,
        frontier_nodes: list[str],
        relation_hints: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ReasoningPath]:
        """Return short candidate reasoning paths rooted at frontier nodes."""

    def get_neighbors_batched(
        self,
        node_ids: list[str],
        include_reverse: bool = True,
        relation_filter: list[str] | None = None,
        limit: int | None = None,
        strict_relation_filter: bool = False,
    ) -> dict[str, list[ExternalGraphNeighbor]]:
        """Return neighbors for a batch of graph nodes."""
        results: dict[str, list[ExternalGraphNeighbor]] = {}
        for node_id in node_ids:
            try:
                results[node_id] = self.get_neighbors(
                    node_id=node_id,
                    include_reverse=include_reverse,
                    relation_filter=relation_filter,
                    limit=limit,
                    strict_relation_filter=strict_relation_filter,
                )
            except TypeError:
                results[node_id] = self.get_neighbors(
                    node_id=node_id,
                    include_reverse=include_reverse,
                    relation_filter=relation_filter,
                    limit=limit,
                )
        return results

    def get_node_edges(
        self,
        node_id: str,
        include_reverse: bool = True,
        include_literals: bool = True,
        limit: int | None = None,
    ) -> list[ExternalGraphNeighbor]:
        """Return precise local edges for one node when available."""
        neighbors = self.get_neighbors(
            node_id=node_id,
            include_reverse=include_reverse,
            relation_filter=None,
            limit=limit,
        )
        if include_literals:
            return neighbors
        return [edge for edge in neighbors if edge.neighbor_id.startswith(("m.", "g."))]

    def resolve_entity_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
        """Return rich entity candidates when the backend supports it."""
        rows: list[dict[str, object]] = []
        for mention in names:
            for node_id in self.resolve_entity_mentions([mention], top_k=top_k):
                rows.append(
                    {
                        "mid": node_id,
                        "name": self.get_entity_display_name(node_id),
                        "mention": mention,
                        "match_type": "backend_default",
                        "score": 0.0,
                    }
                )
        return rows

    def resolve_relation_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
        """Return rich relation candidates when the backend supports it."""
        rows: list[dict[str, object]] = []
        for hint in names:
            for relation_id in self.resolve_relation_hints([hint], top_k=top_k):
                rows.append(
                    {
                        "relation": relation_id,
                        "relation_name": self.get_relation_display_name(relation_id),
                        "hint": hint,
                        "match_type": "backend_default",
                        "score": 0.0,
                    }
                )
        return rows

    def get_node_types(self, node_id: str) -> list[str]:
        """Return coarse types for one node id when available."""
        return []

    def get_node_degree(self, node_id: str) -> int:
        """Return the total degree for one node id when available."""
        return len(self.get_neighbors(node_id=node_id, include_reverse=True, relation_filter=None, limit=0))

    def get_relation_frequency(self, relation_id: str) -> int:
        """Return a coarse global relation frequency when available."""
        return 0

    def collect_related_entities(
        self,
        source_ids: list[str],
        relation_ids: list[str],
        direction: str = "forward",
        limit: int = 100,
        expected_answer_type: str = "",
    ) -> list[dict[str, object]]:
        """Collect entity targets for a fixed relation set with light filtering."""
        rows = self.collect_related_entities_constrained(
            source_ids=source_ids,
            relation_ids=relation_ids,
            direction=direction,
            expected_answer_type=expected_answer_type,
            constraint_terms=[],
            limit=limit,
        )
        return [
            {
                "source_entity_id": row.get("source_entity_id"),
                "target_entity_id": row.get("target_entity_id"),
                "target_label": row.get("target_label"),
                "relation_id": row.get("relation_id"),
            }
            for row in rows
        ]

    def collect_related_entities_constrained(
        self,
        source_ids: list[str],
        relation_ids: list[str],
        direction: str = "forward",
        expected_answer_type: str = "",
        constraint_terms: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Collect relation targets and rank them using lightweight local constraints."""
        relation_ids = [item for item in relation_ids if item]
        if not source_ids or not relation_ids or limit <= 0:
            return []
        normalized_expected_type = " ".join(str(expected_answer_type or "").strip().lower().split())
        normalized_terms = [
            " ".join(str(term).strip().lower().split())
            for term in (constraint_terms or [])
            if str(term).strip()
        ]
        rows: list[dict[str, object]] = []
        for source_id in source_ids:
            try:
                neighbors = self.get_neighbors(
                    node_id=source_id,
                    include_reverse=True,
                    relation_filter=relation_ids,
                    limit=max(limit * 2, len(relation_ids)),
                    strict_relation_filter=True,
                )
            except TypeError:
                neighbors = self.get_neighbors(
                    node_id=source_id,
                    include_reverse=True,
                    relation_filter=relation_ids,
                    limit=max(limit * 2, len(relation_ids)),
                )
            for edge in neighbors:
                if direction == "forward" and edge.reversed:
                    continue
                if direction == "reverse" and not edge.reversed:
                    continue
                target_id = edge.source_id if edge.reversed else edge.neighbor_id
                target_label = self.get_entity_display_name(target_id)
                haystack = " ".join(
                    [
                        " ".join(str(target_label or "").strip().lower().split()),
                        " ".join(" ".join(str(item).strip().lower().split()) for item in self.get_node_types(target_id)),
                    ]
                ).strip()
                score = 0.0
                if normalized_expected_type and normalized_expected_type in haystack:
                    score += 6.0
                for term in normalized_terms:
                    if term and term in haystack:
                        score += 3.0
                rows.append(
                    {
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                        "target_label": target_label,
                        "relation_id": edge.triple.relation,
                        "score": score,
                        "matched_constraints": [term for term in normalized_terms if term and term in haystack],
                    }
                )
        rows.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("relation_id") or ""),
                str(item.get("target_entity_id") or ""),
            )
        )
        deduped: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            key = (
                str(row.get("source_entity_id") or ""),
                str(row.get("relation_id") or ""),
                str(row.get("target_entity_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= limit:
                break
        return deduped
