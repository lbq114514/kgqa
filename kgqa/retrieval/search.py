"""Reusable KGQA search strategies over graph backends."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from kgqa.kg.graph_api import BaseGraphAPI, ExternalGraphNeighbor
from kgqa.retrieval.ranking import PathCandidateScorer, PathReranker, PathSelector, SearchScoringContext
from kgqa.utils.text import deduplicate_paths
from kgqa.utils.types import ReasoningPath, Triple


@dataclass
class SearchRequest:
    """Unified search request shared by all search strategies."""

    seed_entity_ids: list[str]
    target_entity_ids: list[str] = field(default_factory=list)
    relation_hints: list[str] = field(default_factory=list)
    answer_type_hints: list[str] = field(default_factory=list)
    max_depth: int = 2
    beam_width: int = 10
    max_expansions: int = 200
    max_paths: int = 50
    literal_policy: str = "stop"
    high_degree_policy: str = "penalize"
    strict_relation_filter: bool = False


def _neighbor_fetch_limit(beam_width: int) -> int | None:
    """Translate beam width into one backend neighbor cap."""
    return 0 if int(beam_width) <= 0 else int(beam_width)


def _frontier_slice_limit(beam_width: int) -> int | None:
    """Translate beam width into one frontier width cap."""
    return None if int(beam_width) <= 0 else max(1, int(beam_width))


def _path_limit(max_paths: int) -> int | None:
    """Translate path-cap configuration into an optional limit."""
    return None if int(max_paths) <= 0 else int(max_paths)


def _path_text_for_backend(backend: BaseGraphAPI, nodes: list[str], triples: list[Triple]) -> str:
    """Serialize one path using backend-specific relation display names."""
    if not triples or len(nodes) <= 1:
        return " -> ".join(nodes)
    segments: list[str] = [nodes[0]]
    current = nodes[0]
    for index, triple in enumerate(triples):
        next_node = nodes[index + 1] if index + 1 < len(nodes) else triple.tail
        relation = backend.get_relation_display_name(triple.relation)
        if triple.tail == current and triple.head == next_node:
            relation = f"{relation} (reverse)"
        segments.append(relation)
        segments.append(next_node)
        current = next_node
    return " -> ".join(segments)


def _path_from_edge(
    backend: BaseGraphAPI,
    current_node_id: str,
    current_nodes: list[str],
    current_triples: list[Triple],
    current_edge_ids: list[str],
    current_strategy: str,
    edge: ExternalGraphNeighbor,
) -> ReasoningPath:
    current_label = current_nodes[-1]
    if edge.reversed:
        triple = Triple(
            head=edge.triple.head,
            relation=edge.triple.relation,
            tail=current_label,
        )
        next_label = edge.triple.head
    else:
        triple = Triple(
            head=current_label,
            relation=edge.triple.relation,
            tail=edge.triple.tail,
        )
        next_label = edge.triple.tail
    next_nodes = current_nodes + [next_label]
    next_triples = current_triples + [triple]
    terminal_node_id = edge.neighbor_id
    terminal_node_kind = "id" if terminal_node_id.startswith(("m.", "g.")) else "literal"
    return ReasoningPath(
        triples=next_triples,
        nodes=next_nodes,
        text=_path_text_for_backend(backend, next_nodes, next_triples),
        source_stage="retrieval_search",
        edge_ids=current_edge_ids + [f"{current_node_id}:{edge.triple.relation}:{terminal_node_id}"],
        terminal_node_id=terminal_node_id,
        terminal_node_kind=terminal_node_kind,
        search_strategy=current_strategy,
    )


class TwoHopExpansionSearcher:
    """Fast two-hop bootstrap searcher for short evidence discovery."""

    def __init__(self, backend: BaseGraphAPI) -> None:
        self.backend = backend

    def search(self, request: SearchRequest) -> list[ReasoningPath]:
        """Return short two-hop candidate paths from the provided seeds."""
        if request.max_depth <= 1:
            return []
        paths = self.backend.find_two_hop_extensions(
            frontier_nodes=request.seed_entity_ids,
            relation_hints=request.relation_hints,
            limit=_path_limit(request.max_paths),
        )
        for path in paths:
            path.search_strategy = "two_hop"
            path.terminal_node_id = path.nodes[-1] if path.nodes else ""
            path.terminal_node_kind = "id" if path.terminal_node_id.startswith(("m.", "g.")) else "literal"
        return deduplicate_paths(paths)


class ConstrainedBFSSearcher:
    """Generic BFS searcher with literal stop and relation-filter controls."""

    def __init__(self, backend: BaseGraphAPI) -> None:
        self.backend = backend

    def search(self, request: SearchRequest) -> list[ReasoningPath]:
        """Enumerate bounded simple paths from the provided seed entities."""
        queue: deque[tuple[str, list[str], list[str], list[Triple], list[str], int]] = deque()
        path_limit = _path_limit(request.max_paths)
        for seed_id in request.seed_entity_ids:
            queue.append(
                (
                    seed_id,
                    [self.backend.get_entity_display_name(seed_id)],
                    [seed_id],
                    [],
                    [],
                    0,
                )
            )
        results: list[ReasoningPath] = []
        expansions = 0
        while queue and (path_limit is None or len(results) < path_limit) and expansions < request.max_expansions:
            current_id, node_labels, node_ids, triples, edge_ids, depth = queue.popleft()
            if depth >= request.max_depth:
                continue
            for edge in self.backend.get_neighbors(
                node_id=current_id,
                include_reverse=True,
                relation_filter=request.relation_hints,
                limit=_neighbor_fetch_limit(request.beam_width),
                strict_relation_filter=request.strict_relation_filter,
            ):
                if edge.neighbor_id in node_ids:
                    continue
                next_path = _path_from_edge(
                    backend=self.backend,
                    current_node_id=current_id,
                    current_nodes=node_labels,
                    current_triples=triples,
                    current_edge_ids=edge_ids,
                    current_strategy="bfs",
                    edge=edge,
                )
                results.append(next_path)
                expansions += 1
                if (path_limit is not None and len(results) >= path_limit) or expansions >= request.max_expansions:
                    break
                if next_path.terminal_node_kind == "literal" and request.literal_policy == "stop":
                    continue
                queue.append(
                    (
                        edge.neighbor_id,
                        next_path.nodes,
                        node_ids + [edge.neighbor_id],
                        next_path.triples,
                        next_path.edge_ids,
                        depth + 1,
                    )
                )
        return deduplicate_paths(results)


class BeamSearchSearcher:
    """Heuristic beam search over graph expansions."""

    def __init__(self, backend: BaseGraphAPI, scorer: PathCandidateScorer) -> None:
        self.backend = backend
        self.scorer = scorer

    def search(self, request: SearchRequest) -> list[ReasoningPath]:
        """Search multi-hop paths while keeping only the highest-scoring frontier states."""
        context = SearchScoringContext(
            relation_hints=request.relation_hints,
            answer_type_hints=request.answer_type_hints,
            reward_literal_terminals=request.literal_policy != "stop",
        )
        path_limit = _path_limit(request.max_paths)
        frontier: list[tuple[str, list[str], list[str], list[Triple], list[str]]] = []
        for seed_id in request.seed_entity_ids:
            frontier.append((seed_id, [self.backend.get_entity_display_name(seed_id)], [seed_id], [], []))
        results: list[ReasoningPath] = []
        expansions = 0
        for _depth in range(request.max_depth):
            candidate_states: list[tuple[float, tuple[str, list[str], list[str], list[Triple], list[str]]]] = []
            for current_id, node_labels, node_ids, triples, edge_ids in frontier:
                neighbors = self.backend.get_neighbors(
                    node_id=current_id,
                    include_reverse=True,
                    relation_filter=request.relation_hints,
                    limit=_neighbor_fetch_limit(request.beam_width),
                    strict_relation_filter=request.strict_relation_filter,
                )
                for edge in neighbors:
                    if edge.neighbor_id in node_ids:
                        continue
                    next_path = _path_from_edge(
                        backend=self.backend,
                        current_node_id=current_id,
                        current_nodes=node_labels,
                        current_triples=triples,
                        current_edge_ids=edge_ids,
                        current_strategy="beam",
                        edge=edge,
                    )
                    score = self.scorer.score_partial_path(next_path, context)
                    results.append(next_path)
                    expansions += 1
                    if next_path.terminal_node_kind == "literal" and request.literal_policy == "stop":
                        continue
                    candidate_states.append(
                        (
                            score,
                            (
                                edge.neighbor_id,
                                next_path.nodes,
                                node_ids + [edge.neighbor_id],
                                next_path.triples,
                                next_path.edge_ids,
                            ),
                        )
                    )
                    if expansions >= request.max_expansions:
                        break
                if expansions >= request.max_expansions:
                    break
            if not candidate_states or expansions >= request.max_expansions:
                break
            candidate_states.sort(key=lambda item: (-item[0], len(item[1][3]), item[1][1][-1]))
            frontier_limit = _frontier_slice_limit(request.beam_width)
            if frontier_limit is None:
                frontier = [state for _, state in candidate_states]
            else:
                frontier = [state for _, state in candidate_states[:frontier_limit]]
            if path_limit is not None and len(results) >= path_limit:
                break
        deduped = deduplicate_paths(results)
        return deduped if path_limit is None else deduped[:path_limit]


class BidirectionalPathSearcher:
    """Best-effort source-target path search with an explicit target set."""

    def __init__(self, backend: BaseGraphAPI) -> None:
        self.backend = backend
        self._fallback = ConstrainedBFSSearcher(backend)

    def search(self, request: SearchRequest) -> list[ReasoningPath]:
        """Return source-target paths when target ids are provided."""
        if not request.target_entity_ids:
            return []
        all_paths = self._fallback.search(request)
        target_ids = set(request.target_entity_ids)
        filtered = [path for path in all_paths if path.terminal_node_id in target_ids]
        for path in filtered:
            path.search_strategy = "bidirectional"
        path_limit = _path_limit(request.max_paths)
        return filtered if path_limit is None else filtered[:path_limit]


class HybridSearcher:
    """Combine short bootstrap search with beam and constrained fallback expansion."""

    def __init__(
        self,
        backend: BaseGraphAPI,
        scorer: PathCandidateScorer,
        reranker: PathReranker,
        selector: PathSelector,
    ) -> None:
        self.backend = backend
        self.two_hop = TwoHopExpansionSearcher(backend)
        self.beam = BeamSearchSearcher(backend, scorer)
        self.bfs = ConstrainedBFSSearcher(backend)
        self.reranker = reranker
        self.selector = selector

    def search(self, request: SearchRequest) -> list[ReasoningPath]:
        """Run the hybrid strategy and return diverse top-ranked paths."""
        context = SearchScoringContext(
            relation_hints=request.relation_hints,
            answer_type_hints=request.answer_type_hints,
            reward_literal_terminals=request.literal_policy != "stop",
        )
        path_limit = _path_limit(request.max_paths)
        paths = []
        if request.max_depth > 1:
            paths.extend(self.two_hop.search(request))
        paths.extend(self.beam.search(request))
        if path_limit is None or len(paths) < path_limit:
            paths.extend(self.bfs.search(request))
        reranked = self.reranker.rerank(deduplicate_paths(paths), context)
        return self.selector.select(reranked)
