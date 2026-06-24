"""In-memory knowledge graph with path-search helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from kgqa.utils.types import Triple


@dataclass(frozen=True)
class GraphEdge:
    """An edge step used during graph traversal."""

    triple: Triple
    neighbor: str
    reversed: bool


class KnowledgeGraph:
    """Triple-backed knowledge graph with bidirectional traversal helpers."""

    def __init__(self, triples: list[Triple]) -> None:
        self.triples = triples
        self.entities: set[str] = set()
        self.relations: set[str] = set()
        self.adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        self.reverse_adjacency: dict[str, list[GraphEdge]] = defaultdict(list)

        for triple in triples:
            self.entities.add(triple.head)
            self.entities.add(triple.tail)
            self.relations.add(triple.relation)
            self.adjacency[triple.head].append(
                GraphEdge(triple=triple, neighbor=triple.tail, reversed=False)
            )
            self.reverse_adjacency[triple.tail].append(
                GraphEdge(triple=triple, neighbor=triple.head, reversed=True)
            )

    def get_neighbors(self, entity: str, include_reverse: bool = True) -> list[GraphEdge]:
        """Return outgoing graph edges, optionally including reverse traversal edges."""
        neighbors = list(self.adjacency.get(entity, []))
        if include_reverse:
            neighbors.extend(self.reverse_adjacency.get(entity, []))
        return neighbors

    def multi_hop_neighbors(self, source: str, max_depth: int) -> set[str]:
        """Collect nodes within max_depth undirected hops from source."""
        if source not in self.entities:
            return set()
        visited = {source}
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.get_neighbors(node, include_reverse=True):
                if edge.neighbor in visited:
                    continue
                visited.add(edge.neighbor)
                queue.append((edge.neighbor, depth + 1))
        return visited

    def bidirectional_bfs(self, source: str, target: str, max_depth: int) -> list[str]:
        """Return one shortest undirected path between two entities."""
        if source == target:
            return [source]
        if source not in self.entities or target not in self.entities:
            return []
        queue: deque[tuple[str, list[str], int]] = deque([(source, [source], 0)])
        visited = {source}
        while queue:
            node, path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.get_neighbors(node, include_reverse=True):
                if edge.neighbor in visited:
                    continue
                next_path = path + [edge.neighbor]
                if edge.neighbor == target:
                    return next_path
                visited.add(edge.neighbor)
                queue.append((edge.neighbor, next_path, depth + 1))
        return []

    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int,
        max_paths: int = 20,
    ) -> list[tuple[list[str], list[Triple]]]:
        """Find simple paths between two entities up to max_depth hops."""
        if source not in self.entities or target not in self.entities:
            return []
        if source == target:
            return [([source], [])]

        results: list[tuple[list[str], list[Triple]]] = []
        queue: deque[tuple[str, list[str], list[Triple]]] = deque([(source, [source], [])])

        while queue and len(results) < max_paths:
            node, nodes, triples = queue.popleft()
            if len(triples) >= max_depth:
                continue
            for edge in self.get_neighbors(node, include_reverse=True):
                if edge.neighbor in nodes:
                    continue
                next_nodes = nodes + [edge.neighbor]
                next_triples = triples + [edge.triple]
                if edge.neighbor == target:
                    results.append((next_nodes, next_triples))
                    if len(results) >= max_paths:
                        break
                    continue
                queue.append((edge.neighbor, next_nodes, next_triples))
        return results

    def induced_subgraph(self, nodes: set[str]) -> "KnowledgeGraph":
        """Return the subgraph induced by the provided nodes."""
        triples = [
            triple
            for triple in self.triples
            if triple.head in nodes and triple.tail in nodes
        ]
        return KnowledgeGraph(triples)
