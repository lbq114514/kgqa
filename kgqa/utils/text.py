"""Text normalization and path serialization helpers."""

from __future__ import annotations

from typing import Iterable

from kgqa.utils.types import ReasoningPath, Triple


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return " ".join(text.strip().lower().split())


def triple_to_text(triple: Triple) -> str:
    """Serialize a triple into a short string."""
    return f"{triple.head} -> {triple.relation} -> {triple.tail}"


def path_to_text(nodes: list[str], triples: list[Triple]) -> str:
    """Serialize a path using the traversed node order."""
    if not triples or len(nodes) <= 1:
        return " -> ".join(nodes)
    segments: list[str] = [nodes[0]]
    current = nodes[0]
    for index, triple in enumerate(triples):
        next_node = nodes[index + 1] if index + 1 < len(nodes) else triple.tail
        if triple.head == current and triple.tail == next_node:
            relation = triple.relation
        elif triple.tail == current and triple.head == next_node:
            relation = f"{triple.relation} (reverse)"
        else:
            relation = triple.relation
        segments.append(relation)
        segments.append(next_node)
        current = next_node
    return " -> ".join(segments)


def _path_structure_identity(path: ReasoningPath) -> tuple[object, ...]:
    """Return a stable structure key independent of transient path metadata."""
    if path.edge_ids:
        return ("edges", tuple(path.edge_ids))
    return (
        "structure",
        path.text,
        tuple((triple.head, triple.relation, triple.tail) for triple in path.triples),
        tuple(path.nodes),
    )


def _path_metadata_score(path: ReasoningPath) -> tuple[int, int]:
    """Prefer paths that carry concrete graph identity metadata."""
    return (
        1 if path.terminal_node_id else 0,
        len(path.edge_ids),
    )


def deduplicate_paths(paths: Iterable[ReasoningPath]) -> list[ReasoningPath]:
    """Remove duplicate reasoning paths while preserving distinct terminal entities."""
    unique_paths: list[ReasoningPath] = []
    structure_index: dict[tuple[object, ...], list[int]] = {}
    for path in paths:
        structure_key = _path_structure_identity(path)
        matched = False
        for index in structure_index.get(structure_key, []):
            existing = unique_paths[index]
            existing_terminal = existing.terminal_node_id or ""
            current_terminal = path.terminal_node_id or ""
            if existing_terminal and current_terminal and existing_terminal != current_terminal:
                continue
            if _path_metadata_score(path) > _path_metadata_score(existing):
                unique_paths[index] = path
            matched = True
            break
        if matched:
            continue
        unique_paths.append(path)
        structure_index.setdefault(structure_key, []).append(len(unique_paths) - 1)
    return unique_paths
