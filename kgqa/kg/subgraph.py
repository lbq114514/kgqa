"""Question subgraph construction and topic connectivity pruning."""

from __future__ import annotations

from kgqa.kg.graph import KnowledgeGraph


def build_question_subgraph(
    kg: KnowledgeGraph,
    topic_entities: list[str],
    dmax: int,
) -> KnowledgeGraph:
    """Build a local question subgraph by Dmax-hop expansion from topic entities."""
    collected_nodes: set[str] = set()
    for entity in topic_entities:
        collected_nodes.update(kg.multi_hop_neighbors(entity, dmax))
    if not collected_nodes:
        return KnowledgeGraph([])
    return kg.induced_subgraph(collected_nodes)


def prune_subgraph_by_topic_connectivity(
    subgraph: KnowledgeGraph,
    topic_entities: list[str],
) -> KnowledgeGraph:
    """Keep connectivity paths among topic entities and drop clear side branches."""
    if len(topic_entities) < 2:
        return subgraph

    kept_nodes: set[str] = set(topic_entities)
    for index, source in enumerate(topic_entities[:-1]):
        for target in topic_entities[index + 1 :]:
            path = subgraph.bidirectional_bfs(source, target, max_depth=max(len(subgraph.entities), 1))
            if path:
                kept_nodes.update(path)

    if kept_nodes == set(topic_entities):
        return subgraph
    return subgraph.induced_subgraph(kept_nodes)
