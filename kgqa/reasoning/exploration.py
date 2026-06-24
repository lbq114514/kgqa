"""Path exploration stages for topic, supplement, and node expansion search."""

from __future__ import annotations

import json
from itertools import product

from kgqa.kg.entity_linking import link_entities_to_kg, link_relations_to_kg
from kgqa.kg.graph_api import BaseGraphAPI, ExternalGraphNeighbor
from kgqa.kg.graph import KnowledgeGraph
from kgqa.llm.base import BaseLLM
from kgqa.llm.prompts import CWQ_BOOTSTRAP_EXPLORATION_PROMPT, SUPPLEMENT_ENTITY_PROMPT
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.text import deduplicate_paths, path_to_text
from kgqa.utils.types import ExplorationHints, QuestionAnalysisResult, ReasoningPath, SupplementHints, Triple

LOGGER = get_logger(__name__)


def find_paths_between_entities(
    graph: KnowledgeGraph,
    source: str,
    target: str,
    max_depth: int,
    max_paths: int = 20,
) -> list[ReasoningPath]:
    """Find simple paths between two entities within a depth limit."""
    raw_paths = graph.find_paths(source, target, max_depth=max_depth, max_paths=max_paths)
    return [
        ReasoningPath(
            triples=triples,
            nodes=nodes,
            text=path_to_text(nodes, triples),
            source_stage="topic_entity_path_exploration",
        )
        for nodes, triples in raw_paths
    ]


def _merge_path_segments(segments: list[ReasoningPath], source_stage: str) -> ReasoningPath:
    """Merge adjacent entity-to-entity segments into one reasoning path."""
    merged_nodes: list[str] = []
    merged_triples: list[Triple] = []
    for index, segment in enumerate(segments):
        if index == 0:
            merged_nodes.extend(segment.nodes)
            merged_triples.extend(segment.triples)
            continue
        merged_nodes.extend(segment.nodes[1:])
        merged_triples.extend(segment.triples)
    return ReasoningPath(
        triples=merged_triples,
        nodes=merged_nodes,
        text=path_to_text(merged_nodes, merged_triples),
        source_stage=source_stage,
    )


def explore_topic_entity_paths(
    subgraph: KnowledgeGraph,
    ordered_topic_entities: list[str],
    dpredict: int,
    dmax: int,
    max_paths_per_pair: int = 20,
) -> list[ReasoningPath]:
    """Search for reasoning paths that cover the ordered topic entities."""
    if not ordered_topic_entities:
        return []
    if len(ordered_topic_entities) == 1:
        return [
            ReasoningPath(
                triples=[],
                nodes=ordered_topic_entities,
                text=ordered_topic_entities[0],
                source_stage="topic_entity_path_exploration",
            )
        ]

    for depth in range(min(dpredict, dmax), dmax + 1):
        segment_candidates: list[list[ReasoningPath]] = []
        for source, target in zip(ordered_topic_entities, ordered_topic_entities[1:]):
            paths = find_paths_between_entities(
                subgraph,
                source,
                target,
                max_depth=depth,
                max_paths=max_paths_per_pair,
            )
            if not paths:
                segment_candidates = []
                break
            segment_candidates.append(paths)

        if not segment_candidates:
            continue

        merged = [
            _merge_path_segments(list(combo), "topic_entity_path_exploration")
            for combo in product(*segment_candidates)
        ]
        deduped = deduplicate_paths(merged)
        if deduped:
            LOGGER.info("Found %d topic-entity paths at depth %d", len(deduped), depth)
            return deduped
    return []


def annotate_paths_with_relation_matches(
    paths: list[ReasoningPath],
    relation_hints: list[str],
) -> list[ReasoningPath]:
    """Annotate each path with the subset of relation hints it contains."""
    deduped_hints: list[str] = []
    seen_hints: set[str] = set()
    for relation in relation_hints:
        if relation in seen_hints:
            continue
        seen_hints.add(relation)
        deduped_hints.append(relation)

    for path in paths:
        path_relations = {triple.relation for triple in path.triples}
        path.matched_relations = [relation for relation in deduped_hints if relation in path_relations]
    return paths


def sort_paths_by_relation_bias(paths: list[ReasoningPath]) -> list[ReasoningPath]:
    """Softly prioritize paths that match more hinted relations and use fewer hops."""
    ranked = list(enumerate(paths))
    ranked.sort(
        key=lambda item: (
            0 if item[1].matched_relations else 1,
            -len(item[1].matched_relations),
            len(item[1].triples),
            item[0],
        )
    )
    return [path for _, path in ranked]


def _triple_from_external_neighbor(frontier_label: str, edge: ExternalGraphNeighbor) -> Triple:
    """Align one external edge with the current frontier label."""
    if edge.reversed:
        return Triple(
            head=edge.triple.head,
            relation=edge.triple.relation,
            tail=frontier_label,
        )
    return Triple(
        head=frontier_label,
        relation=edge.triple.relation,
        tail=edge.triple.tail,
    )


def generate_supplement_hints_with_llm(
    question: str,
    topic_entities: list[str],
    current_paths: list[ReasoningPath],
    kg_entities: list[str],
    kg_relations: list[str],
    llm: BaseLLM,
) -> SupplementHints:
    """Ask the LLM for likely bridge entities and useful relation hints."""
    current_path_text = json.dumps([path.text for path in current_paths], ensure_ascii=False, indent=2)
    prompt = SUPPLEMENT_ENTITY_PROMPT.format(
        question=question,
        topic_entities=topic_entities,
        current_paths=current_path_text,
    )
    raw = llm.generate(prompt)
    parsed = robust_json_parse(raw, fallback={})

    if isinstance(parsed, list):
        entity_candidates = [str(item).strip() for item in parsed if str(item).strip()]
        relation_candidates: list[str] = []
    elif isinstance(parsed, dict):
        raw_entities = parsed.get("entities") or []
        raw_relations = parsed.get("relations") or []
        entity_candidates = [str(item).strip() for item in raw_entities if str(item).strip()]
        relation_candidates = [str(item).strip() for item in raw_relations if str(item).strip()]
    else:
        entity_candidates = []
        relation_candidates = []

    linked_entities = link_entities_to_kg(entity_candidates, kg_entities, top_k=1)
    linked_relations = link_relations_to_kg(relation_candidates, kg_relations, top_k=3)
    hints = SupplementHints(
        entity_candidates=entity_candidates,
        linked_entities=linked_entities,
        relation_candidates=relation_candidates,
        linked_relations=linked_relations,
    )
    LOGGER.info("Supplement hints: %s", hints)
    return hints


def generate_cwq_bootstrap_hints_with_llm(
    question: str,
    topic_entities: list[str],
    question_analysis: QuestionAnalysisResult,
    llm: BaseLLM,
) -> ExplorationHints:
    """Ask the LLM for the first round of CWQ external-graph exploration hints."""
    prompt = CWQ_BOOTSTRAP_EXPLORATION_PROMPT.format(
        question=question,
        topic_entities=json.dumps(topic_entities, ensure_ascii=False),
        question_analysis=json.dumps(question_analysis.to_dict(), ensure_ascii=False, indent=2),
    )
    raw = llm.generate(prompt)
    parsed = robust_json_parse(raw, fallback={})
    if isinstance(parsed, dict):
        focus_entities = [str(item).strip() for item in parsed.get("focus_entities", []) if str(item).strip()]
        answer_type_hints = [str(item).strip() for item in parsed.get("answer_type_hints", []) if str(item).strip()]
        relation_name_hints = [
            str(item).strip() for item in parsed.get("relation_name_hints", []) if str(item).strip()
        ]
        reasoning_focus = str(parsed.get("reasoning_focus", "")).strip()
    else:
        focus_entities = []
        answer_type_hints = []
        relation_name_hints = []
        reasoning_focus = ""

    hints = ExplorationHints(
        focus_entities=focus_entities,
        answer_type_hints=answer_type_hints,
        relation_name_hints=relation_name_hints,
        reasoning_focus=reasoning_focus,
    )
    LOGGER.info("CWQ bootstrap hints: %s", hints)
    return hints


def explore_supplement_paths(
    subgraph: KnowledgeGraph,
    topic_entities: list[str],
    supplement_hints: SupplementHints,
    dmax: int,
    max_paths_per_pair: int = 20,
) -> list[ReasoningPath]:
    """Re-run ordered path search after inserting linked supplementary entities."""
    if not supplement_hints.linked_entities or not topic_entities:
        return []

    candidate_sequences: list[list[str]] = []
    for entity in supplement_hints.linked_entities:
        if entity in topic_entities:
            continue
        if len(topic_entities) == 1:
            candidate_sequences.append([topic_entities[0], entity])
        else:
            candidate_sequences.append([topic_entities[0], entity, *topic_entities[1:]])

    results: list[ReasoningPath] = []
    for sequence in candidate_sequences:
        results.extend(
            explore_topic_entity_paths(
                subgraph=subgraph,
                ordered_topic_entities=sequence,
                dpredict=min(2, dmax),
                dmax=dmax,
                max_paths_per_pair=max_paths_per_pair,
            )
        )

    for path in results:
        path.source_stage = "llm_supplement_path_exploration"
    deduped = deduplicate_paths(results)
    annotate_paths_with_relation_matches(deduped, supplement_hints.linked_relations)
    return sort_paths_by_relation_bias(deduped)


def explore_graphapi_supplement_paths(
    topic_entities: list[str],
    supplement_hints: SupplementHints,
    graph_api: BaseGraphAPI,
    neighbor_limit: int,
    relation_bias_top_k: int,
) -> list[ReasoningPath]:
    """Query the optional external graph API for short supplement paths."""
    seed_mentions = list(topic_entities)
    seed_mentions.extend(supplement_hints.entity_candidates)
    seed_mentions.extend(supplement_hints.linked_entities)
    frontier_nodes = graph_api.resolve_entity_mentions(seed_mentions, top_k=1)
    relation_hint_pool = supplement_hints.linked_relations or supplement_hints.relation_candidates
    relation_hints = relation_hint_pool[: max(1, relation_bias_top_k)]
    LOGGER.info(
        "GraphAPI supplement seeds=%s resolved_frontiers=%s relation_hints=%s neighbor_limit=%d",
        seed_mentions,
        frontier_nodes,
        relation_hints,
        neighbor_limit,
    )
    if not frontier_nodes:
        LOGGER.info("GraphAPI supplement skipped because no frontier nodes were resolved.")
        return []

    paths = graph_api.find_two_hop_extensions(
        frontier_nodes=frontier_nodes,
        relation_hints=relation_hints,
        limit=neighbor_limit,
    )
    for path in paths:
        path.source_stage = "external_graphapi_supplement"
    deduped = deduplicate_paths(paths)
    annotate_paths_with_relation_matches(deduped, relation_hints)
    LOGGER.info(
        "GraphAPI supplement produced raw=%d deduped=%d preview=%s",
        len(paths),
        len(deduped),
        [path.text for path in deduped[:3]],
    )
    return sort_paths_by_relation_bias(deduped)


def explore_graphapi_bootstrap_paths(
    aligned_entity_ids: list[str],
    aligned_relation_ids: list[str],
    graph_api: BaseGraphAPI,
    neighbor_limit: int,
) -> list[ReasoningPath]:
    """Query the external graph API using pre-aligned CWQ entity and relation ids."""
    LOGGER.info(
        "GraphAPI bootstrap aligned_entity_ids=%s aligned_relation_ids=%s neighbor_limit=%d",
        aligned_entity_ids,
        aligned_relation_ids,
        neighbor_limit,
    )
    if not aligned_entity_ids:
        LOGGER.info("GraphAPI bootstrap skipped because no aligned entity ids were resolved.")
        return []

    paths = graph_api.find_two_hop_extensions(
        frontier_nodes=aligned_entity_ids,
        relation_hints=aligned_relation_ids,
        limit=neighbor_limit,
    )
    for path in paths:
        path.source_stage = "external_graphapi_supplement"
    deduped = deduplicate_paths(paths)
    annotate_paths_with_relation_matches(deduped, aligned_relation_ids)
    LOGGER.info(
        "GraphAPI bootstrap produced raw=%d deduped=%d preview=%s",
        len(paths),
        len(deduped),
        [path.text for path in deduped[:3]],
    )
    return sort_paths_by_relation_bias(deduped)


def expand_nodes_from_paths(
    current_paths: list[ReasoningPath],
    kg: KnowledgeGraph,
    visited_nodes: set[str],
    max_expand_steps: int,
) -> list[ReasoningPath]:
    """Expand one-hop neighbors around current path endpoints to propose new paths."""
    expanded_paths: list[ReasoningPath] = list(current_paths)

    for _ in range(max_expand_steps):
        new_paths: list[ReasoningPath] = []
        for path in expanded_paths:
            frontier_nodes = [path.nodes[0], path.nodes[-1]] if path.nodes else []
            for frontier in frontier_nodes:
                for edge in kg.get_neighbors(frontier, include_reverse=True):
                    if edge.neighbor in visited_nodes:
                        continue
                    visited_nodes.add(edge.neighbor)
                    if frontier == path.nodes[0]:
                        nodes = [edge.neighbor] + path.nodes
                        triples = [edge.triple] + path.triples
                    else:
                        nodes = path.nodes + [edge.neighbor]
                        triples = path.triples + [edge.triple]
                    new_paths.append(
                        ReasoningPath(
                            triples=triples,
                            nodes=nodes,
                            text=path_to_text(nodes, triples),
                            source_stage="node_expand_exploration",
                        )
                    )
        if not new_paths:
            break
        expanded_paths = deduplicate_paths(expanded_paths + new_paths)
    return deduplicate_paths(expanded_paths)


def expand_nodes_with_graph_api(
    current_paths: list[ReasoningPath],
    graph_api: BaseGraphAPI,
    visited_nodes: set[str],
    relation_hints: list[str],
    neighbor_limit: int,
    relation_bias_top_k: int,
) -> list[ReasoningPath]:
    """Expand one hop around path endpoints using the optional external graph API."""
    if not current_paths:
        return []

    hint_subset = relation_hints[: max(1, relation_bias_top_k)]
    new_paths: list[ReasoningPath] = []
    resolved_cache: dict[str, list[str]] = {}
    LOGGER.info(
        "GraphAPI node expand start path_count=%d relation_hints=%s neighbor_limit=%d",
        len(current_paths),
        hint_subset,
        neighbor_limit,
    )

    for path in current_paths:
        if not path.nodes:
            continue
        for is_prefix, frontier_label in ((True, path.nodes[0]), (False, path.nodes[-1])):
            candidate_mids = resolved_cache.get(frontier_label)
            if candidate_mids is None:
                candidate_mids = graph_api.resolve_entity_mentions([frontier_label], top_k=1)
                resolved_cache[frontier_label] = candidate_mids
                LOGGER.info(
                    "GraphAPI node expand frontier=%s resolved_mids=%s",
                    frontier_label,
                    candidate_mids,
                )
            for mid in candidate_mids:
                for edge in graph_api.get_neighbors(
                    mid,
                    include_reverse=True,
                    relation_filter=hint_subset,
                    limit=neighbor_limit,
                ):
                    triple = _triple_from_external_neighbor(frontier_label, edge)
                    next_label = triple.head if edge.reversed else triple.tail
                    if next_label in visited_nodes or next_label in path.nodes:
                        continue
                    if is_prefix:
                        nodes = [next_label] + path.nodes
                        triples = [triple] + path.triples
                    else:
                        nodes = path.nodes + [next_label]
                        triples = path.triples + [triple]
                    visited_nodes.add(next_label)
                    new_paths.append(
                        ReasoningPath(
                            triples=triples,
                            nodes=nodes,
                            text=path_to_text(nodes, triples),
                            source_stage="external_graphapi_node_expand",
                        )
                    )

    deduped = deduplicate_paths(new_paths)
    annotate_paths_with_relation_matches(deduped, hint_subset)
    LOGGER.info(
        "GraphAPI node expand produced raw=%d deduped=%d preview=%s",
        len(new_paths),
        len(deduped),
        [path.text for path in deduped[:3]],
    )
    return sort_paths_by_relation_bias(deduped)
