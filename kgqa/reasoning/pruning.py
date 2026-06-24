"""Candidate path pruning with embeddings and local vLLM selection."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from kgqa.llm.base import BaseLLM
from kgqa.llm.prompts import PRECISE_PATH_SELECTION_PROMPT
from kgqa.reasoning.exploration import annotate_paths_with_relation_matches
from kgqa.utils.embeddings import get_sentence_transformer
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.types import QuestionAnalysisResult, ReasoningPath

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOGGER = get_logger(__name__)

_SCHEMA_NOISE_PREFIXES = (
    "/common/",
    "/freebase/",
    "/user/",
    "/base/",
    "common.",
    "freebase.",
    "user.",
    "base.",
    "type.object.",
)


def set_default_embedding_model(model_name: str) -> None:
    """Override the default embedding model used by this module."""
    global DEFAULT_EMBEDDING_MODEL
    DEFAULT_EMBEDDING_MODEL = model_name


def get_embedding_model(model_name: str):
    """Load and cache the SentenceTransformer backend."""
    return get_sentence_transformer(model_name)


def _encode_silently(model: object, texts: list[str]):
    """Encode text without progress bars when supported by the backend."""
    try:
        return model.encode(texts, show_progress_bar=False)
    except TypeError:
        return model.encode(texts)


def _compact_path_text(path: ReasoningPath) -> str:
    """Compress a path into start entity, relation sequence, and end entity."""
    if not path.nodes:
        return path.text
    if not path.triples or len(path.nodes) == 1:
        return path.nodes[0]
    relations = [triple.relation for triple in path.triples]
    return " | ".join([path.nodes[0], *relations, path.nodes[-1]])


def _serialize_candidate_path(path: ReasoningPath, index: int, compact: bool) -> dict[str, object]:
    """Serialize one candidate path for precise LLM selection."""
    return {
        "index": index,
        "path": _compact_path_text(path) if compact else path.text,
        "matched_relations": path.matched_relations,
        "source_stage": path.source_stage,
    }


def fuzzy_select_paths(
    question_indicator: str,
    candidate_paths: list[ReasoningPath],
    w1: int,
    relation_hints: list[str] | None = None,
) -> list[ReasoningPath]:
    """Use embedding similarity to keep the most relevant candidate paths."""
    if not candidate_paths:
        return []
    model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    query_embedding = np.asarray(_encode_silently(model, [question_indicator]))
    path_embeddings = np.asarray(_encode_silently(model, [path.text for path in candidate_paths]))
    similarities = cosine_similarity(query_embedding, path_embeddings)[0]
    relation_hints = relation_hints or []
    annotate_paths_with_relation_matches(candidate_paths, relation_hints)

    if relation_hints:
        ranked_indices = sorted(
            range(len(candidate_paths)),
            key=lambda index: (
                0 if candidate_paths[index].matched_relations else 1,
                -len(candidate_paths[index].matched_relations),
                -float(similarities[index]),
                index,
            ),
        )[: min(w1, len(candidate_paths))]
    else:
        ranked_indices = np.argsort(similarities)[::-1][: min(w1, len(candidate_paths))]
    selected = [candidate_paths[int(index)] for index in ranked_indices]
    LOGGER.info("Fuzzy selection reduced %d paths to %d", len(candidate_paths), len(selected))
    return selected


def precise_select_paths_with_llm(
    question: str,
    question_analysis: QuestionAnalysisResult,
    candidate_paths: list[ReasoningPath],
    wmax: int,
    supplement_relations: list[str],
    sample_id: str | None,
    compact_prompt_paths: bool,
    llm: BaseLLM,
) -> list[ReasoningPath]:
    """Use the local LLM to choose the most useful candidate paths."""
    if not candidate_paths:
        return []
    indexed_paths = [
        _serialize_candidate_path(path, index, compact=compact_prompt_paths)
        for index, path in enumerate(candidate_paths)
    ]
    prompt = PRECISE_PATH_SELECTION_PROMPT.format(
        question=question,
        question_analysis=json.dumps(question_analysis.to_dict(), ensure_ascii=False),
        supplement_relations=json.dumps(supplement_relations, ensure_ascii=False),
        candidate_paths=json.dumps(indexed_paths, ensure_ascii=False, indent=2),
        wmax=wmax,
    )
    LOGGER.info(
        "Precise selection request sample_id=%s candidate_count=%d compact_prompt_paths=%s prompt_size_chars=%d",
        sample_id or "<none>",
        len(candidate_paths),
        compact_prompt_paths,
        len(prompt),
    )
    raw = llm.generate(prompt)
    parsed = robust_json_parse(raw, fallback=[])
    if isinstance(parsed, list):
        indices = [int(index) for index in parsed if str(index).isdigit()]
        selected = [
            candidate_paths[index]
            for index in indices[:wmax]
            if 0 <= index < len(candidate_paths)
        ]
        if selected:
            LOGGER.info("Precise selection kept %d paths", len(selected))
            return selected
    return candidate_paths[:wmax]


def branch_reduced_selection(
    candidate_paths: list[ReasoningPath],
    wmax: int,
    llm: BaseLLM,
) -> list[ReasoningPath]:
    """Reduce branch explosion by keeping one path per early prefix when possible."""
    if len(candidate_paths) <= wmax:
        return candidate_paths

    working = list(candidate_paths)
    for prefix_len in range(2, max(len(path.nodes) for path in working) + 1):
        grouped: dict[tuple[str, ...], ReasoningPath] = {}
        for path in working:
            key = tuple(path.nodes[: min(prefix_len, len(path.nodes))])
            grouped.setdefault(key, path)
        working = list(grouped.values())
        if len(working) <= wmax:
            return working[:wmax]
    return working[:wmax]


def _path_identity_signature(path: ReasoningPath) -> tuple[object, ...]:
    if path.edge_ids:
        return ("edge_ids", *path.edge_ids)
    triples = tuple((triple.head, triple.relation, triple.tail) for triple in path.triples)
    return ("triples", triples, path.terminal_node_id, path.terminal_node_kind)


def _relation_skeleton(path: ReasoningPath) -> tuple[str, ...]:
    return tuple(triple.relation for triple in path.triples)


def _repeated_relation_penalty(path: ReasoningPath) -> float:
    relations = [triple.relation for triple in path.triples]
    if not relations:
        return 0.0
    return max(0, len(relations) - len(set(relations))) * 0.2


def _schema_noise_penalty(path: ReasoningPath) -> float:
    penalty = 0.0
    for triple in path.triples:
        relation = triple.relation
        if relation.startswith(_SCHEMA_NOISE_PREFIXES):
            penalty += 0.3
    if path.terminal_node_kind == "literal" and not path.matched_answer_type_hints and path.search_score_breakdown.get("answer_score", 0.0) < 0.5:
        penalty += 0.25
    return penalty


def _relation_beam_rank_tuple(path: ReasoningPath) -> tuple[float, float, float, float, int]:
    return (
        float(path.path_score),
        float(path.search_score_breakdown.get("answer_score", 0.0)),
        float(len(path.matched_relations)),
        -(_repeated_relation_penalty(path) + _schema_noise_penalty(path)),
        -len(path.text),
    )


def prune_relation_beam_paths(
    candidate_paths: list[ReasoningPath],
    wmax: int,
) -> list[ReasoningPath]:
    """Consolidate relation-beam evidence without re-running heavy fuzzy or LLM pruning."""
    if not candidate_paths:
        return []
    for path in candidate_paths:
        path.pruning_status = "preserved"

    unique_paths: dict[tuple[object, ...], ReasoningPath] = {}
    duplicate_count = 0
    for path in candidate_paths:
        key = _path_identity_signature(path)
        existing = unique_paths.get(key)
        if existing is None:
            unique_paths[key] = path
            continue
        duplicate_count += 1
        if _relation_beam_rank_tuple(path) > _relation_beam_rank_tuple(existing):
            existing.pruning_status = "relation_beam_consolidated_pruned"
            unique_paths[key] = path
        else:
            path.pruning_status = "relation_beam_consolidated_pruned"
    deduped = list(unique_paths.values())

    by_terminal: dict[tuple[str, str], list[ReasoningPath]] = {}
    for path in deduped:
        by_terminal.setdefault((path.terminal_node_id, path.terminal_node_kind), []).append(path)

    terminal_best: list[ReasoningPath] = []
    collapsed_count = 0
    for group in by_terminal.values():
        ordered = sorted(
            group,
            key=lambda path: (
                len(path.triples),
                _repeated_relation_penalty(path),
                -len(path.matched_relations),
                len(path.text),
            ),
        )
        best = ordered[0]
        terminal_best.append(best)
        for dropped in ordered[1:]:
            dropped.pruning_status = "relation_beam_consolidated_pruned"
            collapsed_count += 1

    terminal_best.sort(key=_relation_beam_rank_tuple, reverse=True)
    selected: list[ReasoningPath] = []
    selected_terminals: set[tuple[str, str]] = set()
    selected_skeletons: set[tuple[str, ...]] = set()
    terminal_groups = by_terminal

    for path in terminal_best:
        terminal_key = (path.terminal_node_id, path.terminal_node_kind)
        if terminal_key in selected_terminals:
            continue
        selected.append(path)
        selected_terminals.add(terminal_key)
        selected_skeletons.add(_relation_skeleton(path))
        if len(selected) >= wmax:
            break

    if len(selected) < wmax:
        for path in terminal_best:
            skeleton = _relation_skeleton(path)
            if path in selected or skeleton in selected_skeletons:
                continue
            selected.append(path)
            selected_skeletons.add(skeleton)
            if len(selected) >= wmax:
                break

    if len(selected) < wmax:
        for path in terminal_best:
            if path in selected:
                continue
            selected.append(path)
            if len(selected) >= wmax:
                break

    if len(selected) < wmax:
        backup_paths: list[ReasoningPath] = []
        for terminal_key, group in terminal_groups.items():
            ordered = sorted(group, key=_relation_beam_rank_tuple, reverse=True)
            backups = [path for path in ordered[1:] if path not in selected]
            backup_paths.extend(backups)
        for path in sorted(backup_paths, key=_relation_beam_rank_tuple, reverse=True):
            if path in selected:
                continue
            selected.append(path)
            if len(selected) >= wmax:
                break

    selected_ids = {id(path) for path in selected}
    for path in candidate_paths:
        if id(path) not in selected_ids:
            path.pruning_status = "relation_beam_consolidated_pruned"
    for path in selected:
        path.pruning_status = "preserved"

    unique_terminal_count = len({(path.terminal_node_id, path.terminal_node_kind) for path in candidate_paths})
    unique_skeleton_count = len({_relation_skeleton(path) for path in candidate_paths})
    LOGGER.info(
        "Relation-beam pruning input path_count=%d unique_terminals=%d unique_skeletons=%d",
        len(candidate_paths),
        unique_terminal_count,
        unique_skeleton_count,
    )
    LOGGER.info("Relation-beam dedup removed=%d remaining=%d", duplicate_count, len(deduped))
    LOGGER.info("Relation-beam answer collapse removed=%d remaining=%d", collapsed_count, len(terminal_best))
    LOGGER.info(
        "Relation-beam final context kept=%d terminals=%s",
        len(selected),
        [path.terminal_node_id for path in selected[: min(len(selected), 10)]],
    )
    return selected


def prune_paths(
    question: str,
    question_analysis: QuestionAnalysisResult,
    candidate_paths: list[ReasoningPath],
    w1: int,
    wmax: int,
    relation_hints: list[str] | None,
    sample_id: str | None,
    compact_precise_path_prompt: bool,
    llm: BaseLLM,
) -> list[ReasoningPath]:
    """Run the full three-stage pruning pipeline."""
    relation_hints = relation_hints or []
    for path in candidate_paths:
        path.pruning_status = "preserved"
        path.matched_relations = []

    fuzzy = fuzzy_select_paths(
        question_analysis.reasoning_indicator,
        candidate_paths,
        w1,
        relation_hints=relation_hints,
    )
    fuzzy_ids = {id(path) for path in fuzzy}
    for path in candidate_paths:
        if id(path) not in fuzzy_ids:
            path.pruning_status = "fuzzy_pruned"

    precise = precise_select_paths_with_llm(
        question,
        question_analysis,
        fuzzy,
        wmax,
        supplement_relations=relation_hints,
        sample_id=sample_id,
        compact_prompt_paths=compact_precise_path_prompt,
        llm=llm,
    )
    precise_ids = {id(path) for path in precise}
    for path in fuzzy:
        if id(path) not in precise_ids:
            path.pruning_status = "precise_pruned"

    branch_reduced = branch_reduced_selection(precise, wmax, llm)
    branch_reduced_ids = {id(path) for path in branch_reduced}
    for path in precise:
        if id(path) not in branch_reduced_ids:
            path.pruning_status = "branch_reduced_pruned"

    for path in branch_reduced:
        path.pruning_status = "preserved"

    LOGGER.info("Pruning produced %d final paths", len(branch_reduced))
    return branch_reduced
