"""Non-LLM ranking and selection utilities for KGQA path retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from kgqa.kg.graph_api import BaseGraphAPI
from kgqa.reasoning.exploration import annotate_paths_with_relation_matches
from kgqa.utils.text import deduplicate_paths
from kgqa.utils.types import ReasoningPath


@dataclass
class SearchScoringContext:
    """Inputs used to score and rerank candidate reasoning paths."""

    relation_hints: list[str] = field(default_factory=list)
    answer_type_hints: list[str] = field(default_factory=list)
    high_degree_threshold: int = 500
    reward_literal_terminals: bool = False
    prefer_shorter_paths: bool = True


class PathCandidateScorer:
    """Heuristic path scorer for non-LLM retrieval and beam search."""

    def __init__(self, backend: BaseGraphAPI | None = None) -> None:
        self.backend = backend

    def score_path(self, path: ReasoningPath, context: SearchScoringContext) -> ReasoningPath:
        """Populate one path with a heuristic score and score breakdown."""
        annotate_paths_with_relation_matches([path], context.relation_hints)
        hops = len(path.triples)
        relation_hits = len(path.matched_relations)
        relation_score = float(relation_hits * 10.0)
        hop_penalty = float(hops) * (-1.5 if context.prefer_shorter_paths else 0.0)
        terminal_kind = path.terminal_node_kind or "id"
        literal_bonus = 0.0
        if terminal_kind == "literal":
            literal_bonus = 2.0 if context.reward_literal_terminals else -1.0
        high_degree_penalty = 0.0
        rarity_bonus = 0.0
        if self.backend is not None:
            for node_id in self._path_node_ids(path):
                degree = self.backend.get_node_degree(node_id)
                if degree >= context.high_degree_threshold:
                    high_degree_penalty -= 2.0
            for triple in path.triples:
                frequency = self.backend.get_relation_frequency(triple.relation)
                if frequency > 0:
                    rarity_bonus += min(3.0, 1.0 / math.log10(frequency + 10.0))
        answer_type_hits = self._match_answer_type_hints(path, context.answer_type_hints)
        path.matched_answer_type_hints = answer_type_hits
        answer_type_bonus = float(len(answer_type_hits) * 6.0)
        breakdown = {
            "relation_hits": relation_score,
            "answer_type_bonus": answer_type_bonus,
            "rarity_bonus": rarity_bonus,
            "literal_bonus": literal_bonus,
            "hop_penalty": hop_penalty,
            "high_degree_penalty": high_degree_penalty,
        }
        path.search_score_breakdown = breakdown
        path.path_score = sum(breakdown.values())
        return path

    def score_partial_path(self, path: ReasoningPath, context: SearchScoringContext) -> float:
        """Return one scalar score for partial beam-search expansion."""
        return self.score_path(path, context).path_score

    @staticmethod
    def _path_node_ids(path: ReasoningPath) -> list[str]:
        node_ids = [path.terminal_node_id] if path.terminal_node_id else []
        if not node_ids and path.nodes:
            return [node for node in path.nodes if node.startswith(("m.", "g."))]
        return [node_id for node_id in node_ids if node_id.startswith(("m.", "g."))]

    @staticmethod
    def _match_answer_type_hints(path: ReasoningPath, answer_type_hints: list[str]) -> list[str]:
        relation_tokens = path.triples[-1].relation.split(".") if path.triples else []
        terminal_text = " ".join(
            [
                path.nodes[-1] if path.nodes else "",
                path.terminal_node_kind,
                *relation_tokens,
            ]
        ).lower()
        matched: list[str] = []
        for hint in answer_type_hints:
            hint_norm = str(hint).strip().lower()
            if hint_norm and hint_norm in terminal_text and hint_norm not in matched:
                matched.append(hint_norm)
        return matched


class PathReranker:
    """Score and sort candidate paths with heuristic ranking."""

    def __init__(self, scorer: PathCandidateScorer) -> None:
        self.scorer = scorer

    def rerank(self, paths: list[ReasoningPath], context: SearchScoringContext) -> list[ReasoningPath]:
        """Return candidate paths ordered by descending heuristic quality."""
        scored = [self.scorer.score_path(path, context) for path in paths]
        scored.sort(
            key=lambda path: (
                -path.path_score,
                0 if path.matched_relations else 1,
                -len(path.matched_answer_type_hints),
                len(path.triples),
                path.text,
            )
        )
        return scored


class PathSelector:
    """Select a diverse, bounded subset of candidate paths."""

    def __init__(self, max_paths: int = 20, prefix_diversity: int = 2) -> None:
        self.max_paths = int(max_paths)
        self.prefix_diversity = max(1, int(prefix_diversity))

    def select(self, paths: list[ReasoningPath]) -> list[ReasoningPath]:
        """Keep top-ranked paths while reducing repeated early-prefix branches."""
        deduped = deduplicate_paths(paths)
        if self.max_paths <= 0:
            return deduped
        if len(deduped) <= self.max_paths:
            return deduped

        selected: list[ReasoningPath] = []
        prefix_counts: dict[tuple[str, ...], int] = {}
        fallback: list[ReasoningPath] = []
        for path in deduped:
            prefix = tuple(path.nodes[: min(self.prefix_diversity, len(path.nodes))])
            count = prefix_counts.get(prefix, 0)
            if count == 0:
                selected.append(path)
                prefix_counts[prefix] = 1
            else:
                fallback.append(path)
            if len(selected) >= self.max_paths:
                return selected[: self.max_paths]

        for path in fallback:
            selected.append(path)
            if len(selected) >= self.max_paths:
                break
        return selected[: self.max_paths]
