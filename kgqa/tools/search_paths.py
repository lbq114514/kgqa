"""Search candidate paths directly through the retrieval toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgqa.retrieval import (
    BeamSearchSearcher,
    BidirectionalPathSearcher,
    ConstrainedBFSSearcher,
    HybridSearcher,
    IndexedSQLiteGraphBackend,
    PathCandidateScorer,
    PathReranker,
    PathSelector,
    SearchRequest,
)


def parse_args() -> argparse.Namespace:
    """Parse search-path CLI arguments."""
    parser = argparse.ArgumentParser(description="Search KGQA candidate paths from one or more seed entities.")
    parser.add_argument("--db-path", required=True, help="Path to SQLite truth database")
    parser.add_argument("--index-dir", help="Optional runtime index directory")
    parser.add_argument("--seed", action="append", required=True, help="Seed entity id or mention")
    parser.add_argument("--target", action="append", help="Optional target entity ids")
    parser.add_argument("--relation-hint", action="append", default=[], help="Optional relation hints")
    parser.add_argument("--answer-type-hint", action="append", default=[], help="Optional answer type hints")
    parser.add_argument(
        "--strategy",
        choices=["hybrid", "beam", "bfs", "two_hop", "bidirectional"],
        default="hybrid",
        help="Search strategy",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--max-expansions", type=int, default=500)
    parser.add_argument("--max-paths", type=int, default=20)
    parser.add_argument("--top-k-candidates", type=int, default=1)
    return parser.parse_args()


def _resolve_seed_ids(backend: IndexedSQLiteGraphBackend, seeds: list[str], top_k: int) -> list[str]:
    resolved: list[str] = []
    for seed in seeds:
        if seed.startswith(("m.", "g.")):
            resolved.append(seed)
            continue
        resolved.extend(backend.resolve_entity_mentions([seed], top_k=top_k))
    seen: set[str] = set()
    deduped: list[str] = []
    for item in resolved:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def main() -> None:
    """Run a standalone path search and print candidate paths as JSON."""
    args = parse_args()
    backend = IndexedSQLiteGraphBackend(
        db_path=Path(args.db_path),
        index_dir=Path(args.index_dir) if args.index_dir else None,
    )
    scorer = PathCandidateScorer(backend)
    reranker = PathReranker(scorer)
    selector = PathSelector(max_paths=args.max_paths)
    searchers = {
        "beam": BeamSearchSearcher(backend, scorer),
        "bfs": ConstrainedBFSSearcher(backend),
        "two_hop": None,
        "bidirectional": BidirectionalPathSearcher(backend),
        "hybrid": HybridSearcher(backend, scorer, reranker, selector),
    }
    try:
        seed_ids = _resolve_seed_ids(backend, args.seed, args.top_k_candidates)
        request = SearchRequest(
            seed_entity_ids=seed_ids,
            target_entity_ids=list(args.target or []),
            relation_hints=list(args.relation_hint or []),
            answer_type_hints=list(args.answer_type_hint or []),
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            max_expansions=args.max_expansions,
            max_paths=args.max_paths,
        )
        if args.strategy == "two_hop":
            from kgqa.retrieval import TwoHopExpansionSearcher

            paths = TwoHopExpansionSearcher(backend).search(request)
        else:
            paths = searchers[args.strategy].search(request)  # type: ignore[union-attr]
        print(
            json.dumps(
                {
                    "seed_ids": seed_ids,
                    "paths": [path.to_dict() for path in paths],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
