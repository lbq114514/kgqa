"""Inspect entity and relation candidates against an indexed backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgqa.retrieval.backend import IndexedSQLiteGraphBackend


def parse_args() -> argparse.Namespace:
    """Parse inspection arguments."""
    parser = argparse.ArgumentParser(description="Inspect entity/relation candidates in KGQA retrieval backend.")
    parser.add_argument("--db-path", required=True, help="Path to SQLite truth database")
    parser.add_argument("--index-dir", help="Optional runtime index directory")
    parser.add_argument("--entity", help="Entity mention to resolve")
    parser.add_argument("--relation", help="Relation hint to resolve")
    parser.add_argument("--top-k", type=int, default=5, help="Candidate cutoff")
    return parser.parse_args()


def main() -> None:
    """Inspect backend resolution outputs for one entity or relation mention."""
    args = parse_args()
    backend = IndexedSQLiteGraphBackend(
        db_path=Path(args.db_path),
        index_dir=Path(args.index_dir) if args.index_dir else None,
    )
    try:
        payload: dict[str, object] = {}
        if args.entity:
            payload["entity_candidates"] = backend.resolve_entity_candidates([args.entity], top_k=args.top_k)
        if args.relation:
            payload["relation_candidates"] = backend.resolve_relation_candidates([args.relation], top_k=args.top_k)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
