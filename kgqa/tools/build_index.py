"""CLI wrapper for building indexed SQLite runtime retrieval artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgqa.retrieval.backend import build_indexed_sqlite_runtime_index
from kgqa.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse build-index arguments."""
    parser = argparse.ArgumentParser(description="Build KGQA runtime adjacency indexes.")
    parser.add_argument("--db-path", required=True, help="Path to source SQLite truth database")
    parser.add_argument("--index-dir", required=True, help="Output directory for runtime index files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runtime index")
    parser.add_argument(
        "--high-degree-threshold",
        type=int,
        default=500,
        help="Degree threshold used to flag high-degree nodes",
    )
    parser.add_argument(
        "--staging-dir",
        default=".kgqa_build_tmp",
        help="Local staging directory used to build the runtime index before moving it to --index-dir.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000, help="Log progress every N rows.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--resume", action="store_true", help="Resume from a partial build when possible.")
    return parser.parse_args()


def main() -> None:
    """Build one indexed runtime artifact set."""
    args = parse_args()
    configure_logging(args.log_level)
    index_db = build_indexed_sqlite_runtime_index(
        db_path=Path(args.db_path),
        index_dir=Path(args.index_dir),
        overwrite=args.overwrite,
        high_degree_threshold=args.high_degree_threshold,
        staging_dir=Path(args.staging_dir),
        progress_every=args.progress_every,
        resume=args.resume,
    )
    print(index_db)


if __name__ == "__main__":
    main()
