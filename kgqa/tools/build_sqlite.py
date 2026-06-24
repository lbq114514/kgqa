"""CLI wrapper for building SQLite truth databases."""

from __future__ import annotations

import argparse
from pathlib import Path

from kgqa.kg.sqlite_graph_api import (
    build_sqlite_graph_db,
    build_sqlite_graph_db_from_processed_freebase,
)
from kgqa.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse build-sqlite arguments."""
    parser = argparse.ArgumentParser(description="Build KGQA SQLite truth databases.")
    parser.add_argument("--entities-csv", required=True, help="Path to entities.csv")
    parser.add_argument("--triples-csv", required=True, help="Path to triples/triplets.csv")
    parser.add_argument("--db-path", required=True, help="Output SQLite file path")
    parser.add_argument(
        "--format",
        choices=["processed_freebase", "legacy_fb"],
        default="processed_freebase",
        help="Input CSV layout.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing SQLite file.")
    parser.add_argument("--fast-mode", action="store_true", help="Use faster, lower-validation bulk-load settings.")
    parser.add_argument(
        "--staging-dir",
        default=".kgqa_build_tmp",
        help="Local staging directory used to build the SQLite file before moving it to --db-path.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000, help="Log progress every N rows.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--resume", action="store_true", help="Resume from a partial build when possible.")
    return parser.parse_args()


def main() -> None:
    """Build one SQLite truth database from CSV inputs."""
    args = parse_args()
    configure_logging(args.log_level)
    builder = (
        build_sqlite_graph_db_from_processed_freebase
        if args.format == "processed_freebase"
        else build_sqlite_graph_db
    )
    builder_kwargs = {
        "entities_csv_path": Path(args.entities_csv),
        "triples_csv_path": Path(args.triples_csv),
        "db_path": Path(args.db_path),
        "overwrite": args.overwrite,
    }
    if args.format == "processed_freebase":
        builder_kwargs["fast_mode"] = args.fast_mode
        builder_kwargs["staging_dir"] = Path(args.staging_dir)
        builder_kwargs["progress_every"] = args.progress_every
        builder_kwargs["resume"] = args.resume
    db_path = builder(**builder_kwargs)
    print(db_path)


if __name__ == "__main__":
    main()
