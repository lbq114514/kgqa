"""Build a SQLite database for processed_nobase Freebase graph triples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.sqlite_graph_api import build_sqlite_graph_db_from_processed_freebase
from kgqa.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a SQLite processed Freebase graph database.")
    parser.add_argument("--entities-csv", required=True, help="Path to entities.csv")
    parser.add_argument("--triples-csv", required=True, help="Path to triplets.csv")
    parser.add_argument("--db-path", required=True, help="Output SQLite database path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing database file.")
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
    """Build the configured SQLite database."""
    args = parse_args()
    configure_logging(args.log_level)
    db_path = build_sqlite_graph_db_from_processed_freebase(
        entities_csv_path=Path(args.entities_csv),
        triples_csv_path=Path(args.triples_csv),
        db_path=Path(args.db_path),
        overwrite=args.overwrite,
        fast_mode=args.fast_mode,
        staging_dir=Path(args.staging_dir),
        progress_every=args.progress_every,
        resume=args.resume,
    )
    print(db_path)


if __name__ == "__main__":
    main()
