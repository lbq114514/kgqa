"""Build a SQLite database for the optional FB2M graph API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.sqlite_graph_api import build_sqlite_graph_db


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a SQLite FB2M graph database.")
    parser.add_argument("--entities-csv", required=True, help="Path to entities.csv")
    parser.add_argument("--triples-csv", required=True, help="Path to triples.csv")
    parser.add_argument("--db-path", required=True, help="Output SQLite database path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing database file.")
    return parser.parse_args()


def main() -> None:
    """Build the configured SQLite database."""
    args = parse_args()
    db_path = build_sqlite_graph_db(
        entities_csv_path=Path(args.entities_csv),
        triples_csv_path=Path(args.triples_csv),
        db_path=Path(args.db_path),
        overwrite=args.overwrite,
    )
    print(db_path)


if __name__ == "__main__":
    main()
