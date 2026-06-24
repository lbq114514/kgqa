"""SQLite-backed optional graph API for Freebase-style triples."""

from __future__ import annotations

import csv
from difflib import SequenceMatcher
import re
import sqlite3
from pathlib import Path
import shutil
from typing import Any

from kgqa.kg.graph_api import BaseGraphAPI, ExternalGraphNeighbor
from kgqa.utils.logging import get_logger
from kgqa.utils.text import deduplicate_paths
from kgqa.utils.types import ReasoningPath, Triple

LOGGER = get_logger(__name__)

_INSERT_BATCH_SIZE = 100_000

ENTITY_MATCH_SCORES = {
    "mid": 100.0,
    "name_exact": 95.0,
    "alias_like": 80.0,
}

RELATION_MATCH_SCORES = {
    "relation_exact": 100.0,
    "relation_norm_exact": 98.0,
    "relation_name_exact": 95.0,
    "relation_name_like": 90.0,
    "relation_id_like": 85.0,
}

_SEPARATOR_RE = re.compile(r"[\s._/\-()|]+")
_RELATION_GENERIC_SEGMENTS = {
    "measurement unit",
    "dated integer",
    "dated percentage",
    "dated money",
    "dated money value",
    "dated string",
    "dated enum",
    "dated date",
    "dated datetime",
}

_DEFAULT_PROGRESS_EVERY = 1_000_000


def normalize_text(text: str | None) -> str:
    """Normalize entity and relation text for robust matching."""
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    if not normalized:
        return ""
    normalized = _SEPARATOR_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _derive_relation_display_name(relation: str | None) -> str:
    """Derive a readable relation label from one canonical relation id."""
    normalized = normalize_text(relation)
    if normalized:
        return normalized
    return str(relation or "").strip()


def verbalize_relation(relation: str | None) -> str:
    """Generate a shorter relation label for path presentation."""
    relation_id = str(relation or "").strip()
    if not relation_id:
        return ""
    segments = [normalize_text(segment) for segment in relation_id.split(".") if normalize_text(segment)]
    if not segments:
        return relation_id
    meaningful = [segment for segment in segments if segment not in _RELATION_GENERIC_SEGMENTS]
    if not meaningful:
        meaningful = segments
    core = meaningful[2:] if len(meaningful) > 2 else meaningful
    if len(meaningful) > 2:
        domain_segments = set(meaningful[:2])
        filtered_core = [segment for segment in core if segment not in domain_segments]
        if filtered_core:
            core = filtered_core
    compact: list[str] = []
    for segment in core:
        if compact and compact[-1] == segment:
            continue
        compact.append(segment)
    if not compact:
        compact = meaningful[-1:]
    if len(compact) == 1:
        return compact[0]
    if len(compact) == 2:
        return " -> ".join(compact)
    return " -> ".join(compact[-2:])


def _is_entity_id(value: str | None) -> bool:
    """Return whether one string looks like a Freebase entity id."""
    stripped = str(value or "").strip()
    return stripped.startswith(("m.", "g."))


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not _table_exists(connection, table_name):
        return False
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def _format_progress_message(
    stage: str,
    processed: int,
    total: int | None = None,
    extra: str = "",
) -> str:
    if total and total > 0:
        percent = min(100.0, (processed / total) * 100.0)
        message = f"[{stage}] {processed:,}/{total:,} ({percent:.1f}%)"
    else:
        message = f"[{stage}] {processed:,}"
    if extra:
        message = f"{message} {extra}"
    return message


def _safe_stream_position(handle: Any) -> int | None:
    buffer = getattr(handle, "buffer", None)
    if buffer is not None:
        try:
            return int(buffer.tell())
        except (OSError, ValueError):
            return None
    try:
        return int(handle.tell())
    except (OSError, ValueError):
        return None


def _maybe_log_stream_progress(
    stage: str,
    processed: int,
    progress_every: int,
    total_bytes: int | None,
    handle: Any,
    extra: str = "",
) -> None:
    if progress_every <= 0 or processed <= 0 or processed % progress_every != 0:
        return
    position = _safe_stream_position(handle)
    if total_bytes and position is not None and total_bytes > 0:
        LOGGER.info(_format_progress_message(stage, position, total_bytes, f"rows={processed:,} {extra}".strip()))
        return
    LOGGER.info(_format_progress_message(stage, processed, None, extra))


def _run_index_statements(
    connection: sqlite3.Connection,
    stage_prefix: str,
    statements: list[tuple[str, str]],
) -> None:
    total = len(statements)
    for index, (label, sql) in enumerate(statements, start=1):
        LOGGER.info(_format_progress_message(f"{stage_prefix}.start", index - 1, total, f"next={label}"))
        connection.execute(sql)
        LOGGER.info(_format_progress_message(f"{stage_prefix}.done", index, total, label))


def _progress_value(connection: sqlite3.Connection, key: str, default: str = "") -> str:
    row = connection.execute("SELECT value FROM build_progress WHERE key = ? LIMIT 1", (key,)).fetchone()
    if row is None:
        return default
    return str(row[0] or default)


def _set_progress_value(connection: sqlite3.Connection, key: str, value: str | int) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO build_progress(key, value) VALUES (?, ?)",
        (key, str(value)),
    )


def _normalize_mid_candidate(mention: str) -> str:
    stripped = mention.strip()
    if stripped.startswith("/m/"):
        return f"m.{stripped[3:]}"
    if stripped.startswith("/g/"):
        return f"g.{stripped[3:]}"
    return stripped


def _row_value(row: sqlite3.Row | tuple[Any, ...], key: str, index: int) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[key]
    return row[index]


def _upsert_best_match(
    existing: dict[str, dict[str, Any]],
    primary_key: str,
    key_value: str,
    payload: dict[str, Any],
) -> None:
    current = existing.get(key_value)
    if current is None or float(payload.get("score", 0.0)) > float(current.get("score", 0.0)):
        existing[key_value] = payload


def resolve_entity_candidates(
    connection: sqlite3.Connection,
    mention: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Resolve one entity mention against the SQLite entity catalog."""
    effective_limit = max(1, int(limit))
    results: dict[str, dict[str, Any]] = {}
    normalized_mention = normalize_text(mention)
    normalized_mid = _normalize_mid_candidate(str(mention))

    if normalized_mid.startswith(("m.", "g.")):
        row = connection.execute(
            "SELECT mid, name, aliases FROM entities WHERE mid = ? LIMIT 1",
            (normalized_mid,),
        ).fetchone()
        if row is not None:
            _upsert_best_match(
                results,
                "mid",
                str(_row_value(row, "mid", 0)),
                {
                    "mid": str(_row_value(row, "mid", 0)),
                    "name": str(_row_value(row, "name", 1) or ""),
                    "aliases": str(_row_value(row, "aliases", 2) or ""),
                    "match_type": "mid",
                    "score": ENTITY_MATCH_SCORES["mid"],
                },
            )

    if normalized_mention:
        if _column_exists(connection, "entities", "name_norm"):
            rows = connection.execute(
                "SELECT mid, name, aliases FROM entities WHERE name_norm = ? LIMIT ?",
                (normalized_mention, effective_limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT mid, name, aliases FROM entities WHERE lower(name) = ? LIMIT ?",
                (mention.strip().lower(), effective_limit),
            ).fetchall()
        for row in rows:
            _upsert_best_match(
                results,
                "mid",
                str(_row_value(row, "mid", 0)),
                {
                    "mid": str(_row_value(row, "mid", 0)),
                    "name": str(_row_value(row, "name", 1) or ""),
                    "aliases": str(_row_value(row, "aliases", 2) or ""),
                    "match_type": "name_exact",
                    "score": ENTITY_MATCH_SCORES["name_exact"],
                },
            )

        if _column_exists(connection, "entities", "aliases_norm"):
            alias_rows = connection.execute(
                "SELECT mid, name, aliases FROM entities WHERE aliases_norm LIKE ? LIMIT ?",
                (f"%{normalized_mention}%", max(effective_limit * 5, 25)),
            ).fetchall()
        else:
            alias_rows = connection.execute(
                "SELECT mid, name, aliases FROM entities WHERE lower(aliases) LIKE ? LIMIT ?",
                (f"%{mention.strip().lower()}%", max(effective_limit * 5, 25)),
            ).fetchall()
        for row in alias_rows:
            _upsert_best_match(
                results,
                "mid",
                str(_row_value(row, "mid", 0)),
                {
                    "mid": str(_row_value(row, "mid", 0)),
                    "name": str(_row_value(row, "name", 1) or ""),
                    "aliases": str(_row_value(row, "aliases", 2) or ""),
                    "match_type": "alias_like",
                    "score": ENTITY_MATCH_SCORES["alias_like"],
                },
            )

    sorted_rows = sorted(
        results.values(),
        key=lambda item: (-float(item["score"]), str(item["mid"])),
    )
    return sorted_rows[:effective_limit]


def _relation_token_overlap_score(normalized_hint: str, normalized_target: str) -> float:
    hint_tokens = set(normalized_hint.split())
    target_tokens = set(normalized_target.split())
    if not hint_tokens or not target_tokens:
        return 0.0
    overlap = len(hint_tokens & target_tokens)
    union = len(hint_tokens | target_tokens)
    return 80.0 * (overlap / union)


def resolve_relation_candidates(
    connection: sqlite3.Connection,
    hint: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Resolve one relation hint against the SQLite relation catalog."""
    effective_limit = max(1, int(limit))
    normalized_hint = normalize_text(hint)
    if not normalized_hint:
        return []

    results: dict[str, dict[str, Any]] = {}
    has_relations_table = _table_exists(connection, "relations")
    relation_field = "relation"
    relation_name_field = "relation_name"
    relation_norm_field = "relation_norm" if _column_exists(connection, "relations", "relation_norm") else ""
    relation_name_norm_field = (
        "relation_name_norm" if _column_exists(connection, "relations", "relation_name_norm") else ""
    )

    if has_relations_table:
        row = connection.execute(
            "SELECT relation, relation_name FROM relations WHERE relation = ? LIMIT 1",
            (hint.strip(),),
        ).fetchone()
        if row is not None:
            _upsert_best_match(
                results,
                "relation",
                str(_row_value(row, "relation", 0)),
                {
                    "relation": str(_row_value(row, "relation", 0)),
                    "relation_name": str(_row_value(row, "relation_name", 1) or ""),
                    "match_type": "relation_exact",
                    "score": RELATION_MATCH_SCORES["relation_exact"],
                },
            )

        if relation_norm_field:
            rows = connection.execute(
                "SELECT relation, relation_name FROM relations WHERE relation_norm = ? LIMIT ?",
                (normalized_hint, effective_limit),
            ).fetchall()
            for row in rows:
                _upsert_best_match(
                    results,
                    "relation",
                    str(_row_value(row, "relation", 0)),
                    {
                        "relation": str(_row_value(row, "relation", 0)),
                        "relation_name": str(_row_value(row, "relation_name", 1) or ""),
                        "match_type": "relation_norm_exact",
                        "score": RELATION_MATCH_SCORES["relation_norm_exact"],
                    },
                )

        if relation_name_norm_field:
            rows = connection.execute(
                "SELECT relation, relation_name FROM relations WHERE relation_name_norm = ? LIMIT ?",
                (normalized_hint, effective_limit),
            ).fetchall()
            for row in rows:
                _upsert_best_match(
                    results,
                    "relation",
                    str(_row_value(row, "relation", 0)),
                    {
                        "relation": str(_row_value(row, "relation", 0)),
                        "relation_name": str(_row_value(row, "relation_name", 1) or ""),
                        "match_type": "relation_name_exact",
                        "score": RELATION_MATCH_SCORES["relation_name_exact"],
                    },
                )

            rows = connection.execute(
                "SELECT relation, relation_name FROM relations WHERE relation_name_norm LIKE ? LIMIT ?",
                (f"%{normalized_hint}%", max(effective_limit * 5, 25)),
            ).fetchall()
            for row in rows:
                _upsert_best_match(
                    results,
                    "relation",
                    str(_row_value(row, "relation", 0)),
                    {
                        "relation": str(_row_value(row, "relation", 0)),
                        "relation_name": str(_row_value(row, "relation_name", 1) or ""),
                        "match_type": "relation_name_like",
                        "score": RELATION_MATCH_SCORES["relation_name_like"],
                    },
                )

        relation_like_rows = connection.execute(
            "SELECT relation, relation_name FROM relations WHERE relation LIKE ? LIMIT ?",
            (f"%{hint.strip()}%", max(effective_limit * 5, 25)),
        ).fetchall()
        for row in relation_like_rows:
            _upsert_best_match(
                results,
                "relation",
                str(_row_value(row, "relation", 0)),
                {
                    "relation": str(_row_value(row, "relation", 0)),
                    "relation_name": str(_row_value(row, "relation_name", 1) or ""),
                    "match_type": "relation_id_like",
                    "score": RELATION_MATCH_SCORES["relation_id_like"],
                },
            )

        if len(results) < effective_limit:
            rows = connection.execute(
                "SELECT relation, relation_name, relation_norm, relation_name_norm FROM relations"
            ).fetchall()
            for row in rows:
                relation = str(_row_value(row, "relation", 0))
                relation_name = str(_row_value(row, "relation_name", 1) or "")
                relation_norm = str(_row_value(row, "relation_norm", 2) or "")
                relation_name_norm = str(_row_value(row, "relation_name_norm", 3) or "")
                token_score = max(
                    _relation_token_overlap_score(normalized_hint, relation_name_norm),
                    _relation_token_overlap_score(normalized_hint, relation_norm),
                )
                sequence_score = max(
                    70.0 * SequenceMatcher(None, normalized_hint, relation_name_norm).ratio(),
                    70.0 * SequenceMatcher(None, normalized_hint, relation_norm).ratio(),
                )
                score = max(token_score, sequence_score)
                if score <= 0.0:
                    continue
                _upsert_best_match(
                    results,
                    "relation",
                    relation,
                    {
                        "relation": relation,
                        "relation_name": relation_name,
                        "match_type": "relation_name_like" if token_score >= sequence_score else "relation_id_like",
                        "score": score,
                    },
                )
    else:
        rows = connection.execute(
            """
            SELECT relation, MAX(COALESCE(NULLIF(relation_name, ''), relation)) AS relation_name
            FROM triples
            GROUP BY relation
            """
        ).fetchall()
        for row in rows:
            relation = str(_row_value(row, "relation", 0))
            relation_name = str(_row_value(row, "relation_name", 1) or relation)
            relation_norm = normalize_text(relation)
            relation_name_norm = normalize_text(relation_name)
            if hint.strip() == relation:
                score = RELATION_MATCH_SCORES["relation_exact"]
                match_type = "relation_exact"
            elif normalized_hint == relation_norm:
                score = RELATION_MATCH_SCORES["relation_norm_exact"]
                match_type = "relation_norm_exact"
            elif normalized_hint == relation_name_norm:
                score = RELATION_MATCH_SCORES["relation_name_exact"]
                match_type = "relation_name_exact"
            elif normalized_hint in relation_name_norm:
                score = RELATION_MATCH_SCORES["relation_name_like"]
                match_type = "relation_name_like"
            elif hint.strip() and hint.strip() in relation:
                score = RELATION_MATCH_SCORES["relation_id_like"]
                match_type = "relation_id_like"
            else:
                token_score = max(
                    _relation_token_overlap_score(normalized_hint, relation_name_norm),
                    _relation_token_overlap_score(normalized_hint, relation_norm),
                )
                sequence_score = max(
                    70.0 * SequenceMatcher(None, normalized_hint, relation_name_norm).ratio(),
                    70.0 * SequenceMatcher(None, normalized_hint, relation_norm).ratio(),
                )
                score = max(token_score, sequence_score)
                if score <= 0.0:
                    continue
                match_type = "relation_name_like" if token_score >= sequence_score else "relation_id_like"
            _upsert_best_match(
                results,
                "relation",
                relation,
                {
                    "relation": relation,
                    "relation_name": relation_name,
                    "match_type": match_type,
                    "score": score,
                },
            )

    sorted_rows = sorted(
        results.values(),
        key=lambda item: (-float(item["score"]), str(item["relation"])),
    )
    return sorted_rows[:effective_limit]


def build_sqlite_graph_db(
    entities_csv_path: str | Path,
    triples_csv_path: str | Path,
    db_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Build a SQLite graph database from CSV files."""
    entities_csv_path = Path(entities_csv_path)
    triples_csv_path = Path(triples_csv_path)
    db_path = Path(db_path)

    if db_path.exists() and not overwrite:
        raise FileExistsError(f"Database already exists at {db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;

            CREATE TABLE entities (
                mid TEXT PRIMARY KEY,
                name TEXT,
                name_norm TEXT,
                aliases TEXT,
                aliases_norm TEXT
            );

            CREATE TABLE relations (
                relation TEXT PRIMARY KEY,
                relation_name TEXT,
                relation_norm TEXT,
                relation_name_norm TEXT
            );

            CREATE TABLE triples (
                head TEXT,
                relation TEXT,
                relation_name TEXT,
                tail TEXT
            );
            """
        )

        with entities_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            batch: list[tuple[str, str, str, str, str]] = []
            for row in reader:
                mid = str(row.get("mid", "")).strip()
                name = str(row.get("name", "")).strip()
                aliases = str(row.get("aliases", "")).strip()
                batch.append(
                    (
                        mid,
                        name,
                        normalize_text(name),
                        aliases,
                        normalize_text(aliases),
                    )
                )
                if len(batch) >= _INSERT_BATCH_SIZE:
                    connection.executemany(
                        "INSERT INTO entities(mid, name, name_norm, aliases, aliases_norm) VALUES (?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO entities(mid, name, name_norm, aliases, aliases_norm) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )

        relation_rows: dict[str, str] = {}
        with triples_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            batch: list[tuple[str, str, str, str]] = []
            for row in reader:
                head = str(row.get("head", "")).strip()
                relation = str(row.get("relation", "")).strip()
                relation_name = str(row.get("relation_name", "")).strip()
                tail = str(row.get("tail", "")).strip()
                batch.append((head, relation, relation_name, tail))
                if relation and relation not in relation_rows:
                    relation_rows[relation] = relation_name
                elif relation and not relation_rows[relation] and relation_name:
                    relation_rows[relation] = relation_name
                if len(batch) >= _INSERT_BATCH_SIZE:
                    connection.executemany(
                        "INSERT INTO triples(head, relation, relation_name, tail) VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO triples(head, relation, relation_name, tail) VALUES (?, ?, ?, ?)",
                    batch,
                )

        connection.executemany(
            "INSERT INTO relations(relation, relation_name, relation_norm, relation_name_norm) VALUES (?, ?, ?, ?)",
            [
                (
                    relation,
                    relation_name,
                    normalize_text(relation),
                    normalize_text(relation_name),
                )
                for relation, relation_name in relation_rows.items()
            ],
        )

        connection.executescript(
            """
            CREATE INDEX idx_triples_head ON triples(head);
            CREATE INDEX idx_triples_tail ON triples(tail);
            CREATE INDEX idx_triples_relation ON triples(relation);
            CREATE INDEX idx_entities_name ON entities(name);
            CREATE INDEX idx_entities_name_norm ON entities(name_norm);
            CREATE INDEX idx_entities_aliases_norm ON entities(aliases_norm);
            CREATE INDEX idx_relations_relation_norm ON relations(relation_norm);
            CREATE INDEX idx_relations_relation_name_norm ON relations(relation_name_norm);
            """
        )
        connection.commit()
    finally:
        connection.close()

    return db_path


def build_sqlite_graph_db_from_processed_freebase(
    entities_csv_path: str | Path,
    triples_csv_path: str | Path,
    db_path: str | Path,
    overwrite: bool = False,
    fast_mode: bool = False,
    staging_dir: str | Path | None = None,
    progress_every: int = _DEFAULT_PROGRESS_EVERY,
    resume: bool = False,
) -> Path:
    """Build a SQLite graph database from processed_nobase Freebase CSV files."""
    entities_csv_path = Path(entities_csv_path)
    triples_csv_path = Path(triples_csv_path)
    db_path = Path(db_path)
    staging_root = Path(staging_dir) if staging_dir is not None else None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_db_path = db_path
    if staging_root is not None:
        LOGGER.info("preparing truth sqlite staging dir root=%s", staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        build_db_path = staging_root / f"{db_path.name}.partial"
        LOGGER.info("truth sqlite staging ready build_db=%s", build_db_path)

    if resume:
        if db_path.exists():
            LOGGER.info("truth sqlite resume found completed target=%s", db_path)
            return db_path
    else:
        if db_path.exists() and not overwrite:
            raise FileExistsError(f"Database already exists at {db_path}")
        if db_path.exists():
            db_path.unlink()
        if build_db_path.exists():
            build_db_path.unlink()

    build_succeeded = False
    connection = sqlite3.connect(str(build_db_path))
    try:
        LOGGER.info(
            "starting truth sqlite build entities=%s triples=%s output=%s fast_mode=%s staging_dir=%s",
            entities_csv_path,
            triples_csv_path,
            db_path,
            fast_mode,
            staging_root,
        )
        insert_batch_size = _INSERT_BATCH_SIZE * (5 if fast_mode else 1)
        pragma_script = """
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
        """
        if fast_mode:
            pragma_script += """
            PRAGMA journal_mode = OFF;
            PRAGMA locking_mode = EXCLUSIVE;
            PRAGMA cache_size = -800000;
            """
        else:
            pragma_script += "PRAGMA journal_mode = WAL;"

        connection.executescript(
            pragma_script
            + """
            CREATE TABLE IF NOT EXISTS build_progress (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS entities (
                mid TEXT PRIMARY KEY,
                name TEXT,
                name_norm TEXT,
                aliases TEXT,
                aliases_norm TEXT,
                types TEXT,
                types_norm TEXT,
                is_cvt INTEGER
            );

            CREATE TABLE IF NOT EXISTS relations (
                relation TEXT PRIMARY KEY,
                relation_name TEXT,
                relation_norm TEXT,
                relation_name_norm TEXT
            );

            CREATE TABLE IF NOT EXISTS triples (
                head TEXT,
                relation TEXT,
                tail TEXT,
                tail_kind TEXT
            );
            """
        )
        LOGGER.info("truth sqlite schema created db=%s", build_db_path)

        entities_total_bytes = entities_csv_path.stat().st_size if entities_csv_path.exists() else None
        if _progress_value(connection, "entities_complete") == "1":
            entity_rows_processed = int(connection.execute("SELECT COUNT(1) FROM entities").fetchone()[0])
            LOGGER.info(_format_progress_message("truth.entities", entity_rows_processed, entity_rows_processed, "resume-skip done"))
        else:
            LOGGER.info(
                "truth entities phase started file=%s size_bytes=%s batch_size=%d",
                entities_csv_path,
                entities_total_bytes,
                insert_batch_size,
            )
            existing_entities = int(connection.execute("SELECT COUNT(1) FROM entities").fetchone()[0])
            if existing_entities:
                LOGGER.info("truth entities resume existing_rows=%d", existing_entities)
            with entities_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                expected_fields = {"id", "name", "aliases", "types", "is_cvt"}
                fieldnames = next(reader, None)
                if set(fieldnames or []) != expected_fields:
                    raise ValueError(
                        "processed Freebase entities.csv must contain exactly: "
                        "id,name,aliases,types,is_cvt"
                    )
                for _ in range(existing_entities):
                    if next(reader, None) is None:
                        break

                batch: list[tuple[str, str, str, str, str, str, str, int]] = []
                entity_rows_processed = existing_entities
                for row_index, row in enumerate(reader, start=existing_entities + 2):
                    entity_rows_processed = row_index - 1
                    mid = str(row[0] if len(row) > 0 else "").strip()
                    if not fast_mode and not _is_entity_id(mid):
                        raise ValueError(
                            f"Invalid processed Freebase entity id at row {row_index}: {mid!r}"
                        )
                    name = str(row[1] if len(row) > 1 else "").strip()
                    aliases = str(row[2] if len(row) > 2 else "").strip()
                    entity_types = str(row[3] if len(row) > 3 else "").strip()
                    is_cvt_raw = str(row[4] if len(row) > 4 else "").strip()
                    if not fast_mode and is_cvt_raw not in {"0", "1"}:
                        raise ValueError(
                            f"Invalid processed Freebase is_cvt value at row {row_index}: {is_cvt_raw!r}"
                        )
                    batch.append(
                        (
                            mid,
                            name,
                            normalize_text(name),
                            aliases,
                            normalize_text(aliases),
                            entity_types,
                            normalize_text(entity_types),
                            1 if is_cvt_raw == "1" else 0,
                        )
                    )
                    if len(batch) >= insert_batch_size:
                        connection.executemany(
                            """
                            INSERT INTO entities(
                                mid, name, name_norm, aliases, aliases_norm, types, types_norm, is_cvt
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            batch,
                        )
                        batch.clear()
                    _maybe_log_stream_progress(
                        stage="truth.entities",
                        processed=row_index - 1,
                        progress_every=progress_every,
                        total_bytes=entities_total_bytes,
                        handle=handle,
                    )
                    if progress_every > 0 and (row_index - 1) % progress_every == 0:
                        _set_progress_value(connection, "entities_rows", row_index - 1)
                        connection.commit()
                if batch:
                    connection.executemany(
                        """
                        INSERT INTO entities(
                            mid, name, name_norm, aliases, aliases_norm, types, types_norm, is_cvt
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
            _set_progress_value(connection, "entities_rows", entity_rows_processed)
            _set_progress_value(connection, "entities_complete", 1)
            connection.commit()
            LOGGER.info(
                _format_progress_message(
                    "truth.entities",
                    entities_total_bytes if entities_total_bytes is not None else entity_rows_processed,
                    entities_total_bytes,
                    f"rows={entity_rows_processed:,} done",
                )
            )

        triples_total_bytes = triples_csv_path.stat().st_size if triples_csv_path.exists() else None
        if _progress_value(connection, "triples_complete") == "1":
            triple_rows_processed = int(connection.execute("SELECT COUNT(1) FROM triples").fetchone()[0])
            LOGGER.info(_format_progress_message("truth.triples", triple_rows_processed, triple_rows_processed, "resume-skip done"))
        else:
            LOGGER.info(
                "truth triples phase started file=%s size_bytes=%s batch_size=%d",
                triples_csv_path,
                triples_total_bytes,
                insert_batch_size,
            )
            existing_triples = int(connection.execute("SELECT COUNT(1) FROM triples").fetchone()[0])
            if existing_triples:
                LOGGER.info("truth triples resume existing_rows=%d", existing_triples)
            with triples_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                expected_fields = {"head", "head_name", "relation", "tail", "tail_name", "tail_kind"}
                fieldnames = next(reader, None)
                if set(fieldnames or []) != expected_fields:
                    raise ValueError(
                        "processed Freebase triplets.csv must contain exactly: "
                        "head,head_name,relation,tail,tail_name,tail_kind"
                    )
                for _ in range(existing_triples):
                    if next(reader, None) is None:
                        break

                batch: list[tuple[str, str, str, str]] = []
                triple_rows_processed = existing_triples
                for row_index, row in enumerate(reader, start=existing_triples + 2):
                    triple_rows_processed = row_index - 1
                    head = str(row[0] if len(row) > 0 else "").strip()
                    if not fast_mode and not _is_entity_id(head):
                        raise ValueError(
                            f"Invalid processed Freebase triple head at row {row_index}: {head!r}"
                        )
                    relation = str(row[2] if len(row) > 2 else "").strip()
                    tail = str(row[3] if len(row) > 3 else "").strip()
                    tail_name = str(row[4] if len(row) > 4 else "").strip()
                    tail_kind = str(row[5] if len(row) > 5 else "").strip().lower()
                    if not fast_mode and tail_kind not in {"id", "literal"}:
                        raise ValueError(
                            f"Invalid processed Freebase tail_kind at row {row_index}: {tail_kind!r}"
                        )
                    if not fast_mode and tail_kind == "id" and not _is_entity_id(tail):
                        raise ValueError(
                            f"Invalid processed Freebase entity tail at row {row_index}: {tail!r}"
                        )
                    if not fast_mode and tail_kind == "literal" and tail_name:
                        LOGGER.warning(
                            "Processed Freebase literal tail has non-empty tail_name row=%d relation=%s tail=%s",
                            row_index,
                            relation,
                            tail,
                        )
                    batch.append((head, relation, tail, tail_kind))
                    if len(batch) >= insert_batch_size:
                        connection.executemany(
                            """
                            INSERT INTO triples(head, relation, tail, tail_kind)
                            VALUES (?, ?, ?, ?)
                            """,
                            batch,
                        )
                        batch.clear()
                    _maybe_log_stream_progress(
                        stage="truth.triples",
                        processed=row_index - 1,
                        progress_every=progress_every,
                        total_bytes=triples_total_bytes,
                        handle=handle,
                    )
                    if progress_every > 0 and (row_index - 1) % progress_every == 0:
                        _set_progress_value(connection, "triples_rows", row_index - 1)
                        connection.commit()
                if batch:
                    connection.executemany(
                        """
                        INSERT INTO triples(head, relation, tail, tail_kind)
                        VALUES (?, ?, ?, ?)
                        """,
                        batch,
                    )
            _set_progress_value(connection, "triples_rows", triple_rows_processed)
            _set_progress_value(connection, "triples_complete", 1)
            connection.commit()
            LOGGER.info(
                _format_progress_message(
                    "truth.triples",
                    triples_total_bytes if triples_total_bytes is not None else triple_rows_processed,
                    triples_total_bytes,
                    f"rows={triple_rows_processed:,} done",
                )
            )

        if _progress_value(connection, "relations_complete") != "1":
            LOGGER.info("truth relations rebuild started")
            connection.execute("DELETE FROM relations")
            relation_rows = [
                (
                    relation_id,
                    _derive_relation_display_name(relation_id),
                    normalize_text(relation_id),
                    normalize_text(_derive_relation_display_name(relation_id)),
                )
                for relation_id, in connection.execute("SELECT DISTINCT relation FROM triples")
            ]
            connection.executemany(
                "INSERT INTO relations(relation, relation_name, relation_norm, relation_name_norm) VALUES (?, ?, ?, ?)",
                relation_rows,
            )
            _set_progress_value(connection, "relations_complete", 1)
            connection.commit()
            LOGGER.info(_format_progress_message("truth.relations", len(relation_rows), len(relation_rows), "done"))
        else:
            relation_count = int(connection.execute("SELECT COUNT(1) FROM relations").fetchone()[0])
            LOGGER.info(_format_progress_message("truth.relations", relation_count, relation_count, "resume-skip done"))

        if _progress_value(connection, "indexes_complete") != "1":
            for index_name in (
                "idx_triples_head",
                "idx_triples_tail",
                "idx_triples_relation",
                "idx_triples_tail_kind",
                "idx_entities_name",
                "idx_entities_name_norm",
                "idx_entities_aliases_norm",
                "idx_entities_types_norm",
                "idx_relations_relation_norm",
                "idx_relations_relation_name_norm",
            ):
                connection.execute(f"DROP INDEX IF EXISTS {index_name}")
            _run_index_statements(
                connection,
                "truth.indexes",
                [
                    ("idx_triples_head", "CREATE INDEX idx_triples_head ON triples(head)"),
                    ("idx_triples_tail", "CREATE INDEX idx_triples_tail ON triples(tail)"),
                    ("idx_triples_relation", "CREATE INDEX idx_triples_relation ON triples(relation)"),
                    ("idx_triples_tail_kind", "CREATE INDEX idx_triples_tail_kind ON triples(tail_kind)"),
                    ("idx_entities_name", "CREATE INDEX idx_entities_name ON entities(name)"),
                    ("idx_entities_name_norm", "CREATE INDEX idx_entities_name_norm ON entities(name_norm)"),
                    ("idx_entities_aliases_norm", "CREATE INDEX idx_entities_aliases_norm ON entities(aliases_norm)"),
                    ("idx_entities_types_norm", "CREATE INDEX idx_entities_types_norm ON entities(types_norm)"),
                    ("idx_relations_relation_norm", "CREATE INDEX idx_relations_relation_norm ON relations(relation_norm)"),
                    (
                        "idx_relations_relation_name_norm",
                        "CREATE INDEX idx_relations_relation_name_norm ON relations(relation_name_norm)",
                    ),
                ],
            )
            _set_progress_value(connection, "indexes_complete", 1)
            connection.commit()
        else:
            LOGGER.info(_format_progress_message("truth.indexes", 10, 10, "resume-skip done"))
        LOGGER.info("truth sqlite commit started db=%s", build_db_path)
        _set_progress_value(connection, "build_complete", 1)
        connection.commit()
        LOGGER.info("truth sqlite build committed db=%s", build_db_path)
        build_succeeded = True
    finally:
        connection.close()
        LOGGER.info("truth sqlite connection closed db=%s", build_db_path)

    if build_db_path != db_path and build_succeeded:
        LOGGER.info("truth sqlite move started src=%s dst=%s", build_db_path, db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(build_db_path), str(db_path))
        LOGGER.info("truth sqlite move finished dst=%s", db_path)
    elif build_db_path != db_path:
        LOGGER.info("truth sqlite partial build preserved path=%s", build_db_path)

    return db_path


class SQLiteGraphAPI(BaseGraphAPI):
    """SQLite-backed graph API for optional external augmentation."""

    def __init__(self, db_path: str | Path, neighbor_limit: int = 25) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Configured graphapi SQLite database does not exist: {self.db_path}"
            )
        self.neighbor_limit = int(neighbor_limit)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self._name_cache: dict[str, str] = {}
        self._relation_display_cache: dict[str, str] = {}
        self._has_relations_table = self._table_exists("relations")
        self._has_entity_name_norm = self._column_exists("entities", "name_norm")
        self._has_entity_aliases_norm = self._column_exists("entities", "aliases_norm")
        self._has_entity_types = self._column_exists("entities", "types")
        self._has_entity_is_cvt = self._column_exists("entities", "is_cvt")
        self._has_triple_head_name = self._column_exists("triples", "head_name")
        self._has_triple_relation_name = self._column_exists("triples", "relation_name")
        self._has_triple_tail_name = self._column_exists("triples", "tail_name")
        self._has_triple_tail_kind = self._column_exists("triples", "tail_kind")
        self._relation_catalog = self._load_relation_catalog()
        LOGGER.info(
            "Initialized SQLiteGraphAPI db_path=%s neighbor_limit=%d relation_catalog=%d relations_table=%s processed_triples=%s",
            self.db_path,
            self.neighbor_limit,
            len(self._relation_catalog),
            self._has_relations_table,
            self._has_triple_head_name and self._has_triple_tail_name and self._has_triple_tail_kind,
        )

    def close(self) -> None:
        """Close the backing SQLite connection."""
        self.connection.close()

    def _table_exists(self, table_name: str) -> bool:
        """Return whether one SQLite table exists."""
        return _table_exists(self.connection, table_name)

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Return whether one SQLite column exists."""
        return _column_exists(self.connection, table_name, column_name)

    def get_neighbors(
        self,
        node_id: str,
        include_reverse: bool = True,
        relation_filter: list[str] | None = None,
        limit: int | None = None,
        strict_relation_filter: bool = False,
    ) -> list[ExternalGraphNeighbor]:
        """Return neighboring edges for one node id."""
        relation_filter = [item for item in (relation_filter or []) if item]
        effective_limit, sql_limit = self._resolve_limit(limit)

        rows: list[ExternalGraphNeighbor] = []
        rows.extend(
            self._query_neighbors(
                column="head",
                node_id=node_id,
                relation_filter=relation_filter,
                sql_limit=sql_limit,
                reversed_edge=False,
                strict_relation_filter=strict_relation_filter,
            )
        )
        if include_reverse:
            rows.extend(
                self._query_neighbors(
                    column="tail",
                    node_id=node_id,
                    relation_filter=relation_filter,
                    sql_limit=sql_limit,
                    reversed_edge=True,
                    strict_relation_filter=strict_relation_filter,
                )
            )

        rows.sort(
            key=lambda item: (
                0 if relation_filter and item.triple.relation in relation_filter else 1,
                item.triple.relation,
                item.neighbor_id,
            )
        )

        deduped: list[ExternalGraphNeighbor] = []
        seen: set[tuple[str, str, str, bool]] = set()
        for item in rows:
            key = (item.source_id, item.neighbor_id, item.triple.relation, item.reversed)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if effective_limit is not None and len(deduped) >= effective_limit:
                break
        LOGGER.info(
            "GraphAPI get_neighbors node_id=%s include_reverse=%s relation_filter=%s strict_relation_filter=%s limit=%s returned=%d",
            node_id,
            include_reverse,
            relation_filter,
            strict_relation_filter,
            "none" if effective_limit is None else effective_limit,
            len(deduped),
        )
        return deduped

    def resolve_entity_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """Return rich entity candidate matches for one or more mentions."""
        candidates: list[dict[str, Any]] = []
        for mention in names:
            for row in resolve_entity_candidates(self.connection, mention, limit=top_k):
                enriched = dict(row)
                enriched["mention"] = mention
                candidates.append(enriched)
        LOGGER.info(
            "GraphAPI resolve_entity_candidates names=%s top_k=%d returned=%d",
            names,
            top_k,
            len(candidates),
        )
        return candidates

    def resolve_entity_mentions(self, names: list[str], top_k: int = 1) -> list[str]:
        """Resolve names to mids using exact name and alias matches."""
        top_k = max(1, int(top_k))
        resolved: list[str] = []
        seen: set[str] = set()
        for mention in names:
            local_count = 0
            for row in resolve_entity_candidates(self.connection, mention, limit=max(top_k, 5)):
                mid = str(row["mid"])
                if mid in seen:
                    continue
                seen.add(mid)
                resolved.append(mid)
                local_count += 1
                if local_count >= top_k:
                    break
        LOGGER.info(
            "GraphAPI resolve_entity_mentions names=%s top_k=%d resolved=%s",
            names,
            top_k,
            resolved,
        )
        return resolved

    def get_entity_display_name(self, node_id: str) -> str:
        """Resolve a mid to its display name."""
        if not _is_entity_id(node_id):
            return node_id
        cached = self._name_cache.get(node_id)
        if cached is not None:
            return cached

        row = self.connection.execute(
            "SELECT name FROM entities WHERE mid = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            self._name_cache[node_id] = node_id
        else:
            name = str(row["name"] or "").strip()
            self._name_cache[node_id] = name or node_id
        return self._name_cache[node_id]

    def resolve_relation_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        """Return rich relation candidate matches for one or more hints."""
        candidates: list[dict[str, Any]] = []
        for hint in names:
            for row in resolve_relation_candidates(self.connection, hint, limit=top_k):
                enriched = dict(row)
                enriched["hint"] = hint
                candidates.append(enriched)
        LOGGER.info(
            "GraphAPI resolve_relation_candidates names=%s top_k=%d returned=%d",
            names,
            top_k,
            len(candidates),
        )
        return candidates

    def resolve_relation_hints(self, names: list[str], top_k: int = 3) -> list[str]:
        """Resolve relation-name hints to canonical Freebase relation ids."""
        top_k = max(1, int(top_k))
        resolved: list[str] = []
        seen: set[str] = set()
        for hint in names:
            local_count = 0
            for row in resolve_relation_candidates(self.connection, hint, limit=max(top_k, 5)):
                relation_id = str(row["relation"])
                if relation_id in seen:
                    continue
                seen.add(relation_id)
                resolved.append(relation_id)
                local_count += 1
                if local_count >= top_k:
                    break
        LOGGER.info(
            "GraphAPI resolve_relation_hints names=%s top_k=%d resolved=%s",
            names,
            top_k,
            resolved,
        )
        return resolved

    def get_relation_display_name(self, relation_id: str) -> str:
        """Resolve a relation id to its readable display name."""
        cached = self._relation_display_cache.get(relation_id)
        if cached is not None:
            return cached

        if self._has_relations_table:
            row = self.connection.execute(
                "SELECT relation_name FROM relations WHERE relation = ? LIMIT 1",
                (relation_id,),
            ).fetchone()
            if row is not None:
                display_name = verbalize_relation(relation_id)
                if display_name:
                    self._relation_display_cache[relation_id] = display_name
                    return display_name

        self._relation_display_cache[relation_id] = verbalize_relation(relation_id)
        return self._relation_display_cache[relation_id]

    def get_node_types(self, node_id: str) -> list[str]:
        """Return entity types when the SQLite entity table exposes them."""
        if not _is_entity_id(node_id) or not self._has_entity_types:
            return []
        row = self.connection.execute(
            "SELECT types FROM entities WHERE mid = ? LIMIT 1",
            (node_id,),
        ).fetchone()
        if row is None:
            return []
        raw = str(row["types"] or "").strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split("|") if item.strip()]

    def collect_related_entities_constrained(
        self,
        source_ids: list[str],
        relation_ids: list[str],
        direction: str = "forward",
        expected_answer_type: str = "",
        constraint_terms: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Collect relation targets and rank them by type and local lexical constraints."""
        source_ids = [item for item in source_ids if item]
        relation_ids = [item for item in relation_ids if item]
        if not source_ids or not relation_ids or int(limit) <= 0:
            return []
        effective_limit = max(1, int(limit))
        normalized_terms = [
            normalize_text(term)
            for term in (constraint_terms or [])
            if normalize_text(term)
        ]
        normalized_expected_type = normalize_text(expected_answer_type)
        relation_placeholders = ", ".join("?" for _ in relation_ids)
        source_placeholders = ", ".join("?" for _ in source_ids)
        if direction == "reverse":
            query = (
                "SELECT tail AS source_entity_id, head AS target_entity_id, relation "
                "FROM triples "
                f"WHERE tail IN ({source_placeholders}) AND relation IN ({relation_placeholders}) "
                "LIMIT ?"
            )
        else:
            query = (
                "SELECT head AS source_entity_id, tail AS target_entity_id, relation "
                "FROM triples "
                f"WHERE head IN ({source_placeholders}) AND relation IN ({relation_placeholders}) "
                "LIMIT ?"
            )
        raw_rows = self.connection.execute(
            query,
            (*source_ids, *relation_ids, max(effective_limit * 8, 200)),
        ).fetchall()
        if not raw_rows:
            return []

        target_ids = [
            str(row["target_entity_id"] or "").strip()
            for row in raw_rows
            if str(row["target_entity_id"] or "").strip().startswith(("m.", "g."))
        ]
        metadata_rows: dict[str, sqlite3.Row] = {}
        if target_ids:
            placeholders = ", ".join("?" for _ in sorted(set(target_ids)))
            select_parts = ["mid", "name"]
            if self._has_entity_types:
                select_parts.append("types")
            if self._has_entity_is_cvt:
                select_parts.append("is_cvt")
            entity_rows = self.connection.execute(
                f"SELECT {', '.join(select_parts)} FROM entities WHERE mid IN ({placeholders})",
                tuple(sorted(set(target_ids))),
            ).fetchall()
            metadata_rows = {str(row["mid"]): row for row in entity_rows}

        scored_rows: list[dict[str, object]] = []
        for row in raw_rows:
            source_id = str(row["source_entity_id"] or "").strip()
            target_id = str(row["target_entity_id"] or "").strip()
            relation_id = str(row["relation"] or "").strip()
            if not source_id or not target_id or not relation_id:
                continue
            metadata = metadata_rows.get(target_id)
            target_label = (
                str(metadata["name"] or "").strip()
                if metadata is not None and str(metadata["name"] or "").strip()
                else self.get_entity_display_name(target_id)
            )
            types = (
                [item.strip() for item in str(metadata["types"] or "").split("|") if item.strip()]
                if metadata is not None and self._has_entity_types
                else []
            )
            haystack = normalize_text(" ".join([target_label, *types, self.get_relation_display_name(relation_id)]))
            matched_constraints = [term for term in normalized_terms if term and term in haystack]
            score = 0.0
            if normalized_expected_type and any(normalized_expected_type in normalize_text(value) for value in types):
                score += 8.0
            elif normalized_expected_type and normalized_expected_type in haystack:
                score += 4.0
            score += 3.0 * len(matched_constraints)
            if metadata is not None and self._has_entity_is_cvt and int(metadata["is_cvt"] or 0):
                score -= 2.0
            scored_rows.append(
                {
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "target_label": target_label,
                    "relation_id": relation_id,
                    "score": round(score, 4),
                    "matched_constraints": matched_constraints,
                }
            )

        scored_rows.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("relation_id") or ""),
                str(item.get("target_entity_id") or ""),
            )
        )
        deduped: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in scored_rows:
            key = (
                str(row.get("source_entity_id") or ""),
                str(row.get("relation_id") or ""),
                str(row.get("target_entity_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= effective_limit:
                break
        LOGGER.info(
            "GraphAPI collect_related_entities_constrained source_count=%d relation_count=%d direction=%s expected_answer_type=%s constraint_terms=%s returned=%d",
            len(source_ids),
            len(relation_ids),
            direction,
            expected_answer_type,
            normalized_terms[:6],
            len(deduped),
        )
        return deduped

    def find_two_hop_extensions(
        self,
        frontier_nodes: list[str],
        relation_hints: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ReasoningPath]:
        """Return short candidate paths rooted at frontier mids."""
        relation_hints = [item for item in (relation_hints or []) if item]
        effective_limit, _ = self._resolve_limit(limit)
        cap = effective_limit if effective_limit is not None else self.neighbor_limit

        paths: list[ReasoningPath] = []
        for frontier_id in frontier_nodes:
            frontier_label = self.get_entity_display_name(frontier_id)
            first_hop_edges = self.get_neighbors(
                frontier_id,
                include_reverse=True,
                relation_filter=relation_hints,
                limit=cap,
                strict_relation_filter=False,
            )
            for first_edge in first_hop_edges:
                first_triple = self._triple_from_edge(frontier_label, first_edge)
                one_hop_nodes = [
                    frontier_label,
                    first_edge.triple.tail if first_triple.head == frontier_label else first_edge.triple.head,
                ]
                paths.append(
                    ReasoningPath(
                        triples=[first_triple],
                        nodes=one_hop_nodes,
                        text=self._format_path_text(one_hop_nodes, [first_triple]),
                        source_stage="external_graphapi_supplement",
                        edge_ids=[f"{frontier_id}:{first_edge.triple.relation}:{first_edge.neighbor_id}"],
                        terminal_node_id=first_edge.neighbor_id,
                        terminal_node_kind="id" if _is_entity_id(first_edge.neighbor_id) else "literal",
                    )
                )
                if not _is_entity_id(first_edge.neighbor_id):
                    continue

                second_hop_edges = self.get_neighbors(
                    first_edge.neighbor_id,
                    include_reverse=True,
                    relation_filter=relation_hints,
                    limit=cap,
                    strict_relation_filter=False,
                )
                intermediate_label = one_hop_nodes[1]
                for second_edge in second_hop_edges:
                    if second_edge.neighbor_id == frontier_id:
                        continue
                    second_triple = self._triple_from_edge(intermediate_label, second_edge)
                    next_label = (
                        second_triple.tail
                        if second_triple.head == intermediate_label
                        else second_triple.head
                    )
                    nodes = [frontier_label, intermediate_label, next_label]
                    triples = [first_triple, second_triple]
                    paths.append(
                        ReasoningPath(
                            triples=triples,
                            nodes=nodes,
                            text=self._format_path_text(nodes, triples),
                            source_stage="external_graphapi_supplement",
                            edge_ids=[
                                f"{frontier_id}:{first_edge.triple.relation}:{first_edge.neighbor_id}",
                                f"{first_edge.neighbor_id}:{second_edge.triple.relation}:{second_edge.neighbor_id}",
                            ],
                            terminal_node_id=second_edge.neighbor_id,
                            terminal_node_kind="id" if _is_entity_id(second_edge.neighbor_id) else "literal",
                        )
                    )
                    if effective_limit is not None and len(paths) >= effective_limit * max(len(frontier_nodes), 1):
                        deduped_paths = deduplicate_paths(paths)
                        LOGGER.info(
                            "GraphAPI find_two_hop_extensions frontier_nodes=%s relation_hints=%s limit=%s returned=%d",
                            frontier_nodes,
                            relation_hints,
                            "none" if effective_limit is None else effective_limit,
                            len(deduped_paths),
                        )
                        return deduped_paths
        deduped_paths = deduplicate_paths(paths)
        LOGGER.info(
            "GraphAPI find_two_hop_extensions frontier_nodes=%s relation_hints=%s limit=%s returned=%d",
            frontier_nodes,
            relation_hints,
            "none" if effective_limit is None else effective_limit,
            len(deduped_paths),
        )
        return deduped_paths

    def expand_paths(
        self,
        seed_nodes: list[str],
        relation_hints: list[str] | None = None,
        max_depth: int = 3,
        strict_relation_filter: bool = True,
        max_paths: int | None = None,
        max_nodes: int | None = None,
    ) -> list[ReasoningPath]:
        """Expand multi-hop reasoning paths from one or more seed mids."""
        relation_hints = [item for item in (relation_hints or []) if item]
        if max_depth <= 0 or not seed_nodes:
            return []

        neighbor_limit: int | None
        if max_paths is None and max_nodes is None:
            neighbor_limit = 0
        else:
            neighbor_limit = self.neighbor_limit

        queue: list[tuple[str, list[str], list[str], list[Triple], int]] = []
        for seed_id in seed_nodes:
            queue.append(
                (
                    seed_id,
                    [self.get_entity_display_name(seed_id)],
                    [seed_id],
                    [],
                    0,
                )
            )

        paths: list[ReasoningPath] = []
        discovered_nodes: set[str] = set(seed_nodes)
        index = 0
        while index < len(queue):
            current_id, node_labels, node_ids, triples, depth = queue[index]
            index += 1
            if depth >= max_depth:
                continue

            current_label = node_labels[-1]
            for edge in self.get_neighbors(
                current_id,
                include_reverse=True,
                relation_filter=relation_hints,
                limit=neighbor_limit,
                strict_relation_filter=strict_relation_filter,
            ):
                if edge.neighbor_id in node_ids:
                    continue
                is_expandable_neighbor = _is_entity_id(edge.neighbor_id)
                if (
                    max_nodes is not None
                    and is_expandable_neighbor
                    and edge.neighbor_id not in discovered_nodes
                    and len(discovered_nodes) >= max_nodes
                ):
                    continue

                next_triple = self._triple_from_edge(current_label, edge)
                next_label = next_triple.tail if next_triple.head == current_label else next_triple.head
                next_node_ids = node_ids + [edge.neighbor_id]
                next_node_labels = node_labels + [next_label]
                next_triples = triples + [next_triple]
                if is_expandable_neighbor:
                    discovered_nodes.add(edge.neighbor_id)
                path = ReasoningPath(
                    triples=next_triples,
                    nodes=next_node_labels,
                    text=self._format_path_text(next_node_labels, next_triples),
                    source_stage="external_graphapi_expansion",
                    edge_ids=[
                        *[
                            f"{source_id}:{triple.relation}:{target_id}"
                            for source_id, triple, target_id in zip(node_ids, next_triples, next_node_ids[1:])
                        ]
                    ],
                    terminal_node_id=edge.neighbor_id,
                    terminal_node_kind="id" if _is_entity_id(edge.neighbor_id) else "literal",
                )
                paths.append(path)
                if max_paths is not None and len(paths) >= max_paths:
                    deduped = deduplicate_paths(paths)
                    LOGGER.info(
                        "GraphAPI expand_paths seed_nodes=%s relation_hints=%s max_depth=%d strict_relation_filter=%s returned=%d",
                        seed_nodes,
                        relation_hints,
                        max_depth,
                        strict_relation_filter,
                        len(deduped),
                    )
                    return deduped
                if is_expandable_neighbor:
                    queue.append(
                        (
                            edge.neighbor_id,
                            next_node_labels,
                            next_node_ids,
                            next_triples,
                            depth + 1,
                        )
                    )

        deduped = deduplicate_paths(paths)
        LOGGER.info(
            "GraphAPI expand_paths seed_nodes=%s relation_hints=%s max_depth=%d strict_relation_filter=%s returned=%d",
            seed_nodes,
            relation_hints,
            max_depth,
            strict_relation_filter,
            len(deduped),
        )
        return deduped

    def _load_relation_catalog(self) -> list[tuple[str, str, str, str]]:
        """Load unique relation ids and display names for relation-name alignment."""
        if self._has_relations_table:
            rows = self.connection.execute(
                """
                SELECT relation, relation_name, relation_norm, relation_name_norm
                FROM relations
                """
            ).fetchall()
            catalog: list[tuple[str, str, str, str]] = []
            for row in rows:
                relation_id = str(row["relation"])
                relation_name = verbalize_relation(relation_id)
                relation_norm = str(row["relation_norm"] or normalize_text(relation_id))
                relation_name_norm = str(
                    row["relation_name_norm"]
                    or normalize_text(str(row["relation_name"] or relation_name))
                )
                self._relation_display_cache[relation_id] = relation_name
                catalog.append((relation_id, relation_name, relation_name_norm, relation_norm))
            return catalog

        rows = self.connection.execute("SELECT DISTINCT relation FROM triples").fetchall()
        catalog: list[tuple[str, str, str, str]] = []
        for row in rows:
            relation_id = str(row["relation"])
            relation_name = verbalize_relation(relation_id)
            self._relation_display_cache[relation_id] = relation_name
            catalog.append(
                (
                    relation_id,
                    relation_name,
                    normalize_text(relation_name),
                    normalize_text(relation_id),
                )
            )
        return catalog

    def _resolve_limit(self, limit: int | None) -> tuple[int | None, int | None]:
        """Resolve user limit into an effective cap and SQL LIMIT value."""
        if limit is None:
            if self.neighbor_limit <= 0:
                return None, None
            return self.neighbor_limit, self.neighbor_limit
        if int(limit) <= 0:
            return None, None
        resolved = int(limit)
        return resolved, resolved

    def _format_path_text(self, nodes: list[str], triples: list[Triple]) -> str:
        """Render one reasoning path with verbalized relation labels."""
        if not triples or len(nodes) <= 1:
            return " -> ".join(nodes)
        segments: list[str] = [nodes[0]]
        current = nodes[0]
        for index, triple in enumerate(triples):
            next_node = nodes[index + 1] if index + 1 < len(nodes) else triple.tail
            relation = self.get_relation_display_name(triple.relation)
            if triple.tail == current and triple.head == next_node:
                relation = f"{relation} (reverse)"
            segments.append(relation)
            segments.append(next_node)
            current = next_node
        return " -> ".join(segments)

    def _query_neighbors(
        self,
        column: str,
        node_id: str,
        relation_filter: list[str],
        sql_limit: int | None,
        reversed_edge: bool,
        strict_relation_filter: bool,
    ) -> list[ExternalGraphNeighbor]:
        where_clauses = [f"{column} = ?"]
        params: list[object] = [node_id]
        order_by = "relation, tail, head"
        if relation_filter and strict_relation_filter:
            placeholders = ", ".join("?" for _ in relation_filter)
            where_clauses.append(f"relation IN ({placeholders})")
            params.extend(relation_filter)
        elif relation_filter:
            placeholders = ", ".join("?" for _ in relation_filter)
            order_by = f"CASE WHEN relation IN ({placeholders}) THEN 0 ELSE 1 END, {order_by}"
            params.extend(relation_filter)

        select_parts = [
            "head",
            "relation",
            "tail",
            "tail_kind" if self._has_triple_tail_kind else "'id' AS tail_kind",
        ]
        query = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM triples WHERE {' AND '.join(where_clauses)} "
            f"ORDER BY {order_by}"
        )
        if sql_limit is not None:
            query += " LIMIT ?"
            params.append(sql_limit)
        rows = self.connection.execute(query, tuple(params)).fetchall()

        neighbors: list[ExternalGraphNeighbor] = []
        for row in rows:
            head_id = str(row["head"])
            tail_id = str(row["tail"])
            relation_id = str(row["relation"])
            relation_name = self.get_relation_display_name(relation_id)
            head_name = self.get_entity_display_name(head_id)
            tail_kind = str(row["tail_kind"] or "id").strip().lower()
            tail_name = tail_id if tail_kind == "literal" else self.get_entity_display_name(tail_id)
            triple = Triple(
                head=head_name,
                relation=relation_id,
                tail=tail_name,
            )
            if reversed_edge:
                source_id = tail_id
                neighbor_id = head_id
            else:
                source_id = head_id
                neighbor_id = tail_id
            neighbors.append(
                ExternalGraphNeighbor(
                    source_id=source_id,
                    neighbor_id=neighbor_id,
                    triple=triple,
                    relation_name=relation_name,
                    reversed=reversed_edge,
                )
            )
        return neighbors

    @staticmethod
    def _triple_from_edge(frontier_label: str, edge: ExternalGraphNeighbor) -> Triple:
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
