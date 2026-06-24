"""Indexed SQLite runtime backend for large-KG retrieval."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import shutil
from typing import Any

from kgqa.kg.graph_api import ExternalGraphNeighbor
from kgqa.kg.relation_beam import (
    EvidenceEdge,
    FrontierExpansion,
    NodeMetadata,
    RelationCandidate,
)
from kgqa.kg.sqlite_graph_api import SQLiteGraphAPI, _derive_relation_display_name, normalize_text
from kgqa.utils.logging import get_logger
from kgqa.utils.types import Triple

LOGGER = get_logger(__name__)

_INDEX_DB_NAME = "runtime_index.sqlite"
_INDEX_BATCH_SIZE = 100_000
_DEFAULT_PROGRESS_EVERY = 1_000_000
_SQLITE_PARAM_BATCH = 500


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


def build_indexed_sqlite_runtime_index(
    db_path: str | Path,
    index_dir: str | Path,
    overwrite: bool = False,
    high_degree_threshold: int = 500,
    staging_dir: str | Path | None = None,
    progress_every: int = _DEFAULT_PROGRESS_EVERY,
    resume: bool = False,
) -> Path:
    """Build runtime lookup and adjacency indexes from one SQLite truth database."""
    db_path = Path(db_path)
    index_dir = Path(index_dir)
    staging_root = Path(staging_dir) if staging_dir is not None else None
    index_dir.mkdir(parents=True, exist_ok=True)
    index_db_path = index_dir / _INDEX_DB_NAME
    build_index_db_path = index_db_path
    if staging_root is not None:
        LOGGER.info("preparing runtime index staging dir root=%s", staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        build_index_db_path = staging_root / f"{index_dir.name}_{_INDEX_DB_NAME}.partial"
        LOGGER.info("runtime index staging ready build_db=%s", build_index_db_path)

    if resume:
        if index_db_path.exists():
            LOGGER.info("runtime index resume found completed target=%s", index_db_path)
            return index_db_path
    else:
        if index_db_path.exists():
            if not overwrite:
                raise FileExistsError(f"Runtime index already exists at {index_db_path}")
            index_db_path.unlink()
        if build_index_db_path.exists():
            build_index_db_path.unlink()

    build_succeeded = False
    source = sqlite3.connect(str(db_path))
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(str(build_index_db_path))
    target.row_factory = sqlite3.Row
    try:
        LOGGER.info(
            "starting runtime index build source=%s index_dir=%s staging_dir=%s",
            db_path,
            index_dir,
            staging_root,
        )
        target.executescript(
            """
            CREATE TABLE IF NOT EXISTS build_progress (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            PRAGMA locking_mode = EXCLUSIVE;
            PRAGMA cache_size = -400000;

            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY,
                head TEXT,
                relation TEXT,
                tail TEXT,
                tail_kind TEXT
            );

            CREATE TABLE IF NOT EXISTS entity_alias (
                alias_norm TEXT,
                mid TEXT,
                entity_name TEXT,
                alias_source TEXT
            );

            CREATE TABLE IF NOT EXISTS relation_lookup (
                lookup_norm TEXT,
                relation TEXT,
                relation_name TEXT,
                lookup_source TEXT
            );

            CREATE TABLE IF NOT EXISTS node_stats (
                mid TEXT PRIMARY KEY,
                in_degree INTEGER,
                out_degree INTEGER,
                literal_out_degree INTEGER,
                degree INTEGER,
                high_degree INTEGER
            );

            CREATE TABLE IF NOT EXISTS relation_stats (
                relation TEXT PRIMARY KEY,
                frequency INTEGER
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        LOGGER.info("runtime index schema created db=%s", build_index_db_path)

        has_tail_kind = _column_exists(source, "triples", "tail_kind")
        edge_total = int(source.execute("SELECT COUNT(1) FROM triples").fetchone()[0])
        entity_total = int(source.execute("SELECT COUNT(1) FROM entities").fetchone()[0])
        relation_total = int(source.execute("SELECT COUNT(1) FROM relations").fetchone()[0])
        LOGGER.info(
            "runtime index source counts edges=%d entities=%d relations=%d batch_size=%d",
            edge_total,
            entity_total,
            relation_total,
            _INDEX_BATCH_SIZE,
        )

        triple_select = [
            "rowid AS sqlite_rowid",
            "head",
            "relation",
            "tail",
            "tail_kind" if has_tail_kind else "'id' AS tail_kind",
        ]
        edge_batch: list[tuple[int, str, str, str, str]] = []
        out_degree: dict[str, int] = {}
        in_degree: dict[str, int] = {}
        literal_out_degree: dict[str, int] = {}
        relation_frequency: dict[str, int] = {}
        if _progress_value(target, "edges_complete") != "1":
            existing_edge_id = int(target.execute("SELECT COALESCE(MAX(edge_id), 0) FROM edges").fetchone()[0])
            if existing_edge_id:
                LOGGER.info("runtime index edges resume existing_edge_id=%d", existing_edge_id)
            rows = source.execute(
                f"SELECT {', '.join(triple_select)} FROM triples WHERE rowid > ?",
                (existing_edge_id,),
            )
            edges_processed = existing_edge_id
            for row in rows:
                edges_processed += 1
                sqlite_rowid = int(row["sqlite_rowid"])
                head = str(row["head"])
                relation = str(row["relation"])
                tail = str(row["tail"])
                tail_kind = str(row["tail_kind"] or "id").strip().lower()
                edge_id = sqlite_rowid
                edge_batch.append((edge_id, head, relation, tail, tail_kind))
                if len(edge_batch) >= _INDEX_BATCH_SIZE:
                    target.executemany(
                        """
                        INSERT INTO edges(edge_id, head, relation, tail, tail_kind)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        edge_batch,
                    )
                    edge_batch.clear()
                if progress_every > 0 and edges_processed % progress_every == 0:
                    _set_progress_value(target, "edges_rows", edges_processed)
                    target.commit()
                    LOGGER.info(_format_progress_message("index.edges", edges_processed, edge_total))
            if edge_batch:
                target.executemany(
                    """
                    INSERT INTO edges(edge_id, head, relation, tail, tail_kind)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    edge_batch,
                )
            _set_progress_value(target, "edges_rows", edge_total)
            _set_progress_value(target, "edges_complete", 1)
            target.commit()
            LOGGER.info(_format_progress_message("index.edges", edge_total, edge_total, "done"))
        else:
            LOGGER.info(_format_progress_message("index.edges", edge_total, edge_total, "resume-skip done"))

        LOGGER.info("runtime index degree/relation stats scan started")
        for row in target.execute("SELECT head, relation, tail, tail_kind FROM edges"):
            head = str(row["head"])
            relation = str(row["relation"])
            tail = str(row["tail"])
            tail_kind = str(row["tail_kind"] or "id").strip().lower()
            out_degree[head] = out_degree.get(head, 0) + 1
            relation_frequency[relation] = relation_frequency.get(relation, 0) + 1
            if tail_kind == "id":
                in_degree[tail] = in_degree.get(tail, 0) + 1
            else:
                literal_out_degree[head] = literal_out_degree.get(head, 0) + 1
        LOGGER.info("runtime index degree/relation stats scan done unique_relations=%d", len(relation_frequency))

        alias_batch: list[tuple[str, str, str, str]] = []
        all_entity_ids: set[str] = set()
        entities_processed = 0
        if _progress_value(target, "entity_alias_complete") != "1":
            LOGGER.info("runtime index entity_alias phase started")
            target.execute("DELETE FROM entity_alias")
            for row in source.execute("SELECT mid, name, aliases FROM entities"):
                entities_processed += 1
                mid = str(row["mid"])
                all_entity_ids.add(mid)
                entity_name = str(row["name"] or "").strip()
                aliases = str(row["aliases"] or "").strip()
                if entity_name:
                    alias_batch.append((normalize_text(entity_name), mid, entity_name, "name"))
                for alias in [item.strip() for item in aliases.split("|") if item.strip()]:
                    alias_batch.append((normalize_text(alias), mid, entity_name or alias, "alias"))
                if len(alias_batch) >= _INDEX_BATCH_SIZE:
                    target.executemany(
                        "INSERT INTO entity_alias(alias_norm, mid, entity_name, alias_source) VALUES (?, ?, ?, ?)",
                        alias_batch,
                    )
                    alias_batch.clear()
                if progress_every > 0 and entities_processed % progress_every == 0:
                    LOGGER.info(_format_progress_message("index.entity_alias", entities_processed, entity_total))
            if alias_batch:
                target.executemany(
                    "INSERT INTO entity_alias(alias_norm, mid, entity_name, alias_source) VALUES (?, ?, ?, ?)",
                    alias_batch,
                )
            _set_progress_value(target, "entity_alias_complete", 1)
            target.commit()
            LOGGER.info(_format_progress_message("index.entity_alias", entities_processed, entity_total, "done"))
        else:
            LOGGER.info(_format_progress_message("index.entity_alias", entity_total, entity_total, "resume-skip done"))
            for row in source.execute("SELECT mid FROM entities"):
                all_entity_ids.add(str(row["mid"]))

        relation_batch: list[tuple[str, str, str, str]] = []
        relations_processed = 0
        if _progress_value(target, "relation_lookup_complete") != "1":
            LOGGER.info("runtime index relation_lookup phase started")
            target.execute("DELETE FROM relation_lookup")
            for row in source.execute("SELECT relation, relation_name FROM relations"):
                relations_processed += 1
                relation = str(row["relation"])
                relation_name = str(row["relation_name"] or "").strip() or _derive_relation_display_name(relation)
                relation_batch.append((normalize_text(relation), relation, relation_name, "relation_id"))
                relation_batch.append((normalize_text(relation_name), relation, relation_name, "relation_name"))
                if len(relation_batch) >= _INDEX_BATCH_SIZE:
                    target.executemany(
                        "INSERT INTO relation_lookup(lookup_norm, relation, relation_name, lookup_source) VALUES (?, ?, ?, ?)",
                        relation_batch,
                    )
                    relation_batch.clear()
                if progress_every > 0 and relations_processed % progress_every == 0:
                    LOGGER.info(_format_progress_message("index.relation_lookup", relations_processed, relation_total))
            if relation_batch:
                target.executemany(
                    "INSERT INTO relation_lookup(lookup_norm, relation, relation_name, lookup_source) VALUES (?, ?, ?, ?)",
                    relation_batch,
                )
            _set_progress_value(target, "relation_lookup_complete", 1)
            target.commit()
            LOGGER.info(_format_progress_message("index.relation_lookup", relations_processed, relation_total, "done"))
        else:
            LOGGER.info(_format_progress_message("index.relation_lookup", relation_total, relation_total, "resume-skip done"))

        node_stats_batch: list[tuple[str, int, int, int, int, int]] = []
        node_total = len(all_entity_ids | set(out_degree) | set(in_degree))
        node_processed = 0
        if _progress_value(target, "node_stats_complete") != "1":
            LOGGER.info("runtime index node_stats phase started total_nodes=%d", node_total)
            target.execute("DELETE FROM node_stats")
            target.execute("DELETE FROM relation_stats")
            for mid in sorted(all_entity_ids | set(out_degree) | set(in_degree)):
                node_processed += 1
                in_value = int(in_degree.get(mid, 0))
                out_value = int(out_degree.get(mid, 0))
                literal_value = int(literal_out_degree.get(mid, 0))
                degree = in_value + out_value
                node_stats_batch.append(
                    (
                        mid,
                        in_value,
                        out_value,
                        literal_value,
                        degree,
                        1 if degree >= high_degree_threshold else 0,
                    )
                )
                if progress_every > 0 and node_processed % progress_every == 0:
                    LOGGER.info(_format_progress_message("index.node_stats", node_processed, node_total))
            target.executemany(
                """
                INSERT INTO node_stats(mid, in_degree, out_degree, literal_out_degree, degree, high_degree)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                node_stats_batch,
            )
            LOGGER.info(_format_progress_message("index.node_stats", node_processed, node_total, "done"))
            target.executemany(
                "INSERT INTO relation_stats(relation, frequency) VALUES (?, ?)",
                [(relation, frequency) for relation, frequency in relation_frequency.items()],
            )
            LOGGER.info(
                _format_progress_message(
                    "index.relation_stats",
                    len(relation_frequency),
                    len(relation_frequency),
                    "done",
                )
            )
            _set_progress_value(target, "node_stats_complete", 1)
            target.commit()
        else:
            LOGGER.info(_format_progress_message("index.node_stats", node_total, node_total, "resume-skip done"))

        target.execute("DELETE FROM metadata")
        target.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("source_db_path", str(db_path)),
                ("high_degree_threshold", str(high_degree_threshold)),
            ],
        )
        LOGGER.info("runtime index metadata written db=%s", build_index_db_path)
        if _progress_value(target, "indexes_complete") != "1":
            for index_name in (
                "idx_edges_head",
                "idx_edges_tail",
                "idx_edges_relation",
                "idx_edges_head_relation",
                "idx_edges_tail_relation",
                "idx_entity_alias_norm",
                "idx_entity_alias_mid",
                "idx_relation_lookup_norm",
                "idx_relation_lookup_relation",
                "idx_node_stats_degree",
            ):
                target.execute(f"DROP INDEX IF EXISTS {index_name}")
            _run_index_statements(
                target,
                "index.indexes",
                [
                    ("idx_edges_head", "CREATE INDEX idx_edges_head ON edges(head)"),
                    ("idx_edges_tail", "CREATE INDEX idx_edges_tail ON edges(tail)"),
                    ("idx_edges_relation", "CREATE INDEX idx_edges_relation ON edges(relation)"),
                    ("idx_edges_head_relation", "CREATE INDEX idx_edges_head_relation ON edges(head, relation)"),
                    ("idx_edges_tail_relation", "CREATE INDEX idx_edges_tail_relation ON edges(tail, relation)"),
                    ("idx_entity_alias_norm", "CREATE INDEX idx_entity_alias_norm ON entity_alias(alias_norm)"),
                    ("idx_entity_alias_mid", "CREATE INDEX idx_entity_alias_mid ON entity_alias(mid)"),
                    (
                        "idx_relation_lookup_norm",
                        "CREATE INDEX idx_relation_lookup_norm ON relation_lookup(lookup_norm)",
                    ),
                    (
                        "idx_relation_lookup_relation",
                        "CREATE INDEX idx_relation_lookup_relation ON relation_lookup(relation)",
                    ),
                    ("idx_node_stats_degree", "CREATE INDEX idx_node_stats_degree ON node_stats(degree)"),
                ],
            )
            _set_progress_value(target, "indexes_complete", 1)
            target.commit()
        else:
            LOGGER.info(_format_progress_message("index.indexes", 10, 10, "resume-skip done"))
        LOGGER.info("runtime index commit started path=%s", build_index_db_path)
        _set_progress_value(target, "build_complete", 1)
        target.commit()
        LOGGER.info("runtime index build committed path=%s", build_index_db_path)
        build_succeeded = True
    finally:
        source.close()
        target.close()
        LOGGER.info("runtime index connections closed source=%s target=%s", db_path, build_index_db_path)

    if build_index_db_path != index_db_path and build_succeeded:
        LOGGER.info("runtime index move started src=%s dst=%s", build_index_db_path, index_db_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(build_index_db_path), str(index_db_path))
        LOGGER.info("runtime index move finished dst=%s", index_db_path)
    elif build_index_db_path != index_db_path and not build_succeeded:
        LOGGER.info("runtime index partial build preserved path=%s", build_index_db_path)

    LOGGER.info("Built runtime index db_path=%s index_db=%s", db_path, index_db_path)
    return index_db_path


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


class IndexedSQLiteGraphBackend(SQLiteGraphAPI):
    """SQLite truth-store backend with optional runtime adjacency indexes."""

    def __init__(
        self,
        db_path: str | Path,
        index_dir: str | Path | None = None,
        neighbor_limit: int = 25,
    ) -> None:
        super().__init__(db_path=db_path, neighbor_limit=neighbor_limit)
        self.index_dir = Path(index_dir) if index_dir else None
        self.index_db_path = self.index_dir / _INDEX_DB_NAME if self.index_dir else None
        self.index_connection: sqlite3.Connection | None = None
        if self.index_db_path and self.index_db_path.exists():
            self.index_connection = sqlite3.connect(str(self.index_db_path))
            self.index_connection.row_factory = sqlite3.Row
            LOGGER.info("Loaded IndexedSQLiteGraphBackend index_db=%s", self.index_db_path)

    def close(self) -> None:
        """Close SQLite and runtime index connections."""
        if self.index_connection is not None:
            self.index_connection.close()
        super().close()

    def get_neighbors(
        self,
        node_id: str,
        include_reverse: bool = True,
        relation_filter: list[str] | None = None,
        limit: int | None = None,
        strict_relation_filter: bool = False,
    ) -> list[ExternalGraphNeighbor]:
        """Return neighboring edges for one node, preferring the runtime index."""
        if self.index_connection is None:
            return super().get_neighbors(
                node_id=node_id,
                include_reverse=include_reverse,
                relation_filter=relation_filter,
                limit=limit,
                strict_relation_filter=strict_relation_filter,
            )
        relation_filter = [item for item in (relation_filter or []) if item]
        effective_limit, sql_limit = self._resolve_limit(limit)
        rows: list[ExternalGraphNeighbor] = []
        rows.extend(
            self._query_index_neighbors(
                direction="forward",
                node_id=node_id,
                relation_filter=relation_filter,
                sql_limit=sql_limit,
                reversed_edge=False,
                strict_relation_filter=strict_relation_filter,
            )
        )
        if include_reverse:
            rows.extend(
                self._query_index_neighbors(
                    direction="reverse",
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
        return deduped

    def get_neighbors_batched(
        self,
        node_ids: list[str],
        include_reverse: bool = True,
        relation_filter: list[str] | None = None,
        limit: int | None = None,
        strict_relation_filter: bool = False,
    ) -> dict[str, list[ExternalGraphNeighbor]]:
        """Return neighbors for a batch of graph nodes."""
        return {
            node_id: self.get_neighbors(
                node_id=node_id,
                include_reverse=include_reverse,
                relation_filter=relation_filter,
                limit=limit,
                strict_relation_filter=strict_relation_filter,
            )
            for node_id in node_ids
        }

    def get_node_edges(
        self,
        node_id: str,
        include_reverse: bool = True,
        include_literals: bool = True,
        limit: int | None = None,
    ) -> list[ExternalGraphNeighbor]:
        """Return exact adjacent edge rows for one node without relation aggregation."""
        rows = self.get_neighbors(
            node_id=node_id,
            include_reverse=include_reverse,
            relation_filter=None,
            limit=limit,
            strict_relation_filter=False,
        )
        if include_literals:
            return rows
        return [row for row in rows if str(row.neighbor_id).startswith(("m.", "g."))]

    def get_node_types(self, node_id: str) -> list[str]:
        """Return entity types when present."""
        if not self._has_entity_types:
            return []
        row = self.connection.execute("SELECT types FROM entities WHERE mid = ? LIMIT 1", (node_id,)).fetchone()
        if row is None:
            return []
        return [item.strip() for item in str(row["types"] or "").split("|") if item.strip()]

    def get_node_degree(self, node_id: str) -> int:
        """Return node degree using runtime stats when available."""
        if self.index_connection is not None:
            row = self.index_connection.execute(
                "SELECT degree FROM node_stats WHERE mid = ? LIMIT 1",
                (node_id,),
            ).fetchone()
            if row is not None:
                return int(row["degree"] or 0)
        return super().get_node_degree(node_id)

    def get_relation_frequency(self, relation_id: str) -> int:
        """Return a coarse global relation frequency when available."""
        if self.index_connection is not None:
            row = self.index_connection.execute(
                "SELECT frequency FROM relation_stats WHERE relation = ? LIMIT 1",
                (relation_id,),
            ).fetchone()
            if row is not None:
                return int(row["frequency"] or 0)
        row = self.connection.execute(
            "SELECT COUNT(1) AS frequency FROM triples WHERE relation = ?",
            (relation_id,),
        ).fetchone()
        return int(row["frequency"] or 0) if row is not None else 0

    def is_probable_cvt_node(
        self,
        node_id: str,
        name: str = "",
        types: tuple[str, ...] = (),
    ) -> bool:
        """Heuristically identify mediator/CVT-like nodes without overcommitting."""
        if not str(node_id).startswith(("m.", "g.")):
            return False
        normalized_name = str(name or "").strip()
        if _column_exists(self.connection, "entities", "is_cvt"):
            row = self.connection.execute(
                "SELECT is_cvt FROM entities WHERE mid = ? LIMIT 1",
                (node_id,),
            ).fetchone()
            if row is not None and str(row["is_cvt"] or "").strip() in {"1", "true", "True"}:
                return True
        cvt_type_markers = (
            "type.cvt",
            "freebase.type_hints.mediator",
            "base.type_ontology.mediator",
        )
        if any(marker in types for marker in cvt_type_markers):
            return True
        # Fall back only for unlabeled record-like nodes; named entities should not be
        # treated as CVTs unless explicit mediator markers are present.
        if normalized_name and normalized_name != node_id:
            return False
        non_record_markers = (
            "location.",
            "people.",
            "organization.",
            "film.",
            "music.",
            "sports.",
            "government.",
            "book.",
            "tv.",
            "common.topic",
        )
        if any(type_id.startswith(non_record_markers) for type_id in types):
            return False
        return True

    def get_nodes_metadata_batched(
        self,
        node_ids: list[str],
    ) -> dict[str, NodeMetadata]:
        """Return batched node metadata for entity and literal frontier nodes."""
        unique_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not unique_node_ids:
            return {}
        entity_rows: dict[str, sqlite3.Row] = {}
        degree_rows: dict[str, sqlite3.Row] = {}
        entity_ids = [node_id for node_id in unique_node_ids if node_id.startswith(("m.", "g."))]
        for start in range(0, len(entity_ids), _SQLITE_PARAM_BATCH):
            batch = entity_ids[start : start + _SQLITE_PARAM_BATCH]
            placeholders = ", ".join("?" for _ in batch)
            for row in self.connection.execute(
                f"SELECT mid, name, types FROM entities WHERE mid IN ({placeholders})",
                tuple(batch),
            ).fetchall():
                entity_rows[str(row["mid"])] = row
            degree_source = self.index_connection or self.connection
            degree_table = "node_stats" if self.index_connection is not None else None
            if degree_table is not None and _table_exists(degree_source, degree_table):
                for row in degree_source.execute(
                    f"SELECT mid, degree FROM {degree_table} WHERE mid IN ({placeholders})",
                    tuple(batch),
                ).fetchall():
                    degree_rows[str(row["mid"])] = row
        metadata: dict[str, NodeMetadata] = {}
        for node_id in unique_node_ids:
            row = entity_rows.get(node_id)
            if row is None:
                metadata[node_id] = NodeMetadata(
                    node_id=node_id,
                    name=node_id,
                    types=(),
                    degree=0,
                    is_literal=True,
                    is_probable_cvt=False,
                )
                continue
            name = str(row["name"] or "").strip()
            types = tuple(item.strip() for item in str(row["types"] or "").split("|") if item.strip())
            degree_row = degree_rows.get(node_id)
            degree = int(degree_row["degree"] or 0) if degree_row is not None else self.get_node_degree(node_id)
            metadata[node_id] = NodeMetadata(
                node_id=node_id,
                name=name or node_id,
                types=types,
                degree=degree,
                is_literal=False,
                is_probable_cvt=self.is_probable_cvt_node(node_id=node_id, name=name, types=types),
            )
        return metadata

    def find_direct_edges_between(
        self,
        source_ids: list[str],
        target_ids: list[str],
        relation_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """Return exact direct source->target edges for the provided entity pairs."""
        unique_source_ids = list(dict.fromkeys(node_id for node_id in source_ids if node_id))
        unique_target_ids = list(dict.fromkeys(node_id for node_id in target_ids if node_id))
        if not unique_source_ids or not unique_target_ids:
            return []
        effective_limit = max(1, int(limit))
        connection = self.index_connection or self.connection
        table_name = "edges" if self.index_connection is not None else "triples"
        rows: list[dict[str, str]] = []
        for source_start in range(0, len(unique_source_ids), _SQLITE_PARAM_BATCH):
            source_batch = unique_source_ids[source_start : source_start + _SQLITE_PARAM_BATCH]
            source_placeholders = ", ".join("?" for _ in source_batch)
            for target_start in range(0, len(unique_target_ids), _SQLITE_PARAM_BATCH):
                target_batch = unique_target_ids[target_start : target_start + _SQLITE_PARAM_BATCH]
                target_placeholders = ", ".join("?" for _ in target_batch)
                where_parts = [
                    f"head IN ({source_placeholders})",
                    f"tail IN ({target_placeholders})",
                ]
                params: list[object] = list(source_batch) + list(target_batch)
                if relation_ids:
                    relation_placeholders = ", ".join("?" for _ in relation_ids)
                    where_parts.append(f"relation IN ({relation_placeholders})")
                    params.extend(relation_ids)
                query = f"""
                    SELECT head, relation, tail
                    FROM {table_name}
                    WHERE {' AND '.join(where_parts)}
                    ORDER BY head, relation, tail
                    LIMIT ?
                """
                params.append(effective_limit)
                for row in connection.execute(query, tuple(params)).fetchall():
                    rows.append(
                        {
                            "source_id": str(row["head"]),
                            "relation_id": str(row["relation"]),
                            "target_id": str(row["tail"]),
                        }
                    )
                    if len(rows) >= effective_limit:
                        return rows
        return rows

    def find_two_hop_paths_between(
        self,
        source_ids: list[str],
        target_ids: list[str],
        relation_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """Return exact forward two-hop source->mid->target paths for entity pairs."""
        unique_source_ids = list(dict.fromkeys(node_id for node_id in source_ids if node_id))
        unique_target_ids = list(dict.fromkeys(node_id for node_id in target_ids if node_id))
        if not unique_source_ids or not unique_target_ids:
            return []
        effective_limit = max(1, int(limit))
        connection = self.index_connection or self.connection
        table_name = "edges" if self.index_connection is not None else "triples"
        tail_kind_expr_1 = "e1.tail_kind" if self.index_connection is not None or self._has_tail_kind else "'id'"
        tail_kind_expr_2 = "e2.tail_kind" if self.index_connection is not None or self._has_tail_kind else "'id'"
        rows: list[dict[str, str]] = []
        for source_start in range(0, len(unique_source_ids), _SQLITE_PARAM_BATCH):
            source_batch = unique_source_ids[source_start : source_start + _SQLITE_PARAM_BATCH]
            source_placeholders = ", ".join("?" for _ in source_batch)
            for target_start in range(0, len(unique_target_ids), _SQLITE_PARAM_BATCH):
                target_batch = unique_target_ids[target_start : target_start + _SQLITE_PARAM_BATCH]
                target_placeholders = ", ".join("?" for _ in target_batch)
                where_parts = [
                    f"e1.head IN ({source_placeholders})",
                    f"e2.tail IN ({target_placeholders})",
                    f"{tail_kind_expr_1} = 'id'",
                    f"{tail_kind_expr_2} = 'id'",
                ]
                params: list[object] = list(source_batch) + list(target_batch)
                if relation_ids:
                    relation_placeholders = ", ".join("?" for _ in relation_ids)
                    where_parts.append(f"e1.relation IN ({relation_placeholders})")
                    where_parts.append(f"e2.relation IN ({relation_placeholders})")
                    params.extend(relation_ids)
                    params.extend(relation_ids)
                query = f"""
                    SELECT
                        e1.head AS source_id,
                        e1.relation AS first_relation_id,
                        e1.tail AS mid_id,
                        e2.relation AS second_relation_id,
                        e2.tail AS target_id
                    FROM {table_name} e1
                    JOIN {table_name} e2
                      ON e1.tail = e2.head
                    WHERE {' AND '.join(where_parts)}
                    ORDER BY source_id, mid_id, target_id, first_relation_id, second_relation_id
                    LIMIT ?
                """
                params.append(effective_limit)
                for row in connection.execute(query, tuple(params)).fetchall():
                    rows.append(
                        {
                            "source_id": str(row["source_id"]),
                            "first_relation_id": str(row["first_relation_id"]),
                            "mid_id": str(row["mid_id"]),
                            "second_relation_id": str(row["second_relation_id"]),
                            "target_id": str(row["target_id"]),
                        }
                    )
                    if len(rows) >= effective_limit:
                        return rows
        return rows

    def collect_related_entities(
        self,
        source_ids: list[str],
        relation_ids: list[str],
        direction: str = "forward",
        limit: int = 100,
        expected_answer_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Collect entity-valued relation targets for one source frontier."""
        unique_source_ids = list(dict.fromkeys(node_id for node_id in source_ids if node_id))
        unique_relation_ids = list(dict.fromkeys(relation_id for relation_id in relation_ids if relation_id))
        if not unique_source_ids or not unique_relation_ids or direction not in {"forward", "reverse"}:
            return []
        connection = self.index_connection or self.connection
        table_name = "edges" if self.index_connection is not None else "triples"
        tail_kind_select = "tail_kind" if self.index_connection is not None or self._has_tail_kind else "'id' AS tail_kind"
        id_only_clause = "tail_kind = 'id'" if self.index_connection is not None or self._has_tail_kind else "1=1"
        effective_limit = max(1, int(limit))
        rows: list[dict[str, Any]] = []
        for source_start in range(0, len(unique_source_ids), _SQLITE_PARAM_BATCH):
            source_batch = unique_source_ids[source_start : source_start + _SQLITE_PARAM_BATCH]
            source_placeholders = ", ".join("?" for _ in source_batch)
            relation_placeholders = ", ".join("?" for _ in unique_relation_ids)
            if direction == "forward":
                query = f"""
                    SELECT head, relation, tail, {tail_kind_select}
                    FROM {table_name}
                    WHERE head IN ({source_placeholders})
                      AND relation IN ({relation_placeholders})
                      AND {id_only_clause}
                    ORDER BY head, relation, tail
                    LIMIT ?
                """
                params: list[object] = [*source_batch, *unique_relation_ids, effective_limit]
                batch_rows = connection.execute(query, tuple(params)).fetchall()
                for row in batch_rows:
                    rows.append(
                        {
                            "source_entity_id": str(row["head"]),
                            "source_label": self.get_entity_display_name(str(row["head"])),
                            "relation_id": str(row["relation"]),
                            "relation_name": self.get_relation_display_name(str(row["relation"])),
                            "target_entity_id": str(row["tail"]),
                            "target_label": self.get_entity_display_name(str(row["tail"])),
                        }
                    )
            else:
                query = f"""
                    SELECT head, relation, tail, {tail_kind_select}
                    FROM {table_name}
                    WHERE tail IN ({source_placeholders})
                      AND relation IN ({relation_placeholders})
                      AND {id_only_clause}
                    ORDER BY tail, relation, head
                    LIMIT ?
                """
                params = [*source_batch, *unique_relation_ids, effective_limit]
                batch_rows = connection.execute(query, tuple(params)).fetchall()
                for row in batch_rows:
                    rows.append(
                        {
                            "source_entity_id": str(row["tail"]),
                            "source_label": self.get_entity_display_name(str(row["tail"])),
                            "relation_id": str(row["relation"]),
                            "relation_name": self.get_relation_display_name(str(row["relation"])),
                            "target_entity_id": str(row["head"]),
                            "target_label": self.get_entity_display_name(str(row["head"])),
                        }
                    )
            if len(rows) >= effective_limit:
                break
        if not rows:
            return rows
        metadata = self.get_nodes_metadata_batched([str(item["target_entity_id"]) for item in rows])
        expected_type_norm = normalize_text(expected_answer_type or "")
        filtered: list[dict[str, Any]] = []
        for item in rows:
            target_id = str(item["target_entity_id"])
            node_metadata = metadata.get(target_id)
            target_types = list(node_metadata.types) if node_metadata is not None else []
            item["target_types"] = target_types
            if expected_type_norm:
                haystack = normalize_text(" ".join([str(item["target_label"]), *target_types]))
                if expected_type_norm not in haystack:
                    continue
            filtered.append(item)
        return filtered[:effective_limit]

    def collect_related_literals(
        self,
        source_ids: list[str],
        relation_ids: list[str],
        direction: str = "forward",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Collect literal-valued relation targets for one source frontier."""
        unique_source_ids = list(dict.fromkeys(node_id for node_id in source_ids if node_id))
        unique_relation_ids = list(dict.fromkeys(relation_id for relation_id in relation_ids if relation_id))
        if not unique_source_ids or not unique_relation_ids or direction != "forward":
            return []
        connection = self.index_connection or self.connection
        table_name = "edges" if self.index_connection is not None else "triples"
        tail_kind_select = "tail_kind" if self.index_connection is not None or self._has_tail_kind else "'literal' AS tail_kind"
        literal_only_clause = "tail_kind = 'literal'" if self.index_connection is not None or self._has_tail_kind else "1=1"
        effective_limit = max(1, int(limit))
        rows: list[dict[str, Any]] = []
        for source_start in range(0, len(unique_source_ids), _SQLITE_PARAM_BATCH):
            source_batch = unique_source_ids[source_start : source_start + _SQLITE_PARAM_BATCH]
            source_placeholders = ", ".join("?" for _ in source_batch)
            relation_placeholders = ", ".join("?" for _ in unique_relation_ids)
            query = f"""
                SELECT head, relation, tail, {tail_kind_select}
                FROM {table_name}
                WHERE head IN ({source_placeholders})
                  AND relation IN ({relation_placeholders})
                  AND {literal_only_clause}
                ORDER BY head, relation, tail
                LIMIT ?
            """
            params: list[object] = [*source_batch, *unique_relation_ids, effective_limit]
            batch_rows = connection.execute(query, tuple(params)).fetchall()
            for row in batch_rows:
                rows.append(
                    {
                        "source_entity_id": str(row["head"]),
                        "source_label": self.get_entity_display_name(str(row["head"])),
                        "relation_id": str(row["relation"]),
                        "relation_name": self.get_relation_display_name(str(row["relation"])),
                        "value": str(row["tail"]),
                    }
                )
            if len(rows) >= effective_limit:
                break
        return rows[:effective_limit]

    def rank_entities_by_numeric_attribute(
        self,
        source_ids: list[str],
        relation_ids: list[str],
        direction: str = "forward",
        operation: str = "argmax",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Collect literal numeric attributes and rank them using SQLite ordering."""
        attribute_rows = self.collect_related_literals(
            source_ids=source_ids,
            relation_ids=relation_ids,
            direction=direction,
            limit=limit,
        )
        parsed_rows: list[dict[str, Any]] = []
        for row in attribute_rows:
            value_text = str(row.get("value") or "").strip()
            try:
                numeric_value = float(value_text.replace(",", ""))
            except ValueError:
                continue
            parsed_rows.append({**row, "numeric_value": numeric_value})
        reverse = operation != "argmin"
        return sorted(
            parsed_rows,
            key=lambda item: (float(item.get("numeric_value", 0.0)), str(item.get("source_entity_id", ""))),
            reverse=reverse,
        )[: max(1, int(limit))]

    def expand_relation_candidates_for_aggregate(
        self,
        seed_ids: list[str],
        expected_answer_type: str = "",
        include_literals: bool = True,
    ) -> list[dict[str, Any]]:
        """Expose frontier relation candidates in one aggregate-planning-friendly shape."""
        candidates = self.get_frontier_relations(
            node_ids=seed_ids,
            include_reverse=True,
            include_literals=include_literals,
        )
        expected_type_norm = normalize_text(expected_answer_type or "")
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            relation_text = normalize_text(
                " ".join(
                    [
                        candidate.relation,
                        candidate.relation_name,
                        *candidate.sample_target_types,
                        *candidate.sample_target_names,
                    ]
                )
            )
            score = float(candidate.support_ratio)
            if expected_type_norm and expected_type_norm in relation_text:
                score += 1.0
            if any(token in relation_text for token in ("population", "date", "time", "number", "rate", "area", "gdp")):
                score += 0.5
            rows.append(
                {
                    "relation_id": candidate.relation,
                    "relation_name": candidate.relation_name,
                    "direction": candidate.direction,
                    "sample_target_types": list(candidate.sample_target_types),
                    "sample_target_names": list(candidate.sample_target_names),
                    "support_ratio": float(candidate.support_ratio),
                    "score": score,
                }
            )
        return sorted(rows, key=lambda item: (-float(item["score"]), item["relation_id"]))

    def get_node_relations(
        self,
        node_id: str,
        include_reverse: bool = True,
        include_literals: bool = True,
    ) -> list[RelationCandidate]:
        """Return available relations for one node without first materializing full neighbors."""
        if not node_id:
            return []
        if self.index_connection is not None:
            return self.get_frontier_relations(
                [node_id],
                include_reverse=include_reverse,
                include_literals=include_literals,
            )
        return self._get_node_relations_fallback(
            node_id=node_id,
            include_reverse=include_reverse,
            include_literals=include_literals,
        )

    def get_frontier_relations(
        self,
        node_ids: list[str],
        include_reverse: bool = True,
        include_literals: bool = True,
    ) -> list[RelationCandidate]:
        """Aggregate available relations over a frontier directly from the runtime edge index."""
        unique_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not unique_node_ids:
            return []
        if self.index_connection is None:
            merged: dict[tuple[str, str], RelationCandidate] = {}
            for node_id in unique_node_ids:
                for candidate in self._get_node_relations_fallback(
                    node_id=node_id,
                    include_reverse=include_reverse,
                    include_literals=include_literals,
                ):
                    key = (candidate.relation, candidate.direction)
                    existing = merged.get(key)
                    if existing is None:
                        merged[key] = candidate
                    else:
                        support = existing.supporting_node_count + candidate.supporting_node_count
                        total = existing.total_neighbor_count + candidate.total_neighbor_count
                        merged[key] = RelationCandidate(
                            relation=existing.relation,
                            relation_name=existing.relation_name,
                            direction=existing.direction,
                            supporting_node_count=support,
                            total_neighbor_count=total,
                            support_ratio=support / max(len(unique_node_ids), 1),
                            global_frequency=max(existing.global_frequency, candidate.global_frequency),
                            sample_target_types=existing.sample_target_types or candidate.sample_target_types,
                            sample_target_names=existing.sample_target_names or candidate.sample_target_names,
                        )
            return sorted(
                merged.values(),
                key=lambda item: (-item.support_ratio, -item.supporting_node_count, item.relation, item.direction),
            )

        aggregation: dict[tuple[str, str], dict[str, Any]] = {}
        metadata_cache: dict[str, NodeMetadata] = {}
        for start in range(0, len(unique_node_ids), _SQLITE_PARAM_BATCH):
            batch = unique_node_ids[start : start + _SQLITE_PARAM_BATCH]
            placeholders = ", ".join("?" for _ in batch)
            forward_where = [f"head IN ({placeholders})"]
            forward_params: list[object] = list(batch)
            if not include_literals:
                forward_where.append("tail_kind = 'id'")
            forward_rows = self.index_connection.execute(
                f"""
                SELECT head, relation, tail, tail_kind
                FROM edges
                WHERE {' AND '.join(forward_where)}
                """,
                tuple(forward_params),
            ).fetchall()
            self._accumulate_relation_rows(
                aggregation=aggregation,
                rows=forward_rows,
                direction="forward",
                support_key="head",
                target_key="tail",
                metadata_cache=metadata_cache,
            )
            if include_reverse:
                reverse_rows = self.index_connection.execute(
                    f"""
                    SELECT tail, relation, head, tail_kind
                    FROM edges
                    WHERE tail IN ({placeholders})
                      AND tail_kind = 'id'
                    """,
                    tuple(batch),
                ).fetchall()
                self._accumulate_relation_rows(
                    aggregation=aggregation,
                    rows=reverse_rows,
                    direction="reverse",
                    support_key="tail",
                    target_key="head",
                    metadata_cache=metadata_cache,
                )
        return self._finalize_relation_candidates(aggregation, frontier_size=len(unique_node_ids))

    def expand_frontier_by_relation(
        self,
        node_ids: list[str],
        relation: str,
        direction: str,
        parent_evidence_paths: dict[str, tuple[EvidenceEdge, ...]] | None = None,
        limit_per_source: int = 8,
        total_limit: int = 100,
    ) -> FrontierExpansion:
        """Expand a frontier by one exact relation action while preserving evidence paths."""
        unique_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if direction not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported direction: {direction}")
        if not unique_node_ids or not relation:
            return FrontierExpansion(next_nodes=(), evidence_paths={}, source_node_count=0, edge_count=0, truncated=False)
        limit_per_source = max(1, int(limit_per_source))
        effective_total_limit = max(1, int(total_limit))
        parent_evidence_paths = parent_evidence_paths or {}
        next_nodes: list[str] = []
        evidence_paths: dict[str, tuple[EvidenceEdge, ...]] = {}
        edge_count = 0
        truncated = False
        if self.index_connection is not None:
            source_rows = self._query_relation_expansion_rows_batched(
                source_ids=unique_node_ids,
                relation=relation,
                direction=direction,
                limit_per_source=limit_per_source,
            )
            source_totals = {
                source_id: rows[0]["source_total"] if rows else 0
                for source_id, rows in source_rows.items()
            }
        else:
            source_rows = {}
            source_totals = {}
            for source_id in unique_node_ids:
                rows = self._query_relation_expansion_rows(
                    source_id=source_id,
                    relation=relation,
                    direction=direction,
                    limit=limit_per_source,
                )
                source_rows[source_id] = rows
                source_totals[source_id] = (
                    self._count_relation_expansion_edges(
                        source_id=source_id,
                        relation=relation,
                        direction=direction,
                    )
                    if rows
                    else 0
                )

        for source_id in unique_node_ids:
            if len(next_nodes) >= effective_total_limit:
                truncated = True
                break
            rows = source_rows.get(source_id, [])
            kept_for_source = 0
            for row in rows:
                if len(next_nodes) >= effective_total_limit:
                    truncated = True
                    break
                target_id = str(row["tail"] if direction == "forward" else row["head"])
                if not target_id:
                    continue
                edge_count += 1
                kept_for_source += 1
                if target_id not in next_nodes:
                    next_nodes.append(target_id)
                if target_id not in evidence_paths:
                    parent_path = parent_evidence_paths.get(source_id, ())
                    evidence_paths[target_id] = parent_path + (
                        EvidenceEdge(
                            source_id=source_id,
                            relation=relation,
                            direction=direction,
                            target_id=target_id,
                        ),
                    )
            if kept_for_source >= limit_per_source and source_totals.get(source_id, 0) > limit_per_source:
                truncated = True
            if len(next_nodes) >= effective_total_limit:
                truncated = True
                break
        return FrontierExpansion(
            next_nodes=tuple(next_nodes),
            evidence_paths=evidence_paths,
            source_node_count=len(unique_node_ids),
            edge_count=edge_count,
            truncated=truncated,
        )

    def resolve_entity_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
        """Use the runtime alias index when available, then fall back to SQLite matching."""
        if self.index_connection is None:
            return super().resolve_entity_candidates(names, top_k=top_k)
        results: list[dict[str, object]] = []
        for mention in names:
            mention_norm = normalize_text(mention)
            if mention_norm:
                rows = self.index_connection.execute(
                    """
                    SELECT mid, entity_name, alias_source
                    FROM entity_alias
                    WHERE alias_norm = ?
                    LIMIT ?
                    """,
                    (mention_norm, max(1, top_k)),
                ).fetchall()
                for row in rows:
                    results.append(
                        {
                            "mid": str(row["mid"]),
                            "name": str(row["entity_name"] or self.get_entity_display_name(str(row["mid"]))),
                            "aliases": "",
                            "mention": mention,
                            "match_type": f"{row['alias_source']}_exact",
                            "score": 95.0 if str(row["alias_source"]) == "name" else 85.0,
                        }
                    )
        if results:
            deduped: dict[str, dict[str, object]] = {}
            for row in results:
                current = deduped.get(str(row["mid"]))
                if current is None or float(row["score"]) > float(current["score"]):
                    deduped[str(row["mid"])] = row
            return sorted(deduped.values(), key=lambda item: (-float(item["score"]), str(item["mid"])))[:top_k]
        return super().resolve_entity_candidates(names, top_k=top_k)

    def resolve_relation_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
        """Use the runtime relation lookup when available, then fall back to SQLite matching."""
        if self.index_connection is None:
            return super().resolve_relation_candidates(names, top_k=top_k)
        results: list[dict[str, object]] = []
        for hint in names:
            hint_norm = normalize_text(hint)
            if not hint_norm:
                continue
            rows = self.index_connection.execute(
                """
                SELECT relation, relation_name, lookup_source
                FROM relation_lookup
                WHERE lookup_norm = ?
                LIMIT ?
                """,
                (hint_norm, max(1, top_k)),
            ).fetchall()
            for row in rows:
                results.append(
                    {
                        "relation": str(row["relation"]),
                        "relation_name": self.get_relation_display_name(str(row["relation"])),
                        "hint": hint,
                        "match_type": f"{row['lookup_source']}_exact",
                        "score": 95.0 if str(row["lookup_source"]) == "relation_name" else 90.0,
                    }
                )
        if results:
            deduped: dict[str, dict[str, object]] = {}
            for row in results:
                current = deduped.get(str(row["relation"]))
                if current is None or float(row["score"]) > float(current["score"]):
                    deduped[str(row["relation"])] = row
            return sorted(
                deduped.values(),
                key=lambda item: (-float(item["score"]), str(item["relation"])),
            )[:top_k]
        return super().resolve_relation_candidates(names, top_k=top_k)

    def _accumulate_relation_rows(
        self,
        aggregation: dict[tuple[str, str], dict[str, Any]],
        rows: list[sqlite3.Row],
        direction: str,
        support_key: str,
        target_key: str,
        metadata_cache: dict[str, NodeMetadata],
    ) -> None:
        target_ids = [
            str(row[target_key])
            for row in rows
            if str(row[target_key] or "").startswith(("m.", "g."))
        ]
        if target_ids:
            metadata_cache.update(self.get_nodes_metadata_batched(target_ids))
        for row in rows:
            relation = str(row["relation"])
            key = (relation, direction)
            bucket = aggregation.setdefault(
                key,
                {
                    "relation": relation,
                    "relation_name": self.get_relation_display_name(relation),
                    "direction": direction,
                    "supporting_nodes": set(),
                    "total_neighbor_count": 0,
                    "sample_target_names": [],
                    "sample_target_types": [],
                },
            )
            support_node = str(row[support_key] or "")
            target_id = str(row[target_key] or "")
            bucket["supporting_nodes"].add(support_node)
            bucket["total_neighbor_count"] += 1
            if len(bucket["sample_target_names"]) < 3:
                if str(row["tail_kind"] or "id").strip().lower() == "literal" and direction == "forward":
                    bucket["sample_target_names"].append(target_id)
                else:
                    metadata = metadata_cache.get(target_id)
                    if metadata is not None:
                        if metadata.name and metadata.name not in bucket["sample_target_names"]:
                            bucket["sample_target_names"].append(metadata.name)
                        for node_type in metadata.types:
                            if node_type and node_type not in bucket["sample_target_types"]:
                                bucket["sample_target_types"].append(node_type)
                            if len(bucket["sample_target_types"]) >= 3:
                                break

    def _finalize_relation_candidates(
        self,
        aggregation: dict[tuple[str, str], dict[str, Any]],
        frontier_size: int,
    ) -> list[RelationCandidate]:
        candidates: list[RelationCandidate] = []
        for (relation, direction), payload in aggregation.items():
            supporting_node_count = len(payload["supporting_nodes"])
            candidates.append(
                RelationCandidate(
                    relation=relation,
                    relation_name=str(payload["relation_name"]),
                    direction=direction,
                    supporting_node_count=supporting_node_count,
                    total_neighbor_count=int(payload["total_neighbor_count"]),
                    support_ratio=supporting_node_count / max(frontier_size, 1),
                    global_frequency=self.get_relation_frequency(relation),
                    sample_target_types=tuple(payload["sample_target_types"][:3]),
                    sample_target_names=tuple(payload["sample_target_names"][:3]),
                )
            )
        candidates.sort(
            key=lambda item: (-item.support_ratio, -item.supporting_node_count, item.relation, item.direction)
        )
        return candidates

    def _get_node_relations_fallback(
        self,
        node_id: str,
        include_reverse: bool,
        include_literals: bool,
    ) -> list[RelationCandidate]:
        aggregation: dict[tuple[str, str], dict[str, Any]] = {}
        forward_where = ["head = ?"]
        forward_params: list[object] = [node_id]
        if self._has_tail_kind and not include_literals:
            forward_where.append("tail_kind = 'id'")
        for row in self.connection.execute(
            f"""
            SELECT head, relation, tail, {('tail_kind' if self._has_tail_kind else "'id' AS tail_kind")}
            FROM triples
            WHERE {' AND '.join(forward_where)}
            """,
            tuple(forward_params),
        ).fetchall():
            self._accumulate_relation_rows(
                aggregation=aggregation,
                rows=[row],
                direction="forward",
                support_key="head",
                target_key="tail",
                metadata_cache={},
            )
        if include_reverse:
            for row in self.connection.execute(
                f"""
                SELECT tail, relation, head, {('tail_kind' if self._has_tail_kind else "'id' AS tail_kind")}
                FROM triples
                WHERE tail = ?
                """,
                (node_id,),
            ).fetchall():
                if str(row["tail_kind"] or "id").strip().lower() != "id":
                    continue
                self._accumulate_relation_rows(
                    aggregation=aggregation,
                    rows=[row],
                    direction="reverse",
                    support_key="tail",
                    target_key="head",
                    metadata_cache={},
                )
        return self._finalize_relation_candidates(aggregation, frontier_size=1)

    def _query_relation_expansion_rows(
        self,
        source_id: str,
        relation: str,
        direction: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        connection = self.index_connection or self.connection
        if direction == "forward":
            if self.index_connection is not None:
                query = """
                    SELECT head, relation, tail, tail_kind
                    FROM edges
                    WHERE head = ? AND relation = ?
                    ORDER BY tail
                    LIMIT ?
                """
            else:
                tail_kind_expr = "tail_kind" if self._has_tail_kind else "'id' AS tail_kind"
                query = f"""
                    SELECT head, relation, tail, {tail_kind_expr}
                    FROM triples
                    WHERE head = ? AND relation = ?
                    ORDER BY tail
                    LIMIT ?
                """
            return connection.execute(query, (source_id, relation, limit)).fetchall()
        if self.index_connection is not None:
            query = """
                SELECT head, relation, tail, tail_kind
                FROM edges
                WHERE tail = ? AND relation = ? AND tail_kind = 'id'
                ORDER BY head
                LIMIT ?
            """
        else:
            tail_kind_expr = "tail_kind" if self._has_tail_kind else "'id' AS tail_kind"
            query = f"""
                SELECT head, relation, tail, {tail_kind_expr}
                FROM triples
                WHERE tail = ? AND relation = ?
                ORDER BY head
                LIMIT ?
            """
        rows = connection.execute(query, (source_id, relation, limit)).fetchall()
        if self.index_connection is None:
            rows = [row for row in rows if str(row["tail_kind"] or "id").strip().lower() == "id"]
        return rows

    def _query_relation_expansion_rows_batched(
        self,
        source_ids: list[str],
        relation: str,
        direction: str,
        limit_per_source: int,
    ) -> dict[str, list[sqlite3.Row]]:
        """Return per-source expansion rows with one indexed SQL query per source batch.

        This preserves the public expand semantics by only changing how rows are fetched.
        Callers still consume rows in the original source order and enforce total_limit in Python.
        """
        if self.index_connection is None:
            return {}
        unique_source_ids = list(dict.fromkeys(source_id for source_id in source_ids if source_id))
        if not unique_source_ids:
            return {}
        if direction not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported direction: {direction}")
        partition_column = "head" if direction == "forward" else "tail"
        order_column = "tail" if direction == "forward" else "head"
        results: dict[str, list[sqlite3.Row]] = {source_id: [] for source_id in unique_source_ids}
        for start in range(0, len(unique_source_ids), _SQLITE_PARAM_BATCH):
            batch = unique_source_ids[start : start + _SQLITE_PARAM_BATCH]
            placeholders = ", ".join("?" for _ in batch)
            where_parts = [f"{partition_column} IN ({placeholders})", "relation = ?"]
            params: list[object] = list(batch)
            params.append(relation)
            if direction == "reverse":
                where_parts.append("tail_kind = 'id'")
            query = f"""
                SELECT head, relation, tail, tail_kind, source_total
                FROM (
                    SELECT
                        head,
                        relation,
                        tail,
                        tail_kind,
                        ROW_NUMBER() OVER (
                            PARTITION BY {partition_column}
                            ORDER BY {order_column}
                        ) AS rn,
                        COUNT(*) OVER (
                            PARTITION BY {partition_column}
                        ) AS source_total
                    FROM edges
                    WHERE {' AND '.join(where_parts)}
                ) ranked
                WHERE rn <= ?
                ORDER BY {partition_column}, {order_column}
            """
            params.append(limit_per_source)
            for row in self.index_connection.execute(query, tuple(params)).fetchall():
                source_id = str(row[partition_column] or "")
                if source_id:
                    results.setdefault(source_id, []).append(row)
        return results

    def _count_relation_expansion_edges(
        self,
        source_id: str,
        relation: str,
        direction: str,
    ) -> int:
        connection = self.index_connection or self.connection
        if direction == "forward":
            row = connection.execute(
                f"SELECT COUNT(1) AS count FROM {('edges' if self.index_connection is not None else 'triples')} WHERE head = ? AND relation = ?",
                (source_id, relation),
            ).fetchone()
        else:
            table = "edges" if self.index_connection is not None else "triples"
            row = connection.execute(
                f"SELECT COUNT(1) AS count FROM {table} WHERE tail = ? AND relation = ?",
                (source_id, relation),
            ).fetchone()
            if self.index_connection is None and row is not None and self._has_tail_kind:
                row = connection.execute(
                    "SELECT COUNT(1) AS count FROM triples WHERE tail = ? AND relation = ? AND tail_kind = 'id'",
                    (source_id, relation),
                ).fetchone()
        return int(row["count"] or 0) if row is not None else 0

    def _query_index_neighbors(
        self,
        direction: str,
        node_id: str,
        relation_filter: list[str],
        sql_limit: int | None,
        reversed_edge: bool,
        strict_relation_filter: bool,
    ) -> list[ExternalGraphNeighbor]:
        if self.index_connection is None:
            return []
        is_forward = direction == "forward"
        column = "head" if is_forward else "tail"
        where_clauses = [f"{column} = ?"]
        params: list[object] = [node_id]
        order_by = "relation, tail, head" if is_forward else "relation, head, tail"
        if relation_filter and strict_relation_filter:
            placeholders = ", ".join("?" for _ in relation_filter)
            where_clauses.append(f"relation IN ({placeholders})")
            params.extend(relation_filter)
        elif relation_filter:
            placeholders = ", ".join("?" for _ in relation_filter)
            order_by = f"CASE WHEN relation IN ({placeholders}) THEN 0 ELSE 1 END, {order_by}"
            params.extend(relation_filter)

        query = f"SELECT * FROM edges WHERE {' AND '.join(where_clauses)} ORDER BY {order_by}"
        if sql_limit is not None:
            query += " LIMIT ?"
            params.append(sql_limit)
        rows = self.index_connection.execute(query, tuple(params)).fetchall()
        neighbors: list[ExternalGraphNeighbor] = []
        for row in rows:
            relation = str(row["relation"])
            relation_name = self.get_relation_display_name(relation)
            if is_forward:
                head_id = str(row["head"])
                tail_id = str(row["tail"])
                tail_kind = str(row["tail_kind"] or "id").strip().lower()
                tail_name = (
                    tail_id
                    if tail_kind == "literal"
                    else self.get_entity_display_name(tail_id)
                )
                head_name = self.get_entity_display_name(head_id)
                source_id, neighbor_id = head_id, tail_id
            else:
                tail_id = str(row["tail"])
                head_id = str(row["head"])
                head_name = self.get_entity_display_name(head_id)
                tail_name = self.get_entity_display_name(tail_id)
                source_id, neighbor_id = tail_id, head_id
            neighbors.append(
                ExternalGraphNeighbor(
                    source_id=source_id,
                    neighbor_id=neighbor_id,
                    triple=Triple(head=head_name, relation=relation, tail=tail_name),
                    relation_name=relation_name,
                    reversed=reversed_edge,
                )
            )
        return neighbors
