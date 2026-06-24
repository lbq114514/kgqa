from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.sqlite_graph_api import build_sqlite_graph_db_from_processed_freebase
from kgqa.retrieval import (
    BeamSearchSearcher,
    ConstrainedBFSSearcher,
    HybridSearcher,
    IndexedSQLiteGraphBackend,
    PathCandidateScorer,
    PathReranker,
    PathSelector,
    SearchRequest,
    SearchScoringContext,
    TwoHopExpansionSearcher,
    build_indexed_sqlite_runtime_index,
)
from kgqa.utils.types import ReasoningPath, Triple


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_runtime_fixture(tmp_path: Path) -> tuple[Path, Path]:
    entities_csv = tmp_path / "entities.csv"
    triples_csv = tmp_path / "triplets.csv"
    db_path = tmp_path / "graph.sqlite"
    index_dir = tmp_path / "runtime_index"
    _write_csv(
        entities_csv,
        ["id", "name", "aliases", "types", "is_cvt"],
        [
            {"id": "m.nick", "name": "Nick Clegg", "aliases": "Nicholas William Peter Clegg", "types": "people.person", "is_cvt": "0"},
            {"id": "m.education", "name": "education record", "aliases": "", "types": "education.education", "is_cvt": "1"},
            {"id": "m.social", "name": "Social anthropology", "aliases": "social anthropology", "types": "education.field_of_study", "is_cvt": "0"},
            {"id": "m.cambridge", "name": "University of Cambridge", "aliases": "cambridge", "types": "education.university", "is_cvt": "0"},
            {"id": "m.year", "name": "", "aliases": "", "types": "type.datetime", "is_cvt": "0"},
        ],
    )
    _write_csv(
        triples_csv,
        ["head", "head_name", "relation", "tail", "tail_name", "tail_kind"],
        [
            {"head": "m.nick", "head_name": "Nick Clegg", "relation": "people.person.education", "tail": "m.education", "tail_name": "education record", "tail_kind": "id"},
            {"head": "m.education", "head_name": "education record", "relation": "education.education.major_field_of_study", "tail": "m.social", "tail_name": "Social anthropology", "tail_kind": "id"},
            {"head": "m.education", "head_name": "education record", "relation": "education.education.institution", "tail": "m.cambridge", "tail_name": "University of Cambridge", "tail_kind": "id"},
            {"head": "m.nick", "head_name": "Nick Clegg", "relation": "people.person.date_of_birth", "tail": "1967-01-07", "tail_name": "", "tail_kind": "literal"},
        ],
    )
    build_sqlite_graph_db_from_processed_freebase(entities_csv, triples_csv, db_path, overwrite=True, fast_mode=True)
    build_indexed_sqlite_runtime_index(db_path=db_path, index_dir=index_dir, overwrite=True, high_degree_threshold=2)
    return db_path, index_dir


def test_build_indexed_sqlite_runtime_index_creates_lookup_tables(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    index_db = index_dir / "runtime_index.sqlite"
    assert index_db.exists()
    connection = sqlite3.connect(str(index_db))
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {"edges", "entity_alias", "relation_lookup", "node_stats", "relation_stats"} <= tables
        edge_columns = [row[1] for row in connection.execute("PRAGMA table_info(edges)").fetchall()]
        assert edge_columns == ["edge_id", "head", "relation", "tail", "tail_kind"]
        edge_row = connection.execute("SELECT edge_id FROM edges ORDER BY edge_id LIMIT 1").fetchone()
        assert isinstance(edge_row[0], int)
        stats_row = connection.execute("SELECT degree, high_degree FROM node_stats WHERE mid = 'm.education'").fetchone()
        assert stats_row == (3, 1)
    finally:
        connection.close()


def test_runtime_index_builder_supports_staging_dir(tmp_path: Path) -> None:
    db_path, _ = _build_runtime_fixture(tmp_path)
    index_dir = tmp_path / "runtime_index_staged"
    staging_dir = tmp_path / "ssd_tmp"
    build_indexed_sqlite_runtime_index(
        db_path=db_path,
        index_dir=index_dir,
        overwrite=True,
        high_degree_threshold=2,
        staging_dir=staging_dir,
    )
    assert (index_dir / "runtime_index.sqlite").exists()


def test_indexed_sqlite_backend_resolves_candidates_and_neighbors(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=10)
    try:
        entity_candidates = backend.resolve_entity_candidates(["Nick Clegg"], top_k=3)
        relation_candidates = backend.resolve_relation_candidates(["major field of study"], top_k=3)
        neighbors = backend.get_neighbors("m.nick", include_reverse=True, relation_filter=[], limit=10)

        assert entity_candidates[0]["mid"] == "m.nick"
        assert relation_candidates[0]["relation"] == "education.education.major_field_of_study"
        assert any(neighbor.triple.tail == "1967-01-07" for neighbor in neighbors)
        assert backend.get_node_degree("m.education") == 3
        assert "education.education" in backend.get_node_types("m.education")
    finally:
        backend.close()


def test_searchers_and_ranker_return_reasonable_paths(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=10)
    scorer = PathCandidateScorer(backend)
    reranker = PathReranker(scorer)
    selector = PathSelector(max_paths=5)
    request = SearchRequest(
        seed_entity_ids=["m.nick"],
        relation_hints=["people.person.education", "education.education.major_field_of_study"],
        answer_type_hints=["field of study"],
        max_depth=3,
        beam_width=5,
        max_expansions=50,
        max_paths=10,
    )
    try:
        bfs_paths = ConstrainedBFSSearcher(backend).search(request)
        beam_paths = BeamSearchSearcher(backend, scorer).search(request)
        two_hop_paths = TwoHopExpansionSearcher(backend).search(request)
        hybrid_paths = HybridSearcher(backend, scorer, reranker, selector).search(request)

        assert bfs_paths
        assert beam_paths
        assert two_hop_paths
        assert hybrid_paths
        assert any(path.search_strategy == "beam" for path in beam_paths)
        assert any("Social anthropology" in path.text for path in hybrid_paths)
        assert any("major field of study" in path.text for path in hybrid_paths)
    finally:
        backend.close()


def test_beam_width_zero_removes_neighbor_fetch_cap(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=1)
    request = SearchRequest(
        seed_entity_ids=["m.nick"],
        relation_hints=[],
        max_depth=1,
        beam_width=0,
        max_expansions=50,
        max_paths=10,
    )
    try:
        bfs_paths = ConstrainedBFSSearcher(backend).search(request)
        relations = {path.triples[0].relation for path in bfs_paths if path.triples}
        assert "people.person.education" in relations
        assert "people.person.date_of_birth" in relations
    finally:
        backend.close()


def test_max_paths_zero_removes_path_cap(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=10)
    scorer = PathCandidateScorer(backend)
    reranker = PathReranker(scorer)
    selector = PathSelector(max_paths=0)
    request = SearchRequest(
        seed_entity_ids=["m.nick"],
        relation_hints=[],
        max_depth=1,
        beam_width=0,
        max_expansions=50,
        max_paths=0,
    )
    try:
        hybrid_paths = HybridSearcher(backend, scorer, reranker, selector).search(request)
        assert len(hybrid_paths) >= 2
        relations = {path.triples[0].relation for path in hybrid_paths if path.triples}
        assert "people.person.education" in relations
        assert "people.person.date_of_birth" in relations
    finally:
        backend.close()


def test_one_hop_hybrid_skips_two_hop_bootstrap(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=10)
    scorer = PathCandidateScorer(backend)
    reranker = PathReranker(scorer)
    selector = PathSelector(max_paths=0)
    request = SearchRequest(
        seed_entity_ids=["m.nick"],
        relation_hints=[],
        max_depth=1,
        beam_width=0,
        max_expansions=50,
        max_paths=0,
    )
    try:
        two_hop_paths = TwoHopExpansionSearcher(backend).search(request)
        hybrid_paths = HybridSearcher(backend, scorer, reranker, selector).search(request)
        assert two_hop_paths == []
        assert hybrid_paths
        assert all(len(path.triples) == 1 for path in hybrid_paths)
        assert all(path.search_strategy != "two_hop" for path in hybrid_paths)
    finally:
        backend.close()


def test_path_candidate_scorer_prefers_relation_hits(tmp_path: Path) -> None:
    db_path, index_dir = _build_runtime_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=10)
    scorer = PathCandidateScorer(backend)
    try:
        hit = ReasoningPath(
            triples=[
                Triple("Nick Clegg", "people.person.education", "education record"),
                Triple("education record", "education.education.major_field_of_study", "Social anthropology"),
            ],
            nodes=["Nick Clegg", "education record", "Social anthropology"],
            text="Nick Clegg -> people.person.education -> education record -> education.education.major_field_of_study -> Social anthropology",
            source_stage="retrieval_search",
            terminal_node_id="m.social",
            terminal_node_kind="id",
        )
        miss = ReasoningPath(
            triples=[Triple("Nick Clegg", "people.person.date_of_birth", "1967-01-07")],
            nodes=["Nick Clegg", "1967-01-07"],
            text="Nick Clegg -> people.person.date_of_birth -> 1967-01-07",
            source_stage="retrieval_search",
            terminal_node_id="1967-01-07",
            terminal_node_kind="literal",
        )
        context = SearchScoringContext(
            relation_hints=["education.education.major_field_of_study"],
            answer_type_hints=["field of study"],
        )
        assert scorer.score_path(hit, context).path_score > scorer.score_path(miss, context).path_score
    finally:
        backend.close()
