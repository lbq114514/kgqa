from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import kgqa.retrieval.backend as backend_module
from kgqa.kg.relation_beam import (
    BaseLLMRelationBeamAdapter,
    CVT_BUNDLE_SOURCE_STAGE,
    CandidateRerankContext,
    DeterministicMockRelationBeamLLM,
    EvidenceEdge,
    FrontierExpansion,
    NodeMetadata,
    RelationAction,
    RelationBeamConfig,
    RelationBeamState,
    RelationCandidate,
    RelationPathStep,
    RelationTypeBeamSearcher,
    SubquestionRelationExplorer,
    build_child_state,
    filter_relation_candidates,
    merge_same_relation_paths,
    prune_relation_beam,
    relation_path_signature,
)
from kgqa.kg.sqlite_graph_api import build_sqlite_graph_db_from_processed_freebase
from kgqa.llm.base import BaseLLM
from kgqa.retrieval import IndexedSQLiteGraphBackend, build_indexed_sqlite_runtime_index


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_relation_beam_fixture(tmp_path: Path) -> tuple[Path, Path]:
    entities_csv = tmp_path / "entities.csv"
    triples_csv = tmp_path / "triplets.csv"
    db_path = tmp_path / "graph.sqlite"
    index_dir = tmp_path / "runtime_index"
    entity_rows = [
        {"id": "m.person_a", "name": "Person A", "aliases": "", "types": "people.person", "is_cvt": "0"},
        {"id": "m.marriage", "name": "", "aliases": "", "types": "people.marriage", "is_cvt": "1"},
        {"id": "m.person_b", "name": "Person B", "aliases": "", "types": "people.person", "is_cvt": "0"},
        {"id": "m.city_c", "name": "City C", "aliases": "", "types": "location.citytown", "is_cvt": "0"},
        {"id": "m.profession_x", "name": "Lawyer", "aliases": "", "types": "people.profession", "is_cvt": "0"},
        {"id": "m.attraction_0", "name": "Opera House", "aliases": "", "types": "travel.tourist_attraction", "is_cvt": "0"},
        {"id": "m.attraction_1", "name": "City Park", "aliases": "", "types": "travel.tourist_attraction", "is_cvt": "0"},
        {"id": "m.attraction_2", "name": "Central Zoo", "aliases": "", "types": "travel.tourist_attraction", "is_cvt": "0"},
        {
            "id": "m.zz_museum",
            "name": "History Museum",
            "aliases": "",
            "types": "architecture.museum|travel.tourist_attraction",
            "is_cvt": "0",
        },
    ]
    for index in range(12):
        entity_rows.append(
            {
                "id": f"m.award_{index}",
                "name": f"Award {index}",
                "aliases": "",
                "types": "award.award_nomination",
                "is_cvt": "0",
            }
        )
    triple_rows = [
        {
            "head": "m.person_a",
            "head_name": "Person A",
            "relation": "people.person.spouse_s",
            "tail": "m.marriage",
            "tail_name": "",
            "tail_kind": "id",
        },
        {
            "head": "m.marriage",
            "head_name": "",
            "relation": "people.marriage.spouse",
            "tail": "m.person_a",
            "tail_name": "Person A",
            "tail_kind": "id",
        },
        {
            "head": "m.marriage",
            "head_name": "",
            "relation": "people.marriage.spouse",
            "tail": "m.person_b",
            "tail_name": "Person B",
            "tail_kind": "id",
        },
        {
            "head": "m.person_b",
            "head_name": "Person B",
            "relation": "people.person.place_of_birth",
            "tail": "m.city_c",
            "tail_name": "City C",
            "tail_kind": "id",
        },
        {
            "head": "m.person_a",
            "head_name": "Person A",
            "relation": "people.person.profession",
            "tail": "m.profession_x",
            "tail_name": "Lawyer",
            "tail_kind": "id",
        },
        {
            "head": "m.person_b",
            "head_name": "Person B",
            "relation": "type.object.name",
            "tail": "Person B",
            "tail_name": "",
            "tail_kind": "literal",
        },
        {
            "head": "m.city_c",
            "head_name": "City C",
            "relation": "travel.travel_destination.tourist_attractions",
            "tail": "m.attraction_0",
            "tail_name": "Opera House",
            "tail_kind": "id",
        },
        {
            "head": "m.city_c",
            "head_name": "City C",
            "relation": "travel.travel_destination.tourist_attractions",
            "tail": "m.attraction_1",
            "tail_name": "City Park",
            "tail_kind": "id",
        },
        {
            "head": "m.city_c",
            "head_name": "City C",
            "relation": "travel.travel_destination.tourist_attractions",
            "tail": "m.attraction_2",
            "tail_name": "Central Zoo",
            "tail_kind": "id",
        },
        {
            "head": "m.city_c",
            "head_name": "City C",
            "relation": "travel.travel_destination.tourist_attractions",
            "tail": "m.zz_museum",
            "tail_name": "History Museum",
            "tail_kind": "id",
        },
        {
            "head": "m.zz_museum",
            "head_name": "History Museum",
            "relation": "architecture.museum.established",
            "tail": "1900",
            "tail_name": "",
            "tail_kind": "literal",
        },
        {
            "head": "m.attraction_0",
            "head_name": "Opera House",
            "relation": "architecture.museum.established",
            "tail": "1800",
            "tail_name": "",
            "tail_kind": "literal",
        },
    ]
    for index in range(12):
        triple_rows.append(
            {
                "head": "m.person_a",
                "head_name": "Person A",
                "relation": "award.award_nomination",
                "tail": f"m.award_{index}",
                "tail_name": f"Award {index}",
                "tail_kind": "id",
            }
        )
    _write_csv(entities_csv, ["id", "name", "aliases", "types", "is_cvt"], entity_rows)
    _write_csv(
        triples_csv,
        ["head", "head_name", "relation", "tail", "tail_name", "tail_kind"],
        triple_rows,
    )
    build_sqlite_graph_db_from_processed_freebase(entities_csv, triples_csv, db_path, overwrite=True, fast_mode=True)
    build_indexed_sqlite_runtime_index(db_path=db_path, index_dir=index_dir, overwrite=True, high_degree_threshold=4)
    return db_path, index_dir


class InvalidRelationLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs: object) -> str:
        if "actions" in prompt:
            return '{"actions":[{"relation":"book.book.editions","direction":"forward","role":"answer","confidence":1.0,"expected_next_type":"city","reason":"bad","protect_for_next_hop":false}]}'
        return '{"states":[]}'


class ExplodingRelationLLM:
    def select_relation_actions(self, **kwargs: object) -> list[RelationAction]:
        raise RuntimeError("boom")

    def rerank_paths(self, **kwargs: object) -> dict[str, dict[str, object]]:
        raise RuntimeError("boom")


class CountingBatchLLM(BaseLLM):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return '{"actions":[{"relation":"rel_0","direction":"forward","role":"answer","confidence":0.9,"expected_next_type":"city","reason":"pick first","protect_for_next_hop":false}]}'


class MarriageRelationLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs: object) -> str:
        return '{"actions":[{"relation":"people.person.spouse_s","direction":"forward","role":"intermediate","confidence":0.95,"expected_next_type":"person","reason":"marriage cvt","protect_for_next_hop":true}]}'


def _base_state(
    state_id: str = "s0",
    path: tuple[RelationPathStep, ...] = (),
    protected_until_hop: int = 0,
    answer_score: float = 0.0,
    contains_answer_candidates: bool = False,
    semantic_score: float = 0.6,
) -> RelationBeamState:
    return RelationBeamState(
        state_id=state_id,
        frontier_nodes=("m.person_a",),
        relation_path=path,
        evidence_paths={"m.person_a": ()},
        source_nodes=("m.person_a",),
        hop=len(path),
        semantic_score=semantic_score,
        type_score=0.4,
        answer_score=answer_score,
        diversity_score=0.8,
        cost_penalty=0.1,
        protected_until_hop=protected_until_hop,
        contains_answer_candidates=contains_answer_candidates,
        path_reason=state_id,
        visited_relation_signatures=frozenset((step.relation, step.direction) for step in path),
    )


def test_get_node_relations_and_frontier_aggregation(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        node_relations = backend.get_node_relations("m.person_a", include_reverse=True, include_literals=True)
        relation_index = {(item.relation, item.direction): item for item in node_relations}
        assert ("people.person.spouse_s", "forward") in relation_index
        assert relation_index[("people.person.spouse_s", "forward")].supporting_node_count == 1
        assert relation_index[("award.award_nomination", "forward")].total_neighbor_count == 12

        frontier_relations = backend.get_frontier_relations(["m.person_a", "m.person_b"], include_reverse=True)
        frontier_index = {(item.relation, item.direction): item for item in frontier_relations}
        assert frontier_index[("people.person.place_of_birth", "forward")].support_ratio == 0.5
        assert frontier_index[("people.person.spouse_s", "forward")].support_ratio == 0.5
    finally:
        backend.close()


def test_get_frontier_relations_merges_batches_without_duplicate_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    monkeypatch.setattr(backend_module, "_SQLITE_PARAM_BATCH", 2)
    try:
        node_ids = ["m.person_a", "m.person_b", "m.person_a", "m.person_b"]
        relations = backend.get_frontier_relations(node_ids, include_reverse=False)
        relation_index = {(item.relation, item.direction): item for item in relations}
        assert relation_index[("people.person.spouse_s", "forward")].supporting_node_count == 1
        assert relation_index[("people.person.spouse_s", "forward")].support_ratio == 0.5
    finally:
        backend.close()


def test_get_node_edges_returns_precise_neighbors(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        edges = backend.get_node_edges("m.marriage", include_reverse=False, include_literals=True)
        relation_pairs = {(edge.triple.relation, edge.neighbor_id) for edge in edges}
        assert ("people.marriage.spouse", "m.person_a") in relation_pairs
        assert ("people.marriage.spouse", "m.person_b") in relation_pairs
    finally:
        backend.close()


def test_expand_frontier_by_relation_limits_and_evidence(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        expansion = backend.expand_frontier_by_relation(
            ["m.person_a"],
            relation="award.award_nomination",
            direction="forward",
            parent_evidence_paths={"m.person_a": ()},
            limit_per_source=3,
            total_limit=5,
        )
        assert len(expansion.next_nodes) == 3
        assert expansion.edge_count == 3
        assert expansion.truncated is True
        first_target = expansion.next_nodes[0]
        assert expansion.evidence_paths[first_target] == (
            EvidenceEdge(
                source_id="m.person_a",
                relation="award.award_nomination",
                direction="forward",
                target_id=first_target,
            ),
        )
    finally:
        backend.close()


def test_expand_frontier_reranks_typed_candidates_before_truncation(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        baseline = backend.expand_frontier_by_relation(
            ["m.city_c"],
            relation="travel.travel_destination.tourist_attractions",
            direction="forward",
            parent_evidence_paths={"m.city_c": ()},
            limit_per_source=2,
            total_limit=2,
        )
        assert baseline.next_nodes == ("m.attraction_0", "m.attraction_1")

        reranked = backend.expand_frontier_by_relation(
            ["m.city_c"],
            relation="travel.travel_destination.tourist_attractions",
            direction="forward",
            parent_evidence_paths={"m.city_c": ()},
            limit_per_source=2,
            total_limit=2,
            rerank_context=CandidateRerankContext(
                expected_answer_type="museum",
                expected_next_type="museum",
                relation_hint_ids=("architecture.museum.established",),
                subquestion="What museum was established latest?",
            ),
            overfetch_factor=3,
            max_overfetch=10,
            min_candidates_for_rerank=1,
        )
        assert reranked.next_nodes[0] == "m.zz_museum"
        assert reranked.evidence_paths["m.zz_museum"] == (
            EvidenceEdge(
                source_id="m.city_c",
                relation="travel.travel_destination.tourist_attractions",
                direction="forward",
                target_id="m.zz_museum",
            ),
        )
    finally:
        backend.close()


def test_expand_frontier_empty_rerank_context_preserves_order(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        expansion = backend.expand_frontier_by_relation(
            ["m.city_c"],
            relation="travel.travel_destination.tourist_attractions",
            direction="forward",
            parent_evidence_paths={"m.city_c": ()},
            limit_per_source=2,
            total_limit=2,
            rerank_context=CandidateRerankContext(),
            overfetch_factor=3,
            max_overfetch=10,
            min_candidates_for_rerank=1,
        )
        assert expansion.next_nodes == ("m.attraction_0", "m.attraction_1")
    finally:
        backend.close()


def test_rank_entities_by_numeric_attribute_orders_in_sqlite(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        argmax_rows = backend.rank_entities_by_numeric_attribute(
            source_ids=["m.attraction_0", "m.zz_museum"],
            relation_ids=["architecture.museum.established"],
            operation="argmax",
            limit=2,
        )
        assert [row["source_entity_id"] for row in argmax_rows] == ["m.zz_museum", "m.attraction_0"]
        assert [row["numeric_value"] for row in argmax_rows] == [1900.0, 1800.0]

        argmin_rows = backend.rank_entities_by_numeric_attribute(
            source_ids=["m.attraction_0", "m.zz_museum"],
            relation_ids=["architecture.museum.established"],
            operation="argmin",
            limit=2,
        )
        assert [row["source_entity_id"] for row in argmin_rows] == ["m.attraction_0", "m.zz_museum"]
    finally:
        backend.close()


def test_backend_find_direct_edges_between_and_two_hop_paths(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        direct_rows = backend.find_direct_edges_between(
            source_ids=["m.person_b"],
            target_ids=["m.city_c"],
            relation_ids=["people.person.place_of_birth"],
        )
        assert direct_rows == [
            {
                "source_id": "m.person_b",
                "relation_id": "people.person.place_of_birth",
                "target_id": "m.city_c",
            }
        ]

        two_hop_rows = backend.find_two_hop_paths_between(
            source_ids=["m.person_a"],
            target_ids=["m.person_b"],
            limit=10,
        )
        assert two_hop_rows == [
            {
                "source_id": "m.person_a",
                "first_relation_id": "people.person.spouse_s",
                "mid_id": "m.marriage",
                "second_relation_id": "people.marriage.spouse",
                "target_id": "m.person_b",
            }
        ]
    finally:
        backend.close()


def test_get_nodes_metadata_batched_marks_literals_and_probable_cvt(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    try:
        metadata = backend.get_nodes_metadata_batched(["m.marriage", "m.city_c", "1990"])
        assert metadata["m.marriage"].is_probable_cvt is True
        assert metadata["m.city_c"].is_literal is False
        assert metadata["1990"].is_literal is True
    finally:
        backend.close()


def test_immediate_reverse_relation_is_filtered() -> None:
    state = _base_state(
        path=(
            RelationPathStep(
                relation="people.person.spouse_s",
                relation_name="spouse",
                direction="forward",
                role="intermediate",
                confidence=0.8,
            ),
        )
    )
    candidates = [
        RelationCandidate("people.person.spouse_s", "spouse", "reverse", 1, 1, 1.0, 10),
        RelationCandidate("people.marriage.spouse", "spouse", "forward", 1, 2, 1.0, 10),
    ]
    filtered = filter_relation_candidates(state, candidates, expected_answer_type="city")
    assert [item.relation for item in filtered] == ["people.marriage.spouse"]


def test_merge_same_relation_paths_unions_frontiers() -> None:
    step = RelationPathStep("people.person.spouse_s", "spouse", "forward", "intermediate", 0.7)
    state_a = _base_state(state_id="a", path=(step,))
    state_b = _base_state(state_id="b", path=(step,))
    state_b = RelationBeamState(
        **{
            **state_b.__dict__,
            "frontier_nodes": ("m.marriage", "m.person_b"),
            "evidence_paths": {"m.marriage": (), "m.person_b": ()},
        }
    )
    merged = merge_same_relation_paths([state_a, state_b], max_nodes_per_path=10)
    assert len(merged) == 1
    assert set(merged[0].frontier_nodes) >= {"m.person_a", "m.marriage", "m.person_b"}


def test_prune_relation_beam_preserves_protected_and_diversity() -> None:
    spouse_step = RelationPathStep("people.person.spouse_s", "spouse", "forward", "intermediate", 0.9)
    profession_step = RelationPathStep("people.person.profession", "profession", "forward", "answer", 0.95)
    award_step = RelationPathStep("award.award_nomination", "award", "forward", "answer", 0.93)
    protected = _base_state("protected", (spouse_step,), protected_until_hop=2, semantic_score=0.4)
    answer1 = _base_state("answer1", (profession_step,), answer_score=0.9, contains_answer_candidates=True)
    answer2 = _base_state("answer2", (award_step,), answer_score=0.88, contains_answer_candidates=True)
    spouse_variant = _base_state("spouse_variant", (spouse_step,), semantic_score=0.7)
    pruned = prune_relation_beam([protected, answer1, answer2, spouse_variant], current_hop=1, beam_width=3)
    signatures = [relation_path_signature(item) for item in pruned]
    assert relation_path_signature(protected) in signatures
    assert relation_path_signature(answer1) in signatures
    assert relation_path_signature(answer2) in signatures


def test_build_child_state_matches_movie_to_film_type_alias() -> None:
    parent = _base_state()
    action = RelationAction(
        relation="film.film_character.portrayed_in_films.film.performance.film",
        relation_name="film",
        direction="forward",
        role="intermediate",
        confidence=0.9,
        expected_next_type="movie",
        reason="bridge to film",
        protect_for_next_hop=False,
    )
    child = build_child_state(
        parent=parent,
        action=action,
        expansion=FrontierExpansion(
            next_nodes=("m.film",),
            evidence_paths={"m.film": ()},
            source_node_count=1,
            edge_count=1,
            truncated=False,
        ),
        metadata={
            "m.film": NodeMetadata(
                node_id="m.film",
                name="Forrest Gump",
                types=("film.film",),
                degree=12,
                is_literal=False,
                is_probable_cvt=False,
            ),
        },
        max_nodes_per_path=10,
        state_index=1,
        expected_answer_type="movie",
    )
    assert child.answer_score >= 0.82


def test_three_hop_search_reaches_answer_through_cvt(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    searcher = RelationTypeBeamSearcher(
        graph=backend,
        llm=DeterministicMockRelationBeamLLM(),
        config=RelationBeamConfig(max_hops=3, beam_width=6, relations_per_state=3, relation_retrieval_top_k=5),
    )
    try:
        states = searcher.search(
            original_question="Where was the spouse of Person A born?",
            subquestion="Where was the spouse of Person A born?",
            input_nodes=["m.person_a"],
            expected_answer_type="city",
        )
        assert states
        assert any("m.city_c" in state.frontier_nodes for state in states)
        assert all(state.hop <= 3 for state in states)
    finally:
        backend.close()


def test_llm_invalid_relation_is_ignored_and_falls_back(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    adapter = BaseLLMRelationBeamAdapter(InvalidRelationLLM())
    searcher = RelationTypeBeamSearcher(
        graph=backend,
        llm=adapter,
        config=RelationBeamConfig(max_hops=2, beam_width=4, relations_per_state=2, relation_retrieval_top_k=4),
    )
    try:
        states = searcher.search(
            original_question="Where was the spouse of Person A born?",
            subquestion="Where was the spouse of Person A born?",
            input_nodes=["m.person_a"],
            expected_answer_type="city",
        )
        assert states
        assert any(state.relation_path for state in states)
    finally:
        backend.close()


def test_relation_action_selection_batches_large_candidate_lists() -> None:
    llm = CountingBatchLLM()
    adapter = BaseLLMRelationBeamAdapter(llm, relation_batch_size=100)
    candidates = [
        RelationCandidate(
            relation=f"rel_{index}",
            relation_name=f"Relation {index}",
            direction="forward",
            supporting_node_count=1,
            total_neighbor_count=1,
            support_ratio=1.0,
            global_frequency=1,
        )
        for index in range(205)
    ]
    actions = adapter.select_relation_actions(
        original_question="Where was the spouse of Person A born?",
        subquestion="Where was the spouse of Person A born?",
        expected_answer_type="city",
        state=_base_state(),
        frontier_summary={"node_count": 1, "entity_count": 1, "literal_count": 0, "probable_cvt_ratio": 0.0},
        candidates=candidates,
        anchor_mentions=[],
        relation_hint_names=[],
        resolved_dependencies=[],
        remaining_hops=2,
        top_k=3,
    )
    assert len(llm.prompts) == 3
    assert actions


def test_llm_exception_uses_fallback(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    searcher = RelationTypeBeamSearcher(
        graph=backend,
        llm=ExplodingRelationLLM(),
        config=RelationBeamConfig(max_hops=2, beam_width=4, relations_per_state=2, relation_retrieval_top_k=4),
    )
    try:
        states = searcher.search(
            original_question="Who is Person A married to?",
            subquestion="Who is Person A married to?",
            input_nodes=["m.person_a"],
            expected_answer_type="person",
        )
        assert states
        assert any(state.relation_path for state in states)
    finally:
        backend.close()


def test_no_answer_returns_best_explainable_path(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    searcher = RelationTypeBeamSearcher(
        graph=backend,
        llm=DeterministicMockRelationBeamLLM(),
        config=RelationBeamConfig(max_hops=2, beam_width=4, relations_per_state=2, relation_retrieval_top_k=4),
    )
    try:
        states = searcher.search(
            original_question="What galaxy is Person A associated with?",
            subquestion="What galaxy is Person A associated with?",
            input_nodes=["m.person_a"],
            expected_answer_type="galaxy",
        )
        assert states
        assert all(state.hop <= 2 for state in states)
    finally:
        backend.close()


def test_subquestion_relation_explorer_returns_candidate_paths(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    explorer = SubquestionRelationExplorer(
        graph=backend,
        llm=DeterministicMockRelationBeamLLM(),
        config=RelationBeamConfig(max_hops=3, beam_width=6),
    )
    try:
        result = explorer.explore(
            original_question="Where was the spouse of Person A born?",
            subquestion_id="sq1",
            subquestion_text="Where was the spouse of Person A born?",
            input_entities=["m.person_a"],
            expected_answer_type="city",
        )
        assert result.candidate_paths
        assert "m.city_c" in result.answer_entities
        assert result.searched_hops <= 3
    finally:
        backend.close()


def test_subquestion_relation_explorer_emits_cvt_bundle_path(tmp_path: Path) -> None:
    db_path, index_dir = _build_relation_beam_fixture(tmp_path)
    backend = IndexedSQLiteGraphBackend(db_path=db_path, index_dir=index_dir, neighbor_limit=20)
    explorer = SubquestionRelationExplorer(
        graph=backend,
        llm=BaseLLMRelationBeamAdapter(MarriageRelationLLM()),
        config=RelationBeamConfig(max_hops=1, beam_width=4, relations_per_state=2, relation_retrieval_top_k=4),
    )
    try:
        result = explorer.explore(
            original_question="Who is Person A married to?",
            subquestion_id="sq1",
            subquestion_text="Who is Person A married to?",
            input_entities=["m.person_a"],
            expected_answer_type="person",
        )
        assert result.candidate_paths
        bundle_paths = [path for path in result.candidate_paths if path.source_stage == CVT_BUNDLE_SOURCE_STAGE]
        assert bundle_paths
        bundle = bundle_paths[0]
        relations = [triple.relation for triple in bundle.triples]
        assert "people.person.spouse_s" in relations
        assert "people.marriage.spouse" in relations
        assert bundle.terminal_node_id == "m.person_b"
        assert "[CVT:" in bundle.text
        assert "{spouse:" in bundle.text
        assert "spouse: [Person A, Person B]" in bundle.text
    finally:
        backend.close()
