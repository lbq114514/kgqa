from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.graph import KnowledgeGraph
from kgqa.llm.base import BaseLLM
from kgqa.reasoning.pipeline import KGQAPipeline
from kgqa.reasoning.question_analysis import analyze_question
from kgqa.reasoning.subquestion_solvers import (
    AggregateSolver,
    ConstrainedCollectSolver,
    ExploreSolver,
    VerifySolver,
    SubquestionExecutionContext,
    SubquestionSolverRouter,
)
from kgqa.utils.types import Triple
from kgqa.utils.types import EntityMentionSpec, RelationHintSpec, SubQuestionSpec


class StaticLLM(BaseLLM):
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self.response


class Candidate:
    def __init__(
        self,
        relation: str,
        relation_name: str,
        direction: str = "forward",
        support_ratio: float = 1.0,
        global_frequency: int = 1,
        sample_target_types: tuple[str, ...] = (),
        sample_target_names: tuple[str, ...] = (),
    ) -> None:
        self.relation = relation
        self.relation_name = relation_name
        self.direction = direction
        self.supporting_node_count = 1
        self.total_neighbor_count = 1
        self.support_ratio = support_ratio
        self.global_frequency = global_frequency
        self.sample_target_types = sample_target_types
        self.sample_target_names = sample_target_names


class Expansion:
    def __init__(self, evidence_paths: dict[str, tuple[object, ...]]) -> None:
        self.next_nodes = tuple(evidence_paths.keys())
        self.evidence_paths = evidence_paths
        self.source_node_count = 1
        self.edge_count = len(evidence_paths)
        self.truncated = False


class Edge:
    def __init__(self, source_id: str, relation: str, direction: str, target_id: str) -> None:
        self.source_id = source_id
        self.relation = relation
        self.direction = direction
        self.target_id = target_id


class FakeIndexedGraphAPI:
    def __init__(
        self,
        relation_candidates: list[Candidate],
        expansions: dict[tuple[str, str], dict[str, tuple[object, ...]]],
        labels: dict[str, str],
        mention_map: dict[str, list[str]] | None = None,
        neighbors: dict[str, list[object]] | None = None,
    ) -> None:
        self._relation_candidates = relation_candidates
        self._expansions = expansions
        self._labels = labels
        self._mention_map = mention_map or {}
        self._neighbors = neighbors or {}
        self._constrained_rows: list[dict[str, object]] = []

    def get_frontier_relations(self, node_ids, include_reverse=True, include_literals=True):
        return list(self._relation_candidates)

    def expand_frontier_by_relation(self, node_ids, relation, direction, parent_evidence_paths=None, limit_per_source=8, total_limit=100):
        return Expansion(self._expansions.get((relation, direction), {}))

    def get_entity_display_name(self, node_id: str) -> str:
        return self._labels.get(node_id, node_id)

    def resolve_entity_mentions(self, names, top_k=1):
        resolved = []
        for name in names:
            resolved.extend(self._mention_map.get(name, [])[:top_k])
        return resolved

    def get_neighbors(self, node_id, include_reverse=True, relation_filter=None, limit=None, strict_relation_filter=False):
        rows = list(self._neighbors.get(node_id, []))
        if relation_filter:
            rows = [row for row in rows if row.triple.relation in relation_filter]
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows

    def collect_related_entities_constrained(
        self,
        source_ids,
        relation_ids,
        direction="forward",
        expected_answer_type="",
        constraint_terms=None,
        limit=100,
    ):
        return list(self._constrained_rows)[:limit]


class AggregateGraphAPI:
    def __init__(self, mode: str = "established") -> None:
        self.selected_relation_ids: list[str] = []
        self.mode = mode

    def expand_relation_candidates_for_aggregate(self, seed_ids, expected_answer_type="", include_literals=True):
        if self.mode == "postgraduates":
            return [
                {
                    "relation_id": "education.educational_institution.total_enrollment",
                    "relation_name": "total enrollment",
                    "direction": "forward",
                    "sample_target_types": [],
                    "sample_target_names": ["40000"],
                    "support_ratio": 1.0,
                    "score": 20.0,
                },
                {
                    "relation_id": "education.university.number_of_postgraduates",
                    "relation_name": "number of postgraduates",
                    "direction": "forward",
                    "sample_target_types": [],
                    "sample_target_names": ["856"],
                    "support_ratio": 0.5,
                    "score": 1.0,
                },
            ]
        return [
            {
                "relation_id": "architecture.museum.type_of_museum",
                "relation_name": "type of museum",
                "direction": "forward",
                "sample_target_types": ["architecture.type_of_museum"],
                "sample_target_names": ["History museum"],
                "support_ratio": 1.0,
                "score": 20.0,
            },
            {
                "relation_id": "architecture.museum.established",
                "relation_name": "established",
                "direction": "forward",
                "sample_target_types": [],
                "sample_target_names": ["1891"],
                "support_ratio": 0.5,
                "score": 1.0,
            },
        ]

    def rank_entities_by_numeric_attribute(self, source_ids, relation_ids, direction="forward", operation="argmax", limit=100):
        self.selected_relation_ids = list(relation_ids)
        if self.mode == "postgraduates":
            rows = [
                {
                    "source_entity_id": "m.purdue",
                    "source_label": "Purdue University",
                    "relation_id": "education.university.number_of_postgraduates",
                    "relation_name": "number of postgraduates",
                    "value": "856",
                    "numeric_value": 856.0,
                },
                {
                    "source_entity_id": "m.stanford_gsb",
                    "source_label": "Stanford Graduate School of Business",
                    "relation_id": "education.university.number_of_postgraduates",
                    "relation_name": "number of postgraduates",
                    "value": "809",
                    "numeric_value": 809.0,
                },
            ]
            return sorted(rows, key=lambda row: row["numeric_value"], reverse=operation != "argmin")[:limit]
        rows = [
            {
                "source_entity_id": "m.old",
                "source_label": "Old Museum",
                "relation_id": "architecture.museum.established",
                "value": "1800-01-01",
                "numeric_value": 1800.0,
            },
            {
                "source_entity_id": "m.new",
                "source_label": "New Museum",
                "relation_id": "architecture.museum.established",
                "value": "1891-01-01",
                "numeric_value": 1891.0,
            },
        ]
        return sorted(rows, key=lambda row: row["numeric_value"], reverse=operation != "argmin")[:limit]


def _make_pipeline(monkeypatch) -> KGQAPipeline:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: object())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: object())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: object())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: object())
    return KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=StaticLLM("{}"),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"search": {"subquestion_explorer": "relation_beam"}},
            "search": {"dmax": 3, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )


def test_question_analysis_falls_back_to_explore_solver_type() -> None:
    analysis = analyze_question(
        question="What is the population of France?",
        llm=StaticLLM(
            '{"topic_entities":["France"],"split_questions":["What is the population of France?"],"reasoning_indicator":"lookup population","ordered_topic_entities":["France"],"predicted_depth":1,"sub_questions":[{"id":"sq1","question":"What is the population of France?","topic_entities":[{"name":"France","aliases":[],"expected_type":"country","role":"topic_entity"}],"interested_nodes":[],"interested_relations":[{"name":"population","aliases":[],"freebase_like_ids":[],"direction":"topic_entity -> answer","description":"population"}],"expected_answer_type":"population","expected_hop":1,"depends_on":[]}]}'
        ),
        dmax=3,
        topic_entities_override=["France"],
    )

    assert analysis.sub_questions[0].solver_type == "explore"


def test_question_analysis_derives_local_topic_entities_from_interested_nodes() -> None:
    class FakeLLM(BaseLLM):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return (
                '{"topic_entities":["Libya"],'
                '"split_questions":["Which country uses \'Libya, Libya, Libya\' as its national anthem?"],'
                '"reasoning_indicator":"find the country by anthem",'
                '"ordered_topic_entities":["Libya"],'
                '"predicted_depth":1,'
                '"sub_questions":[{"id":"sq1","question":"Which country uses \'Libya, Libya, Libya\' as its national anthem?",'
                '"topic_entities":[],"interested_nodes":[{"name":"Libya, Libya, Libya","aliases":[],"expected_type":"anthem","role":"constraint_or_anchor"}],'
                '"interested_relations":[{"name":"national anthem of","aliases":[],"freebase_like_ids":[],"direction":"interested_node -> answer","description":"country with this anthem"}],'
                '"expected_answer_type":"country","expected_hop":1,"depends_on":[]}]}'
            )

    analysis = analyze_question(
        question="Which country uses Libya, Libya, Libya as its national anthem?",
        llm=FakeLLM(),
        dmax=3,
    )

    assert [item.name for item in analysis.sub_questions[0].local_topic_entities] == ["Libya, Libya, Libya"]


def test_question_analysis_refines_lookup_without_seed_to_explore() -> None:
    class FakeLLM(BaseLLM):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return (
                '{"topic_entities":[],"split_questions":["Which country uses \'Libya, Libya, Libya\' as its national anthem?"],'
                '"reasoning_indicator":"find the country by anthem","ordered_topic_entities":[],"predicted_depth":1,'
                '"sub_questions":[{"id":"sq1","question":"Which country uses \'Libya, Libya, Libya\' as its national anthem?",'
                '"topic_entities":[],"local_topic_entities":[],"interested_nodes":[{"name":"Libya, Libya, Libya","aliases":[],"expected_type":"anthem","role":"constraint_or_anchor"}],'
                '"interested_relations":[{"name":"national anthem of","aliases":[],"freebase_like_ids":[],"direction":"interested_node -> answer","description":"country with this anthem"}],'
                '"expected_answer_type":"country","expected_hop":1,"depends_on":[],"solver_type":"lookup","solver_reason":"known seed direct attribute retrieval"}]}'
            )

    analysis = analyze_question(
        question="Which country uses Libya, Libya, Libya as its national anthem?",
        llm=FakeLLM(),
        dmax=3,
    )

    assert analysis.sub_questions[0].solver_type == "explore"


def test_question_analysis_refines_verify_without_candidate_to_explore() -> None:
    class FakeLLM(BaseLLM):
        def generate(self, prompt: str, **kwargs: object) -> str:
            return (
                '{"topic_entities":[],"split_questions":["Which country uses \'Libya, Libya, Libya\' as its national anthem?"],'
                '"reasoning_indicator":"find the country by anthem","ordered_topic_entities":[],"predicted_depth":1,'
                '"sub_questions":[{"id":"sq1","question":"Which country uses \'Libya, Libya, Libya\' as its national anthem?",'
                '"topic_entities":[],"local_topic_entities":[],"interested_nodes":[],"interested_relations":[],'
                '"expected_answer_type":"country","expected_hop":1,"depends_on":[],"solver_type":"verify","solver_reason":"verify constraint"}]}'
            )

    analysis = analyze_question(
        question="Which country uses Libya, Libya, Libya as its national anthem?",
        llm=FakeLLM(),
        dmax=3,
    )

    assert analysis.sub_questions[0].solver_type == "explore"


def test_build_next_subquestion_inputs_preserves_solver_fields(monkeypatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    sub_question = SubQuestionSpec(
        id="sq2",
        question="What is the population of that country?",
        depends_on=["sq1"],
        solver_type="explore",
        solver_reason="attribute retrieval",
    )
    run_state = type("State", (), {"resolved_sub_answers": {"sq1": {"literals": ["France"], "predicted_answers": ["France"]}}})()
    enriched = pipeline._build_next_subquestion_inputs(sub_question, run_state)

    assert enriched.solver_type == "explore"
    assert enriched.solver_reason == "attribute retrieval"


def test_build_next_subquestion_inputs_preserves_downstream_filters(monkeypatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    sub_question = SubQuestionSpec(
        id="sq1",
        question="Which country is the anthem from?",
        downstream_filters=["religion", "practiced religion"],
    )
    run_state = type("State", (), {"resolved_sub_answers": {}})()

    enriched = pipeline._build_next_subquestion_inputs(sub_question, run_state)

    assert enriched.downstream_filters == ["religion", "practiced religion"]


def test_pipeline_attaches_downstream_filters_to_upstream_subquestion(monkeypatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    analysis = analyze_question(
        question="What religions are practiced in the country of the Afghan National Anthem?",
        llm=StaticLLM(
            '{"topic_entities":["Afghan National Anthem"],"split_questions":["Which country is the Afghan National Anthem from?","What religions are practiced in that country?"],"reasoning_indicator":"find country then religion","ordered_topic_entities":["Afghan National Anthem"],"predicted_depth":2,"sub_questions":[{"id":"sq1","question":"Which country is the Afghan National Anthem from?","topic_entities":[{"name":"Afghan National Anthem","aliases":[],"expected_type":"anthem","role":"topic_entity"}],"interested_relations":[{"name":"country of origin","aliases":[],"freebase_like_ids":[],"direction":"topic_entity -> answer","description":"country"}],"expected_answer_type":"country","expected_hop":1,"depends_on":[]},{"id":"sq2","question":"What religions are practiced in that country?","interested_relations":[{"name":"practiced religion","aliases":["religion"],"freebase_like_ids":[],"direction":"country -> answer","description":"religions practiced"}],"expected_answer_type":"religion","expected_hop":1,"depends_on":["sq1"]}]}'
        ),
        dmax=3,
        topic_entities_override=["Afghan National Anthem"],
    )

    updated = pipeline._attach_downstream_filters_to_analysis(analysis)

    assert "practiced religion" in updated.sub_questions[0].downstream_filters
    assert "religion" in updated.sub_questions[0].downstream_filters


def test_build_next_subquestion_inputs_prefers_local_topic_entities(monkeypatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    sub_question = SubQuestionSpec(
        id="sq1",
        question="Which country uses 'Libya, Libya, Libya' as its national anthem?",
        topic_entities=[EntityMentionSpec(name="Libya", role="topic_entity")],
        local_topic_entities=[EntityMentionSpec(name="Libya, Libya, Libya", role="topic_entity")],
        interested_nodes=[EntityMentionSpec(name="Libya, Libya, Libya", role="constraint_or_anchor")],
    )
    run_state = type("State", (), {"resolved_sub_answers": {}})()

    enriched = pipeline._build_next_subquestion_inputs(sub_question, run_state)

    assert [item.name for item in enriched.topic_entities] == ["Libya, Libya, Libya"]
    assert [item.name for item in enriched.local_topic_entities] == ["Libya, Libya, Libya"]


def test_build_next_subquestion_inputs_drops_weak_global_topics_for_dependency_steps(monkeypatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    sub_question = SubQuestionSpec(
        id="sq2",
        question="Who is the leader of that country?",
        topic_entities=[EntityMentionSpec(name="Libya", role="topic_entity")],
        depends_on=["sq1"],
    )
    run_state = type("State", (), {"resolved_sub_answers": {"sq1": {}}})()

    enriched = pipeline._build_next_subquestion_inputs(sub_question, run_state)

    assert enriched.topic_entities == []
    assert enriched.local_topic_entities == []


def test_router_upgrades_lookup_solver_request_to_explore() -> None:
    router = SubquestionSolverRouter()
    sub_question = SubQuestionSpec(id="sq1", question="What is the population of France?", solver_type="lookup")
    context = SubquestionExecutionContext(
        original_question="What is the population of France?",
        sub_question=sub_question,
        graph_api=None,
        step_seed_ids=["m.france"],
        constraint_target_ids=[],
        relation_ids=[],
        resolved_sub_answers={},
        execute_explore=lambda: [],
    )

    assert isinstance(router.select_solver(sub_question, context), ExploreSolver)


def test_router_selects_explore_solver() -> None:
    router = SubquestionSolverRouter()
    sub_question = SubQuestionSpec(id="sq1", question="Which actor played the kid in Forrest Gump?", solver_type="explore")
    context = SubquestionExecutionContext(
        original_question=sub_question.question,
        sub_question=sub_question,
        graph_api=None,
        step_seed_ids=["m.movie"],
        constraint_target_ids=[],
        relation_ids=[],
        resolved_sub_answers={},
        execute_explore=lambda: [],
    )

    assert isinstance(router.select_solver(sub_question, context), ExploreSolver)


def test_router_selects_constrained_collect_solver() -> None:
    router = SubquestionSolverRouter()
    sub_question = SubQuestionSpec(id="sq1", question="Which country fits the filter?", solver_type="constrained_collect")
    context = SubquestionExecutionContext(
        original_question=sub_question.question,
        sub_question=sub_question,
        graph_api=FakeIndexedGraphAPI([], {}, {}),
        step_seed_ids=["m.seed"],
        constraint_target_ids=[],
        relation_ids=["location.country.religions"],
        resolved_sub_answers={},
        execute_explore=lambda: [],
    )

    assert isinstance(router.select_solver(sub_question, context), ConstrainedCollectSolver)


def test_router_upgrades_lookup_without_stable_seed_to_explore() -> None:
    router = SubquestionSolverRouter()
    sub_question = SubQuestionSpec(
        id="sq1",
        question="Which country uses Libya, Libya, Libya as its national anthem?",
        solver_type="lookup",
        interested_nodes=[EntityMentionSpec(name="Libya, Libya, Libya", role="constraint_or_anchor")],
    )
    context = SubquestionExecutionContext(
        original_question=sub_question.question,
        sub_question=sub_question,
        graph_api=None,
        step_seed_ids=[],
        constraint_target_ids=[],
        relation_ids=[],
        resolved_sub_answers={},
        execute_explore=lambda: [],
    )

    assert isinstance(router.select_solver(sub_question, context), ExploreSolver)


def test_router_upgrades_verify_without_candidate_to_explore() -> None:
    router = SubquestionSolverRouter()
    sub_question = SubQuestionSpec(
        id="sqv",
        question="Is that country in Central America?",
        solver_type="verify",
    )
    context = SubquestionExecutionContext(
        original_question=sub_question.question,
        sub_question=sub_question,
        graph_api=None,
        step_seed_ids=[],
        constraint_target_ids=[],
        relation_ids=[],
        resolved_sub_answers={},
        execute_explore=lambda: [],
    )

    assert isinstance(router.select_solver(sub_question, context), ExploreSolver)


def test_aggregate_solver_supports_in_memory_argmax() -> None:
    solver = AggregateSolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Which country has the highest population?",
            sub_question=SubQuestionSpec(id="sq3", question="Which country has the highest population?", solver_type="aggregate"),
            graph_api=None,
            step_seed_ids=[],
            constraint_target_ids=[],
            relation_ids=[],
            resolved_sub_answers={
                "sq2": {
                    "attribute_rows": [
                        {"source_entity_id": "m.a", "source_label": "A", "value": "10"},
                        {"source_entity_id": "m.b", "source_label": "B", "value": "20"},
                    ]
                }
            },
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "aggregate"
    assert result.primary_entity_ids == ["m.b"]
    assert result.candidate_paths
    assert result.candidate_paths[0].source_stage == "sql_aggregate"


def test_aggregate_solver_reranks_backend_relations_by_superlative_hint() -> None:
    solver = AggregateSolver({"aggregate": {"graph_sql": {"selected_relation_top_k": 1}}})
    graph_api = AggregateGraphAPI()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Which museum was established latest?",
            sub_question=SubQuestionSpec(
                id="sq2",
                question="Of those museums, which one was established the latest?",
                solver_type="aggregate",
                depends_on=["sq1"],
                interested_relations=[
                    RelationHintSpec(
                        name="establishment date",
                        aliases=["founded", "date of establishment"],
                        description="the date the museum was established",
                    )
                ],
                expected_answer_type="museum",
            ),
            graph_api=graph_api,
            step_seed_ids=[],
            constraint_target_ids=[],
            relation_ids=[],
            resolved_sub_answers={
                "sq1": {"entity_ids": ["m.old", "m.new"], "entity_set_role": "candidate_answers"}
            },
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "aggregate"
    assert graph_api.selected_relation_ids == ["architecture.museum.established"]
    assert result.primary_entity_ids == ["m.new"]
    assert result.structured_outputs["set_operation_hint"] == "argmax"
    assert result.candidate_paths
    assert result.candidate_paths[0].triple_facts[0].relation_id == "architecture.museum.established"


def test_aggregate_solver_prefers_argmin_over_number_of_count() -> None:
    solver = AggregateSolver({"aggregate": {"graph_sql": {"selected_relation_top_k": 1}}})
    graph_api = AggregateGraphAPI(mode="postgraduates")
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Which college has the smallest number of postgraduates?",
            sub_question=SubQuestionSpec(
                id="sq2",
                question="Which college has the smallest number of postgraduates?",
                solver_type="aggregate",
                depends_on=["sq1"],
                interested_relations=[
                    RelationHintSpec(name="number of postgraduates", aliases=["postgraduates"])
                ],
                expected_answer_type="university",
            ),
            graph_api=graph_api,
            step_seed_ids=[],
            relation_ids=[],
            resolved_sub_answers={
                "sq1": {"entity_ids": ["m.purdue", "m.stanford_gsb"], "entity_set_role": "candidate_answers"}
            },
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "aggregate"
    assert result.solver_debug["operation"] == "argmin"
    assert graph_api.selected_relation_ids == ["education.university.number_of_postgraduates"]
    assert result.primary_entity_ids == ["m.stanford_gsb"]


def test_aggregate_solver_supports_in_memory_intersection() -> None:
    solver = AggregateSolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Which institution is the same?",
            sub_question=SubQuestionSpec(id="sq3", question="Which institution is the same?", solver_type="aggregate", depends_on=["sq1", "sq2"]),
            graph_api=None,
            step_seed_ids=[],
            constraint_target_ids=[],
            relation_ids=[],
            resolved_sub_answers={
                "sq1": {"entity_ids": ["m.uw", "m.other"], "entity_set_role": "comparison_candidates"},
                "sq2": {"entity_ids": ["m.uw"], "entity_set_role": "comparison_candidates"},
            },
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "aggregate"
    assert result.primary_entity_ids == ["m.uw"]


def test_verify_solver_direct_verification() -> None:
    class Neighbor:
        def __init__(self, source_id: str, neighbor_id: str, relation: str, head: str, tail: str, reversed: bool = False):
            self.source_id = source_id
            self.neighbor_id = neighbor_id
            self.triple = Triple(head=head, relation=relation, tail=tail)
            self.relation_name = relation
            self.reversed = reversed

    graph_api = FakeIndexedGraphAPI(
        relation_candidates=[],
        expansions={},
        labels={"m.france": "France", "m.paris": "Paris"},
        mention_map={"Paris": ["m.paris"]},
        neighbors={
            "m.france": [Neighbor("m.france", "m.paris", "location.country.capital", "France", "Paris")]
        },
    )
    solver = VerifySolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Is Paris the capital of France?",
            sub_question=SubQuestionSpec(
                id="sq1",
                question="Is Paris the capital of France?",
                interested_nodes=[EntityMentionSpec(name="Paris", role="constraint_or_anchor")],
                interested_relations=[RelationHintSpec(name="capital", freebase_like_ids=["location.country.capital"])],
                solver_type="verify",
            ),
            graph_api=graph_api,
            step_seed_ids=["m.france"],
            relation_ids=["location.country.capital"],
            resolved_sub_answers={},
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "verify"
    assert result.primary_entity_ids == ["m.paris"]
    assert result.structured_outputs["verification_rows"][0]["verification_mode"] == "direct_verify"


def test_verify_solver_exact_direct_sqlite_pair_check() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE triples (head TEXT, relation TEXT, tail TEXT, tail_kind TEXT)")
    conn.execute("INSERT INTO triples(head, relation, tail, tail_kind) VALUES (?, ?, ?, ?)", ("m.france", "location.country.capital", "m.paris", "id"))

    class GraphAPI:
        def __init__(self) -> None:
            self.connection = conn

        def get_entity_display_name(self, node_id: str) -> str:
            return {"m.france": "France", "m.paris": "Paris"}.get(node_id, node_id)

        def resolve_entity_mentions(self, names, top_k=1):
            return ["m.paris"] if names and names[0] == "Paris" else []

        def get_neighbors(self, *args, **kwargs):
            raise AssertionError("direct verify should not need neighbor expansion when exact pair exists")

    solver = VerifySolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Is Paris the capital of France?",
            sub_question=SubQuestionSpec(
                id="sq1",
                question="Is Paris the capital of France?",
                interested_nodes=[EntityMentionSpec(name="Paris", role="constraint_or_anchor")],
                interested_relations=[RelationHintSpec(name="capital", freebase_like_ids=["location.country.capital"])],
                solver_type="verify",
            ),
            graph_api=GraphAPI(),
            step_seed_ids=["m.france"],
            relation_ids=["location.country.capital"],
            resolved_sub_answers={},
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "verify"
    assert result.primary_entity_ids == ["m.paris"]
    assert result.structured_outputs["verification_rows"][0]["verification_mode"] == "direct_verify"


def test_verify_solver_prefers_backend_pair_and_two_hop_helpers() -> None:
    class GraphAPI:
        def get_entity_display_name(self, node_id: str) -> str:
            return {
                "m.movie": "Forrest Gump",
                "m.performance": "Performance",
                "m.actor": "Michael Connor Humphreys",
            }.get(node_id, node_id)

        def resolve_entity_mentions(self, names, top_k=1):
            return ["m.actor"] if names and names[0] == "Michael Connor Humphreys" else []

        def find_direct_edges_between(self, source_ids, target_ids, relation_ids=None, limit=100):
            return []

        def find_two_hop_paths_between(self, source_ids, target_ids, relation_ids=None, limit=100):
            return [
                {
                    "source_id": "m.movie",
                    "first_relation_id": "film.film.starring",
                    "mid_id": "m.performance",
                    "second_relation_id": "film.performance.actor",
                    "target_id": "m.actor",
                }
            ]

        def get_neighbors(self, *args, **kwargs):
            raise AssertionError("backend path helpers should run before neighbor expansion")

    solver = VerifySolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Was Michael Connor Humphreys in Forrest Gump?",
            sub_question=SubQuestionSpec(
                id="sq2",
                question="Was Michael Connor Humphreys in Forrest Gump?",
                interested_nodes=[EntityMentionSpec(name="Michael Connor Humphreys", role="constraint_or_anchor")],
                solver_type="verify",
            ),
            graph_api=GraphAPI(),
            step_seed_ids=["m.movie"],
            relation_ids=[],
            resolved_sub_answers={},
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "verify"
    assert result.structured_outputs["solver_execution_mode"] == "short_path_verify"
    assert result.primary_entity_ids == ["m.actor"]


def test_verify_solver_short_path_verification() -> None:
    class Neighbor:
        def __init__(self, source_id: str, neighbor_id: str, relation: str, head: str, tail: str, reversed: bool = False):
            self.source_id = source_id
            self.neighbor_id = neighbor_id
            self.triple = Triple(head=head, relation=relation, tail=tail)
            self.relation_name = relation
            self.reversed = reversed

    graph_api = FakeIndexedGraphAPI(
        relation_candidates=[],
        expansions={},
        labels={"m.movie": "Forrest Gump", "m.performance": "performance", "m.actor": "Michael Connor Humphreys"},
        mention_map={"Michael Connor Humphreys": ["m.actor"]},
        neighbors={
            "m.movie": [Neighbor("m.movie", "m.performance", "film.film.starring", "Forrest Gump", "performance")],
            "m.performance": [Neighbor("m.performance", "m.actor", "film.performance.actor", "performance", "Michael Connor Humphreys")],
        },
    )
    solver = VerifySolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Was Michael Connor Humphreys in Forrest Gump?",
            sub_question=SubQuestionSpec(
                id="sq2",
                question="Was Michael Connor Humphreys in Forrest Gump?",
                interested_nodes=[EntityMentionSpec(name="Michael Connor Humphreys", role="constraint_or_anchor")],
                solver_type="verify",
            ),
            graph_api=graph_api,
            step_seed_ids=["m.movie"],
            relation_ids=[],
            resolved_sub_answers={},
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "verify"
    assert result.structured_outputs["solver_execution_mode"] == "short_path_verify"
    assert result.primary_entity_ids == ["m.actor"]


def test_verify_solver_fallback_without_targets() -> None:
    solver = VerifySolver()
    fallback_path = []
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Does that satisfy the condition?",
            sub_question=SubQuestionSpec(id="sqv", question="Does that satisfy the condition?", solver_type="verify"),
            graph_api=None,
            step_seed_ids=["m.seed"],
            relation_ids=[],
            resolved_sub_answers={},
            execute_explore=lambda: fallback_path,
        )
    )

    assert result.solver_type == "explore"
    assert result.solver_debug["fallback_reason"] == "verify_no_target_entities"


def test_constrained_collect_solver_returns_ranked_entities() -> None:
    graph_api = FakeIndexedGraphAPI(
        relation_candidates=[],
        expansions={},
        labels={"m.country": "Afghanistan", "m.religion": "Islam", "m.other": "Hinduism"},
    )
    graph_api._constrained_rows = [
        {
            "source_entity_id": "m.country",
            "target_entity_id": "m.religion",
            "target_label": "Islam",
            "relation_id": "location.country.religions",
            "score": 9.0,
            "matched_constraints": ["religion"],
        },
        {
            "source_entity_id": "m.country",
            "target_entity_id": "m.other",
            "target_label": "Hinduism",
            "relation_id": "location.country.religions",
            "score": 4.0,
            "matched_constraints": [],
        },
    ]
    solver = ConstrainedCollectSolver()

    result = solver.run(
        SubquestionExecutionContext(
            original_question="What religions are practiced in that country?",
            sub_question=SubQuestionSpec(
                id="sq2",
                question="What religions are practiced in that country?",
                solver_type="constrained_collect",
                interested_relations=[RelationHintSpec(name="practiced religion", aliases=["religion"])],
                expected_answer_type="religion",
                downstream_filters=["religion"],
            ),
            graph_api=graph_api,
            step_seed_ids=["m.country"],
            relation_ids=["location.country.religions"],
            resolved_sub_answers={},
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "constrained_collect"
    assert result.primary_entity_ids == ["m.religion", "m.other"]
    assert result.structured_outputs["solver_execution_mode"] == "constrained_collect"
    assert result.candidate_paths[0].nodes[-1] == "Islam"


def test_aggregate_solver_count() -> None:
    solver = AggregateSolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="How many institutions are there?",
            sub_question=SubQuestionSpec(id="sqc", question="How many institutions are there?", solver_type="aggregate"),
            graph_api=None,
            step_seed_ids=[],
            constraint_target_ids=[],
            relation_ids=[],
            resolved_sub_answers={
                "sq1": {"entity_ids": ["m.a", "m.b"], "entity_set_role": "comparison_candidates"},
                "sq2": {"entity_ids": ["m.b", "m.c"], "entity_set_role": "comparison_candidates"},
            },
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "aggregate"
    assert result.primary_literals == ["3"]


def test_aggregate_solver_fallback_when_missing_inputs() -> None:
    solver = AggregateSolver()
    result = solver.run(
        SubquestionExecutionContext(
            original_question="Which item is best?",
            sub_question=SubQuestionSpec(id="sqf", question="Which item is best?", solver_type="aggregate"),
            graph_api=None,
            step_seed_ids=[],
            constraint_target_ids=[],
            relation_ids=[],
            resolved_sub_answers={},
            execute_explore=lambda: [],
        )
    )

    assert result.solver_type == "explore"
    assert result.solver_debug["fallback_reason"] == "aggregate_unknown_operation"
