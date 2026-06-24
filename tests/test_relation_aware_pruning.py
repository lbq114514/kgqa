from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.entity_linking import link_relations_to_kg
from kgqa.kg.graph_api import ExternalGraphNeighbor
from kgqa.kg.graph import KnowledgeGraph
from kgqa.llm.base import BaseLLM
from kgqa.reasoning.pipeline import KGQAPipeline
from kgqa.reasoning.exploration import (
    explore_supplement_paths,
    generate_cwq_bootstrap_hints_with_llm,
    generate_supplement_hints_with_llm,
)
from kgqa.reasoning.pruning import precise_select_paths_with_llm, prune_paths, prune_relation_beam_paths
from kgqa.reasoning.question_analysis import analyze_question, plan_agentic_step
from kgqa.kg.relation_beam import NodeMetadata, RelationCandidate
from kgqa.utils.types import AgenticRunState, AgenticStepResult
from kgqa.utils.types import (
    EntityMentionSpec,
    ExplorationHints,
    QuestionAnalysisResult,
    ReasoningPath,
    RelationHintSpec,
    SubQuestionSpec,
    SupplementHints,
    Triple,
)


class FakeEmbeddingModel:
    """A tiny keyword-based encoder for deterministic similarity tests."""

    KEYWORDS = (
        "time",
        "zone",
        "containedby",
        "contain",
        "subject",
        "other",
        "united",
        "states",
        "america",
    )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = (
                text.lower()
                .replace(".", " ")
                .replace("_", " ")
                .replace("-", " ")
            )
            vector = [float(normalized.count(keyword)) for keyword in self.KEYWORDS]
            vector.append(1.0)
            vectors.append(vector)
        return vectors


class StaticLLM(BaseLLM):
    """Return a single fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self.response


class CaptureLLM(BaseLLM):
    """Capture the prompt while returning a fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_prompt = prompt
        return self.response


class SupervisingLLM(BaseLLM):
    """Return deterministic supervision JSON for dependency entities and relations."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        if "You filter upstream resolved entities for one KGQA sub-question." in prompt:
            return '{"selected_entity_ids":["m.team"]}'
        if "You disambiguate entity candidates for one KGQA mention in context." in prompt:
            return '{"selected_entity_ids":["m.city_nijmegen"]}'
        if "You select useful relation choices for one KGQA sub-question." in prompt:
            return '{"selected_relation_ids":["sports.sports_team.championships"]}'
        return "{}"


def make_path(*triples: Triple, source_stage: str = "topic_entity_path_exploration") -> ReasoningPath:
    nodes = [triples[0].head] if triples else ["seed"]
    for triple in triples:
        if triple.head == nodes[-1]:
            nodes.append(triple.tail)
        elif triple.tail == nodes[-1]:
            nodes.append(triple.head)
        else:
            nodes.append(triple.tail)
    text = " -> ".join(
        [nodes[0], *[f"{triple.relation} -> {nodes[index + 1]}" for index, triple in enumerate(triples)]]
    )
    return ReasoningPath(
        triples=list(triples),
        nodes=nodes,
        text=text,
        source_stage=source_stage,
    )


def test_link_relations_to_kg_uses_soft_top_k_and_dedup(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())

    linked = link_relations_to_kg(
        candidates=["time zone", "timezone"],
        kg_relations=[
            "location.location.containedby",
            "location.location.time_zones",
        ],
        top_k=2,
    )

    assert linked[0] == "location.location.time_zones"
    assert linked.count("location.location.time_zones") == 1


def test_generate_supplement_hints_parses_entities_and_relations(monkeypatch) -> None:
    monkeypatch.setattr(
        "kgqa.reasoning.exploration.link_entities_to_kg",
        lambda candidates, kg_entities, top_k=1: [candidate for candidate in candidates if candidate in kg_entities],
    )
    monkeypatch.setattr(
        "kgqa.reasoning.exploration.link_relations_to_kg",
        lambda candidates, kg_relations, top_k=3: ["location.location.time_zones"] if candidates else [],
    )

    hints = generate_supplement_hints_with_llm(
        question="what time zone is new york under",
        topic_entities=["New York"],
        current_paths=[],
        kg_entities=["United States of America"],
        kg_relations=["location.location.time_zones"],
        llm=StaticLLM('{"entities":["United States of America"],"relations":["time zone"]}'),
    )

    assert hints == SupplementHints(
        entity_candidates=["United States of America"],
        linked_entities=["United States of America"],
        relation_candidates=["time zone"],
        linked_relations=["location.location.time_zones"],
    )


def test_generate_cwq_bootstrap_hints_parses_structured_fields() -> None:
    analysis = QuestionAnalysisResult(
        split_questions=["Which country contains Northern District?", "What government type does it use?"],
        reasoning_indicator="Find the parent country, then the government type.",
        ordered_topic_entities=["Northern District"],
        predicted_depth=2,
        topic_entities=["Northern District"],
    )

    hints = generate_cwq_bootstrap_hints_with_llm(
        question="What type of government is used in the country with Northern District?",
        topic_entities=["Northern District"],
        question_analysis=analysis,
        llm=StaticLLM(
            '{"focus_entities":["Northern District"],'
            '"answer_type_hints":["government type","country"],'
            '"relation_name_hints":["administrative parent","government type"],'
            '"reasoning_focus":"Resolve the containing country first, then its government type."}'
        ),
    )

    assert hints == ExplorationHints(
        focus_entities=["Northern District"],
        answer_type_hints=["government type", "country"],
        relation_name_hints=["administrative parent", "government type"],
        aligned_entities=[],
        aligned_relations=[],
        reasoning_focus="Resolve the containing country first, then its government type.",
    )


def test_analyze_question_parses_sub_questions_and_keeps_legacy_fields() -> None:
    analysis = analyze_question(
        question="What type of government is used in the country with Northern District?",
        llm=StaticLLM(
            '{"topic_entities":["Northern District"],'
            '"split_questions":["Which country contains Northern District?",'
            '"What type of government does that country use?"],'
            '"sub_questions":['
            '{'
            '"id":"sq1",'
            '"question":"Which country contains Northern District?",'
            '"topic_entities":[{"name":"Northern District","aliases":[],"expected_type":"district","role":"topic_entity"}],'
            '"interested_nodes":[],'
            '"interested_relations":[{"name":"administrative parent","aliases":[],"freebase_like_ids":[],"direction":"topic_entity -> answer","description":"country containing Northern District"}],'
            '"expected_answer_type":"country",'
            '"expected_hop":1,'
            '"depends_on":[]'
            '}],'
            '"reasoning_indicator":"Find the parent country, then the government type.",'
            '"ordered_topic_entities":["Northern District"],'
            '"predicted_depth":2}'
        ),
        dmax=3,
        topic_entities_override=["Northern District"],
    )

    assert analysis.split_questions == [
        "Which country contains Northern District?",
        "What type of government does that country use?",
    ]
    assert analysis.predicted_depth == 2
    assert analysis.topic_entities == ["Northern District"]
    assert len(analysis.sub_questions) == 1
    assert analysis.sub_questions[0].id == "sq1"
    assert analysis.sub_questions[0].topic_entities[0].name == "Northern District"
    assert analysis.sub_questions[0].interested_relations[0].direction == "topic_entity -> answer"
    assert analysis.sub_questions[0].expected_hop == 1


def test_analyze_question_synthesizes_single_sub_question_for_legacy_schema() -> None:
    analysis = analyze_question(
        question="what did nick clegg study at university",
        llm=StaticLLM(
            '{"topic_entities":["Nick Clegg"],'
            '"split_questions":["What did Nick Clegg study at university?"],'
            '"reasoning_indicator":"Find Nick Clegg education records and field of study.",'
            '"ordered_topic_entities":["Nick Clegg"],'
            '"predicted_depth":1}'
        ),
        dmax=3,
        topic_entities_override=["Nick Clegg"],
    )

    assert analysis.split_questions == ["What did Nick Clegg study at university?"]
    assert len(analysis.sub_questions) == 1
    assert analysis.sub_questions[0].id == "sq1"
    assert analysis.sub_questions[0].expected_hop == 1
    assert analysis.sub_questions[0].question == "what did nick clegg study at university"
    assert analysis.sub_questions[0].topic_entities[0].name == "Nick Clegg"


def test_plan_agentic_step_parses_single_step_and_carryover() -> None:
    analysis = QuestionAnalysisResult(
        split_questions=["Which country contains Northern District?"],
        reasoning_indicator="Find the containing country, then the government type.",
        ordered_topic_entities=["Northern District"],
        predicted_depth=2,
        topic_entities=["Northern District"],
    )
    state = AgenticRunState(
        original_question="What type of government is used in the country with Northern District?",
        global_goal="Find the containing country, then the government type.",
        current_frontier_entities=["m.israel"],
        current_text_constraints=["Israel"],
        loop_index=1,
    )

    step = plan_agentic_step(
        question=state.original_question,
        llm=StaticLLM(
            '{"step_id":"step_2",'
            '"question":"What type of government does Israel use?",'
            '"goal":"Find the government type of the country found earlier.",'
            '"depends_on_step_ids":["step_1"],'
            '"topic_entity_mentions":[{"name":"Israel","aliases":[],"expected_type":"country","role":"topic_entity"}],'
            '"relation_hints":[{"name":"government type","aliases":[],"freebase_like_ids":[],"direction":"country -> answer","description":"government type"}],'
            '"expected_answer_type":"government type",'
            '"carryover_constraints":["Israel"],'
            '"stop_if_answered":true,'
            '"strategy":"external_graph"}'
        ),
        question_analysis=analysis,
        run_state=state,
        dmax=3,
        max_loops=5,
    )

    assert step.step_id == "step_2"
    assert step.topic_entity_mentions[0].name == "Israel"
    assert step.relation_hints[0].name == "government type"
    assert step.carryover_constraints == ["Israel"]
    assert step.stop_if_answered is True


def test_plan_agentic_step_prompt_omits_full_retrieved_paths() -> None:
    analysis = QuestionAnalysisResult(
        split_questions=["q"],
        reasoning_indicator="r",
        ordered_topic_entities=["A"],
        predicted_depth=2,
        topic_entities=["A"],
    )
    state = AgenticRunState(
        original_question="q",
        global_goal="r",
        loop_index=1,
    )
    state.step_history.append(
        AgenticStepResult(
            step_id="step_1",
            retrieved_paths=[
                make_path(
                    Triple("A", "rel", "B"),
                    Triple("B", "rel2", "C"),
                )
            ],
            summarized_evidence=[{"summary_type": "sub_question", "key_triples": [], "evidence": ["A -> B"]}],
        )
    )
    llm = CaptureLLM(
        '{"step_id":"step_2","question":"q2","topic_entity_mentions":[{"name":"B","aliases":[],"expected_type":"","role":"topic_entity"}],"relation_hints":[]}'
    )

    plan_agentic_step(
        question="q",
        llm=llm,
        question_analysis=analysis,
        run_state=state,
        dmax=3,
        max_loops=5,
    )

    assert "retrieved_paths" not in llm.last_prompt


def test_explore_supplement_paths_prioritizes_relation_hits() -> None:
    graph = KnowledgeGraph(
        [
            Triple("New York", "location.location.containedby", "United States of America"),
            Triple("New York", "book.written_work.subjects", "United States of America"),
        ]
    )
    hints = SupplementHints(
        linked_entities=["United States of America"],
        linked_relations=["location.location.containedby"],
    )

    paths = explore_supplement_paths(
        subgraph=graph,
        topic_entities=["New York"],
        supplement_hints=hints,
        dmax=2,
        max_paths_per_pair=10,
    )

    assert len(paths) == 2
    assert paths[0].matched_relations == ["location.location.containedby"]
    assert paths[1].matched_relations == []


def test_prune_paths_prefers_relation_hits_but_falls_back_without_hints(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())

    hit_path = make_path(Triple("A", "rel_hit", "B"))
    hit_path.text = "question hit relation"
    other_path = make_path(Triple("A", "rel_other", "B"))
    other_path.text = "other evidence path"
    analysis = QuestionAnalysisResult(
        split_questions=["q"],
        reasoning_indicator="other",
        ordered_topic_entities=["A"],
        predicted_depth=1,
        topic_entities=["A"],
    )

    without_relation_hints = prune_paths(
        question="q",
        question_analysis=analysis,
        candidate_paths=[hit_path, other_path],
        w1=1,
        wmax=1,
        relation_hints=[],
        sample_id="sample-without-rel",
        compact_precise_path_prompt=False,
        llm=StaticLLM("[0]"),
    )
    assert without_relation_hints[0].triples[0].relation == "rel_other"

    hit_path = make_path(Triple("A", "rel_hit", "B"))
    hit_path.text = "question hit relation"
    other_path = make_path(Triple("A", "rel_other", "B"))
    other_path.text = "other evidence path"
    with_relation_hints = prune_paths(
        question="q",
        question_analysis=analysis,
        candidate_paths=[hit_path, other_path],
        w1=1,
        wmax=1,
        relation_hints=["rel_hit"],
        sample_id="sample-with-rel",
        compact_precise_path_prompt=False,
        llm=StaticLLM("[0]"),
    )

    assert with_relation_hints[0].triples[0].relation == "rel_hit"
    assert with_relation_hints[0].matched_relations == ["rel_hit"]


def test_prune_relation_beam_paths_collapses_same_terminal_to_shorter_path() -> None:
    short_path = make_path(
        Triple("A", "sports.team.championships", "2012 World Series"),
        source_stage="relation_beam",
    )
    short_path.terminal_node_id = "m.ws2012"
    short_path.terminal_node_kind = "id"
    short_path.matched_relations = ["sports.team.championships"]
    short_path.path_score = 0.8

    long_path = make_path(
        Triple("A", "sports.team.history", "season"),
        Triple("season", "sports.team.championships", "2012 World Series"),
        source_stage="relation_beam",
    )
    long_path.terminal_node_id = "m.ws2012"
    long_path.terminal_node_kind = "id"
    long_path.matched_relations = ["sports.team.championships"]
    long_path.path_score = 0.9

    selected = prune_relation_beam_paths([long_path, short_path], wmax=3)

    assert selected == [short_path]
    assert long_path.pruning_status == "relation_beam_consolidated_pruned"
    assert short_path.pruning_status == "preserved"


def test_prune_relation_beam_paths_preserves_diverse_terminals() -> None:
    path1 = make_path(Triple("A", "rel.answer", "B"), source_stage="relation_beam")
    path1.terminal_node_id = "m.answer1"
    path1.terminal_node_kind = "id"
    path1.path_score = 0.9

    path2 = make_path(Triple("A", "rel.answer", "C"), source_stage="relation_beam")
    path2.terminal_node_id = "m.answer2"
    path2.terminal_node_kind = "id"
    path2.path_score = 0.8

    path3 = make_path(Triple("A", "rel.answer", "D"), source_stage="relation_beam")
    path3.terminal_node_id = "m.answer3"
    path3.terminal_node_kind = "id"
    path3.path_score = 0.7

    selected = prune_relation_beam_paths([path1, path2, path3], wmax=2)

    assert len(selected) == 2
    assert {path.terminal_node_id for path in selected} == {"m.answer1", "m.answer2"}


def test_prune_relation_beam_paths_does_not_drop_empty_matched_relations() -> None:
    beam_path = make_path(
        Triple("Forrest Gump", "film.film.starring", "Michael Connor Humphreys"),
        source_stage="relation_beam",
    )
    beam_path.terminal_node_id = "m.actor"
    beam_path.terminal_node_kind = "id"
    beam_path.matched_relations = []
    beam_path.path_score = 0.75

    selected = prune_relation_beam_paths([beam_path], wmax=3)

    assert selected == [beam_path]
    assert beam_path.pruning_status == "preserved"


def test_relation_beam_pruning_path_does_not_call_fuzzy_or_precise(monkeypatch) -> None:
    calls = {"fuzzy": 0, "precise": 0}

    def fake_fuzzy(*args, **kwargs):
        calls["fuzzy"] += 1
        return []

    def fake_precise(*args, **kwargs):
        calls["precise"] += 1
        return []

    monkeypatch.setattr("kgqa.reasoning.pruning.fuzzy_select_paths", fake_fuzzy)
    monkeypatch.setattr("kgqa.reasoning.pruning.precise_select_paths_with_llm", fake_precise)
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=StaticLLM("{}"),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"search": {"subquestion_explorer": "relation_beam"}},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )
    analysis = QuestionAnalysisResult(
        split_questions=["q"],
        reasoning_indicator="r",
        ordered_topic_entities=["A"],
        predicted_depth=1,
        topic_entities=["A"],
    )
    beam_path = make_path(Triple("A", "rel", "B"), source_stage="relation_beam")
    beam_path.terminal_node_id = "m.b"
    beam_path.terminal_node_kind = "id"

    pruned = pipeline._prune_paths_for_context(
        question="q",
        question_analysis=analysis,
        candidate_paths=[beam_path],
        w1=5,
        wmax=2,
        relation_hints=[],
        sample_id="sample",
        trace=[],
        depth=1,
        stage_note="test",
    )

    assert pruned == [beam_path]
    assert calls == {"fuzzy": 0, "precise": 0}


def test_relation_beam_pruning_keeps_beam_priority_with_mixed_sources(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=StaticLLM("{}"),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"search": {"subquestion_explorer": "relation_beam"}},
            "search": {"dmax": 2, "w1": 5, "wmax": 1, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )
    analysis = QuestionAnalysisResult(
        split_questions=["q"],
        reasoning_indicator="r",
        ordered_topic_entities=["A"],
        predicted_depth=1,
        topic_entities=["A"],
    )
    beam_path = make_path(Triple("A", "film.film.starring", "Actor"), source_stage="relation_beam")
    beam_path.terminal_node_id = "m.actor"
    beam_path.terminal_node_kind = "id"
    beam_path.path_score = 0.9
    strict_path = make_path(Triple("A", "award.award_nominee", "Nominee"), source_stage="external_graphapi_strict_probe")
    strict_path.terminal_node_id = "m.nominee"
    strict_path.terminal_node_kind = "id"

    trace: list[object] = []
    pruned = pipeline._prune_paths_for_context(
        question="q",
        question_analysis=analysis,
        candidate_paths=[beam_path, strict_path],
        w1=5,
        wmax=1,
        relation_hints=[],
        sample_id="sample",
        trace=trace,
        depth=1,
        stage_note="test",
    )

    assert pruned == [beam_path]
    assert strict_path.pruning_status == "relation_beam_consolidated_pruned"
    assert beam_path.pruning_status == "preserved"
    assert "relation_beam_evidence_consolidation" in trace[-1].note
    assert "mixed_sources=True" in trace[-1].note


def test_legacy_pruning_still_uses_existing_flow(monkeypatch) -> None:
    calls = {"fuzzy": 0}
    real_fuzzy = __import__("kgqa.reasoning.pruning", fromlist=["fuzzy_select_paths"]).fuzzy_select_paths

    def counting_fuzzy(*args, **kwargs):
        calls["fuzzy"] += 1
        return real_fuzzy(*args, **kwargs)

    monkeypatch.setattr("kgqa.reasoning.pruning.fuzzy_select_paths", counting_fuzzy)
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())

    path = make_path(Triple("A", "rel", "B"))
    path.text = "evidence"
    analysis = QuestionAnalysisResult(
        split_questions=["q"],
        reasoning_indicator="evidence",
        ordered_topic_entities=["A"],
        predicted_depth=1,
        topic_entities=["A"],
    )

    selected = prune_paths(
        question="q",
        question_analysis=analysis,
        candidate_paths=[path],
        w1=1,
        wmax=1,
        relation_hints=[],
        sample_id="legacy",
        compact_precise_path_prompt=False,
        llm=StaticLLM("[0]"),
    )

    assert selected == [path]
    assert calls["fuzzy"] == 1


def test_precise_selection_can_use_compact_path_prompt() -> None:
    llm = CaptureLLM("[0]")
    path = make_path(
        Triple("Nick Clegg", "people.person.education", "education record"),
        Triple("education record", "education.education.major_field_of_study", "Social anthropology"),
    )
    analysis = QuestionAnalysisResult(
        split_questions=["What did Nick Clegg study at university?"],
        reasoning_indicator="Find education then field of study.",
        ordered_topic_entities=["Nick Clegg"],
        predicted_depth=2,
        topic_entities=["Nick Clegg"],
    )

    selected = precise_select_paths_with_llm(
        question="what did nick clegg study at university",
        question_analysis=analysis,
        candidate_paths=[path],
        wmax=1,
        supplement_relations=["education.education.major_field_of_study"],
        sample_id="compact-path-sample",
        compact_prompt_paths=True,
        llm=llm,
    )

    assert selected == [path]
    assert "Nick Clegg | people.person.education | education.education.major_field_of_study | Social anthropology" in llm.last_prompt
    assert path.text not in llm.last_prompt


def test_local_hint_rerank_prefers_paths_matching_interested_nodes_and_relations(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())
    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=StaticLLM("{}"),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"max_paths": 5},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )
    hit_path = make_path(
        Triple("San Francisco Giants", "sports.sports_team.championships", "2012 World Series"),
        source_stage="external_graphapi_supplement",
    )
    miss_path = make_path(
        Triple("San Francisco Giants", "award.award_nominee.award_nominations.award.award_nomination.year", "2013"),
        source_stage="external_graphapi_supplement",
    )
    sub_question = SubQuestionSpec(
        id="sq2",
        question="When did the San Francisco Giants last win the World Series?",
        topic_entities=[EntityMentionSpec(name="San Francisco Giants", role="topic_entity")],
        interested_nodes=[EntityMentionSpec(name="World Series", role="constraint_or_anchor")],
        interested_relations=[RelationHintSpec(name="championships", aliases=["won World Series"])],
        expected_answer_type="date",
    )

    reranked = pipeline._rerank_candidate_paths_with_local_hints(
        sub_question=sub_question,
        candidate_paths=[miss_path, hit_path],
        trace=[],
        depth=1,
    )

    assert reranked[0].triples[0].relation == "sports.sports_team.championships"


def test_local_relation_probe_selects_championship_relation(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    class LocalProbeGraphAPI:
        def get_neighbors(
            self,
            node_id: str,
            include_reverse: bool = True,
            relation_filter: list[str] | None = None,
            limit: int | None = None,
            strict_relation_filter: bool = False,
        ) -> list[ExternalGraphNeighbor]:
            if node_id != "m.team":
                return []
            neighbors = [
                ExternalGraphNeighbor(
                    source_id="m.team",
                    neighbor_id="m.ws2012",
                    triple=Triple(
                        head="San Francisco Giants",
                        relation="sports.sports_team.championships",
                        tail="2012 World Series",
                    ),
                    relation_name="championships",
                    reversed=False,
                ),
                ExternalGraphNeighbor(
                    source_id="m.team",
                    neighbor_id="2013",
                    triple=Triple(
                        head="San Francisco Giants",
                        relation="award.award_nominee.award_nominations.award.award_nomination.year",
                        tail="2013",
                    ),
                    relation_name="award year",
                    reversed=False,
                ),
            ]
            if relation_filter and strict_relation_filter:
                return [edge for edge in neighbors if edge.triple.relation in relation_filter]
            return neighbors

        def get_relation_display_name(self, relation_id: str) -> str:
            return {
                "sports.sports_team.championships": "championships",
                "award.award_nominee.award_nominations.award.award_nomination.year": "award year",
                "time.event.end_date": "end date",
            }.get(relation_id, relation_id)

        def expand_paths(
            self,
            seed_nodes: list[str],
            relation_hints: list[str] | None = None,
            max_depth: int = 3,
            strict_relation_filter: bool = True,
            max_paths: int | None = None,
            max_nodes: int | None = None,
        ) -> list[ReasoningPath]:
            if "sports.sports_team.championships" not in (relation_hints or []):
                return []
            return [
                ReasoningPath(
                    triples=[
                        Triple("San Francisco Giants", "sports.sports_team.championships", "2012 World Series"),
                        Triple("2012 World Series", "time.event.end_date", "2012-10-28"),
                    ],
                    nodes=["San Francisco Giants", "2012 World Series", "2012-10-28"],
                    text="San Francisco Giants -> championships -> 2012 World Series -> end date -> 2012-10-28",
                    source_stage="external_graphapi_expansion",
                )
            ]

        def find_two_hop_extensions(
            self,
            frontier_nodes: list[str],
            relation_hints: list[str] | None = None,
            limit: int | None = None,
        ) -> list[ReasoningPath]:
            if "sports.sports_team.championships" not in (relation_hints or []):
                return []
            return [
                ReasoningPath(
                    triples=[
                        Triple("San Francisco Giants", "sports.sports_team.championships", "2012 World Series"),
                    ],
                    nodes=["San Francisco Giants", "2012 World Series"],
                    text="San Francisco Giants -> championships -> 2012 World Series",
                    source_stage="external_graphapi_supplement",
                )
            ]

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=StaticLLM("{}"),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False, "neighbor_limit": 50, "cwq_max_paths": 10, "cwq_max_nodes": 10},
            "retrieval": {"max_paths": 5},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
            "agentic": {"local_relation_probe_top_k": 2, "max_loops": 3},
        },
    )
    sub_question = SubQuestionSpec(
        id="sq2",
        question="When did that team last win the World Series?",
        topic_entities=[EntityMentionSpec(name="San Francisco Giants", role="topic_entity")],
        interested_nodes=[EntityMentionSpec(name="World Series", role="constraint_or_anchor")],
        interested_relations=[RelationHintSpec(name="last won", aliases=["last championship year"])],
        expected_answer_type="date",
    )
    trace: list = []
    graph_api = LocalProbeGraphAPI()

    relation_ids = pipeline._select_local_relation_probe_ids(
        sub_question=sub_question,
        step_seed_ids=["m.team"],
        graph_api=graph_api,
        candidate_paths=[],
        trace=trace,
        depth=1,
        attempt_index=1,
    )
    probed_paths = pipeline._probe_paths_with_local_relations(
        sub_question=sub_question,
        step_seed_ids=["m.team"],
        relation_ids=relation_ids,
        graph_api=graph_api,
        trace=trace,
        depth=1,
    )

    assert relation_ids == ["sports.sports_team.championships"]
    assert any(path.text.endswith("2012-10-28") for path in probed_paths)


def test_dependency_entity_filter_prefers_primary_upstream_answer(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    class LabelsOnlyGraphAPI:
        def get_entity_display_name(self, node_id: str) -> str:
            return {
                "m.team": "San Francisco Giants",
                "m.player": "Brett Bochy",
            }.get(node_id, node_id)

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=SupervisingLLM(),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"max_paths": 5},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )
    sub_question = SubQuestionSpec(
        id="sq2",
        question="When did that team last win the World Series?",
        expected_answer_type="date",
        depends_on=["sq1"],
    )
    selected = pipeline._filter_dependency_seed_ids_with_llm(
        sub_question=sub_question,
        dependency_seed_ids=["m.team", "m.player"],
        resolved_sub_answers={
            "sq1": {
                "answer": "San Francisco Giants",
                "entity_ids": ["m.team", "m.player"],
            }
        },
        graph_api=LabelsOnlyGraphAPI(),
        trace=[],
        depth=1,
    )

    assert selected == ["m.team"]


def test_relation_supervision_overrides_heuristic_drift(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=SupervisingLLM(),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"max_paths": 5},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )
    sub_question = SubQuestionSpec(
        id="sq2",
        question="When did that team last win the World Series?",
        interested_nodes=[EntityMentionSpec(name="World Series", role="constraint_or_anchor")],
        expected_answer_type="date",
        depends_on=["sq1"],
    )
    selected = pipeline._supervise_local_relation_probe_ids_with_llm(
        sub_question=sub_question,
        scored_rows=[
            (
                20.0,
                {
                    "relation_id": "award.award_nominee.award_nominations.award.award_nomination.notes_description",
                    "display_name": "award nomination notes description",
                    "targets": ["2002 World Series Game 6"],
                },
            ),
            (
                12.0,
                {
                    "relation_id": "sports.sports_team.championships",
                    "display_name": "championships",
                    "targets": ["2010 World Series", "2012 World Series"],
                },
            ),
        ],
        beam_suggested_ids=[],
        trace=[],
        depth=1,
    )

    assert selected == ["sports.sports_team.championships"]


def test_relation_supervision_falls_back_when_llm_selects_nothing(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    class WeakOnlyLLM(BaseLLM):
        def generate(self, prompt: str, **kwargs: object) -> str:
            if "You select useful relation choices for one KGQA sub-question." in prompt:
                return '{"selected_relation_ids":[]}'
            return "{}"

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=WeakOnlyLLM(),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"max_paths": 5},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )
    selected = pipeline._supervise_local_relation_probe_ids_with_llm(
        sub_question=SubQuestionSpec(
            id="sq2",
            question="What religions does that country practice?",
            expected_answer_type="religion",
        ),
        scored_rows=[
            (12.0, {"relation_id": "book.book.editions", "display_name": "editions", "targets": ["Some book"]}),
            (11.0, {"relation_id": "music.recording.tracks", "display_name": "tracks", "targets": ["Some song"]}),
        ],
        beam_suggested_ids=["music.recording.tracks"],
        trace=[],
        depth=1,
    )

    assert selected == ["music.recording.tracks"]


def test_sqlite_entity_resolution_reranks_by_type_and_relation_context(monkeypatch) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    class FakeGraphAPI:
        def resolve_entity_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
            assert top_k >= 10
            return [
                {"mid": "m.city_nijmegen", "name": "Nijmegen", "mention": "Nijmegen", "match_type": "name_exact", "score": 95.0},
                {"mid": "m.tv_nijmegen", "name": "Nijmegen", "mention": "Nijmegen", "match_type": "name_exact", "score": 95.0},
            ]

        def get_nodes_metadata_batched(self, node_ids: list[str]) -> dict[str, NodeMetadata]:
            return {
                "m.city_nijmegen": NodeMetadata(
                    node_id="m.city_nijmegen",
                    name="Nijmegen",
                    types=("location.citytown", "location.location"),
                    degree=8,
                    is_literal=False,
                    is_probable_cvt=False,
                ),
                "m.tv_nijmegen": NodeMetadata(
                    node_id="m.tv_nijmegen",
                    name="Nijmegen",
                    types=("tv.tv_series_episode",),
                    degree=11,
                    is_literal=False,
                    is_probable_cvt=False,
                ),
            }

        def get_node_relations(self, node_id: str, include_reverse: bool = True, include_literals: bool = False) -> list[RelationCandidate]:
            if node_id == "m.city_nijmegen":
                return [
                    RelationCandidate(
                        relation="location.location.containedby",
                        relation_name="contained by",
                        direction="forward",
                        supporting_node_count=1,
                        total_neighbor_count=2,
                        support_ratio=1.0,
                        global_frequency=10,
                        sample_target_types=("location.country",),
                        sample_target_names=("Netherlands",),
                    )
                ]
            return [
                RelationCandidate(
                    relation="tv.tv_program.episodes",
                    relation_name="episodes",
                    direction="reverse",
                    supporting_node_count=1,
                    total_neighbor_count=1,
                    support_ratio=1.0,
                    global_frequency=10,
                    sample_target_types=("tv.tv_program",),
                    sample_target_names=("Nederland zingt",),
                )
            ]

        def get_entity_display_name(self, entity_id: str) -> str:
            return entity_id

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=SupervisingLLM(),
        config={
            "embedding": {"model": "fake"},
            "graphapi": {"enabled": False},
            "retrieval": {"max_paths": 5, "entity_candidate_top_k": 10, "entity_candidate_selected_top_k": 1},
            "search": {"dmax": 2, "w1": 5, "wmax": 2, "max_expand_steps": 1, "max_paths_per_pair": 5},
        },
    )

    resolved = pipeline._resolve_entity_specs_with_sqlite(
        specs=[EntityMentionSpec(name="Nijmegen", expected_type="city", role="topic_entity")],
        graph_api=FakeGraphAPI(),
        context_text="What country bordering France contains an airport that serves Nijmegen?",
    )

    assert [item.id_or_mid for item in resolved] == ["m.city_nijmegen"]
