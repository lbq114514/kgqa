from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.llm.base import BaseLLM
from kgqa.reasoning.answering import check_sufficiency, generate_answer
from kgqa.reasoning.pipeline import KGQAPipeline
from kgqa.reasoning.summarization import summarize_paths
from kgqa.utils.types import (
    AnswerGrounding,
    AnswerResult,
    EntityMentionSpec,
    QuestionAnalysisResult,
    ReasoningPath,
    SubQuestionSpec,
    Triple,
    TripleFact,
)


class CaptureLLM(BaseLLM):
    """Capture the last prompt and return a fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_prompt = prompt
        return self.response


class SequenceLLM(BaseLLM):
    """Return responses in order and capture prompts."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _analysis() -> QuestionAnalysisResult:
    return QuestionAnalysisResult(
        split_questions=["What did Nick Clegg study at university?"],
        reasoning_indicator="Find the education record, then the major field of study.",
        ordered_topic_entities=["Nick Clegg"],
        predicted_depth=2,
        topic_entities=["Nick Clegg"],
    )


def _sports_team_analysis() -> QuestionAnalysisResult:
    return QuestionAnalysisResult(
        split_questions=[
            "Who is Peyton Manning's dad?",
            "For what sports team that makes its home at Mercedes-Benz Superdome did he play?",
        ],
        reasoning_indicator="Find Peyton Manning's father, then the matching team.",
        ordered_topic_entities=["Peyton Manning", "Mercedes-Benz Superdome"],
        predicted_depth=2,
        topic_entities=["Peyton Manning", "Mercedes-Benz Superdome"],
    )


def _paths() -> list[ReasoningPath]:
    return [
        ReasoningPath(
            triples=[
                Triple("Nick Clegg", "people.person.education", "education record"),
                Triple(
                    "education record",
                    "education.education.major_field_of_study",
                    "Social anthropology",
                ),
            ],
            nodes=["Nick Clegg", "education record", "Social anthropology"],
            text=(
                "Nick Clegg -> people.person.education -> education record -> "
                "education.education.major_field_of_study -> Social anthropology"
            ),
            source_stage="external_graphapi_node_expand",
        )
    ]


def test_summarization_prompt_includes_question_analysis_context() -> None:
    llm = CaptureLLM(
        '{"question":"q","question_focus":"q","key_triples":[],"evidence":[]}'
    )

    summarize_paths(
        question="what did nick clegg study at university",
        topic_entities=["Nick Clegg"],
        question_analysis=_analysis(),
        pruned_paths=_paths(),
        llm=llm,
    )

    assert '"predicted_depth": 2' in llm.last_prompt
    assert '"reasoning_indicator": "Find the education record, then the major field of study."' in llm.last_prompt
    assert 'Topic entities: ["Nick Clegg"]' in llm.last_prompt
    assert "Sub-question context:" in llm.last_prompt


def test_answering_and_sufficiency_prompts_include_search_context() -> None:
    summarized_paths = [
        {
            "question": "what did nick clegg study at university",
            "question_focus": "field of study of Nick Clegg",
            "key_triples": [
                {
                    "head": "education record",
                    "relation": "education.education.major_field_of_study",
                    "tail": "Social anthropology",
                }
            ],
            "evidence": [
                "education record -> education.education.major_field_of_study -> Social anthropology"
            ],
        }
    ]

    sufficiency_llm = CaptureLLM('{"sufficient": true, "reason": "enough"}')
    answer_llm = CaptureLLM(
        '{"predicted_answers":["Social anthropology"],"answer":"Social anthropology","supporting_paths":[]}'
    )

    check_sufficiency(
        question="what did nick clegg study at university",
        topic_entities=["Nick Clegg"],
        question_analysis=_analysis(),
        dmax=3,
        dpredict=2,
        split_questions=_analysis().split_questions,
        summarized_paths=summarized_paths,
        llm=sufficiency_llm,
    )
    generate_answer(
        question="what did nick clegg study at university",
        topic_entities=["Nick Clegg"],
        question_analysis=_analysis(),
        dmax=3,
        dpredict=2,
        split_questions=_analysis().split_questions,
        summarized_paths=summarized_paths,
        llm=answer_llm,
    )

    for prompt in (sufficiency_llm.last_prompt, answer_llm.last_prompt):
        assert '"predicted_depth":2' in prompt
        assert '"current_evaluation_depth":2' in prompt
        assert '"dmax":3' in prompt
        assert '"ordered_topic_entities":[' in prompt
        assert '"sub_questions":[]' in prompt
        assert '"reasoning_indicator":"Find the education record, then the major field of study."' in prompt
        assert "Task analysis:" in prompt
        assert "How to use the context:" in prompt


def test_generate_answer_filters_failure_placeholder_answers() -> None:
    llm = CaptureLLM(
        '{"predicted_answers":["insufficient evidence"],"answer":"insufficient evidence","supporting_paths":[]}'
    )

    result = generate_answer(
        question="what did nick clegg study at university",
        topic_entities=["Nick Clegg"],
        question_analysis=_analysis(),
        dmax=3,
        dpredict=2,
        split_questions=_analysis().split_questions,
        summarized_paths=[],
        llm=llm,
    )

    assert result.sufficient is False
    assert result.predicted_answers == []
    assert result.answer == "insufficient evidence"


def test_location_entity_query_names_include_explicit_comma_qualified_span() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    spec = EntityMentionSpec(name="Vienna", expected_type="city", role="topic_entity")

    names = pipeline._build_entity_query_names(
        spec,
        context_text="What museum is located in Vienna, Austria,and was established the latest?",
    )

    assert names[:3] == ["Vienna, Austria", "Vienna Austria", "Vienna"]


def test_entity_query_names_ignore_comma_span_for_non_location_types() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    spec = EntityMentionSpec(name="Mercury", expected_type="song", role="topic_entity")

    names = pipeline._build_entity_query_names(
        spec,
        context_text="Who wrote Mercury, Venus and Mars?",
    )

    assert "Mercury, Venus" not in names
    assert names[0] == "Mercury"


def test_entity_query_names_ignore_comma_span_without_location_type() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    spec = EntityMentionSpec(name="Mercury", role="topic_entity")

    names = pipeline._build_entity_query_names(
        spec,
        context_text="Mercury, Venus and Mars are nearby mentions.",
    )

    assert "Mercury, Venus" not in names
    assert names[0] == "Mercury"


def test_character_entity_query_names_include_terminal_typo_variant() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    spec = EntityMentionSpec(name="Luke Castelland", expected_type="character", role="constraint_or_anchor")

    names = pipeline._build_entity_query_names(
        spec,
        context_text="What film does Logan Lerman play the character Luke Castelland in?",
    )

    assert "Luke Castellan" in names
    assert "Luke Castellan character" in names


def test_constraint_plan_matches_non_terminal_fact_entity() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    sub_question = type(
        "SubQuestion",
        (),
        {
            "interested_nodes": [EntityMentionSpec(name="Luke Castellan", expected_type="character")],
            "interested_relations": [],
        },
    )()
    plan = pipeline._constraint_plan_for_subquestion(sub_question, constraint_target_ids=["m.character"])
    path = ReasoningPath(
        triples=[
            Triple(head="Logan Lerman", relation="film.actor.film", tail="m.performance"),
            Triple(head="m.performance", relation="film.performance.film", tail="Percy Jackson"),
            Triple(head="m.performance", relation="film.performance.character", tail="Luke Castellan"),
        ],
        nodes=["Logan Lerman", "m.performance", "Percy Jackson", "Luke Castellan"],
        text="Logan Lerman -> film -> [CVT] ; [CVT] {film: Percy Jackson; character: Luke Castellan}",
        source_stage="relation_beam_cvt_bundle",
        triple_facts=[
            TripleFact(
                head_id="m.performance",
                head_label="m.performance",
                relation_id="film.performance.character",
                tail_id="m.character",
                tail_label="Luke Castellan",
                tail_kind="id",
            )
        ],
        terminal_node_id="m.film",
        terminal_node_kind="id",
    )

    assert pipeline._path_matches_constraint_plan(path, plan) is True


def test_subquestion_grounding_payload_preserves_full_answer_set() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    pipeline.config = {"agentic": {"max_carryover_entities": 2, "max_carryover_text_values": 2}}
    grounding = AnswerGrounding(
        answer_texts=["Islam", "Hinduism", "Catholicism", "Protestantism"],
        entity_ids=["m.islam", "m.hinduism", "m.catholicism", "m.protestantism"],
        literal_values=["86.1", "1.8", "3.0", "5.7"],
        primary_answer_text="Islam",
    )

    entity_ids, literals, answer = pipeline._project_grounding_to_payload(grounding)

    assert entity_ids == ["m.islam", "m.hinduism", "m.catholicism", "m.protestantism"]
    assert literals == ["Islam", "Hinduism", "Catholicism", "Protestantism", "86.1", "1.8", "3.0", "5.7"]
    assert answer == "Islam"


def test_subquestion_grounding_aligns_entity_ids_to_subanswer_text() -> None:
    class Metadata:
        def __init__(self, name: str, types: tuple[str, ...]) -> None:
            self.name = name
            self.types = types
            self.is_probable_cvt = False
            self.degree = 1

    class GraphAPI:
        metadata = {
            "m.movie": Metadata("Forrest Gump", ("film.film",)),
            "m.actor": Metadata("Kevin Mangan", ("film.actor", "people.person")),
        }

        def get_nodes_metadata_batched(self, ids: list[str]) -> dict[str, Metadata]:
            return {entity_id: self.metadata[entity_id] for entity_id in ids if entity_id in self.metadata}

        def get_entity_display_name(self, entity_id: str) -> str:
            metadata = self.metadata.get(entity_id)
            return metadata.name if metadata is not None else entity_id

        def resolve_entity_mentions(self, labels: list[str], top_k: int = 1) -> list[str]:
            mapping = {"Forrest Gump": "m.movie", "Kevin Mangan": "m.actor"}
            return [mapping[label] for label in labels if label in mapping]

    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    pipeline.config = {"answer_filtering": {"type_filtering": {"enabled": True}}}
    sub_question = SubQuestionSpec(
        id="sq1",
        question="Which movie has a character named Jenny's Father?",
        expected_answer_type="movie",
    )
    summary = {
        "key_triples": [
            {"head": "Jenny's Father", "relation": "film.performance.film", "tail": "Forrest Gump"},
            {"head": "Jenny's Father", "relation": "film.performance.actor", "tail": "Kevin Mangan"},
        ],
        "answer_view_facts": [
            {
                "head_id": "m.character",
                "head_label": "Jenny's Father",
                "relation_id": "film.performance.film",
                "tail_id": "m.movie",
                "tail_label": "Forrest Gump",
                "tail_kind": "id",
            },
            {
                "head_id": "m.character",
                "head_label": "Jenny's Father",
                "relation_id": "film.performance.actor",
                "tail_id": "m.actor",
                "tail_label": "Kevin Mangan",
                "tail_kind": "id",
            },
        ],
    }

    grounding, _stats = pipeline._build_subquestion_grounding(
        graph_api=GraphAPI(),
        sub_question=sub_question,
        pruned_paths=[],
        step_summaries=[summary],
        answer_result=AnswerResult(
            sufficient=True,
            answer="Forrest Gump",
            predicted_answers=["Forrest Gump"],
            resolved_literals=["Forrest Gump"],
        ),
    )

    assert grounding.primary_answer_text == "Forrest Gump"
    assert grounding.entity_ids == ["m.movie"]
    assert grounding.answer_texts == ["Forrest Gump"]


def test_final_answer_uses_llm_list_without_grounding_projection() -> None:
    llm = SequenceLLM(
        [
            '{"sufficient":true,"reason":"enough","primary_answer":"New Orleans Saints","answer_candidates":["New Orleans Saints"]}',
            '{"predicted_answers":["New Orleans Saints"],"answer":"New Orleans Saints","supporting_paths":[]}',
        ]
    )
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    pipeline.llm = llm
    pipeline.config = {}
    summarized_paths = [
        {
            "question": "For what sports team that makes its home at Mercedes-Benz Superdome did Peyton Manning's dad play?",
            "summary_type": "aggregate",
            "key_triples": [
                {
                    "head": "Archie Manning",
                    "relation": "sports.sports_team_roster.team",
                    "tail": "Houston Oilers",
                },
                {
                    "head": "Archie Manning",
                    "relation": "sports.sports_team_roster.team",
                    "tail": "New Orleans Saints",
                },
            ],
            "answer_view_facts": [
                {
                    "head_id": "m.archie",
                    "head_label": "Archie Manning",
                    "relation_id": "sports.sports_team_roster.team",
                    "tail_id": "m.oilers",
                    "tail_label": "Houston Oilers",
                    "tail_kind": "id",
                },
                {
                    "head_id": "m.archie",
                    "head_label": "Archie Manning",
                    "relation_id": "sports.sports_team_roster.team",
                    "tail_id": "m.saints",
                    "tail_label": "New Orleans Saints",
                    "tail_kind": "id",
                },
                {
                    "head_id": "m.archie",
                    "head_label": "Archie Manning",
                    "relation_id": "sports.sports_team_roster.team",
                    "tail_id": "m.vikings",
                    "tail_label": "Minnesota Vikings",
                    "tail_kind": "id",
                },
            ],
        }
    ]

    answer_result, sufficient = pipeline._answer_from_summaries(
        question="For what sports team that makes its home at Mercedes-Benz Superdome did Peyton Manning's dad play?",
        topic_entities=["Peyton Manning", "Mercedes-Benz Superdome"],
        question_analysis=_sports_team_analysis(),
        summarized_paths=summarized_paths,
        dmax=3,
        depth=2,
        trace=[],
    )

    assert sufficient is True
    assert answer_result.predicted_answers == ["New Orleans Saints"]
    assert answer_result.answer == "New Orleans Saints"
    assert answer_result.grounding is None


def test_aggregate_summary_does_not_mark_all_key_facts_as_answer_view() -> None:
    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    item = {
        "sub_question_id": "sq2",
        "key_triples": [
            {
                "head": "Archie Manning",
                "relation": "sports.sports_team_roster.team",
                "tail": "New Orleans Saints",
            },
            {
                "head": "Archie Manning",
                "relation": "sports.sports_team_roster.from",
                "tail": "1977",
            },
        ],
        "answer_view_facts": [
            {
                "head_id": "m.archie",
                "head_label": "Archie Manning",
                "relation_id": "sports.sports_team_roster.team",
                "tail_id": "m.saints",
                "tail_label": "New Orleans Saints",
                "tail_kind": "id",
            },
            {
                "head_id": "m.archie",
                "head_label": "Archie Manning",
                "relation_id": "sports.sports_team_roster.from",
                "tail_id": "",
                "tail_label": "1977",
                "tail_kind": "literal",
            },
        ],
        "evidence": [
            "Archie Manning -> sports.sports_team_roster.team -> New Orleans Saints",
            "Archie Manning -> sports.sports_team_roster.from -> 1977",
        ],
    }

    aggregate = pipeline._aggregate_subquestion_summaries("question", [item])

    assert len(aggregate["key_triple_facts"]) == 2
    assert aggregate["answer_view_facts"] == []


def test_summary_tail_resolution_skips_empty_summary_triples() -> None:
    class FakeGraphAPI:
        def resolve_entity_mentions(self, mentions: list[str], top_k: int = 3) -> list[str]:
            del top_k
            return {
                "Soviet Union": ["m.soviet_union"],
                "1922": ["m.1922"],
            }.get(mentions[0], [])

    pipeline = KGQAPipeline.__new__(KGQAPipeline)
    step_summaries = [
        {
            "question": "empty summary",
            "key_triples": [],
        },
        {
            "question": "anthem country",
            "key_triples": [
                {
                    "head": "m.0h_1h58",
                    "relation": "government.national_anthem_of_a_country.official_anthem_since",
                    "tail": "1922",
                },
                {
                    "head": "m.0h_1h58",
                    "relation": "government.national_anthem_of_a_country.country",
                    "tail": "Soviet Union",
                },
            ],
        },
    ]

    entity_ids = pipeline._resolve_summary_tail_entity_ids(
        graph_api=FakeGraphAPI(),
        expected_type="",
        step_summaries=step_summaries,
        labels={"soviet union"},
    )

    assert entity_ids == ["m.soviet_union"]
