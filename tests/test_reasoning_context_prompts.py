from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.llm.base import BaseLLM
from kgqa.reasoning.answering import check_sufficiency, generate_answer
from kgqa.reasoning.summarization import summarize_paths
from kgqa.utils.types import QuestionAnalysisResult, ReasoningPath, Triple


class CaptureLLM(BaseLLM):
    """Capture the last prompt and return a fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.last_prompt = prompt
        return self.response


def _analysis() -> QuestionAnalysisResult:
    return QuestionAnalysisResult(
        split_questions=["What did Nick Clegg study at university?"],
        reasoning_indicator="Find the education record, then the major field of study.",
        ordered_topic_entities=["Nick Clegg"],
        predicted_depth=2,
        topic_entities=["Nick Clegg"],
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
        assert '"predicted_depth": 2' in prompt
        assert '"current_evaluation_depth": 2' in prompt
        assert '"dmax": 3' in prompt
        assert '"ordered_topic_entities": [' in prompt
        assert '"sub_questions": []' in prompt
        assert '"reasoning_indicator": "Find the education record, then the major field of study."' in prompt
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
