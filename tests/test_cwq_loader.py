from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.datasets.cwq import load_cwq_samples


def test_load_cwq_samples_reads_question_and_answer_text_only(tmp_path: Path) -> None:
    path = tmp_path / "cwq.jsonl"
    rows = [
        {
            "id": "sample-1",
            "question": "Who founded Example Corp?",
            "answers": [
                {"kb_id": "m.1", "text": "Alice Example"},
                {"kb_id": "m.2", "text": "Bob Example"},
            ],
            "entities": [1, 2],
            "subgraph": {"tuples": [], "entities": []},
        },
        {
            "id": "sample-2",
            "question": "Where is Sample University?",
            "answers": [
                {"kb_id": "m.3", "text": "Sample City"},
                {"kb_id": "m.4"},
                "ignored",
            ],
            "entities": [3],
            "subgraph": {"tuples": [], "entities": []},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    samples = load_cwq_samples(path)

    assert len(samples) == 2
    assert samples[0].sample_id == "sample-1"
    assert samples[0].question == "Who founded Example Corp?"
    assert samples[0].answers == ["Alice Example", "Bob Example"]
    assert samples[1].answers == ["Sample City"]


def test_load_cwq_samples_honors_limit(tmp_path: Path) -> None:
    path = tmp_path / "cwq.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"sample-{index}",
                    "question": f"question-{index}",
                    "answers": [{"text": f"answer-{index}"}],
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )

    samples = load_cwq_samples(path, limit=2)

    assert [sample.sample_id for sample in samples] == ["sample-0", "sample-1"]
