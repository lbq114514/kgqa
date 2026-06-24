"""CWQ JSONL loader for external-only evaluation mode."""

from __future__ import annotations

import json
from pathlib import Path

from kgqa.utils.types import CWQSample


def load_cwq_samples(path: str | Path, limit: int | None = None) -> list[CWQSample]:
    """Load CWQ JSONL samples, keeping only question text and answer texts."""
    samples: list[CWQSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            answers = [
                str(item.get("text", "")).strip()
                for item in payload.get("answers", [])
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            samples.append(
                CWQSample(
                    sample_id=str(payload.get("id", f"sample-{index}")),
                    question=str(payload.get("question", "")),
                    answers=answers,
                )
            )
    return samples
