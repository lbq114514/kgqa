"""Robust JSON parsing helpers for noisy local LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def robust_json_parse(text: str, fallback: Any = None) -> Any:
    """Parse JSON from raw LLM text with a few recovery heuristics."""
    if not text:
        return fallback

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for pattern in (r"(\{.*\})", r"(\[.*\])"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return fallback
