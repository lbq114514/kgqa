from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import _graphapi_is_enabled, _validate_cwq_runtime_options


def test_graphapi_is_enabled_reads_runtime_flag() -> None:
    assert _graphapi_is_enabled({"graphapi": {"enabled": True}}) is True
    assert _graphapi_is_enabled({"graphapi": {"enabled": False}}) is False
    assert _graphapi_is_enabled({}) is False


def test_validate_cwq_runtime_options_requires_graphapi() -> None:
    with pytest.raises(ValueError, match="graphapi.enabled=true"):
        _validate_cwq_runtime_options({"graphapi": {"enabled": False}}, use_q_entities=False)


def test_validate_cwq_runtime_options_rejects_q_entities() -> None:
    with pytest.raises(ValueError, match="does not support --use-q-entities"):
        _validate_cwq_runtime_options({"graphapi": {"enabled": True}}, use_q_entities=True)


def test_validate_cwq_runtime_options_accepts_external_only_mode() -> None:
    _validate_cwq_runtime_options({"graphapi": {"enabled": True}}, use_q_entities=False)
