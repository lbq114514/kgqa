from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg import entity_linking
from kgqa.reasoning import pruning
from kgqa.utils import embeddings


def test_embedding_model_is_shared_across_modules(monkeypatch) -> None:
    embeddings._MODEL_CACHE.clear()

    calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            calls.append(model_name)

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    model_a = entity_linking.get_embedding_model("shared-model")
    model_b = pruning.get_embedding_model("shared-model")

    assert model_a is model_b
    assert calls == ["shared-model"]
