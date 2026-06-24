"""Shared embedding-model loading helpers."""

from __future__ import annotations

from threading import Lock

from kgqa.utils.logging import get_logger

LOGGER = get_logger(__name__)

_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = Lock()


def get_sentence_transformer(model_name: str):
    """Load and cache a SentenceTransformer instance once per process."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Install dependencies from requirements.txt."
            ) from exc

        LOGGER.info("Loading shared SentenceTransformer model from %s", model_name)
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        return model
