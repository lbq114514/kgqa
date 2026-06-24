"""Topic entity extraction and KG entity linking."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from kgqa.llm.base import BaseLLM
from kgqa.llm.prompts import ENTITY_EXTRACTION_PROMPT
from kgqa.utils.embeddings import get_sentence_transformer
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.logging import get_logger
from kgqa.utils.text import normalize_text

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOGGER = get_logger(__name__)


def set_default_embedding_model(model_name: str) -> None:
    """Override the default embedding model used by this module."""
    global DEFAULT_EMBEDDING_MODEL
    DEFAULT_EMBEDDING_MODEL = model_name


def get_embedding_model(model_name: str):
    """Load and cache the SentenceTransformer backend."""
    return get_sentence_transformer(model_name)


def _encode_silently(model: object, texts: list[str]):
    """Encode text without progress bars when supported by the backend."""
    try:
        return model.encode(texts, show_progress_bar=False)
    except TypeError:
        return model.encode(texts)


def extract_entities_with_llm(question: str, llm: BaseLLM) -> list[str]:
    """Extract entity mentions from the question with the local LLM."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
    raw = llm.generate(prompt)
    parsed = robust_json_parse(raw, fallback=[])
    if isinstance(parsed, list):
        entities = [str(item).strip() for item in parsed if str(item).strip()]
        LOGGER.info("Extracted entity candidates: %s", entities)
        return entities
    return []


def _link_candidates(
    candidates: list[str],
    choices: list[str],
    top_k: int,
    candidate_label: str,
    choice_label: str,
) -> list[str]:
    """Link candidate strings to the closest KG choices with embedding similarity."""
    if not candidates or not choices:
        return []

    model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    candidate_embeddings = _encode_silently(model, candidates)
    choice_embeddings = _encode_silently(model, choices)
    similarities = cosine_similarity(np.asarray(candidate_embeddings), np.asarray(choice_embeddings))

    linked: list[str] = []
    for row_index, candidate in enumerate(candidates):
        ranked_indices = np.argsort(similarities[row_index])[::-1][:top_k]
        for index in ranked_indices:
            linked_choice = choices[int(index)]
            linked.append(linked_choice)
            LOGGER.info(
                "Linked %s candidate %r to KG %s %r with score %.4f",
                candidate_label,
                candidate,
                choice_label,
                linked_choice,
                similarities[row_index][int(index)],
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in linked:
        key = normalize_text(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def link_entities_to_kg(candidates: list[str], kg_entities: list[str], top_k: int = 1) -> list[str]:
    """Link extracted entity candidates to the closest KG entities."""
    return _link_candidates(
        candidates=candidates,
        choices=kg_entities,
        top_k=top_k,
        candidate_label="entity",
        choice_label="entity",
    )


def link_relations_to_kg(candidates: list[str], kg_relations: list[str], top_k: int = 3) -> list[str]:
    """Link relation hints to the closest KG relation names."""
    return _link_candidates(
        candidates=candidates,
        choices=kg_relations,
        top_k=top_k,
        candidate_label="relation",
        choice_label="relation",
    )
