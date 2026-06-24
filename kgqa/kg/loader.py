"""Load a triple-based knowledge graph from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from kgqa.kg.graph import KnowledgeGraph
from kgqa.utils.types import Triple


def load_knowledge_graph(path: str | Path) -> KnowledgeGraph:
    """Load triples from a JSON file into a KnowledgeGraph."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    triples = [
        Triple(head=item["head"], relation=item["relation"], tail=item["tail"])
        for item in payload
    ]
    return KnowledgeGraph(triples)
