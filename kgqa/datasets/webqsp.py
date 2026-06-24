"""WebQSP JSONL loader and sample-to-KG conversion."""

from __future__ import annotations

import json
from pathlib import Path

from kgqa.kg.graph import KnowledgeGraph
from kgqa.utils.types import Triple, WebQSPSample


def _graph_rows_to_triples(graph_rows: list[list[str]]) -> list[Triple]:
    """Convert WebQSP graph rows into Triple objects."""
    triples: list[Triple] = []
    for row in graph_rows:
        if not isinstance(row, list) or len(row) != 3:
            continue
        head, relation, tail = (str(item) for item in row)
        triples.append(Triple(head=head, relation=relation, tail=tail))
    return triples


def load_webqsp_samples(path: str | Path, limit: int | None = None) -> list[WebQSPSample]:
    """Load WebQSP JSONL samples from disk."""
    samples: list[WebQSPSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            samples.append(
                WebQSPSample(
                    sample_id=str(payload.get("id", f"sample-{index}")),
                    question=str(payload.get("question", "")),
                    answers=[str(item) for item in payload.get("answer", [])],
                    q_entities=[str(item) for item in payload.get("q_entity", [])],
                    a_entities=[str(item) for item in payload.get("a_entity", [])],
                    graph_triples=_graph_rows_to_triples(payload.get("graph", [])),
                )
            )
    return samples


def build_sample_kg(sample: WebQSPSample) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a WebQSP sample-local graph."""
    return KnowledgeGraph(sample.graph_triples)
