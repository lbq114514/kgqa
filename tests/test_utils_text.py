from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.utils.text import deduplicate_paths
from kgqa.utils.types import ReasoningPath, Triple


def _path(*, terminal_node_id: str, edge_ids: list[str] | None = None) -> ReasoningPath:
    return ReasoningPath(
        triples=[Triple(head="Jenny's Father", relation="film.film_character.portrayed_in_films", tail="Forrest Gump")],
        nodes=["Jenny's Father", "Forrest Gump"],
        text="Jenny's Father -> film.film_character.portrayed_in_films -> Forrest Gump",
        source_stage="test",
        edge_ids=edge_ids or [],
        terminal_node_id=terminal_node_id,
        terminal_node_kind="id",
    )


def test_deduplicate_paths_keeps_same_text_with_distinct_terminal_ids() -> None:
    film_path = _path(terminal_node_id="m.0bdjd")
    music_path = _path(terminal_node_id="m.0111z9n0")

    deduped = deduplicate_paths([film_path, music_path])

    assert len(deduped) == 2
    assert {path.terminal_node_id for path in deduped} == {"m.0bdjd", "m.0111z9n0"}


def test_deduplicate_paths_collapses_same_edge_identity() -> None:
    path_a = _path(terminal_node_id="m.0bdjd", edge_ids=["n1:r:m.0bdjd"])
    path_b = _path(terminal_node_id="m.0bdjd", edge_ids=["n1:r:m.0bdjd"])

    deduped = deduplicate_paths([path_a, path_b])

    assert len(deduped) == 1
