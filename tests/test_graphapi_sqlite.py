from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.graph import KnowledgeGraph
from kgqa.kg.graph_api import BaseGraphAPI, ExternalGraphNeighbor
from kgqa.kg.sqlite_graph_api import (
    SQLiteGraphAPI,
    build_sqlite_graph_db,
    build_sqlite_graph_db_from_processed_freebase,
    normalize_text,
    resolve_entity_candidates,
    resolve_relation_candidates,
    verbalize_relation,
)
from kgqa.llm.base import BaseLLM
from kgqa.reasoning.pipeline import KGQAPipeline
from kgqa.utils.types import EntityMentionSpec, ReasoningPath, RelationHintSpec, SubQuestionSpec, Triple


class FakeEmbeddingModel:
    """A tiny deterministic encoder for tests."""

    KEYWORDS = (
        "nick",
        "clegg",
        "education",
        "study",
        "university",
        "anthropology",
        "field",
        "major",
    )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = (
                text.lower()
                .replace(".", " ")
                .replace("_", " ")
                .replace("-", " ")
            )
            vector = [float(normalized.count(keyword)) for keyword in self.KEYWORDS]
            vector.append(1.0)
            vectors.append(vector)
        return vectors


class RoutingLLM(BaseLLM):
    """Return deterministic JSON by prompt type."""

    def __init__(self) -> None:
        self.sufficiency_calls = 0

    def generate(self, prompt: str, **kwargs: object) -> str:
        if "You are a KGQA planner operating in a step-by-step agent loop." in prompt:
            return (
                '{"step_id":"step_1",'
                '"question":"What did Nick Clegg study at university?",'
                '"goal":"Find Nick Clegg education records and field of study.",'
                '"depends_on_step_ids":[],'
                '"topic_entity_mentions":[{"name":"Nick Clegg","aliases":[],"expected_type":"person","role":"topic_entity"}],'
                '"relation_hints":['
                '{"name":"education","aliases":[],"freebase_like_ids":[],"direction":"topic_entity -> answer","description":"education record"},'
                '{"name":"major field of study","aliases":[],"freebase_like_ids":[],"direction":"education record -> answer","description":"field of study"}'
                '],'
                '"expected_answer_type":"field of study",'
                '"carryover_constraints":[],'
                '"stop_if_answered":true,'
                '"strategy":"external_graph"}'
            )
        if "You analyze a KGQA question" in prompt:
            return (
                '{"topic_entities":["Nick Clegg"],'
                '"split_questions":["What did Nick Clegg study at university?"],'
                '"reasoning_indicator":"Find Nick Clegg education records and field of study.",'
                '"ordered_topic_entities":["Nick Clegg"],'
                '"predicted_depth":1}'
            )
        if "You propose bridge entities" in prompt:
            return (
                '{"entities":["University of Cambridge","Field of study"],'
                '"relations":["people.person.education","education.education.major_field_of_study"]}'
            )
        if "You select the most useful candidate reasoning paths" in prompt:
            return "[0, 1, 2]"
        if "You summarize reasoning paths" in prompt:
            if "Social anthropology" in prompt:
                return (
                    '{"question":"what did nick clegg study at university",'
                    '"question_focus":"field of study of Nick Clegg",'
                    '"key_triples":['
                    '{"head":"Nick Clegg","relation":"people.person.education","tail":"education record"},'
                    '{"head":"education record","relation":"education.education.major_field_of_study","tail":"Social anthropology"}'
                    '],'
                    '"evidence":['
                    '"Nick Clegg -> people.person.education -> education record",'
                    '"education record -> education.education.major_field_of_study -> Social anthropology"'
                    "]}"
                )
            return (
                '{"question":"what did nick clegg study at university",'
                '"question_focus":"Nick Clegg education",'
                '"key_triples":[],"evidence":[]}'
            )
        if "You judge whether the provided evidence is sufficient" in prompt:
            self.sufficiency_calls += 1
            if self.sufficiency_calls >= 3 and "Social anthropology" in prompt:
                return '{"sufficient": true, "reason": "Field of study evidence found."}'
            return '{"sufficient": false, "reason": "Need more evidence."}'
        if "You answer a question using only the provided evidence" in prompt:
            if "Social anthropology" in prompt:
                return (
                    '{"predicted_answers":["Social anthropology"],'
                    '"answer":"Social anthropology",'
                    '"supporting_paths":['
                    '"Nick Clegg -> people.person.education -> education record -> '
                    'education.education.major_field_of_study -> Social anthropology"'
                    "]}"
                )
            return '{"predicted_answers":[],"answer":"insufficient","supporting_paths":[]}'
        return "{}"


class CWQRoutingLLM(BaseLLM):
    """Deterministic responses for the CWQ sub-question SQLite flow."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        if "You are a KGQA planner operating in a step-by-step agent loop." in prompt:
            if '"loop_index": 0' in prompt:
                return (
                    '{"step_id":"step_1",'
                    '"question":"Which country contains Northern District?",'
                    '"goal":"Find the country containing Northern District.",'
                    '"depends_on_step_ids":[],'
                    '"topic_entity_mentions":[{"name":"Northern District","aliases":[],"expected_type":"district","role":"topic_entity"}],'
                    '"relation_hints":[{"name":"administrative parent","aliases":[],"freebase_like_ids":[],"direction":"topic_entity -> answer","description":"country containing Northern District"}],'
                    '"expected_answer_type":"country",'
                    '"carryover_constraints":[],'
                    '"stop_if_answered":false,'
                    '"strategy":"external_graph"}'
                )
            return (
                '{"step_id":"step_2",'
                '"question":"What type of government does Israel use?",'
                '"goal":"Find the government type of the country found in the previous step.",'
                '"depends_on_step_ids":["step_1"],'
                '"topic_entity_mentions":[{"name":"Israel","aliases":[],"expected_type":"country","role":"topic_entity"}],'
                '"relation_hints":[{"name":"government type","aliases":[],"freebase_like_ids":[],"direction":"country -> answer","description":"government type of Israel"}],'
                '"expected_answer_type":"government type",'
                '"carryover_constraints":["Israel"],'
                '"stop_if_answered":true,'
                '"strategy":"external_graph"}'
            )
        if "You analyze a KGQA question" in prompt:
            return (
                '{"topic_entities":["Northern District"],'
                '"split_questions":["Which country contains Northern District?",'
                '"What type of government does that country use?"],'
                '"sub_questions":['
                '{'
                '"id":"sq1",'
                '"question":"Which country contains Northern District?",'
                '"topic_entities":[{"name":"Northern District","aliases":[],"expected_type":"district","role":"topic_entity"}],'
                '"interested_nodes":[],'
                '"interested_relations":[{"name":"administrative parent","aliases":[],"freebase_like_ids":[],"direction":"topic_entity -> answer","description":"country containing Northern District"}],'
                '"expected_answer_type":"country",'
                '"depends_on":[]'
                '},'
                '{'
                '"id":"sq2",'
                '"question":"What type of government does that country use?",'
                '"topic_entities":[],'
                '"interested_nodes":[{"name":"Northern District","aliases":[],"expected_type":"district","role":"constraint_or_anchor"}],'
                '"interested_relations":[{"name":"government type","aliases":[],"freebase_like_ids":[],"direction":"country -> answer","description":"government type of the containing country"}],'
                '"expected_answer_type":"government type",'
                '"depends_on":["sq1"]'
                '}],'
                '"reasoning_indicator":"Find the country containing Northern District, then its government type.",'
                '"ordered_topic_entities":["Northern District"],'
                '"predicted_depth":2}'
            )
        if "You select the most useful candidate reasoning paths" in prompt:
            return "[0]"
        if "You summarize reasoning paths" in prompt:
            if "Parliamentary system" not in prompt:
                return (
                    '{"question":"What type of government is used in the country with Northern District?",'
                    '"question_focus":"country containing Northern District",'
                    '"key_triples":['
                    '{"head":"Northern District","relation":"location.location.containedby","tail":"Israel"}'
                    '],'
                    '"evidence":['
                    '"Northern District -> location.location.containedby -> Israel"'
                    "]}"
                )
            return (
                '{"question":"What type of government is used in the country with Northern District?",'
                '"question_focus":"government type of the country containing Northern District",'
                '"key_triples":['
                '{"head":"Northern District","relation":"location.location.containedby","tail":"Israel"},'
                '{"head":"Israel","relation":"government.government.form_of_government","tail":"Parliamentary system"}'
                '],'
                '"evidence":['
                '"Northern District -> location.location.containedby -> Israel",'
                '"Israel -> government.government.form_of_government -> Parliamentary system"'
                "]}"
            )
        if "You judge whether the provided evidence is sufficient" in prompt:
            if "Parliamentary system" in prompt:
                return '{"sufficient": true, "reason": "The evidence directly identifies the country and its government type."}'
            return '{"sufficient": false, "reason": "Need the government type relation."}'
        if "You answer a question using only the provided evidence" in prompt:
            if "Parliamentary system" not in prompt:
                return '{"predicted_answers":[],"answer":"insufficient","supporting_paths":[]}'
            return (
                '{"predicted_answers":["Parliamentary system"],'
                '"answer":"Parliamentary system",'
                '"supporting_paths":['
                '"Northern District -> location.location.containedby -> Israel",'
                '"Israel -> government.government.form_of_government -> Parliamentary system"'
                "]}"
            )
        return "{}"


class RecordingGraphAPI(BaseGraphAPI):
    """Small in-memory graph API used to validate CWQ sub-question behavior."""

    def __init__(self) -> None:
        self.entity_seed_calls: list[list[str]] = []
        self.relation_hint_calls: list[list[str]] = []
        self.bootstrap_relation_ids: list[str] = []
        self.neighbor_relation_ids: list[str] = []
        self.expand_calls: list[dict[str, object]] = []

    def resolve_entity_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for name in names:
            if name == "Northern District":
                rows.append(
                    {
                        "mid": "m.northern_district",
                        "name": "Northern District",
                        "mention": name,
                        "match_type": "name_exact",
                        "score": 95.0,
                    }
                )
            elif name == "Israel":
                rows.append(
                    {
                        "mid": "m.israel",
                        "name": "Israel",
                        "mention": name,
                        "match_type": "name_exact",
                        "score": 95.0,
                    }
                )
        return rows[:top_k]

    def resolve_relation_candidates(self, names: list[str], top_k: int = 5) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for name in names:
            if name == "administrative parent":
                rows.extend(
                    [
                        {
                            "relation": "base.cocktails.cocktail_ingredient.cocktails_with_this_ingredient",
                            "relation_name": "cocktails with this ingredient",
                            "hint": name,
                            "match_type": "relation_name_like",
                            "score": 98.0,
                        },
                        {
                            "relation": "location.location.containedby",
                            "relation_name": "administrative parent",
                            "hint": name,
                            "match_type": "relation_name_exact",
                            "score": 95.0,
                        },
                    ]
                )
            elif name in {"contained by country", "contained by", "country containing", "part of country"}:
                rows.append(
                    {
                        "relation": "location.location.containedby",
                        "relation_name": "contained by country",
                        "hint": name,
                        "match_type": "relation_name_exact",
                        "score": 95.0,
                    }
                )
            elif name == "government type":
                rows.extend(
                    [
                        {
                            "relation": "base.peleton.cyclist.races_won",
                            "relation_name": "races won",
                            "hint": name,
                            "match_type": "relation_name_like",
                            "score": 99.0,
                        },
                        {
                            "relation": "government.government.form_of_government",
                            "relation_name": "government type",
                            "hint": name,
                            "match_type": "relation_name_exact",
                            "score": 95.0,
                        },
                    ]
                )
        return rows[:top_k]

    def get_neighbors(
        self,
        node_id: str,
        include_reverse: bool = True,
        relation_filter: list[str] | None = None,
        limit: int | None = None,
        strict_relation_filter: bool = False,
    ) -> list[ExternalGraphNeighbor]:
        self.neighbor_relation_ids.extend(relation_filter or [])
        if node_id == "m.israel":
            return [
                ExternalGraphNeighbor(
                    source_id="m.israel",
                    neighbor_id="m.parliamentary",
                    triple=Triple(
                        head="Israel",
                        relation="government.government.form_of_government",
                        tail="Parliamentary system",
                    ),
                    relation_name="government type",
                    reversed=False,
                )
            ]
        return []

    def resolve_entity_mentions(self, names: list[str], top_k: int = 1) -> list[str]:
        self.entity_seed_calls.append(list(names))
        resolved: list[str] = []
        seen: set[str] = set()
        for name in names:
            local: list[str] = []
            if name == "Northern District":
                local.append("m.northern_district")
            elif name == "Israel":
                local.append("m.israel")
            count = 0
            for item in local:
                if item in seen:
                    continue
                seen.add(item)
                resolved.append(item)
                count += 1
                if count >= top_k:
                    break
        return resolved

    def get_entity_display_name(self, node_id: str) -> str:
        return {
            "m.northern_district": "Northern District",
            "m.israel": "Israel",
            "m.parliamentary": "Parliamentary system",
        }.get(node_id, node_id)

    def resolve_relation_hints(self, names: list[str], top_k: int = 3) -> list[str]:
        self.relation_hint_calls.append(list(names))
        resolved = []
        for name in names:
            if name == "administrative parent":
                resolved.append("location.location.containedby")
            elif name == "government type":
                resolved.append("government.government.form_of_government")
        return resolved[:top_k]

    def get_relation_display_name(self, relation_id: str) -> str:
        return {
            "location.location.containedby": "administrative parent",
            "government.government.form_of_government": "government type",
        }.get(relation_id, relation_id)

    def find_two_hop_extensions(
        self,
        frontier_nodes: list[str],
        relation_hints: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ReasoningPath]:
        self.bootstrap_relation_ids.extend(relation_hints or [])
        return [
            ReasoningPath(
                triples=[
                    Triple("Northern District", "location.location.containedby", "Israel"),
                ],
                nodes=["Northern District", "Israel"],
                text="Northern District -> location.location.containedby -> Israel",
                source_stage="external_graphapi_supplement",
            )
        ]

    def expand_paths(
        self,
        seed_nodes: list[str],
        relation_hints: list[str] | None = None,
        max_depth: int = 3,
        strict_relation_filter: bool = True,
        max_paths: int | None = None,
        max_nodes: int | None = None,
    ) -> list[ReasoningPath]:
        self.expand_calls.append(
            {
                "seed_nodes": list(seed_nodes),
                "relation_hints": list(relation_hints or []),
                "max_depth": max_depth,
                "strict_relation_filter": strict_relation_filter,
                "max_paths": max_paths,
                "max_nodes": max_nodes,
            }
        )
        if (
            "m.northern_district" in seed_nodes
            and max_depth == 2
            and not relation_hints
        ):
            return [
                ReasoningPath(
                    triples=[
                        Triple("Northern District", "location.location.containedby", "Israel"),
                    ],
                    nodes=["Northern District", "Israel"],
                    text="Northern District -> location.location.containedby -> Israel",
                    source_stage="external_graphapi_expansion",
                ),
                ReasoningPath(
                    triples=[
                        Triple("Northern District", "location.location.containedby", "Israel"),
                        Triple("Israel", "government.government.form_of_government", "Parliamentary system"),
                    ],
                    nodes=["Northern District", "Israel", "Parliamentary system"],
                    text=(
                        "Northern District -> location.location.containedby -> Israel -> "
                        "government.government.form_of_government -> Parliamentary system"
                    ),
                    source_stage="external_graphapi_expansion",
                ),
            ]
        if "m.israel" in seed_nodes and max_depth == 2:
            return [
                ReasoningPath(
                    triples=[
                        Triple("Israel", "government.government.form_of_government", "Parliamentary system"),
                    ],
                    nodes=["Israel", "Parliamentary system"],
                    text="Israel -> government.government.form_of_government -> Parliamentary system",
                    source_stage="external_graphapi_expansion",
                )
            ]
        return []


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_test_graph_db(tmp_path: Path) -> Path:
    entities_csv = tmp_path / "entities.csv"
    triples_csv = tmp_path / "triples.csv"
    db_path = tmp_path / "fb2m.sqlite"

    _write_csv(
        entities_csv,
        ["mid", "name", "aliases"],
        [
            {"mid": "m.nick", "name": "nick clegg", "aliases": "nicholas william peter clegg"},
            {"mid": "m.education", "name": "education record", "aliases": ""},
            {"mid": "m.social", "name": "Social anthropology", "aliases": "social anthropology"},
            {"mid": "m.cambridge", "name": "University of Cambridge", "aliases": "cambridge"},
            {"mid": "m.archive", "name": "field of study archive", "aliases": ""},
        ],
    )
    _write_csv(
        triples_csv,
        ["head", "relation", "relation_name", "tail"],
        [
            {
                "head": "m.nick",
                "relation": "people.person.education",
                "relation_name": "education",
                "tail": "m.education",
            },
            {
                "head": "m.education",
                "relation": "education.education.major_field_of_study",
                "relation_name": "major field of study",
                "tail": "m.social",
            },
            {
                "head": "m.education",
                "relation": "education.education.institution",
                "relation_name": "institution",
                "tail": "m.cambridge",
            },
            {
                "head": "m.archive",
                "relation": "education.field_of_study.students_majoring",
                "relation_name": "students majoring",
                "tail": "m.social",
            },
        ],
    )
    build_sqlite_graph_db(entities_csv, triples_csv, db_path, overwrite=True)
    return db_path


def _build_old_schema_graph_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(
            """
            CREATE TABLE entities (
                mid TEXT PRIMARY KEY,
                name TEXT,
                aliases TEXT
            );
            CREATE TABLE triples (
                head TEXT,
                relation TEXT,
                relation_name TEXT,
                tail TEXT
            );
            """
        )
        connection.executemany(
            "INSERT INTO entities(mid, name, aliases) VALUES (?, ?, ?)",
            [
                ("m.obama", "Barack Obama", "President Obama"),
                ("m.hawaii", "Hawaii", "State of Hawaii"),
            ],
        )
        connection.executemany(
            "INSERT INTO triples(head, relation, relation_name, tail) VALUES (?, ?, ?, ?)",
            [
                ("m.obama", "people.person.place_of_birth", "Place Of Birth", "m.hawaii"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return db_path


def _build_processed_freebase_graph_db(tmp_path: Path) -> Path:
    entities_csv = tmp_path / "processed_entities.csv"
    triples_csv = tmp_path / "processed_triplets.csv"
    db_path = tmp_path / "processed.sqlite"

    _write_csv(
        entities_csv,
        ["id", "name", "aliases", "types", "is_cvt"],
        [
            {
                "id": "m.obama",
                "name": "Barack Obama",
                "aliases": "President Obama|Barack Hussein Obama II",
                "types": "people.person|common.topic",
                "is_cvt": "0",
            },
            {
                "id": "m.honolulu",
                "name": "Honolulu",
                "aliases": "",
                "types": "location.citytown|common.topic",
                "is_cvt": "0",
            },
            {
                "id": "m.hawaii",
                "name": "Hawaii",
                "aliases": "State of Hawaii",
                "types": "location.location|common.topic",
                "is_cvt": "0",
            },
            {
                "id": "m.michelle",
                "name": "Michelle Obama",
                "aliases": "",
                "types": "people.person|common.topic",
                "is_cvt": "0",
            },
            {
                "id": "g.statement",
                "name": "",
                "aliases": "",
                "types": "common.notable_for",
                "is_cvt": "1",
            },
        ],
    )
    _write_csv(
        triples_csv,
        ["head", "head_name", "relation", "tail", "tail_name", "tail_kind"],
        [
            {
                "head": "m.obama",
                "head_name": "Barack Obama",
                "relation": "people.person.place_of_birth",
                "tail": "m.honolulu",
                "tail_name": "Honolulu",
                "tail_kind": "id",
            },
            {
                "head": "m.obama",
                "head_name": "Barack Obama",
                "relation": "people.person.date_of_birth",
                "tail": "1961-08-04",
                "tail_name": "",
                "tail_kind": "literal",
            },
            {
                "head": "m.obama",
                "head_name": "Barack Obama",
                "relation": "people.person.spouse_s.people.marriage.spouse",
                "tail": "m.michelle",
                "tail_name": "Michelle Obama",
                "tail_kind": "id",
            },
            {
                "head": "m.honolulu",
                "head_name": "Honolulu",
                "relation": "location.location.containedby",
                "tail": "m.hawaii",
                "tail_name": "",
                "tail_kind": "id",
            },
            {
                "head": "m.obama",
                "head_name": "Barack Obama",
                "relation": "common.topic.notable_for",
                "tail": "g.statement",
                "tail_name": "",
                "tail_kind": "id",
            },
        ],
    )
    build_sqlite_graph_db_from_processed_freebase(entities_csv, triples_csv, db_path, overwrite=True)
    return db_path


def _make_config(db_path: str, enabled: bool) -> dict:
    return {
        "embedding": {"model": "fake-model"},
        "search": {
            "dmax": 3,
            "w1": 30,
            "wmax": 3,
            "max_expand_steps": 1,
            "max_paths_per_pair": 20,
        },
        "graphapi": {
            "enabled": enabled,
            "backend": "sqlite",
            "db_path": db_path,
            "augment_stages": ["supplement", "node_expand"],
            "neighbor_limit": 10,
            "relation_bias_top_k": 5,
            "cwq_max_paths": 10,
            "cwq_max_nodes": 10,
        },
    }


def test_sqlite_graph_api_resolves_mentions_and_paths(tmp_path: Path) -> None:
    db_path = _build_test_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    resolved = api.resolve_entity_mentions(["Nick Clegg"], top_k=1)
    assert resolved == ["m.nick"]

    neighbors = api.get_neighbors(
        "m.nick",
        include_reverse=True,
        relation_filter=["people.person.education"],
        limit=10,
    )
    assert len(neighbors) == 1
    assert neighbors[0].neighbor_id == "m.education"
    assert neighbors[0].triple.relation == "people.person.education"

    paths = api.find_two_hop_extensions(
        frontier_nodes=["m.nick"],
        relation_hints=["people.person.education", "education.education.major_field_of_study"],
        limit=10,
    )
    assert any("Social anthropology" in path.text for path in paths)
    social_paths = [path for path in paths if "Social anthropology" in path.text]
    assert social_paths
    assert any(path.terminal_node_id == "m.social" for path in social_paths)
    assert all(path.edge_ids for path in social_paths)


def test_normalize_text_handles_relation_style_variants() -> None:
    assert normalize_text("People.Person.Place_Of_Birth") == "people person place of birth"
    assert normalize_text(" place-of-birth ") == "place of birth"
    assert normalize_text(None) == ""


def test_verbalize_relation_shortens_freebase_relation_ids() -> None:
    assert verbalize_relation("people.person.place_of_birth") == "place of birth"
    assert verbalize_relation(
        "education.educational_institution.total_enrollment.measurement_unit.dated_integer.number"
    ) == "total enrollment -> number"


def test_build_sqlite_graph_db_adds_norm_columns_and_relations_table(tmp_path: Path) -> None:
    db_path = _build_test_graph_db(tmp_path)
    connection = sqlite3.connect(str(db_path))
    try:
        entity_columns = [row[1] for row in connection.execute("PRAGMA table_info(entities)").fetchall()]
        relation_columns = [row[1] for row in connection.execute("PRAGMA table_info(relations)").fetchall()]
        assert entity_columns == ["mid", "name", "name_norm", "aliases", "aliases_norm"]
        assert relation_columns == ["relation", "relation_name", "relation_norm", "relation_name_norm"]

        row = connection.execute(
            "SELECT name_norm, aliases_norm FROM entities WHERE mid = 'm.nick'"
        ).fetchone()
        assert row == ("nick clegg", "nicholas william peter clegg")

        relation_row = connection.execute(
            "SELECT relation_name, relation_norm, relation_name_norm FROM relations WHERE relation = ?",
            ("education.education.major_field_of_study",),
        ).fetchone()
        assert relation_row == (
            "major field of study",
            "education education major field of study",
            "major field of study",
        )
    finally:
        connection.close()


def test_build_processed_freebase_sqlite_graph_db_preserves_schema_and_relation_names(tmp_path: Path) -> None:
    db_path = _build_processed_freebase_graph_db(tmp_path)
    connection = sqlite3.connect(str(db_path))
    try:
        entity_columns = [row[1] for row in connection.execute("PRAGMA table_info(entities)").fetchall()]
        relation_columns = [row[1] for row in connection.execute("PRAGMA table_info(relations)").fetchall()]
        triple_columns = [row[1] for row in connection.execute("PRAGMA table_info(triples)").fetchall()]
        assert entity_columns == ["mid", "name", "name_norm", "aliases", "aliases_norm", "types", "types_norm", "is_cvt"]
        assert relation_columns == ["relation", "relation_name", "relation_norm", "relation_name_norm"]
        assert triple_columns == ["head", "relation", "tail", "tail_kind"]

        entity_row = connection.execute(
            "SELECT types, types_norm, is_cvt FROM entities WHERE mid = 'g.statement'"
        ).fetchone()
        assert entity_row == ("common.notable_for", "common notable for", 1)

        triple_row = connection.execute(
            """
            SELECT head, relation, tail, tail_kind
            FROM triples
            WHERE relation = 'people.person.date_of_birth'
            """
        ).fetchone()
        assert triple_row == ("m.obama", "people.person.date_of_birth", "1961-08-04", "literal")
    finally:
        connection.close()


def test_build_processed_freebase_sqlite_graph_db_supports_fast_mode(tmp_path: Path) -> None:
    entities_csv = tmp_path / "processed_entities.csv"
    triples_csv = tmp_path / "processed_triplets.csv"
    db_path = tmp_path / "processed_fast.sqlite"
    _write_csv(
        entities_csv,
        ["id", "name", "aliases", "types", "is_cvt"],
        [{"id": "m.obama", "name": "Barack Obama", "aliases": "", "types": "people.person", "is_cvt": "0"}],
    )
    _write_csv(
        triples_csv,
        ["head", "head_name", "relation", "tail", "tail_name", "tail_kind"],
        [{"head": "m.obama", "head_name": "Barack Obama", "relation": "people.person.place_of_birth", "tail": "m.honolulu", "tail_name": "Honolulu", "tail_kind": "id"}],
    )
    build_sqlite_graph_db_from_processed_freebase(
        entities_csv,
        triples_csv,
        db_path,
        overwrite=True,
        fast_mode=True,
    )
    connection = sqlite3.connect(str(db_path))
    try:
        triple_columns = [row[1] for row in connection.execute("PRAGMA table_info(triples)").fetchall()]
        assert triple_columns == ["head", "relation", "tail", "tail_kind"]
    finally:
        connection.close()


def test_build_processed_freebase_sqlite_graph_db_supports_staging_dir(tmp_path: Path) -> None:
    entities_csv = tmp_path / "processed_entities.csv"
    triples_csv = tmp_path / "processed_triplets.csv"
    db_path = tmp_path / "processed_staged.sqlite"
    staging_dir = tmp_path / "ssd_tmp"
    _write_csv(
        entities_csv,
        ["id", "name", "aliases", "types", "is_cvt"],
        [{"id": "m.obama", "name": "Barack Obama", "aliases": "", "types": "people.person", "is_cvt": "0"}],
    )
    _write_csv(
        triples_csv,
        ["head", "head_name", "relation", "tail", "tail_name", "tail_kind"],
        [{"head": "m.obama", "head_name": "Barack Obama", "relation": "people.person.place_of_birth", "tail": "m.honolulu", "tail_name": "Honolulu", "tail_kind": "id"}],
    )
    build_sqlite_graph_db_from_processed_freebase(
        entities_csv,
        triples_csv,
        db_path,
        overwrite=True,
        fast_mode=True,
        staging_dir=staging_dir,
    )
    assert db_path.exists()


def test_sqlite_graph_api_initializes_with_legacy_schema(tmp_path: Path) -> None:
    db_path = _build_old_schema_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    assert api._table_exists("entities") is True
    assert api._table_exists("relations") is False
    assert api.resolve_entity_mentions(["Barack Obama"], top_k=1) == ["m.obama"]
    assert api.resolve_relation_hints(["place of birth"], top_k=1) == ["people.person.place_of_birth"]


def test_resolve_candidate_helpers_return_ranked_matches(tmp_path: Path) -> None:
    db_path = _build_test_graph_db(tmp_path)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        entity_rows = resolve_entity_candidates(connection, "Social anthropology", limit=5)
        assert entity_rows[0]["mid"] == "m.social"
        assert entity_rows[0]["match_type"] == "name_exact"

        relation_rows = resolve_relation_candidates(connection, "major field of study", limit=5)
        assert relation_rows[0]["relation"] == "education.education.major_field_of_study"
        assert relation_rows[0]["match_type"] in {"relation_name_exact", "relation_name_like"}
    finally:
        connection.close()


def test_sqlite_graph_api_resolves_relation_hints(tmp_path: Path) -> None:
    db_path = _build_test_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    resolved = api.resolve_relation_hints(["major field of study", "education"], top_k=2)

    assert "education.education.major_field_of_study" in resolved
    assert "people.person.education" in resolved


def test_sqlite_graph_api_resolves_barack_obama_and_place_of_birth(tmp_path: Path) -> None:
    entities_csv = tmp_path / "entities.csv"
    triples_csv = tmp_path / "triples.csv"
    db_path = tmp_path / "barack.sqlite"
    _write_csv(
        entities_csv,
        ["mid", "name", "aliases"],
        [
            {"mid": "m.02mjmr", "name": "Barack Obama", "aliases": "President Obama|Barack Hussein Obama II"},
            {"mid": "m.03gh4", "name": "Honolulu", "aliases": ""},
        ],
    )
    _write_csv(
        triples_csv,
        ["head", "relation", "relation_name", "tail"],
        [
            {
                "head": "m.02mjmr",
                "relation": "people.person.place_of_birth",
                "relation_name": "Place Of Birth",
                "tail": "m.03gh4",
            }
        ],
    )
    build_sqlite_graph_db(entities_csv, triples_csv, db_path, overwrite=True)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    assert api.resolve_entity_mentions(["Barack Obama"], top_k=1) == ["m.02mjmr"]
    assert api.resolve_relation_hints(["place of birth"], top_k=1) == ["people.person.place_of_birth"]


def test_get_neighbors_supports_strict_relation_filter_and_soft_bias(tmp_path: Path) -> None:
    db_path = _build_test_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    strict_neighbors = api.get_neighbors(
        "m.education",
        include_reverse=False,
        relation_filter=["education.education.major_field_of_study"],
        limit=10,
        strict_relation_filter=True,
    )
    assert strict_neighbors
    assert all(
        neighbor.triple.relation == "education.education.major_field_of_study"
        for neighbor in strict_neighbors
    )

    soft_neighbors = api.get_neighbors(
        "m.education",
        include_reverse=False,
        relation_filter=["education.education.major_field_of_study"],
        limit=10,
        strict_relation_filter=False,
    )
    assert soft_neighbors[0].triple.relation == "education.education.major_field_of_study"
    assert any(
        neighbor.triple.relation == "education.education.institution"
        for neighbor in soft_neighbors
    )


def test_expand_paths_returns_multi_hop_paths_without_simple_cycles(tmp_path: Path) -> None:
    db_path = _build_test_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    paths = api.expand_paths(
        seed_nodes=["m.nick"],
        relation_hints=[
            "people.person.education",
            "education.education.major_field_of_study",
            "education.field_of_study.students_majoring",
        ],
        max_depth=3,
        strict_relation_filter=True,
        max_paths=20,
    )

    hop_counts = {len(path.triples) for path in paths}
    assert 1 in hop_counts
    assert 2 in hop_counts
    assert 3 in hop_counts
    assert all(path.source_stage == "external_graphapi_expansion" for path in paths)
    assert all(len(path.nodes) == len(set(path.nodes)) for path in paths)
    social_paths = [path for path in paths if path.nodes and path.nodes[-1] == "Social anthropology"]
    assert social_paths
    assert any(path.terminal_node_id == "m.social" for path in social_paths)
    assert all(path.edge_ids for path in social_paths)


def test_processed_freebase_graph_api_uses_entity_names_and_literal_tails(tmp_path: Path) -> None:
    db_path = _build_processed_freebase_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    neighbors = api.get_neighbors(
        "m.obama",
        include_reverse=True,
        relation_filter=[
            "people.person.date_of_birth",
            "people.person.spouse_s.people.marriage.spouse",
            "people.person.place_of_birth",
        ],
        limit=10,
    )
    triples = {neighbor.triple.relation: neighbor.triple for neighbor in neighbors}

    assert triples["people.person.date_of_birth"].head == "Barack Obama"
    assert triples["people.person.date_of_birth"].tail == "1961-08-04"
    assert triples["people.person.spouse_s.people.marriage.spouse"].tail == "Michelle Obama"
    assert triples["people.person.place_of_birth"].tail == "Honolulu"
    assert next(
        neighbor.relation_name for neighbor in neighbors if neighbor.triple.relation == "people.person.place_of_birth"
    ) == "place of birth"


def test_processed_freebase_graph_api_falls_back_to_entity_names_and_stops_at_literals(tmp_path: Path) -> None:
    db_path = _build_processed_freebase_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    honolulu_neighbors = api.get_neighbors(
        "m.honolulu",
        include_reverse=False,
        relation_filter=["location.location.containedby"],
        limit=10,
    )
    assert honolulu_neighbors
    assert honolulu_neighbors[0].triple.tail == "Hawaii"

    paths = api.expand_paths(
        seed_nodes=["m.obama"],
        relation_hints=[],
        max_depth=3,
        strict_relation_filter=False,
        max_paths=20,
    )
    assert any(path.nodes == ["Barack Obama", "1961-08-04"] for path in paths)
    assert not any(path.nodes[:2] == ["Barack Obama", "1961-08-04"] and len(path.nodes) > 2 for path in paths)
    assert any("Michelle Obama" in path.text for path in paths)
    assert any("place of birth" in path.text for path in paths)


def test_processed_freebase_graph_api_resolves_entities_and_relations(tmp_path: Path) -> None:
    db_path = _build_processed_freebase_graph_db(tmp_path)
    api = SQLiteGraphAPI(db_path, neighbor_limit=10)

    assert api.resolve_entity_mentions(["Barack Obama"], top_k=1) == ["m.obama"]
    assert api.resolve_relation_hints(["date of birth", "spouse"], top_k=2)


def test_pipeline_graphapi_accepts_legacy_fb2m_db_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())
    db_path = _build_test_graph_db(tmp_path)
    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=RoutingLLM(),
        config={
            "embedding": {"model": "fake-model"},
            "graphapi": {
                "enabled": True,
                "backend": "sqlite",
                "fb2m_db_path": str(db_path),
            },
        },
    )

    assert isinstance(pipeline.graph_api, SQLiteGraphAPI)


def test_pipeline_does_not_touch_graphapi_when_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=RoutingLLM(),
        config=_make_config(str(tmp_path / "missing.sqlite"), enabled=False),
    )

    result = pipeline.run_with_metadata(
        question="what did nick clegg study at university",
        sample_id="WebQTrn-143",
        gold_answers=["Social anthropology"],
        entity_source="provided",
        topic_entities_override=["Nick Clegg"],
    )

    assert pipeline.graph_api is None
    assert all("graphapi" not in step.note.lower() for step in result.search_trace)
    assert not any(path.source_stage.startswith("external_graphapi") for path in result.candidate_paths)


def test_pipeline_optional_graphapi_augments_webqtrn_143(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    db_path = _build_test_graph_db(tmp_path)
    baseline_pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=RoutingLLM(),
        config=_make_config(str(db_path), enabled=False),
    )
    enabled_pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=RoutingLLM(),
        config=_make_config(str(db_path), enabled=True),
    )

    baseline = baseline_pipeline.run_with_metadata(
        question="what did nick clegg study at university",
        sample_id="WebQTrn-143",
        gold_answers=["Social anthropology"],
        entity_source="provided",
        topic_entities_override=["Nick Clegg"],
    )
    enabled = enabled_pipeline.run_with_metadata(
        question="what did nick clegg study at university",
        sample_id="WebQTrn-143",
        gold_answers=["Social anthropology"],
        entity_source="provided",
        topic_entities_override=["Nick Clegg"],
    )

    assert baseline.answer == "insufficient"
    assert enabled.answer == "Social anthropology"
    assert not any(path.source_stage.startswith("external_graphapi") for path in baseline.candidate_paths)
    assert any(step.stage == "webqsp_topic_entity_resolution" for step in enabled.search_trace)
    assert any(step.stage == "webqsp_graph_construction" for step in enabled.search_trace)
    assert any(path.source_stage == "external_graphapi_expansion" for path in enabled.candidate_paths)
    assert len(enabled.candidate_paths) > len(baseline.candidate_paths)


def test_webqsp_pipeline_uses_external_sqlite_flow_when_graphapi_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())

    db_path = _build_test_graph_db(tmp_path)
    config = _make_config(str(db_path), enabled=True)
    config.setdefault("datasets", {}).setdefault("webqsp", {})["external_graph_only"] = True
    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=RoutingLLM(),
        config=config,
    )

    result = pipeline.run_with_metadata(
        question="what did nick clegg study at university",
        sample_id="WebQTrn-143",
        gold_answers=["Social anthropology"],
        dataset_name="webqsp",
        entity_source="provided",
        topic_entities_override=["Nick Clegg"],
    )

    assert result.answer == "Social anthropology"
    assert any(step.stage == "webqsp_topic_entity_resolution" for step in result.search_trace)
    assert any(step.stage == "webqsp_graph_construction" for step in result.search_trace)
    assert any(path.source_stage == "external_graphapi_expansion" for path in result.candidate_paths)


def test_cwq_pipeline_uses_subquestion_sqlite_expansion(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())
    recording_graph_api = RecordingGraphAPI()
    monkeypatch.setattr("kgqa.reasoning.pipeline.KGQAPipeline._build_graph_api", lambda self: recording_graph_api)

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=CWQRoutingLLM(),
        config=_make_config(str(tmp_path / "unused.sqlite"), enabled=True),
    )

    result = pipeline.run_with_metadata(
        question="What type of government is used in the country with Northern District?",
        sample_id="cwq-sample",
        gold_answers=["Parliamentary system"],
        dataset_name="cwq",
        entity_source="provided",
        topic_entities_override=["Northern District"],
    )

    assert result.answer == "Parliamentary system"
    assert recording_graph_api.expand_calls[0]["seed_nodes"] == ["m.northern_district"]
    assert recording_graph_api.expand_calls[0]["relation_hints"] == []
    assert recording_graph_api.expand_calls[1]["seed_nodes"] == ["m.israel"]
    assert recording_graph_api.expand_calls[0]["strict_relation_filter"] is False
    assert recording_graph_api.expand_calls[0]["max_depth"] == 2
    assert "government.government.form_of_government" in recording_graph_api.bootstrap_relation_ids
    assert result.candidate_paths[0].source_stage == "external_graphapi_expansion"
    assert result.search_trace[0].stage == "cwq_topic_entity_resolution"
    assert any(step.stage == "cwq_sub_question_alignment" for step in result.search_trace)
    assert any(step.stage == "cwq_alignment_summary" for step in result.search_trace)
    assert any(step.stage == "cwq_graph_construction" for step in result.search_trace)
    assert any(step.stage == "path_summarization" for step in result.search_trace)
    assert any(step.stage == "local_relation_probe" for step in result.search_trace)
    assert result.summarized_paths[0]["summary_type"] == "question"
    assert result.alignment_debug[0]["stage"] == "question_topic_entity_alignment"
    assert result.alignment_debug[1]["stage"] == "sub_question_alignment"
    assert "combined_pruning_relation_ids=[]" in next(
        step.note for step in result.search_trace if step.stage == "cwq_alignment_summary"
    )
    assert any(step.stage == "local_hint_rerank" for step in result.search_trace)
    assert any("explore_relation_ids=[]" in step.note for step in result.search_trace if step.stage == "cwq_graph_construction")
    assert any("primary_paths=1" in step.note for step in result.search_trace if step.stage == "cwq_graph_construction")
    assert result.exploration_hints.aligned_relations == []
    assert any(item["sub_question_id"] == "step_1" for item in result.summarized_paths if "sub_question_id" in item)
    assert any(item["sub_question_id"] == "step_2" for item in result.summarized_paths if "sub_question_id" in item)


def test_cwq_relation_reranking_prefers_contextual_country_relation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("kgqa.kg.entity_linking.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pruning.get_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_entity_embedding_model", lambda _: FakeEmbeddingModel())
    monkeypatch.setattr("kgqa.reasoning.pipeline.get_pruning_embedding_model", lambda _: FakeEmbeddingModel())
    recording_graph_api = RecordingGraphAPI()
    monkeypatch.setattr("kgqa.reasoning.pipeline.KGQAPipeline._build_graph_api", lambda self: recording_graph_api)

    pipeline = KGQAPipeline(
        kg=KnowledgeGraph([]),
        llm=RoutingLLM(),
        config=_make_config(str(tmp_path / "unused.sqlite"), enabled=True),
    )
    sub_question = SubQuestionSpec(
        id="sq1",
        question="Which nation has the Alta Verapaz Department?",
        topic_entities=[EntityMentionSpec(name="Alta Verapaz Department", role="topic_entity")],
        interested_relations=[
            RelationHintSpec(
                name="contained by country",
                aliases=["administrative parent"],
                direction="topic_entity -> answer",
                description="country containing Alta Verapaz Department",
            )
        ],
        expected_answer_type="country",
    )

    resolved = pipeline._resolve_cwq_sub_question_with_sqlite(sub_question=sub_question, graph_api=recording_graph_api)

    assert resolved.selected_relation_ids == []
    assert resolved.relation_candidates == []
    assert resolved.resolution_debug["relation_hints_mode"] == "local_path_filter_only"
