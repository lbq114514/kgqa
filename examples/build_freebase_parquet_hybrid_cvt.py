#!/usr/bin/env python3
"""Build processed Freebase CSVs with one retrieval-friendly main view plus optional CVT sidecars.

This script is intentionally compatible with the KGQA processed Freebase loaders:

- entities.csv columns: ``id,name,aliases,types,is_cvt``
- triplets.csv columns: ``head,head_name,relation,tail,tail_name,tail_kind``

The main ``triplets.csv`` intentionally contains only one view at a time:

- ``flattened``: CVT-expanded compound relations for retrieval
- ``raw``: original raw edges for structure-preserving builds

Raw CVT-connected edges can optionally be written to a separate sidecar CSV so they
do not pollute the main retrieval graph.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, TextIO, Tuple

import pyarrow.parquet as pq


FREEBASE_NS = "http://rdf.freebase.com/ns/"

NAME_PRED = "type.object.name"
ALIAS_PRED = "common.topic.alias"
TYPE_PRED = "type.object.type"

KEEP_META_PREDICATES = {NAME_PRED, ALIAS_PRED, TYPE_PRED}
DROP_PREFIXES = ("freebase.", "user.", "dataworld.")
DROP_EXACT = {
    "type.object.key",
    "type.object.permission",
    "type.object.creator",
    "type.object.timestamp",
    "type.object.attribution",
    "common.topic.webpage",
    "common.topic.topic_equivalent_webpage",
    "common.topic.image",
    "common.topic.description",
}
ORDINARY_ENTITY_TYPES = {"common.topic", "type.topic"}

EDGE_FIELDNAMES = ["head", "relation", "tail", "tail_kind"]
TRIPLET_FIELDNAMES = ["head", "head_name", "relation", "tail", "tail_name", "tail_kind"]
ENTITY_FIELDNAMES = ["id", "name", "aliases", "types", "is_cvt"]

CVT_BUCKETS = 64
TRIPLET_BUCKETS = 128
ENTITY_BUCKETS = 64

DEFAULT_MEDIATOR_HINTS = (
    ".mediator",
    "_mediator",
    "government.government_position_held",
    "people.marriage",
    "award.award_nomination",
    "business.employment_tenure",
    "film.performance",
    "sports.sports_team_roster",
)


@dataclass
class ParseStats:
    processed_rows: int = 0
    temp_edges_written: int = 0
    skipped_subject_not_mid: int = 0
    skipped_predicate_dropped: int = 0


@dataclass
class NodeStats:
    in_count: int = 0
    out_count: int = 0
    out_to_id_count: int = 0
    out_to_literal_count: int = 0
    in_domains: Set[str] = field(default_factory=set)
    out_domains: Set[str] = field(default_factory=set)


@dataclass
class BucketGroup:
    paths: List[str]
    files: List[TextIO]
    writers: List[csv.writer]


def strip_freebase_ns(uri: str) -> Optional[str]:
    if uri.startswith(FREEBASE_NS):
        return uri[len(FREEBASE_NS) :]
    return None


def is_mid_or_gid(value: str) -> bool:
    return value.startswith("m.") or value.startswith("g.")


def keep_predicate(pred: str, drop_base: bool = False) -> bool:
    if pred in KEEP_META_PREDICATES:
        return True
    if pred in DROP_EXACT:
        return False
    if pred.startswith(DROP_PREFIXES):
        return False
    if drop_base and pred.startswith("base."):
        return False
    if pred.startswith("type.") or pred.startswith("common."):
        return False
    return "." in pred


def log_progress(message: str) -> None:
    print(message, file=sys.stderr)


def relation_domain_prefix(relation: str) -> str:
    return relation.split(".", 1)[0]


def stable_bucket(values: Tuple[str, ...], bucket_count: int) -> int:
    payload = "\x1f".join(values).encode("utf-8", "surrogatepass")
    return zlib.crc32(payload) % bucket_count


def create_bucket_group(temp_dir: Path, prefix: str, bucket_count: int) -> BucketGroup:
    paths: List[str] = []
    files: List[TextIO] = []
    writers: List[csv.writer] = []
    for index in range(bucket_count):
        path = temp_dir / f"{prefix}_{index:03d}.csv"
        handle = open(path, "w", newline="", encoding="utf-8")
        paths.append(str(path))
        files.append(handle)
        writers.append(csv.writer(handle))
    return BucketGroup(paths=paths, files=files, writers=writers)


def close_bucket_group(group: BucketGroup) -> None:
    for handle in group.files:
        handle.close()


def _update_node_stats_out_id(
    node_stats: DefaultDict[str, NodeStats],
    subject_short: str,
    predicate_short: str,
) -> None:
    stats = node_stats[subject_short]
    stats.out_count += 1
    stats.out_to_id_count += 1
    stats.out_domains.add(relation_domain_prefix(predicate_short))


def _update_node_stats_out_literal(
    node_stats: DefaultDict[str, NodeStats],
    subject_short: str,
    predicate_short: str,
) -> None:
    stats = node_stats[subject_short]
    stats.out_count += 1
    stats.out_to_literal_count += 1
    stats.out_domains.add(relation_domain_prefix(predicate_short))


def _update_node_stats_in(
    node_stats: DefaultDict[str, NodeStats],
    object_short: str,
    predicate_short: str,
) -> None:
    stats = node_stats[object_short]
    stats.in_count += 1
    stats.in_domains.add(relation_domain_prefix(predicate_short))


def collect_pass1(
    parquet_dir: Path,
    args: argparse.Namespace,
    edge_path: str,
) -> Tuple[
    Dict[str, str],
    DefaultDict[str, Set[str]],
    DefaultDict[str, Set[str]],
    DefaultDict[str, NodeStats],
    ParseStats,
]:
    names: Dict[str, str] = {}
    aliases: DefaultDict[str, Set[str]] = defaultdict(set)
    types: DefaultDict[str, Set[str]] = defaultdict(set)
    node_stats: DefaultDict[str, NodeStats] = defaultdict(NodeStats)
    stats = ParseStats()

    parquet_files = sorted(parquet_dir.glob("data-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data-*.parquet files found in {parquet_dir}")
    log_progress(f"[pass1] found {len(parquet_files)} parquet files")

    with open(edge_path, "w", newline="", encoding="utf-8") as edge_file:
        writer = csv.writer(edge_file)
        writer.writerow(EDGE_FIELDNAMES)

        for file_index, file_path in enumerate(parquet_files, start=1):
            parquet_file = pq.ParquetFile(str(file_path))
            log_progress(
                f"[pass1] file {file_index}/{len(parquet_files)} {file_path.name} "
                f"({parquet_file.metadata.num_rows:,} rows)"
            )
            for batch in parquet_file.iter_batches():
                subjects = batch.column("subject").to_pylist()
                predicates = batch.column("predicate").to_pylist()
                objects = batch.column("object").to_pylist()
                object_types = batch.column("object_type").to_pylist()
                object_languages = batch.column("object_language").to_pylist()

                for row_index in range(batch.num_rows):
                    stats.processed_rows += 1
                    subject_short = strip_freebase_ns(subjects[row_index])
                    if subject_short is None or not is_mid_or_gid(subject_short):
                        stats.skipped_subject_not_mid += 1
                        continue

                    predicate_short = strip_freebase_ns(predicates[row_index])
                    if predicate_short is None or not keep_predicate(predicate_short, drop_base=args.drop_base):
                        stats.skipped_predicate_dropped += 1
                        continue

                    object_type = object_types[row_index]
                    object_value = objects[row_index]
                    object_language = object_languages[row_index]

                    if predicate_short == NAME_PRED:
                        if object_type == "literal" and object_language == "en" and object_value and subject_short not in names:
                            names[subject_short] = object_value
                        continue

                    if predicate_short == ALIAS_PRED:
                        if object_type == "literal" and object_language == "en" and object_value:
                            aliases[subject_short].add(object_value)
                        continue

                    if predicate_short == TYPE_PRED:
                        if object_type == "uri":
                            object_short = strip_freebase_ns(object_value)
                            if object_short is not None:
                                types[subject_short].add(sys.intern(object_short))
                        continue

                    if object_type == "uri":
                        object_short = strip_freebase_ns(object_value)
                        if object_short is not None and is_mid_or_gid(object_short):
                            writer.writerow([subject_short, predicate_short, object_short, "id"])
                            stats.temp_edges_written += 1
                            _update_node_stats_out_id(node_stats, subject_short, predicate_short)
                            _update_node_stats_in(node_stats, object_short, predicate_short)
                    elif object_type == "literal":
                        if object_value is not None:
                            writer.writerow([subject_short, predicate_short, str(object_value), "literal"])
                            stats.temp_edges_written += 1
                            _update_node_stats_out_literal(node_stats, subject_short, predicate_short)

    return names, aliases, types, node_stats, stats


def detect_cvt_nodes(
    names: Dict[str, str],
    aliases: DefaultDict[str, Set[str]],
    types: DefaultDict[str, Set[str]],
    node_stats: DefaultDict[str, NodeStats],
    mediator_type_hints: Tuple[str, ...],
) -> Tuple[Set[str], int, List[Tuple[str, int, NodeStats, str]]]:
    cvt_nodes: Set[str] = set()
    candidate_count = 0
    debug_examples: List[Tuple[str, int, NodeStats, str]] = []

    for node, stats in node_stats.items():
        if not is_mid_or_gid(node):
            continue
        if stats.in_count < 1 or stats.out_count < 1:
            continue
        if node in names or aliases.get(node):
            continue

        candidate_count += 1
        node_types = types.get(node, set())
        score = 0
        reasons: List[str] = []

        if 1 <= len(node_types) <= 3:
            score += 2
            reasons.append("1_3_types")
        if node_types and all(value not in ORDINARY_ENTITY_TYPES for value in node_types):
            score += 1
            reasons.append("non_topic_types")
        if stats.in_count * stats.out_count >= 2:
            score += 1
            reasons.append("multi_role_structure")
        if stats.out_count >= 2:
            score += 1
            reasons.append("multiple_outgoing")
        if stats.out_to_literal_count >= 1:
            score += 1
            reasons.append("literal_outgoing")
        if stats.in_domains and stats.out_domains and stats.in_domains != stats.out_domains:
            score += 1
            reasons.append("mixed_relation_domains")
        if node.startswith("g."):
            score += 2
            reasons.append("g_prefix")
        if stats.out_count == 1 and stats.in_count == 1 and len(node_types) == 0:
            score -= 2
            reasons.append("weak_1x1_without_types")
        if any(value in ORDINARY_ENTITY_TYPES for value in node_types):
            score -= 2
            reasons.append("ordinary_topic_type")
        if node_types and any(any(hint in value for hint in mediator_type_hints) for value in node_types):
            score += 3
            reasons.append("mediator_or_schema_hint")

        if stats.in_count + stats.out_count < 3:
            continue
        if stats.out_count == 1 and stats.in_count == 1 and len(node_types) == 0 and stats.out_to_literal_count == 0:
            continue
        if score < 3:
            continue

        cvt_nodes.add(node)
        if len(debug_examples) < 10:
            debug_examples.append((node, score, stats, ",".join(sorted(reasons))))

    return cvt_nodes, candidate_count, debug_examples


def write_triplet_candidate(
    bucket_group: BucketGroup,
    head: str,
    relation: str,
    tail: str,
    tail_kind: str,
) -> None:
    bucket = stable_bucket((head, relation, tail, tail_kind), len(bucket_group.writers))
    bucket_group.writers[bucket].writerow([head, relation, tail, tail_kind])


def route_edges_to_buckets(
    edge_path: str,
    cvt_nodes: Set[str],
    emit_main_cvt_raw: bool,
    cvt_bucket_group: BucketGroup,
    triplet_bucket_group: BucketGroup,
    raw_cvt_triplet_bucket_group: BucketGroup | None,
    log_every: int,
) -> int:
    processed_edges = 0
    with open(edge_path, "r", newline="", encoding="utf-8") as edge_file:
        reader = csv.DictReader(edge_file)
        for row in reader:
            processed_edges += 1
            head = row["head"]
            relation = row["relation"]
            tail = row["tail"]
            tail_kind = row["tail_kind"]

            head_is_cvt = head in cvt_nodes
            tail_is_cvt = tail_kind == "id" and tail in cvt_nodes
            involves_cvt = head_is_cvt or tail_is_cvt

            if not involves_cvt or emit_main_cvt_raw:
                write_triplet_candidate(triplet_bucket_group, head, relation, tail, tail_kind)
            if involves_cvt and raw_cvt_triplet_bucket_group is not None:
                write_triplet_candidate(raw_cvt_triplet_bucket_group, head, relation, tail, tail_kind)

            if tail_is_cvt:
                bucket = stable_bucket((tail,), len(cvt_bucket_group.writers))
                cvt_bucket_group.writers[bucket].writerow(["I", tail, head, relation, "", ""])
            if head_is_cvt:
                bucket = stable_bucket((head,), len(cvt_bucket_group.writers))
                cvt_bucket_group.writers[bucket].writerow(["O", head, "", relation, tail, tail_kind])

            if processed_edges % log_every == 0:
                log_progress(f"[pass2] routed_edges={processed_edges:,}")

    log_progress(f"[pass2] done routed_edges={processed_edges:,}")
    return processed_edges


def expand_cvt_buckets(
    cvt_bucket_group: BucketGroup,
    triplet_bucket_group: BucketGroup,
    flatten_cvt: bool,
) -> int:
    if not flatten_cvt:
        return 0

    flattened_triplets = 0
    log_stride = max(1, len(cvt_bucket_group.paths) // 8)
    for bucket_index, path in enumerate(cvt_bucket_group.paths, start=1):
        incoming: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)
        outgoing: DefaultDict[str, List[Tuple[str, str, str]]] = defaultdict(list)

        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                role, cvt, head, relation, tail, tail_kind = row
                if role == "I":
                    incoming[cvt].append((head, relation))
                else:
                    outgoing[cvt].append((relation, tail, tail_kind))

        for cvt, in_edges in incoming.items():
            out_edges = outgoing.get(cvt, [])
            if not out_edges:
                continue
            for head, left_relation in in_edges:
                for right_relation, tail, tail_kind in out_edges:
                    if tail_kind == "id" and head == tail:
                        continue
                    write_triplet_candidate(
                        triplet_bucket_group,
                        head,
                        f"{left_relation}.{right_relation}",
                        tail,
                        tail_kind,
                    )
                    flattened_triplets += 1

        if bucket_index % log_stride == 0 or bucket_index == len(cvt_bucket_group.paths):
            log_progress(
                f"[pass3] expanded_cvt_bucket={bucket_index}/{len(cvt_bucket_group.paths)} "
                f"flattened_triplets_so_far={flattened_triplets:,}"
            )
    return flattened_triplets


def finalize_triplets(
    triplet_bucket_group: BucketGroup,
    entity_bucket_group: BucketGroup,
    triplets_path: str,
    names: Dict[str, str],
    collect_entity_ids: bool = True,
) -> int:
    output_triplets = 0
    log_stride = max(1, len(triplet_bucket_group.paths) // 8)
    with open(triplets_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(TRIPLET_FIELDNAMES)

        for bucket_index, path in enumerate(triplet_bucket_group.paths, start=1):
            seen: Set[Tuple[str, str, str, str]] = set()
            with open(path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if not row:
                        continue
                    head, relation, tail, tail_kind = row
                    key = (head, relation, tail, tail_kind)
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.writerow(
                        [
                            head,
                            names.get(head, ""),
                            relation,
                            tail,
                            names.get(tail, "") if tail_kind == "id" else "",
                            tail_kind,
                        ]
                    )
                    if collect_entity_ids:
                        entity_bucket_group.writers[stable_bucket((head,), len(entity_bucket_group.writers))].writerow([head])
                        if tail_kind == "id":
                            entity_bucket_group.writers[stable_bucket((tail,), len(entity_bucket_group.writers))].writerow([tail])
                    output_triplets += 1

            if bucket_index % log_stride == 0 or bucket_index == len(triplet_bucket_group.paths):
                log_progress(
                    f"[pass4] finalized_triplet_bucket={bucket_index}/{len(triplet_bucket_group.paths)} "
                    f"output_triplets_so_far={output_triplets:,}"
                )

    return output_triplets


def finalize_entities(
    entity_bucket_group: BucketGroup,
    entities_path: str,
    names: Dict[str, str],
    aliases: DefaultDict[str, Set[str]],
    types: DefaultDict[str, Set[str]],
    cvt_nodes: Set[str],
) -> int:
    final_entities = 0
    log_stride = max(1, len(entity_bucket_group.paths) // 8)
    with open(entities_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(ENTITY_FIELDNAMES)

        for bucket_index, path in enumerate(entity_bucket_group.paths, start=1):
            seen_entities: Set[str] = set()
            with open(path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if not row:
                        continue
                    entity_id = row[0]
                    if entity_id in seen_entities:
                        continue
                    seen_entities.add(entity_id)
                    writer.writerow(
                        [
                            entity_id,
                            names.get(entity_id, ""),
                            "|".join(sorted(aliases.get(entity_id, set()))),
                            "|".join(sorted(types.get(entity_id, set()))),
                            "1" if entity_id in cvt_nodes else "0",
                        ]
                    )
                    final_entities += 1

            if bucket_index % log_stride == 0 or bucket_index == len(entity_bucket_group.paths):
                log_progress(
                    f"[pass5] finalized_entity_bucket={bucket_index}/{len(entity_bucket_group.paths)} "
                    f"final_entities_so_far={final_entities:,}"
                )

    return final_entities


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build KGQA-compatible Freebase CSVs with both raw and flattened CVT edges."
    )
    parser.add_argument("--parquet-dir", required=True, help="Directory containing data-*.parquet files")
    parser.add_argument("--entities", default="entities.csv", help="Output entities.csv path")
    parser.add_argument("--triplets", default="triplets.csv", help="Output triplets.csv path")
    parser.add_argument("--drop-base", action="store_true", help="Drop base.* predicates during preprocessing")
    parser.add_argument(
        "--main-triplet-view",
        choices=("flattened", "raw"),
        default="flattened",
        help="Which single view to write into the main triplets.csv (default: flattened).",
    )
    parser.add_argument(
        "--raw-cvt-triplets",
        default="",
        help="Optional sidecar CSV path for raw CVT-connected edges; not used by the standard builders.",
    )
    parser.add_argument(
        "--flatten-cvt",
        action="store_true",
        default=True,
        help="Enable flattened CVT compound relation generation when the main view is flattened (default: enabled).",
    )
    parser.add_argument(
        "--no-flatten-cvt",
        dest="flatten_cvt",
        action="store_false",
        help="Disable flattened CVT compound relation generation.",
    )
    parser.add_argument(
        "--mediator-type-hint",
        action="append",
        default=[],
        help="Extra substring hint used to score CVT-like nodes from their type ids.",
    )
    parser.add_argument("--log-every", type=int, default=1_000_000, help="Log progress every N routed edges.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.log_every <= 0:
        parser.error("--log-every must be a positive integer")

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.is_dir():
        parser.error(f"--parquet-dir does not exist or is not a directory: {parquet_dir}")

    entities_path = Path(args.entities)
    triplets_path = Path(args.triplets)
    raw_cvt_triplets_path = Path(args.raw_cvt_triplets) if args.raw_cvt_triplets else None
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    triplets_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_cvt_triplets_path is not None:
        raw_cvt_triplets_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(__file__).resolve().parent / ".tmp_freebase_kgqa_hybrid"
    temp_dir.mkdir(parents=True, exist_ok=True)

    edge_temp = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix="freebase_kgqa_edges_",
        suffix=".csv",
        dir=str(temp_dir),
        delete=False,
    )
    edge_path = edge_temp.name
    edge_temp.close()

    cvt_bucket_group = create_bucket_group(temp_dir, "cvt_bucket", CVT_BUCKETS)
    triplet_bucket_group = create_bucket_group(temp_dir, "triplet_bucket", TRIPLET_BUCKETS)
    entity_bucket_group = create_bucket_group(temp_dir, "entity_bucket", ENTITY_BUCKETS)
    raw_cvt_triplet_group = create_bucket_group(temp_dir, "raw_cvt_triplet_bucket", TRIPLET_BUCKETS) if args.raw_cvt_triplets else None

    temp_paths = [edge_path, *cvt_bucket_group.paths, *triplet_bucket_group.paths, *entity_bucket_group.paths]
    if raw_cvt_triplet_group is not None:
        temp_paths.extend(raw_cvt_triplet_group.paths)
    mediator_type_hints = tuple(DEFAULT_MEDIATOR_HINTS + tuple(str(value) for value in args.mediator_type_hint))
    emit_main_cvt_raw = args.main_triplet_view == "raw"
    flatten_main_cvt = args.main_triplet_view == "flattened" and args.flatten_cvt

    try:
        log_progress(
            f"[temp] edge_csv={edge_path} temp_dir={temp_dir} "
            f"main_triplet_view={args.main_triplet_view} flatten_main_cvt={flatten_main_cvt} "
            f"raw_cvt_sidecar={bool(args.raw_cvt_triplets)}"
        )
        names, aliases, types, node_stats, pass1_stats = collect_pass1(parquet_dir, args, edge_path)

        cvt_nodes, candidate_count, debug_examples = detect_cvt_nodes(
            names=names,
            aliases=aliases,
            types=types,
            node_stats=node_stats,
            mediator_type_hints=mediator_type_hints,
        )
        log_progress(
            f"[pass2] cvt_nodes={len(cvt_nodes):,} candidate_nodes={candidate_count:,} "
            f"main_triplet_view={args.main_triplet_view} flatten_main_cvt={flatten_main_cvt}"
        )
        for node, score, stats_entry, reasons in debug_examples:
            log_progress(
                f"[pass2] sample_cvt node={node} score={score} "
                f"in={stats_entry.in_count} out={stats_entry.out_count} reasons={reasons}"
            )

        route_edges_to_buckets(
            edge_path=edge_path,
            cvt_nodes=cvt_nodes,
            emit_main_cvt_raw=emit_main_cvt_raw,
            cvt_bucket_group=cvt_bucket_group,
            triplet_bucket_group=triplet_bucket_group,
            raw_cvt_triplet_bucket_group=raw_cvt_triplet_group,
            log_every=max(1, args.log_every // 10),
        )
        close_bucket_group(cvt_bucket_group)

        flattened_triplets = expand_cvt_buckets(
            cvt_bucket_group=cvt_bucket_group,
            triplet_bucket_group=triplet_bucket_group,
            flatten_cvt=flatten_main_cvt,
        )
        close_bucket_group(triplet_bucket_group)
        if raw_cvt_triplet_group is not None:
            close_bucket_group(raw_cvt_triplet_group)

        output_triplets = finalize_triplets(
            triplet_bucket_group=triplet_bucket_group,
            entity_bucket_group=entity_bucket_group,
            triplets_path=str(triplets_path),
            names=names,
        )
        raw_cvt_triplets = 0
        if raw_cvt_triplet_group is not None:
            raw_cvt_triplets = finalize_triplets(
                triplet_bucket_group=raw_cvt_triplet_group,
                entity_bucket_group=entity_bucket_group,
                triplets_path=str(raw_cvt_triplets_path),
                names=names,
                collect_entity_ids=False,
            )
        close_bucket_group(entity_bucket_group)

        final_entities = finalize_entities(
            entity_bucket_group=entity_bucket_group,
            entities_path=str(entities_path),
            names=names,
            aliases=aliases,
            types=types,
            cvt_nodes=cvt_nodes,
        )

        log_progress(
            f"[done] processed_rows={pass1_stats.processed_rows:,} "
            f"temp_edges={pass1_stats.temp_edges_written:,} "
            f"cvt_nodes={len(cvt_nodes):,} "
            f"flattened_triplets={flattened_triplets:,} "
            f"output_triplets={output_triplets:,} "
            f"raw_cvt_triplets={raw_cvt_triplets:,} "
            f"final_entities={final_entities:,}"
        )
    finally:
        groups = [cvt_bucket_group, triplet_bucket_group, entity_bucket_group]
        if raw_cvt_triplet_group is not None:
            groups.append(raw_cvt_triplet_group)
        for group in groups:
            for handle in group.files:
                if not handle.closed:
                    handle.close()
        for path in temp_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
