"""End-to-end training-free KGQA pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any

from kgqa.kg.entity_linking import (
    extract_entities_with_llm,
    get_embedding_model as get_entity_embedding_model,
    link_entities_to_kg,
    set_default_embedding_model as set_entity_embedding_model,
)
from kgqa.kg.graph import KnowledgeGraph
from kgqa.kg.graph_api import BaseGraphAPI
from kgqa.kg.relation_beam import (
    BaseLLMRelationBeamAdapter,
    RelationBeamConfig,
    SubquestionRelationExplorer,
    CVT_BUNDLE_SOURCE_STAGE,
    expand_cvt_bundle_paths,
)
from kgqa.kg.sqlite_graph_api import SQLiteGraphAPI
from kgqa.retrieval import (
    HybridSearcher,
    IndexedSQLiteGraphBackend,
    PathCandidateScorer,
    PathReranker,
    PathSelector,
    SearchRequest,
)
from kgqa.kg.subgraph import build_question_subgraph, prune_subgraph_by_topic_connectivity
from kgqa.llm.base import BaseLLM
from kgqa.reasoning.answering import (
    check_subquestion_sufficiency,
    check_sufficiency,
    generate_answer,
    generate_subquestion_answer,
)
from kgqa.reasoning.exploration import (
    annotate_paths_with_relation_matches,
    expand_nodes_from_paths,
    expand_nodes_with_graph_api,
    explore_graphapi_bootstrap_paths,
    explore_supplement_paths,
    explore_graphapi_supplement_paths,
    explore_topic_entity_paths,
    generate_supplement_hints_with_llm,
    sort_paths_by_relation_bias,
)
from kgqa.reasoning.pruning import (
    get_embedding_model as get_pruning_embedding_model,
    prune_relation_beam_paths,
    prune_paths,
    set_default_embedding_model as set_pruning_embedding_model,
)
from kgqa.reasoning.question_analysis import analyze_question, plan_agentic_step
from kgqa.reasoning.subquestion_solvers import (
    SubquestionExecutionContext,
    SubquestionSolverResult,
    SubquestionSolverRouter,
)
from kgqa.reasoning.summarization import summarize_paths
from kgqa.utils.json_utils import robust_json_parse
from kgqa.utils.text import deduplicate_paths
from kgqa.utils.logging import get_logger
from kgqa.utils.types import (
    AgenticPlanStep,
    AgenticRunState,
    AgenticStepResult,
    AnswerResult,
    EntityMentionSpec,
    ExplorationHints,
    PipelineResult,
    QuestionAnalysisResult,
    ReasoningPath,
    ResolvedCandidate,
    ResolvedSubQuestion,
    SearchTraceStep,
    SubQuestionSpec,
    SupplementHints,
)

LOGGER = get_logger(__name__)

DEPENDENCY_ENTITY_FILTER_PROMPT = """You filter upstream resolved entities for one KGQA sub-question.
Current sub-question:
{sub_question}

Expected answer type:
{expected_answer_type}

Resolved dependencies:
{resolved_dependencies}

Candidate dependency entities:
{candidate_entities}

Return only JSON:
{{
  "selected_entity_ids": ["..."]
}}

Rules:
- Select only entities that should be used as input seeds for the current sub-question.
- Prefer the primary entity answer from dependencies, not incidental related entities.
- Reject entities from the wrong semantic role or wrong type.
- Select from the provided candidate entity ids only.
- If unsure, keep the most directly answer-like entity.
"""

ENTITY_CANDIDATE_DISAMBIGUATION_PROMPT = """You disambiguate entity candidates for one KGQA mention in context.
Question or sub-question:
{question}

Mention:
{mention}

Expected type:
{expected_type}

Role:
{role}

Candidate entities:
{candidate_entities}

Return only JSON:
{{
  "selected_entity_ids": ["..."]
}}

Rules:
- Select only from the provided candidate entity ids.
- Prefer entities whose types and local relations best match the current question context.
- Reject candidates from the wrong semantic domain even when the surface name matches exactly.
- Use local relation summaries as evidence for disambiguation.
- Keep the result short; usually one id is best, at most two if the context is genuinely ambiguous.
"""

RELATION_SUPERVISION_PROMPT = """You select useful relation choices for one KGQA sub-question.
Current sub-question:
{sub_question}

Expected answer type:
{expected_answer_type}

Resolved dependencies:
{resolved_dependencies}

Candidate relations from the current one-hop neighborhood:
{candidate_relations}

Return only JSON:
{{
  "selected_relation_ids": ["..."]
}}

Rules:
- Select only relations that are directly useful for strict probing of the current sub-question.
- Prefer relations whose sample targets match both the expected answer type and the question semantics.
- Reject relations from the wrong semantic domain even if they share surface words with the question.
- Use semantic-domain reasoning in a general way:
  if the sub-question is about country / religion / language / government / location / airport / education / sports result / date,
  then relations whose sample targets are mainly books, TV episodes, songs, tracks, albums, or other clearly unrelated media objects are usually off_topic unless those targets are themselves the expected answer type.
- Prefer relations that move toward the answer type over side facts such as awards, metadata, ceremony details, or bookkeeping descriptions.
- Select from the provided relation ids only.
- You may select multiple relations, but keep the list short.
- If no relation is clearly useful, return an empty list.
"""

RELATION_BEAM_ENTITY_HINT_PROMPT = """You predict short hypothesis entity hints for KG relation search.
Sub-question: {sub_question}
Expected answer type: {expected_answer_type}
Resolved dependencies: {resolved_dependencies}

Return JSON only:
{{
  "entity_hints": ["..."]
}}

Rules:
- Output short noun phrases or entity-like mentions, not full sentences.
- Hints may be implicit bridge entities, role labels, alternate mentions, or likely character/object names.
- Prefer hints grounded in the current sub-question and resolved dependency evidence.
- Do not repeat the expected answer type alone.
- Return at most 3 hints.
"""

NOISE_RELATION_HINTS = {
    "cycling",
    "cyclist",
    "award",
    "cocktail",
    "injury",
    "interest",
    "cereal",
    "olympics",
    "formula_1",
}

CONTEXTUAL_RELATION_QUERY_EXPANSIONS = {
    "leader": [
        "leader",
        "religious leader",
        "head of state",
        "president",
        "prime minister",
        "organization leadership",
    ],
    "mascot": [
        "mascot",
        "sports team mascot",
        "team mascot",
        "organization mascot",
    ],
    "national anthem": [
        "national anthem",
        "anthem of country",
        "country national anthem",
    ],
    "contained by country": [
        "contained by country",
        "administrative parent",
        "contained by",
        "country containing",
        "part of country",
    ],
    "located in region": [
        "located in region",
        "contained by region",
        "administrative parent region",
        "region containing country",
    ],
}


def _path_preview_payload(paths: list["ReasoningPath"], max_paths: int = 8, max_triples: int = 4) -> list[dict[str, Any]]:
    """Build a compact candidate-path preview for logs before pruning."""
    preview: list[dict[str, Any]] = []
    for path in paths[:max_paths]:
        preview.append(
            {
                "source_stage": path.source_stage,
                "matched_relations": list(path.matched_relations[:4]),
                "text": path.text,
                "triples": [
                    {
                        "head": triple.head,
                        "relation": triple.relation,
                        "tail": triple.tail,
                    }
                    for triple in path.triples[:max_triples]
                ],
            }
        )
    return preview


def _path_matches_debug_keyword(path: "ReasoningPath", keywords: list[str]) -> bool:
    """Return True when a path touches any debug keyword in text or triples."""
    haystacks = [path.text.lower(), *[triple.head.lower() for triple in path.triples], *[triple.relation.lower() for triple in path.triples], *[triple.tail.lower() for triple in path.triples]]
    return any(keyword in haystack for keyword in keywords for haystack in haystacks)


@dataclass
class KGQAPipeline:
    """Orchestrates the full KGQA search, pruning, summarization, and answering flow."""

    kg: KnowledgeGraph
    llm: BaseLLM
    config: dict

    def __post_init__(self) -> None:
        embedding_model = self.config["embedding"]["model"]
        set_entity_embedding_model(embedding_model)
        set_pruning_embedding_model(embedding_model)
        get_entity_embedding_model(embedding_model)
        get_pruning_embedding_model(embedding_model)
        self.graph_api = self._build_graph_api()
        self.retrieval_backend = self.graph_api
        self.path_candidate_scorer = PathCandidateScorer(self.retrieval_backend)
        self.path_reranker = PathReranker(self.path_candidate_scorer)
        self.path_selector = PathSelector(max_paths=self._retrieval_max_paths())
        self.hybrid_searcher = (
            HybridSearcher(
                backend=self.retrieval_backend,
                scorer=self.path_candidate_scorer,
                reranker=self.path_reranker,
                selector=self.path_selector,
            )
            if self.retrieval_backend is not None
            else None
        )
        self.subquestion_solver_router = SubquestionSolverRouter(config=self.config)
        LOGGER.info(
            "KGQAPipeline graphapi enabled=%s stages=%s neighbor_limit=%d relation_bias_top_k=%d compact_precise_path_prompt=%s",
            self.graph_api is not None,
            self.config.get("graphapi", {}).get("augment_stages", []),
            self._graphapi_neighbor_limit(),
            self._graphapi_relation_bias_top_k(),
            self._use_compact_precise_path_prompt(),
        )

    def run(self, question: str) -> PipelineResult:
        """Run the complete pipeline and return a structured result."""
        return self.run_with_metadata(question=question)

    def run_with_metadata(
        self,
        question: str,
        sample_id: str | None = None,
        gold_answers: list[str] | None = None,
        dataset_name: str = "webqsp",
        entity_source: str = "llm",
        topic_entities_override: list[str] | None = None,
    ) -> PipelineResult:
        """Run the complete pipeline and attach optional dataset metadata."""
        search_cfg = self.config["search"]
        dmax = int(search_cfg.get("dmax", 3))
        w1 = int(search_cfg.get("w1", 80))
        wmax = int(search_cfg.get("wmax", 3))
        max_expand_steps = int(search_cfg.get("max_expand_steps", 2))
        max_paths_per_pair = int(search_cfg.get("max_paths_per_pair", 20))

        if self._agentic_enabled():
            use_external_only = dataset_name == "cwq" or (
                dataset_name == "webqsp" and self._webqsp_uses_external_graph_only()
            )
            return self._run_agentic_loop(
                question=question,
                sample_id=sample_id,
                gold_answers=gold_answers,
                dataset_name=dataset_name,
                entity_source=entity_source,
                topic_entities_override=topic_entities_override,
                dmax=dmax,
                w1=w1,
                wmax=wmax,
                max_expand_steps=max_expand_steps,
                max_paths_per_pair=max_paths_per_pair,
                use_external_only=use_external_only,
                trace_prefix="cwq" if dataset_name == "cwq" else "webqsp",
            )

        if dataset_name == "cwq":
            return self._run_sqlite_external_only(
                question=question,
                sample_id=sample_id,
                gold_answers=gold_answers,
                dmax=dmax,
                w1=w1,
                wmax=wmax,
                max_expand_steps=max_expand_steps,
                entity_source=entity_source,
                topic_entities_override=topic_entities_override,
                trace_prefix="cwq",
            )

        if dataset_name == "webqsp" and self._webqsp_uses_external_graph_only():
            return self._run_sqlite_external_only(
                question=question,
                sample_id=sample_id,
                gold_answers=gold_answers,
                dmax=dmax,
                w1=w1,
                wmax=wmax,
                max_expand_steps=max_expand_steps,
                entity_source=entity_source,
                topic_entities_override=topic_entities_override,
                trace_prefix="webqsp",
            )

        if entity_source == "provided" and topic_entities_override:
            topic_entities = list(topic_entities_override)
        else:
            topic_candidates = extract_entities_with_llm(question, self.llm)
            topic_entities = link_entities_to_kg(topic_candidates, sorted(self.kg.entities), top_k=1)
            if not topic_entities:
                topic_entities = topic_candidates

        subgraph = build_question_subgraph(self.kg, topic_entities, dmax=dmax)
        subgraph = prune_subgraph_by_topic_connectivity(subgraph, topic_entities)

        analysis = analyze_question(
            question,
            self.llm,
            dmax=dmax,
            topic_entities_override=topic_entities,
        )
        dpredict = min(analysis.predicted_depth, dmax)
        supplement_hints = SupplementHints()

        trace: list[SearchTraceStep] = []
        candidate_paths = explore_topic_entity_paths(
            subgraph=subgraph,
            ordered_topic_entities=analysis.ordered_topic_entities or topic_entities,
            dpredict=dpredict,
            dmax=dmax,
            max_paths_per_pair=max_paths_per_pair,
        )
        trace.append(
            SearchTraceStep(
                stage="topic_entity_path_exploration",
                depth=dpredict,
                candidate_count=len(candidate_paths),
                pruned_count=0,
                note="Initial topic-entity path exploration.",
            )
        )

        pruned_paths, summarized_paths, answer_result, sufficient = self._evaluate_paths(
            question=question,
            topic_entities=analysis.topic_entities or topic_entities,
            question_analysis=analysis,
            candidate_paths=candidate_paths,
            w1=w1,
            wmax=wmax,
            relation_hints=[],
            sample_id=sample_id,
            trace=trace,
            dmax=dmax,
            depth=dpredict,
        )
        if sufficient:
            return self._build_pipeline_result(
                question=question,
                topic_entities=analysis.topic_entities or topic_entities,
                question_analysis=analysis,
                dmax=dmax,
                dpredict=dpredict,
                candidate_paths=candidate_paths,
                pruned_paths=pruned_paths,
                summarized_paths=summarized_paths,
                answer=answer_result.answer,
                sufficient=True,
                confidence="high",
                supplement_hints=supplement_hints,
                sample_id=sample_id,
                gold_answers=gold_answers,
                predicted_answers=answer_result.predicted_answers,
                search_trace=trace,
            )

        supplement_hints = generate_supplement_hints_with_llm(
            question=question,
            topic_entities=analysis.topic_entities or topic_entities,
            current_paths=candidate_paths,
            kg_entities=sorted(subgraph.entities),
            kg_relations=sorted(self.kg.relations),
            llm=self.llm,
        )
        supplement_paths = explore_supplement_paths(
            subgraph=subgraph,
            topic_entities=analysis.ordered_topic_entities or topic_entities,
            supplement_hints=supplement_hints,
            dmax=dmax,
            max_paths_per_pair=max_paths_per_pair,
        )
        external_supplement_paths: list[ReasoningPath] = []
        if self._graphapi_enabled_for_stage("supplement"):
            external_supplement_paths = explore_graphapi_supplement_paths(
                topic_entities=analysis.ordered_topic_entities or (analysis.topic_entities or topic_entities),
                supplement_hints=supplement_hints,
                graph_api=self.graph_api,
                neighbor_limit=self._graphapi_neighbor_limit(),
                relation_bias_top_k=self._graphapi_relation_bias_top_k(),
            )
        merged_supplement_paths = deduplicate_paths(supplement_paths + external_supplement_paths)
        if self._graphapi_enabled_for_stage("supplement"):
            LOGGER.info(
                "GraphAPI supplement merged sample=%d external=%d total=%d sample_id=%s",
                len(supplement_paths),
                len(external_supplement_paths),
                len(merged_supplement_paths),
                sample_id,
            )
        supplement_note = "Supplementary bridge-entity exploration."
        if self._graphapi_enabled_for_stage("supplement"):
            supplement_note = (
                "Supplementary bridge-entity exploration with external graphapi. "
                f"sample={len(supplement_paths)} external={len(external_supplement_paths)} "
                f"merged={len(merged_supplement_paths)}"
            )
        trace.append(
            SearchTraceStep(
                stage="llm_supplement_path_exploration",
                depth=dmax,
                candidate_count=len(merged_supplement_paths),
                pruned_count=0,
                note=supplement_note,
            )
        )

        all_candidate_paths = deduplicate_paths(candidate_paths + merged_supplement_paths)
        pruned_paths, summarized_paths, answer_result, sufficient = self._evaluate_paths(
            question=question,
            topic_entities=topic_entities,
            question_analysis=analysis,
            candidate_paths=all_candidate_paths,
            w1=w1,
            wmax=wmax,
            relation_hints=self._supplement_relation_hints(supplement_hints),
            sample_id=sample_id,
            trace=trace,
            dmax=dmax,
            depth=dmax,
        )
        if sufficient:
            return self._build_pipeline_result(
                question=question,
                topic_entities=analysis.topic_entities or topic_entities,
                question_analysis=analysis,
                dmax=dmax,
                dpredict=dpredict,
                candidate_paths=all_candidate_paths,
                pruned_paths=pruned_paths,
                summarized_paths=summarized_paths,
                answer=answer_result.answer,
                sufficient=True,
                confidence="high",
                supplement_hints=supplement_hints,
                sample_id=sample_id,
                gold_answers=gold_answers,
                predicted_answers=answer_result.predicted_answers,
                search_trace=trace,
            )

        visited_nodes = {node for path in all_candidate_paths for node in path.nodes}
        sample_expanded_paths = expand_nodes_from_paths(
            current_paths=all_candidate_paths,
            kg=self.kg,
            visited_nodes=visited_nodes,
            max_expand_steps=max_expand_steps,
        )
        external_expanded_paths: list[ReasoningPath] = []
        if self._graphapi_enabled_for_stage("node_expand"):
            external_expanded_paths = expand_nodes_with_graph_api(
                current_paths=all_candidate_paths,
                graph_api=self.graph_api,
                visited_nodes=visited_nodes,
                relation_hints=supplement_hints.linked_relations or supplement_hints.relation_candidates,
                neighbor_limit=self._graphapi_neighbor_limit(),
                relation_bias_top_k=self._graphapi_relation_bias_top_k(),
            )
        expanded_paths = deduplicate_paths(sample_expanded_paths + external_expanded_paths)
        if self._graphapi_enabled_for_stage("node_expand"):
            LOGGER.info(
                "GraphAPI node expand merged sample=%d external=%d total=%d sample_id=%s",
                len(sample_expanded_paths),
                len(external_expanded_paths),
                len(expanded_paths),
                sample_id,
            )
        node_expand_note = "Endpoint one-hop expansion."
        if self._graphapi_enabled_for_stage("node_expand"):
            node_expand_note = (
                "Endpoint one-hop expansion with external graphapi. "
                f"sample={len(sample_expanded_paths)} external={len(external_expanded_paths)} "
                f"merged={len(expanded_paths)}"
            )
        trace.append(
            SearchTraceStep(
                stage="node_expand_exploration",
                depth=dmax,
                candidate_count=len(expanded_paths),
                pruned_count=0,
                note=node_expand_note,
            )
        )

        pruned_paths, summarized_paths, answer_result, sufficient = self._evaluate_paths(
            question=question,
            topic_entities=topic_entities,
            question_analysis=analysis,
            candidate_paths=expanded_paths,
            w1=w1,
            wmax=wmax,
            relation_hints=self._supplement_relation_hints(supplement_hints),
            sample_id=sample_id,
            trace=trace,
            dmax=dmax,
            depth=dmax,
        )
        if sufficient:
            confidence = "high"
            answer = answer_result.answer
        else:
            confidence = "low"
            answer = answer_result.answer if answer_result.answer else "insufficient"

        return self._build_pipeline_result(
            question=question,
            topic_entities=topic_entities,
            question_analysis=analysis,
            dmax=dmax,
            dpredict=dpredict,
            candidate_paths=expanded_paths,
            pruned_paths=pruned_paths,
            summarized_paths=summarized_paths,
            answer=answer,
            sufficient=sufficient,
            confidence=confidence,
            supplement_hints=supplement_hints,
            sample_id=sample_id,
            gold_answers=gold_answers,
            predicted_answers=answer_result.predicted_answers,
            search_trace=trace,
        )

    def _run_agentic_loop(
        self,
        question: str,
        sample_id: str | None,
        gold_answers: list[str] | None,
        dataset_name: str,
        entity_source: str,
        topic_entities_override: list[str] | None,
        dmax: int,
        w1: int,
        wmax: int,
        max_expand_steps: int,
        max_paths_per_pair: int,
        use_external_only: bool,
        trace_prefix: str,
    ) -> PipelineResult:
        """Run the step-by-step agentic loop as a strict dependency-ordered sub-question solver."""
        graph_api = self._require_sqlite_graph_api() if self.graph_api is not None else None
        if use_external_only and graph_api is None:
            raise ValueError("Agentic external-only execution requires graphapi to be enabled.")

        try:
            if entity_source == "provided" and topic_entities_override:
                topic_entities = list(topic_entities_override)
            else:
                topic_candidates = extract_entities_with_llm(question, self.llm)
                topic_entities = link_entities_to_kg(topic_candidates, sorted(self.kg.entities), top_k=1)
                if not topic_entities:
                    topic_entities = topic_candidates

            analysis = analyze_question(
                question,
                self.llm,
                dmax=dmax,
                topic_entities_override=topic_entities if topic_entities else None,
            )
            analysis = self._attach_downstream_filters_to_analysis(analysis)
            if not topic_entities:
                topic_entities = list(analysis.topic_entities or analysis.ordered_topic_entities)

            dpredict = min(analysis.predicted_depth, dmax)
            trace: list[SearchTraceStep] = []
            alignment_debug: list[dict[str, Any]] = []
            sub_question_trace_ids = {
                sub_question.id: f"step_{index}"
                for index, sub_question in enumerate(analysis.sub_questions, start=1)
            }

            topic_seed_ids: list[str] = []
            if graph_api is not None and topic_entities:
                topic_entity_candidates = self._resolve_entity_specs_with_sqlite(
                    specs=[EntityMentionSpec(name=name, role="topic_entity") for name in topic_entities],
                    graph_api=graph_api,
                    context_text=question,
                )
                topic_seed_ids = self._deduplicate_strings(
                    [candidate.id_or_mid for candidate in topic_entity_candidates]
                )
                alignment_debug.append(
                    {
                        "stage": "question_topic_entity_alignment",
                        "topic_entities": list(topic_entities),
                        "resolved_topic_entity_candidates": [
                            candidate.to_dict() for candidate in topic_entity_candidates
                        ],
                        "seed_node_ids": topic_seed_ids,
                    }
                )
                trace.append(
                    SearchTraceStep(
                        stage=f"{trace_prefix}_topic_entity_resolution",
                        depth=min(dmax, 2),
                        candidate_count=len(topic_entity_candidates),
                        pruned_count=len(topic_seed_ids),
                        note=f"topic_entities={topic_entities} seed_node_ids={topic_seed_ids}",
                    )
                )

            subgraph = None
            if not use_external_only:
                subgraph = build_question_subgraph(self.kg, topic_entities, dmax=dmax)
                subgraph = prune_subgraph_by_topic_connectivity(subgraph, topic_entities)

            run_state = AgenticRunState(
                original_question=question,
                global_goal=analysis.reasoning_indicator or question,
                current_frontier_entities=list(topic_seed_ids),
                current_text_constraints=list(topic_entities),
                pending_sub_question_ids=[sub_question.id for sub_question in analysis.sub_questions],
            )

            all_candidate_paths: list[ReasoningPath] = []
            final_pruned_paths: list[ReasoningPath] = []
            final_summaries: list[dict[str, Any]] = []
            final_answer_result = self._empty_answer_result()
            sufficient = False
            attempt_budget = self._agentic_max_subquestion_attempts()
            loop_index = 0

            while loop_index < self._agentic_max_loops():
                next_sub_question = self._select_next_ready_sub_question(
                    analysis=analysis,
                    run_state=run_state,
                )
                if next_sub_question is None:
                    break

                blocked_dependency_ids = [
                    dependency_id
                    for dependency_id in next_sub_question.depends_on
                    if dependency_id in run_state.failed_sub_question_ids
                ]
                if blocked_dependency_ids:
                    LOGGER.info(
                        "Sub-question blocked sub_question_id=%s blocked_dependencies=%s",
                        next_sub_question.id,
                        blocked_dependency_ids,
                    )
                    run_state.failed_sub_question_ids.append(next_sub_question.id)
                    run_state.pending_sub_question_ids = [
                        item for item in run_state.pending_sub_question_ids if item != next_sub_question.id
                    ]
                    break

                retry_failure_mode = ""
                for attempt_index in range(1, attempt_budget + 1):
                    loop_index += 1
                    run_state.loop_index = loop_index - 1
                    run_state.current_sub_question_id = next_sub_question.id
                    LOGGER.info(
                        "Sub-question start sub_question_id=%s depends_on=%s attempt=%d",
                        next_sub_question.id,
                        next_sub_question.depends_on,
                        attempt_index,
                    )

                    sub_question = self._build_next_subquestion_inputs(
                        sub_question=next_sub_question,
                        run_state=run_state,
                    )
                    if retry_failure_mode in {
                        "relation_found_but_branching_too_large",
                        "upstream_set_too_large_with_downstream_filter",
                    }:
                        sub_question = replace(
                            sub_question,
                            solver_type="constrained_collect",
                            execution_hints={
                                **dict(sub_question.execution_hints),
                                "failure_mode": retry_failure_mode,
                                "retry_attempt": attempt_index,
                            },
                        )
                    step_analysis = self._analysis_for_sub_question(analysis, sub_question)
                    trace.append(
                        SearchTraceStep(
                            stage="agentic_sub_question_start",
                            depth=min(dmax, loop_index),
                            candidate_count=len(run_state.evidence_bank),
                            pruned_count=len(run_state.step_history),
                            note=(
                                f"sub_question_id={sub_question.id} depends_on={sub_question.depends_on} "
                                f"attempt={attempt_index}"
                            ),
                        )
                    )

                    resolved_sub_question = None
                    dependency_seed_ids: list[str] = []
                    if graph_api is not None:
                        resolved_sub_question = self._resolve_cwq_sub_question_with_sqlite(
                            sub_question=sub_question,
                            graph_api=graph_api,
                        )
                        alignment_debug.append(
                            {
                                "stage": "sub_question_alignment",
                                "sub_question_id": sub_question.id,
                                "question": sub_question.question,
                                "attempt": attempt_index,
                                "resolved": resolved_sub_question.to_dict(),
                            }
                        )
                        trace.append(
                            SearchTraceStep(
                                stage=f"{trace_prefix}_sub_question_alignment",
                                depth=min(dmax, loop_index),
                                candidate_count=(
                                    len(resolved_sub_question.topic_entity_candidates)
                                    + len(resolved_sub_question.interested_node_candidates)
                                    + len(resolved_sub_question.relation_candidates)
                                ),
                                pruned_count=0,
                                note=(
                                    f"{sub_question.id}: seeds={resolved_sub_question.seed_node_ids} "
                                    f"selected_relations={resolved_sub_question.selected_relation_ids or resolved_sub_question.relation_ids} "
                                    f"depends_on={resolved_sub_question.depends_on} attempt={attempt_index}"
                                ),
                            )
                        )
                        raw_dependency_seed_ids = self._dependency_seed_ids_for_sub_question(
                            sub_question=sub_question,
                            dependency_seed_ids={
                                key: list(value.get("entity_ids", []))
                                for key, value in run_state.resolved_sub_answers.items()
                            },
                        )
                        dependency_seed_ids = self._filter_dependency_seed_ids_with_llm(
                            sub_question=sub_question,
                            dependency_seed_ids=raw_dependency_seed_ids,
                            resolved_sub_answers=run_state.resolved_sub_answers,
                            graph_api=graph_api,
                            trace=trace,
                            depth=min(dmax, loop_index),
                        )

                    if dependency_seed_ids:
                        step_seed_ids = list(dependency_seed_ids)
                    elif resolved_sub_question is not None:
                        resolved_seed_ids = self._deduplicate_strings(resolved_sub_question.seed_node_ids)
                        if resolved_seed_ids:
                            step_seed_ids = resolved_seed_ids
                        elif self._allow_question_topic_seed_fallback(sub_question):
                            step_seed_ids = list(topic_seed_ids)
                        else:
                            step_seed_ids = []
                    else:
                        step_seed_ids = list(topic_seed_ids)

                    if resolved_sub_question is not None:
                        relation_ids = self._relation_ids_for_attempt(
                            resolved_sub_question=resolved_sub_question,
                            attempt_index=attempt_index,
                            failure_mode=retry_failure_mode,
                        )
                    else:
                        relation_ids = self._deduplicate_strings(
                            [hint.name for hint in sub_question.interested_relations if hint.name]
                        )

                    trace.append(
                        SearchTraceStep(
                            stage=f"{trace_prefix}_alignment_summary",
                            depth=min(dmax, loop_index),
                            candidate_count=len(step_seed_ids) + len(relation_ids),
                            pruned_count=len(step_seed_ids),
                            note=(
                                f"sub_question_id={sub_question.id} combined_seed_ids={step_seed_ids} "
                                f"combined_pruning_relation_ids={relation_ids} explore_relation_ids=[] "
                                f"attempt={attempt_index}"
                            ),
                        )
                    )

                    strategy = self._strategy_for_attempt(attempt_index, retry_failure_mode)
                    plan_step_context = AgenticPlanStep(
                        step_id=f"{sub_question.id}_attempt_{attempt_index}",
                        question=sub_question.question,
                        goal=sub_question.question,
                        depends_on_step_ids=list(sub_question.depends_on),
                        topic_entity_mentions=self._effective_sub_question_topics(sub_question),
                        relation_hints=list(sub_question.interested_relations),
                        expected_answer_type=sub_question.expected_answer_type,
                        carryover_constraints=list(run_state.current_text_constraints),
                        downstream_filters=list(sub_question.downstream_filters),
                        stop_if_answered=False,
                        strategy=strategy,
                    )
                    solver_result = self._execute_subquestion_solver_step(
                        question=sub_question.question,
                        topic_entities=[
                            entity.name for entity in self._effective_sub_question_topics(sub_question) if entity.name
                        ]
                        or topic_entities,
                        subgraph=subgraph,
                        graph_api=graph_api,
                        plan_step=plan_step_context,
                        step_analysis=step_analysis,
                        step_seed_ids=step_seed_ids,
                        relation_ids=relation_ids,
                        sub_question=sub_question,
                        resolved_sub_answers=run_state.resolved_sub_answers,
                        dmax=dmax,
                        max_expand_steps=max_expand_steps,
                        max_paths_per_pair=max_paths_per_pair,
                        trace=trace,
                        trace_prefix=trace_prefix,
                    )
                    candidate_paths = list(solver_result.candidate_paths)
                    local_relation_ids = (
                        self._select_local_relation_probe_ids(
                            sub_question=sub_question,
                            step_seed_ids=step_seed_ids,
                            graph_api=graph_api,
                            candidate_paths=candidate_paths,
                            trace=trace,
                            depth=min(dmax, loop_index),
                            attempt_index=attempt_index,
                        )
                        if solver_result.allow_local_relation_probe
                        else []
                    )
                    if local_relation_ids and graph_api is not None:
                        strict_probe_paths = self._probe_paths_with_local_relations(
                            sub_question=sub_question,
                            step_seed_ids=step_seed_ids,
                            relation_ids=local_relation_ids,
                            graph_api=graph_api,
                            trace=trace,
                            depth=min(dmax, loop_index),
                        )
                        candidate_paths = deduplicate_paths(strict_probe_paths + candidate_paths)
                        candidate_paths = self._rerank_candidate_paths_with_local_hints(
                            sub_question=sub_question,
                            candidate_paths=candidate_paths,
                            trace=trace,
                            depth=min(dmax, loop_index),
                        )
                    effective_relation_ids = self._deduplicate_strings(relation_ids + local_relation_ids)
                    all_candidate_paths = deduplicate_paths(all_candidate_paths + candidate_paths)

                    pruned_paths = self._prune_paths_for_context(
                        question=sub_question.question,
                        question_analysis=step_analysis,
                        candidate_paths=candidate_paths,
                        w1=w1,
                        wmax=wmax,
                        relation_hints=effective_relation_ids,
                        sample_id=sample_id,
                        trace=trace,
                        depth=min(dmax, loop_index),
                        stage_note=f"Sub-question pruning for {sub_question.id} attempt={attempt_index}.",
                    )
                    final_pruned_paths = pruned_paths
                    step_summaries = self._summarize_paths_for_context(
                        question=question,
                        topic_entities=topic_entities,
                        question_analysis=step_analysis,
                        pruned_paths=pruned_paths,
                        trace=trace,
                        depth=min(dmax, loop_index),
                        stage_note=f"Sub-question summarization for {sub_question.id} attempt={attempt_index}.",
                        sub_question=sub_question,
                        evidence_bank=run_state.evidence_bank,
                        plan_step_context=plan_step_context.to_dict(),
                    )
                    step_summaries = [
                        {
                            **item,
                            "sub_question_id": sub_question_trace_ids.get(
                                str(item.get("sub_question_id") or sub_question.id),
                                str(item.get("sub_question_id") or sub_question.id),
                            ),
                        }
                        if isinstance(item, dict)
                        else item
                        for item in step_summaries
                    ]

                    sub_sufficiency = check_subquestion_sufficiency(
                        sub_question=sub_question,
                        topic_entities=[
                            entity.name for entity in self._effective_sub_question_topics(sub_question) if entity.name
                        ]
                        or topic_entities,
                        question_analysis=step_analysis,
                        dmax=dmax,
                        dpredict=min(dmax, loop_index),
                        summarized_paths=step_summaries,
                        exploration_hints=ExplorationHints(),
                        llm=self.llm,
                        agentic_state=run_state,
                    )
                    run_state.sufficiency_history.append(dict(sub_sufficiency))
                    sub_answer_result = generate_subquestion_answer(
                        sub_question=sub_question,
                        topic_entities=[
                            entity.name for entity in self._effective_sub_question_topics(sub_question) if entity.name
                        ]
                        or topic_entities,
                        question_analysis=step_analysis,
                        dmax=dmax,
                        dpredict=min(dmax, loop_index),
                        summarized_paths=step_summaries,
                        exploration_hints=ExplorationHints(),
                        llm=self.llm,
                        agentic_state=run_state,
                    )
                    sub_answer_result = self._backfill_subquestion_answer(
                        sub_question=sub_question,
                        pruned_paths=pruned_paths,
                        step_summaries=step_summaries,
                        answer_result=sub_answer_result,
                    )
                    output_entities, output_literals, type_filter_stats = self._canonicalize_subquestion_outputs(
                        graph_api=graph_api,
                        sub_question=sub_question,
                        pruned_paths=pruned_paths,
                        step_summaries=step_summaries,
                        answer_result=sub_answer_result,
                        structured_outputs=solver_result.structured_outputs,
                        primary_entity_ids=solver_result.primary_entity_ids,
                        primary_literals=solver_result.primary_literals,
                    )
                    sub_answer_result.reason = str(sub_sufficiency.get("reason", ""))
                    output_type_ok = self._subquestion_outputs_match_expected_type(
                        graph_api=graph_api,
                        sub_question=sub_question,
                        entity_ids=output_entities,
                        literals=output_literals,
                        answer_result=sub_answer_result,
                    )
                    soft_type_ok = False
                    if not output_type_ok:
                        soft_type_ok = self._subquestion_outputs_soft_match_expected_type(
                            graph_api=graph_api,
                            sub_question=sub_question,
                            entity_ids=output_entities,
                            literals=output_literals,
                            answer_result=sub_answer_result,
                            step_summaries=step_summaries,
                        )
                        if soft_type_ok:
                            output_type_ok = True

                    sub_resolved = output_type_ok and (
                        (
                            bool(sub_sufficiency.get("sufficient", False))
                            and bool(sub_answer_result.predicted_answers)
                        )
                        or (
                            bool(sub_answer_result.predicted_answers)
                            and self._summaries_have_fact_evidence(step_summaries)
                        )
                    )
                    trace.append(
                        SearchTraceStep(
                            stage="subquestion_answering",
                            depth=min(dmax, loop_index),
                            candidate_count=len(step_summaries),
                            pruned_count=len(step_summaries),
                            sufficient=sub_resolved,
                            note=(
                                sub_answer_result.reason
                                if output_type_ok and not soft_type_ok
                                else (
                                    f"{sub_answer_result.reason} Soft type pass-through accepted for expected_answer_type={sub_question.expected_answer_type}."
                                    if soft_type_ok
                                    else f"{sub_answer_result.reason} Type check failed for expected_answer_type={sub_question.expected_answer_type}."
                                )
                            ),
                        )
                    )
                    run_state.current_frontier_entities = list(
                        output_entities[: self._agentic_max_carryover_entities()]
                    )
                    run_state.current_text_constraints = list(
                        output_literals[: self._agentic_max_carryover_text_values()]
                    )

                    step_status = (
                        "resolved" if sub_resolved else ("failed" if attempt_index >= attempt_budget else "pending")
                    )
                    step_result = AgenticStepResult(
                        step_id=plan_step_context.step_id,
                        sub_question_id=sub_question.id,
                        sub_question_text=sub_question.question,
                        resolved_entity_ids=list(step_seed_ids),
                        resolved_relation_ids=list(effective_relation_ids),
                        retrieved_paths=list(candidate_paths),
                        summarized_evidence=list(step_summaries),
                        candidate_answers=list(sub_answer_result.predicted_answers),
                        sub_answer=sub_answer_result.answer,
                        sub_answer_entities=list(output_entities),
                        sub_answer_literals=list(output_literals),
                        status=step_status,
                        attempt_count=attempt_index,
                        depends_on_step_ids=list(sub_question.depends_on),
                        failure_reason="" if sub_resolved else sub_answer_result.reason,
                        carryover_entities=list(output_entities),
                        carryover_text_values=list(output_literals),
                        is_step_resolved=sub_resolved,
                        solver_type=solver_result.solver_type,
                        solver_debug={**dict(solver_result.solver_debug), **type_filter_stats},
                    )
                    run_state.step_history.append(step_result)

                    if sub_resolved:
                        run_state.evidence_bank.extend(step_summaries)
                        run_state.resolved_sub_answers[sub_question.id] = {
                            "answer": sub_answer_result.answer,
                            "predicted_answers": list(sub_answer_result.predicted_answers),
                            "entity_ids": list(output_entities),
                            "literals": list(output_literals),
                            "supporting_paths": list(sub_answer_result.supporting_paths),
                            "evidence_relations": self._collect_summary_relation_ids(step_summaries),
                            "evidence_text": self._collect_summary_evidence_text(step_summaries),
                            "depends_on": list(sub_question.depends_on),
                            "solver_type": solver_result.solver_type,
                            "solver_reason": sub_question.solver_reason,
                            "solver_debug": dict(solver_result.solver_debug),
                            "entity_set_role": str(
                                solver_result.structured_outputs.get("entity_set_role") or "candidate_answers"
                            ),
                            "attribute_rows": (
                                list(solver_result.structured_outputs.get("attribute_rows", []))
                                if isinstance(
                                    solver_result.structured_outputs.get("attribute_rows", []),
                                    list,
                                )
                                else []
                            ),
                            "set_operation_hint": str(
                                solver_result.structured_outputs.get("set_operation_hint") or "none"
                            ),
                            "verification_rows": (
                                list(solver_result.structured_outputs.get("verification_rows", []))
                                if isinstance(
                                    solver_result.structured_outputs.get("verification_rows", []),
                                    list,
                                )
                                else []
                            ),
                            "aggregate_rows": (
                                list(solver_result.structured_outputs.get("aggregate_rows", []))
                                if isinstance(
                                    solver_result.structured_outputs.get("aggregate_rows", []),
                                    list,
                                )
                                else []
                            ),
                            "solver_execution_mode": str(
                                solver_result.structured_outputs.get("solver_execution_mode") or ""
                            ),
                            "type_filter_before_count": int(
                                type_filter_stats.get("type_filter_before_count", 0)
                            ),
                            "type_filter_after_count": int(
                                type_filter_stats.get("type_filter_after_count", 0)
                            ),
                        }
                        run_state.pending_sub_question_ids = [
                            item for item in run_state.pending_sub_question_ids if item != sub_question.id
                        ]
                        LOGGER.info(
                            "Sub-question resolved sub_question_id=%s sub_answer=%s entity_ids=%s literal_values=%s",
                            sub_question.id,
                            sub_answer_result.answer,
                            output_entities,
                            output_literals,
                        )
                        break

                    retry_failure_mode = self._diagnose_subquestion_failure_mode(
                        sub_question=sub_question,
                        step_seed_ids=step_seed_ids,
                        relation_ids=effective_relation_ids,
                        candidate_paths=candidate_paths,
                        output_entities=output_entities,
                        output_type_ok=output_type_ok,
                        solver_result=solver_result,
                    )
                    trace.append(
                        SearchTraceStep(
                            stage="failure_diagnosis",
                            depth=min(dmax, loop_index),
                            candidate_count=len(candidate_paths),
                            pruned_count=len(output_entities),
                            note=(
                                f"sub_question_id={sub_question.id} failure_mode={retry_failure_mode or 'generic_retry'} "
                                f"attempt={attempt_index} relation_ids={effective_relation_ids[:5]} "
                                f"downstream_filters={sub_question.downstream_filters[:5]}"
                            ),
                        )
                    )
                    if attempt_index < attempt_budget:
                        LOGGER.info(
                            "Sub-question retry sub_question_id=%s attempt=%d reason=%s next_strategy=%s",
                            sub_question.id,
                            attempt_index,
                            sub_answer_result.reason,
                            self._strategy_for_attempt(attempt_index + 1, retry_failure_mode),
                        )
                        continue

                    LOGGER.info(
                        "Sub-question blocked sub_question_id=%s attempt=%d reason=%s",
                        sub_question.id,
                        attempt_index,
                        sub_answer_result.reason,
                    )
                    if sub_question.id not in run_state.failed_sub_question_ids:
                        run_state.failed_sub_question_ids.append(sub_question.id)
                    run_state.pending_sub_question_ids = [
                        item for item in run_state.pending_sub_question_ids if item != sub_question.id
                    ]
                    break

                if run_state.failed_sub_question_ids:
                    break

            resolved_ids = set(run_state.resolved_sub_answers)
            required_ids = [sub_question.id for sub_question in analysis.sub_questions]
            can_finalize_strict = bool(
                required_ids and all(sub_question_id in resolved_ids for sub_question_id in required_ids)
            )
            can_finalize_partial = bool(
                not can_finalize_strict
                and run_state.resolved_sub_answers
                and self._best_effort_answering_enabled()
            )

            if can_finalize_strict or can_finalize_partial:
                aggregate_summary = self._aggregate_subquestion_summaries(question, run_state.evidence_bank)
                aggregate_summary["summary_type"] = "question"
                aggregate_summary["resolved_sub_answers"] = dict(run_state.resolved_sub_answers)
                final_summaries = [aggregate_summary, *run_state.evidence_bank]
                trace.append(
                    SearchTraceStep(
                        stage="path_summarization",
                        depth=min(dmax, max(loop_index, 1)),
                        candidate_count=len(final_pruned_paths),
                        pruned_count=len(final_summaries),
                        note="aggregate evidence updated after strict sub-question execution",
                    )
                )
                LOGGER.info(
                    "Final synthesis resolved_sub_questions=%s",
                    list(run_state.resolved_sub_answers),
                )
                trace.append(
                    SearchTraceStep(
                        stage="final_synthesis",
                        depth=min(dmax, max(loop_index, 1)),
                        candidate_count=len(run_state.evidence_bank),
                        pruned_count=len(final_summaries),
                        note=(
                            f"resolved_sub_questions={list(run_state.resolved_sub_answers)} "
                            f"mode={'strict' if can_finalize_strict else 'partial'}"
                        ),
                    )
                )
                sufficiency = check_sufficiency(
                    question=question,
                    topic_entities=topic_entities,
                    question_analysis=analysis,
                    dmax=dmax,
                    dpredict=min(dmax, max(loop_index, 1)),
                    split_questions=analysis.split_questions,
                    summarized_paths=final_summaries,
                    exploration_hints=ExplorationHints(),
                    llm=self.llm,
                    agentic_state=run_state,
                )
                sufficient = bool(sufficiency.get("sufficient", False)) if can_finalize_strict else False
                final_answer_result = generate_answer(
                    question=question,
                    topic_entities=topic_entities,
                    question_analysis=analysis,
                    dmax=dmax,
                    dpredict=min(dmax, max(loop_index, 1)),
                    split_questions=analysis.split_questions,
                    summarized_paths=final_summaries,
                    exploration_hints=ExplorationHints(),
                    llm=self.llm,
                    agentic_state=run_state,
                )
                final_answer_result = self._filter_final_answer_result_by_type(
                    graph_api=graph_api,
                    question_analysis=analysis,
                    answer_result=final_answer_result,
                )
                if not sufficient:
                    final_answer_result = self._apply_best_effort_answer_selection(
                        question_analysis=analysis,
                        pruned_paths=final_pruned_paths,
                        candidate_paths=all_candidate_paths,
                        summarized_paths=final_summaries,
                        answer_result=final_answer_result,
                        agentic_state=run_state,
                        trace=trace,
                        depth=min(dmax, max(loop_index, 1)),
                    )
                    final_answer_result = self._filter_final_answer_result_by_type(
                        graph_api=graph_api,
                        question_analysis=analysis,
                        answer_result=final_answer_result,
                    )
                trace.append(
                    SearchTraceStep(
                        stage="question_answering",
                        depth=min(dmax, max(loop_index, 1)),
                        candidate_count=len(final_summaries),
                        pruned_count=len(final_summaries),
                        sufficient=sufficient,
                        note=(
                            f"{str(sufficiency.get('reason', ''))} "
                            if can_finalize_strict
                            else (
                                "partial final synthesis from "
                                f"resolved_sub_questions={list(run_state.resolved_sub_answers)}"
                            )
                        ).strip(),
                    )
                )
                if final_answer_result.answer:
                    run_state.final_answer = final_answer_result.answer
            else:
                aggregate_summary = self._aggregate_subquestion_summaries(question, run_state.evidence_bank)
                aggregate_summary["summary_type"] = "question"
                aggregate_summary["resolved_sub_answers"] = dict(run_state.resolved_sub_answers)
                final_summaries = (
                    [aggregate_summary, *run_state.evidence_bank]
                    if run_state.evidence_bank
                    else [aggregate_summary]
                )

            confidence = "high" if sufficient else "low"
            answer = final_answer_result.answer if final_answer_result.answer else "insufficient"
            return self._build_pipeline_result(
                question=question,
                topic_entities=topic_entities,
                question_analysis=analysis,
                dmax=dmax,
                dpredict=dpredict,
                candidate_paths=all_candidate_paths,
                pruned_paths=final_pruned_paths,
                summarized_paths=final_summaries,
                answer=answer,
                sufficient=sufficient,
                confidence=confidence,
                sample_id=sample_id,
                gold_answers=gold_answers,
                predicted_answers=final_answer_result.predicted_answers,
                search_trace=trace,
                alignment_debug=alignment_debug,
            )
        except Exception:
            LOGGER.exception(
                "Agentic loop crashed question=%s sample_id=%s dataset=%s",
                question,
                sample_id,
                dataset_name,
            )
            return self._build_pipeline_result(
                question=question,
                topic_entities=[],
                question_analysis=self._empty_question_analysis(question),
                dmax=dmax,
                dpredict=dmax,
                candidate_paths=[],
                pruned_paths=[],
                summarized_paths=[],
                answer="error",
                sufficient=False,
                confidence="error",
                sample_id=sample_id,
                gold_answers=gold_answers,
                predicted_answers=[],
                search_trace=[],
                alignment_debug=[],
            )

    def _empty_answer_result(self) -> AnswerResult:
        """Return a stable empty answer payload for agentic initialization."""
        return AnswerResult(
            sufficient=False,
            answer="insufficient",
            predicted_answers=[],
            resolved_entity_mentions=[],
            resolved_literals=[],
            supporting_paths=[],
        )

    def _empty_question_analysis(self, question: str) -> QuestionAnalysisResult:
        """Return a minimal QuestionAnalysisResult for graceful error handling."""
        return QuestionAnalysisResult(
            split_questions=[],
            reasoning_indicator=question,
            ordered_topic_entities=[],
            predicted_depth=1,
            topic_entities=[],
            sub_questions=[],
        )

    def _agentic_max_subquestion_attempts(self) -> int:
        """Return the retry budget for each individual sub-question."""
        return max(1, min(3, self._agentic_max_loops()))

    def _select_next_ready_sub_question(
        self,
        analysis: QuestionAnalysisResult,
        run_state: AgenticRunState,
    ) -> SubQuestionSpec | None:
        """Pick the next unresolved sub-question whose dependencies are all resolved."""
        pending_ids = set(run_state.pending_sub_question_ids)
        resolved_ids = set(run_state.resolved_sub_answers)
        for sub_question in analysis.sub_questions:
            if sub_question.id not in pending_ids:
                continue
            if all(dependency_id in resolved_ids for dependency_id in sub_question.depends_on):
                return sub_question
        return None

    def _build_next_subquestion_inputs(
        self,
        sub_question: SubQuestionSpec,
        run_state: AgenticRunState,
    ) -> SubQuestionSpec:
        """Inject resolved dependency answers into the current sub-question context."""
        dependency_literals: list[str] = []
        for dependency_id in sub_question.depends_on:
            dependency_literals.extend(run_state.resolved_sub_answers.get(dependency_id, {}).get("literals", []))
            dependency_literals.extend(run_state.resolved_sub_answers.get(dependency_id, {}).get("predicted_answers", []))
        dependency_literals = self._deduplicate_strings(
            [value for value in dependency_literals if self._is_propagatable_text(value)]
        )
        dependency_mentions = [
            EntityMentionSpec(name=value, role="constraint_or_anchor")
            for value in dependency_literals
        ]
        local_topic_entities = list(sub_question.local_topic_entities)
        if not local_topic_entities and not sub_question.topic_entities and sub_question.interested_nodes:
            local_topic_entities = [
                EntityMentionSpec(
                    name=entity.name,
                    aliases=list(entity.aliases),
                    expected_type=entity.expected_type,
                    role="topic_entity",
                )
                for entity in sub_question.interested_nodes
                if entity.name
            ]
        dependency_topic_entities = [EntityMentionSpec(name=value, role="topic_entity") for value in dependency_literals]
        if local_topic_entities:
            topic_entities = list(local_topic_entities)
        elif sub_question.depends_on:
            topic_entities = list(dependency_topic_entities)
        else:
            topic_entities = list(sub_question.topic_entities) or list(dependency_topic_entities)
        return SubQuestionSpec(
            id=sub_question.id,
            question=sub_question.question,
            topic_entities=topic_entities,
            local_topic_entities=local_topic_entities,
            interested_nodes=list(sub_question.interested_nodes) + dependency_mentions,
            interested_relations=list(sub_question.interested_relations),
            expected_answer_type=sub_question.expected_answer_type,
            expected_hop=sub_question.expected_hop,
            depends_on=list(sub_question.depends_on),
            solver_type=sub_question.solver_type,
            solver_reason=sub_question.solver_reason,
            downstream_filters=list(sub_question.downstream_filters),
            execution_hints=dict(sub_question.execution_hints),
        )

    def _attach_downstream_filters_to_analysis(
        self,
        analysis: QuestionAnalysisResult,
    ) -> QuestionAnalysisResult:
        """Propagate later-step filter terms back onto upstream dependency-producing sub-questions."""
        if not analysis.sub_questions:
            return analysis
        updated_sub_questions: list[SubQuestionSpec] = []
        for sub_question in analysis.sub_questions:
            filters = list(sub_question.downstream_filters)
            for candidate in analysis.sub_questions:
                if sub_question.id not in candidate.depends_on:
                    continue
                filters.extend(self._downstream_filter_terms_from_subquestion(candidate))
            filters = self._deduplicate_strings(filters)
            if filters != list(sub_question.downstream_filters):
                updated_sub_questions.append(replace(sub_question, downstream_filters=filters))
            else:
                updated_sub_questions.append(sub_question)
        analysis.sub_questions = updated_sub_questions
        return analysis

    def _downstream_filter_terms_from_subquestion(self, sub_question: SubQuestionSpec) -> list[str]:
        """Extract structured downstream filter terms from one dependent sub-question."""
        return self._deduplicate_strings(
            [
                *[node.name for node in sub_question.interested_nodes if node.name],
                *[relation.name for relation in sub_question.interested_relations if relation.name],
                *[
                    alias
                    for relation in sub_question.interested_relations
                    for alias in relation.aliases
                    if alias
                ],
                *[entity.name for entity in sub_question.local_topic_entities if entity.name],
                str(sub_question.expected_answer_type or "").strip(),
            ]
        )

    @staticmethod
    def _effective_sub_question_topics(sub_question: SubQuestionSpec) -> list[EntityMentionSpec]:
        """Return the strongest topic-like entities for one sub-question."""
        if sub_question.local_topic_entities:
            return list(sub_question.local_topic_entities)
        return list(sub_question.topic_entities)

    @classmethod
    def _allow_question_topic_seed_fallback(cls, sub_question: SubQuestionSpec) -> bool:
        """Only use question-level topic seeds when the sub-question has no local anchors."""
        return not (
            sub_question.depends_on
            or sub_question.local_topic_entities
            or sub_question.interested_nodes
        )

    def _execute_subquestion_solver_step(
        self,
        question: str,
        topic_entities: list[str],
        subgraph: Any,
        graph_api: Any,
        plan_step: AgenticPlanStep,
        step_analysis: QuestionAnalysisResult,
        step_seed_ids: list[str],
        relation_ids: list[str],
        sub_question: SubQuestionSpec | None,
        resolved_sub_answers: dict[str, dict[str, Any]],
        dmax: int,
        max_expand_steps: int,
        max_paths_per_pair: int,
        trace: list[SearchTraceStep],
        trace_prefix: str,
    ) -> SubquestionSolverResult:
        """Route the current sub-question to one typed solver."""
        target_sub_question = sub_question or SubQuestionSpec(
            id=plan_step.step_id,
            question=plan_step.question,
            topic_entities=list(plan_step.topic_entity_mentions),
            local_topic_entities=list(plan_step.topic_entity_mentions),
            interested_relations=list(plan_step.relation_hints),
            expected_answer_type=plan_step.expected_answer_type,
            expected_hop=1,
            depends_on=list(plan_step.depends_on_step_ids),
            downstream_filters=list(plan_step.downstream_filters),
        )
        context = SubquestionExecutionContext(
            original_question=question,
            sub_question=target_sub_question,
            graph_api=graph_api,
            step_seed_ids=list(step_seed_ids),
            relation_ids=list(relation_ids),
            resolved_sub_answers=resolved_sub_answers,
            execute_explore=lambda: self._execute_agentic_retrieval_step(
                question=plan_step.question,
                topic_entities=topic_entities,
                subgraph=subgraph,
                graph_api=graph_api,
                plan_step=plan_step,
                step_analysis=step_analysis,
                step_seed_ids=step_seed_ids,
                relation_ids=relation_ids,
                sub_question=target_sub_question,
                resolved_sub_answers=resolved_sub_answers,
                dmax=dmax,
                max_expand_steps=max_expand_steps,
                max_paths_per_pair=max_paths_per_pair,
                trace=trace,
                trace_prefix=trace_prefix,
            ),
        )
        result = self.subquestion_solver_router.run(context)
        trace.append(
            SearchTraceStep(
                stage="subquestion_solver_selected",
                depth=min(dmax, max(1, int(target_sub_question.expected_hop or 1))),
                candidate_count=len(step_seed_ids),
                pruned_count=len(result.candidate_paths),
                note=(
                    f"sub_question_id={target_sub_question.id} solver_type={result.solver_type} "
                    f"reason={target_sub_question.solver_reason or ''} seed_count={len(step_seed_ids)} "
                    f"debug={result.solver_debug}"
                ),
            )
        )
        return result

    def _relation_ids_for_attempt(
        self,
        resolved_sub_question: ResolvedSubQuestion,
        attempt_index: int,
        failure_mode: str = "",
    ) -> list[str]:
        """Relax relation constraints across retries while keeping the first attempt precise."""
        if failure_mode in {
            "relation_found_but_branching_too_large",
            "upstream_set_too_large_with_downstream_filter",
        }:
            return self._deduplicate_strings(
                resolved_sub_question.selected_relation_ids or resolved_sub_question.relation_ids
            )
        if attempt_index <= 1:
            return self._deduplicate_strings(
                resolved_sub_question.selected_relation_ids or resolved_sub_question.relation_ids
            )
        if attempt_index == 2:
            return self._deduplicate_strings(
                resolved_sub_question.relation_ids or resolved_sub_question.selected_relation_ids
            )
        return self._deduplicate_strings(
            [
                *resolved_sub_question.relation_ids,
                *resolved_sub_question.selected_relation_ids,
            ]
        )

    def _strategy_for_attempt(self, attempt_index: int, failure_mode: str = "") -> str:
        """Map retry count to retrieval strategy labels for traceability."""
        if failure_mode in {
            "relation_found_but_branching_too_large",
            "upstream_set_too_large_with_downstream_filter",
        }:
            return "constrained_collect"
        if attempt_index <= 1:
            return "auto"
        if attempt_index == 2:
            return "supplement"
        return "node_expand"

    def _diagnose_subquestion_failure_mode(
        self,
        *,
        sub_question: SubQuestionSpec,
        step_seed_ids: list[str],
        relation_ids: list[str],
        candidate_paths: list[ReasoningPath],
        output_entities: list[str],
        output_type_ok: bool,
        solver_result: SubquestionSolverResult,
    ) -> str:
        """Classify retry-worthy failures so the next attempt can choose a better execution mode."""
        relation_set = self._deduplicate_strings(
            [
                relation
                for path in candidate_paths[:20]
                for relation in path.matched_relations or [triple.relation for triple in path.triples]
                if relation
            ]
        )
        has_downstream_filters = bool(sub_question.downstream_filters)
        if (
            relation_ids
            and len(candidate_paths) >= self._agentic_branching_path_threshold()
            and len(relation_set) <= max(1, len(relation_ids) + 1)
            and (not output_entities or not output_type_ok)
            and solver_result.solver_type == "explore"
        ):
            return "relation_found_but_branching_too_large"
        if (
            sub_question.depends_on
            and len(step_seed_ids) >= self._agentic_large_dependency_seed_threshold()
            and has_downstream_filters
            and solver_result.solver_type == "explore"
        ):
            return "upstream_set_too_large_with_downstream_filter"
        if candidate_paths and not output_type_ok:
            return "type_mismatch_after_retrieval"
        if relation_ids and not candidate_paths:
            return "relation_not_confident"
        return "generic_retry"

    def _collect_subquestion_output_labels(
        self,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        step_summaries: list[dict[str, Any]],
        answer_result: AnswerResult,
        structured_outputs: dict[str, Any] | None = None,
        primary_literals: list[str] | None = None,
    ) -> list[str]:
        """Collect raw answer-like labels before canonical entity resolution."""
        structured_outputs = structured_outputs or {}
        labels: list[str] = []
        labels.extend(primary_literals or [])
        labels.extend(getattr(answer_result, "resolved_literals", []))
        labels.extend(getattr(answer_result, "predicted_answers", []))
        labels.extend(getattr(answer_result, "resolved_entity_mentions", []))
        labels.extend(
            [
                str(row.get("value") or "").strip()
                for row in structured_outputs.get("attribute_rows", [])
                if isinstance(row, dict) and str(row.get("value") or "").strip()
            ]
        )
        labels.extend(
            self._extract_answer_like_labels(
                sub_question=sub_question,
                pruned_paths=pruned_paths,
                summaries=step_summaries,
            )
        )
        return self._deduplicate_strings([label for label in labels if self._is_propagatable_text(label)])

    def _canonicalize_subquestion_outputs(
        self,
        graph_api: Any,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        step_summaries: list[dict[str, Any]],
        answer_result: AnswerResult,
        structured_outputs: dict[str, Any] | None = None,
        primary_entity_ids: list[str] | None = None,
        primary_literals: list[str] | None = None,
    ) -> tuple[list[str], list[str], dict[str, int]]:
        """Resolve one canonical set of sub-question entities/labels before state propagation."""
        labels = self._collect_subquestion_output_labels(
            sub_question=sub_question,
            pruned_paths=pruned_paths,
            step_summaries=step_summaries,
            answer_result=answer_result,
            structured_outputs=structured_outputs,
            primary_literals=primary_literals,
        )
        evidence_entity_ids = self._extract_subquestion_entity_ids_from_evidence(
            graph_api=graph_api,
            sub_question=sub_question,
            pruned_paths=pruned_paths,
            step_summaries=step_summaries,
            answer_result=answer_result,
            literals=labels,
        )
        candidate_entity_ids = self._deduplicate_strings(
            [
                *evidence_entity_ids,
                *(primary_entity_ids or []),
            ]
        )

        type_filter_stats = {
            "type_filter_before_count": len(candidate_entity_ids) or len(labels),
            "type_filter_after_count": len(candidate_entity_ids) or len(labels),
        }
        if self._filter_subquestion_candidates_enabled():
            candidate_entity_ids, labels, type_filter_stats = self._filter_candidates_by_expected_type(
                graph_api=graph_api,
                expected_type=sub_question.expected_answer_type,
                entity_ids=candidate_entity_ids,
                labels=labels,
            )

        canonical_labels = self._deduplicate_strings(
            [label for label in labels if self._is_propagatable_text(label)]
        )[: self._agentic_max_carryover_text_values()]
        canonical_entity_ids = self._deduplicate_strings(candidate_entity_ids)[: self._agentic_max_carryover_entities()]

        if canonical_entity_ids and graph_api is not None and hasattr(graph_api, "get_nodes_metadata_batched"):
            metadata_by_id = graph_api.get_nodes_metadata_batched(canonical_entity_ids)
            metadata_labels = self._deduplicate_strings(
                [
                    str(
                        (
                            metadata_by_id.get(entity_id).name
                            if metadata_by_id.get(entity_id) is not None
                            else graph_api.get_entity_display_name(entity_id)
                        )
                        or ""
                    ).strip()
                    for entity_id in canonical_entity_ids
                    if str(
                        (
                            metadata_by_id.get(entity_id).name
                            if metadata_by_id.get(entity_id) is not None
                            else graph_api.get_entity_display_name(entity_id)
                        )
                        or ""
                    ).strip()
                ]
            )
            if metadata_labels:
                canonical_labels = metadata_labels[: self._agentic_max_carryover_text_values()]

        if canonical_labels:
            answer_result.predicted_answers = list(canonical_labels)
            answer_result.resolved_literals = self._deduplicate_strings(
                list(canonical_labels) + list(answer_result.resolved_literals)
            )[: self._agentic_max_carryover_text_values()]
            answer_result.resolved_entity_mentions = list(canonical_labels)
            if (
                not self._is_propagatable_text(answer_result.answer)
                or self._normalize_relation_text(answer_result.answer)
                not in {self._normalize_relation_text(label) for label in canonical_labels}
            ):
                answer_result.answer = canonical_labels[0]

        return canonical_entity_ids, canonical_labels, type_filter_stats

    def _extract_subquestion_entity_ids_from_evidence(
        self,
        graph_api: Any,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        step_summaries: list[dict[str, Any]],
        answer_result: AnswerResult,
        literals: list[str],
    ) -> list[str]:
        """Prefer summary-level answer tails over intermediate path terminals."""
        if graph_api is None:
            return []
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        labels = {
            self._normalize_relation_text(value)
            for value in [
                *getattr(answer_result, "predicted_answers", []),
                *getattr(answer_result, "resolved_literals", []),
                *getattr(answer_result, "resolved_entity_mentions", []),
                *literals,
            ]
            if self._is_propagatable_text(str(value))
        }
        summary_entity_ids = self._resolve_summary_tail_entity_ids(
            graph_api=graph_api,
            expected_type=expected_type,
            step_summaries=step_summaries,
            labels=labels,
        )
        if summary_entity_ids:
            return summary_entity_ids

        entity_ids: list[str] = []
        for path in pruned_paths:
            terminal_id = str(path.terminal_node_id or "").strip()
            if not terminal_id or path.terminal_node_kind != "id":
                continue
            terminal_label = self._normalize_relation_text(
                str(graph_api.get_entity_display_name(terminal_id) or terminal_id)
            )
            if labels and terminal_label and terminal_label not in labels:
                continue
            entity_ids.append(terminal_id)
        if entity_ids:
            return self._deduplicate_strings(entity_ids)

        if literals:
            return self._deduplicate_strings(graph_api.resolve_entity_mentions(literals[:3], top_k=1))
        return []

    def _resolve_summary_tail_entity_ids(
        self,
        graph_api: Any,
        expected_type: str,
        step_summaries: list[dict[str, Any]],
        labels: set[str],
    ) -> list[str]:
        """Resolve answer-like summary tails, preferring expected-type matches over CVT mids."""
        if not hasattr(graph_api, "resolve_entity_mentions"):
            return []

        preferred_tail_labels: list[str] = []
        fallback_tail_labels: list[str] = []
        for item in step_summaries:
            if not isinstance(item, dict):
                continue
            for triple in item.get("key_triples", []):
                if not isinstance(triple, dict):
                    continue
                tail = str(triple.get("tail") or "").strip()
                if not tail:
                    continue
                normalized_tail = self._normalize_relation_text(tail)
                fallback_tail_labels.append(tail)
                if labels and normalized_tail and normalized_tail in labels:
                    preferred_tail_labels.append(tail)
        summary_tail_labels = self._deduplicate_strings(preferred_tail_labels or fallback_tail_labels)
        if not summary_tail_labels:
            return []

        candidate_ids: list[str] = []
        for label in summary_tail_labels[:10]:
            candidate_ids.extend(graph_api.resolve_entity_mentions([label], top_k=3))
        candidate_ids = self._deduplicate_strings(candidate_ids)
        if not candidate_ids:
            return []
        if not expected_type or not hasattr(graph_api, "get_nodes_metadata_batched"):
            return candidate_ids

        metadata_by_id = graph_api.get_nodes_metadata_batched(candidate_ids)
        typed_ids = [
            entity_id
            for entity_id in candidate_ids
            if self._entity_matches_expected_type(metadata_by_id.get(entity_id), expected_type)
        ]
        return self._deduplicate_strings(typed_ids or candidate_ids)

    def _best_effort_answering_enabled(self) -> bool:
        """Return whether low-confidence best-effort answer selection is enabled."""
        answering_cfg = self.config.get("answering", {})
        best_effort_cfg = answering_cfg.get("best_effort", {})
        if isinstance(best_effort_cfg, dict):
            return bool(best_effort_cfg.get("enabled", False))
        return bool(best_effort_cfg)

    def _is_best_effort_candidate_text(self, value: str) -> bool:
        """Keep node-like answer text and reject generic failure/explanation text."""
        if not self._is_propagatable_text(value):
            return False
        lowered = str(value).strip().lower()
        blocked_markers = (
            "i am sorry",
            "the provided evidence",
            "there is no information",
            "does not contain information",
            "cannot determine",
            "unable to determine",
            "no factual information",
        )
        return not any(marker in lowered for marker in blocked_markers)

    def _collect_best_effort_candidates_from_subanswers(
        self,
        question_analysis: QuestionAnalysisResult,
        agentic_state: AgenticRunState | None,
    ) -> list[str]:
        """Prefer the latest resolved sub-answer labels from the agentic execution state."""
        if agentic_state is None or not agentic_state.resolved_sub_answers:
            return []
        ordered_ids = [sub_question.id for sub_question in question_analysis.sub_questions]
        for sub_question_id in reversed(ordered_ids):
            payload = agentic_state.resolved_sub_answers.get(sub_question_id, {})
            if not isinstance(payload, dict):
                continue
            values = [
                *[str(value).strip() for value in payload.get("predicted_answers", []) if str(value).strip()],
                *[str(value).strip() for value in payload.get("literals", []) if str(value).strip()],
                str(payload.get("answer") or "").strip(),
            ]
            filtered = [value for value in values if self._is_best_effort_candidate_text(value)]
            if filtered:
                return self._deduplicate_strings(filtered)
        return []

    def _collect_best_effort_candidates_from_paths(
        self,
        paths: list[ReasoningPath],
    ) -> list[str]:
        """Collect terminal-node labels from highest-scoring paths."""
        if not paths:
            return []
        ranked_paths = sorted(paths, key=lambda path: float(getattr(path, "path_score", 0.0)), reverse=True)
        labels: list[str] = []
        for path in ranked_paths:
            if path.nodes:
                labels.append(str(path.nodes[-1]).strip())
            if len(labels) >= 8:
                break
        return self._deduplicate_strings([label for label in labels if self._is_best_effort_candidate_text(label)])

    def _collect_best_effort_candidates_from_summaries(
        self,
        summarized_paths: list[dict[str, Any]],
    ) -> list[str]:
        """Collect answer-like tail labels from summarized evidence."""
        labels: list[str] = []
        for item in summarized_paths:
            if not isinstance(item, dict):
                continue
            for triple in item.get("key_triples", []):
                if not isinstance(triple, dict):
                    continue
                tail = str(triple.get("tail") or "").strip()
                if tail:
                    labels.append(tail)
        return self._deduplicate_strings([label for label in labels if self._is_best_effort_candidate_text(label)])

    def _apply_best_effort_answer_selection(
        self,
        *,
        question_analysis: QuestionAnalysisResult,
        pruned_paths: list[ReasoningPath],
        candidate_paths: list[ReasoningPath],
        summarized_paths: list[dict[str, Any]],
        answer_result: AnswerResult,
        agentic_state: AgenticRunState | None,
        trace: list[SearchTraceStep],
        depth: int,
    ) -> AnswerResult:
        """Inject a low-confidence answer from evidence when strict answering is insufficient."""
        if not self._best_effort_answering_enabled():
            return answer_result
        if answer_result.predicted_answers and self._is_best_effort_candidate_text(answer_result.answer):
            return answer_result
        candidates = self._deduplicate_strings(
            [
                *self._collect_best_effort_candidates_from_subanswers(question_analysis, agentic_state),
                *[str(value).strip() for value in answer_result.predicted_answers if str(value).strip()],
                *[str(value).strip() for value in answer_result.resolved_literals if str(value).strip()],
                *[str(value).strip() for value in answer_result.resolved_entity_mentions if str(value).strip()],
                *self._collect_best_effort_candidates_from_paths(pruned_paths),
                *self._collect_best_effort_candidates_from_paths(candidate_paths),
                *self._collect_best_effort_candidates_from_summaries(summarized_paths),
            ]
        )
        candidates = [value for value in candidates if self._is_best_effort_candidate_text(value)]
        if not candidates:
            return answer_result
        if not answer_result.predicted_answers:
            answer_result.predicted_answers = list(candidates)
        if not self._is_best_effort_candidate_text(answer_result.answer):
            answer_result.answer = candidates[0]
        answer_result.resolved_literals = self._deduplicate_strings(
            list(answer_result.resolved_literals) + list(candidates[:3])
        )
        trace.append(
            SearchTraceStep(
                stage="best_effort_answering",
                depth=depth,
                candidate_count=len(candidate_paths),
                pruned_count=min(len(candidates), 3),
                sufficient=False,
                note=f"selected={candidates[:3]}",
            )
        )
        return answer_result

    def _question_expected_answer_type(self, question_analysis: QuestionAnalysisResult) -> str:
        """Approximate the final expected answer type from the last planned sub-question."""
        if not question_analysis.sub_questions:
            return ""
        for sub_question in reversed(question_analysis.sub_questions):
            expected_type = str(sub_question.expected_answer_type or "").strip()
            if expected_type:
                return expected_type
        return ""

    def _filter_final_answer_result_by_type(
        self,
        graph_api: Any,
        question_analysis: QuestionAnalysisResult,
        answer_result: AnswerResult,
    ) -> AnswerResult:
        """Filter final answer candidates by the inferred whole-question expected type."""
        if not self._filter_final_candidates_enabled():
            return answer_result
        expected_type = self._question_expected_answer_type(question_analysis)
        if not expected_type:
            return answer_result
        _, filtered_labels, _ = self._filter_candidates_by_expected_type(
            graph_api=graph_api,
            expected_type=expected_type,
            entity_ids=[],
            labels=[
                *answer_result.predicted_answers,
                *answer_result.resolved_literals,
                *answer_result.resolved_entity_mentions,
            ],
        )
        if not filtered_labels:
            return answer_result
        answer_result.predicted_answers = list(filtered_labels)
        answer_result.resolved_literals = self._deduplicate_strings(
            list(answer_result.resolved_literals) + list(filtered_labels)
        )
        answer_result.answer = filtered_labels[0]
        return answer_result

    def _subquestion_outputs_match_expected_type(
        self,
        graph_api: Any,
        sub_question: SubQuestionSpec,
        entity_ids: list[str],
        literals: list[str],
        answer_result: AnswerResult,
    ) -> bool:
        """Require resolved outputs to match the declared answer type before marking the step resolved."""
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        if not expected_type:
            return True
        if graph_api is None or not hasattr(graph_api, "get_nodes_metadata_batched"):
            return True

        candidate_ids = self._deduplicate_strings(list(entity_ids))
        if not candidate_ids and hasattr(graph_api, "resolve_entity_mentions"):
            label_candidates = self._deduplicate_strings(
                [
                    *[str(value).strip() for value in getattr(answer_result, "predicted_answers", []) if str(value).strip()],
                    *[str(value).strip() for value in getattr(answer_result, "resolved_literals", []) if str(value).strip()],
                    *[str(value).strip() for value in literals if str(value).strip()],
                ]
            )
            if label_candidates:
                candidate_ids = self._deduplicate_strings(graph_api.resolve_entity_mentions(label_candidates[:3], top_k=1))
        if not candidate_ids:
            return False

        metadata_by_id = graph_api.get_nodes_metadata_batched(candidate_ids)
        for entity_id in candidate_ids:
            metadata = metadata_by_id.get(entity_id)
            if metadata is None:
                continue
            if self._entity_matches_expected_type(metadata, expected_type):
                return True
        return False

    def _subquestion_outputs_soft_match_expected_type(
        self,
        *,
        graph_api: Any,
        sub_question: SubQuestionSpec,
        entity_ids: list[str],
        literals: list[str],
        answer_result: AnswerResult,
        step_summaries: list[dict[str, Any]],
    ) -> bool:
        """Allow direct-evidence answers through when type hints are semantically close but metadata is coarse."""
        if not self._allow_soft_type_pass_through():
            return False
        if not self._summaries_have_fact_evidence(step_summaries):
            return False
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        if not expected_type:
            return False

        alias_tokens = {
            token
            for alias in self._expected_type_aliases(expected_type)
            for token in self._normalize_relation_text(alias).split()
            if token and len(token) > 2
        }
        lexical_alias_map = {
            "sports team": {"team", "club", "franchise"},
            "educational institution": {"school", "college", "university", "institution"},
            "college": {"college", "university", "school", "institute"},
            "university": {"university", "college", "institute", "school"},
            "school": {"school", "college", "university", "academy"},
            "movie": {"movie", "film", "feature film", "motion picture"},
            "film": {"film", "movie", "feature film", "motion picture"},
            "tv series": {"series", "show", "television"},
            "radio station": {"radio", "station", "fm", "am"},
            "date": {"date", "year", "month"},
            "year": {"year"},
        }
        alias_tokens.update(lexical_alias_map.get(expected_type, set()))
        if not alias_tokens:
            return False

        candidate_texts = self._deduplicate_strings(
            [
                *entity_ids,
                *literals,
                *answer_result.predicted_answers,
                *answer_result.resolved_literals,
                *answer_result.resolved_entity_mentions,
            ]
        )
        if graph_api is not None and hasattr(graph_api, "resolve_entity_mentions"):
            resolved_ids = list(entity_ids)
            label_candidates = [value for value in candidate_texts if self._is_propagatable_text(value)]
            for label in label_candidates[:5]:
                resolved_ids.extend(graph_api.resolve_entity_mentions([label], top_k=1))
            resolved_ids = self._deduplicate_strings(resolved_ids)
            if resolved_ids and hasattr(graph_api, "get_nodes_metadata_batched"):
                metadata_by_id = graph_api.get_nodes_metadata_batched(resolved_ids)
                for entity_id in resolved_ids:
                    metadata = metadata_by_id.get(entity_id)
                    if metadata is None:
                        continue
                    type_text = self._metadata_type_text(metadata)
                    name_text = self._metadata_name_text(metadata)
                    if any(token in type_text for token in alias_tokens):
                        return True
                    if self._expected_type_allows_name_match(expected_type) and any(token in name_text for token in alias_tokens):
                        return True

        for value in candidate_texts:
            haystack = self._normalize_relation_text(value)
            if any(token in haystack for token in alias_tokens):
                return True
        return False

    def _expected_type_aliases(self, expected_type: str) -> set[str]:
        normalized = self._normalize_relation_text(expected_type)
        alias_map = {
            "country": (
                "location country",
                "country",
                "sovereign state",
                "nation",
                "location administrative division",
            ),
            "nation": (
                "location country",
                "country",
                "sovereign state",
                "nation",
                "location administrative division",
            ),
            "person": ("people person", "person", "deceased person", "celebrities celebrity"),
            "city": (
                "location citytown",
                "citytown",
                "city",
                "town",
                "location administrative division",
            ),
            "state": (
                "state",
                "us state",
                "location us state",
                "administrative division",
                "location administrative division",
                "first level administrative division",
            ),
            "government": ("government", "governmental jurisdiction"),
            "organization": ("organization", "institution", "company", "non profit organization"),
            "sports team": ("sports sports team", "sports team", "team", "franchise", "club"),
            "educational institution": (
                "education educational institution",
                "educational institution",
                "institution",
                "college",
                "university",
                "school",
            ),
            "college": (
                "education educational institution",
                "educational institution",
                "college",
                "university",
                "school",
                "institute",
            ),
            "university": (
                "education educational institution",
                "educational institution",
                "university",
                "college",
                "school",
                "institute",
            ),
            "school": ("education educational institution", "educational institution", "school", "academy"),
            "religion": ("religion", "religious denomination", "faith"),
            "language": ("language", "human language", "dialect"),
            "movie": ("movie", "film", "feature film", "motion picture"),
            "film": ("film", "movie", "feature film", "motion picture"),
            "book": ("book", "written work"),
            "tv series": ("tv tv program", "tv series", "television series", "series", "show", "tv program"),
            "radio station": ("broadcast radio station", "radio station", "radio", "station", "broadcast"),
            "date": ("datetime", "date", "time", "year", "month", "day"),
            "year": ("datetime", "date", "time", "year"),
            "number": ("number", "integer", "float", "dated percentage", "measurement unit dated integer"),
            "boolean": ("boolean",),
        }
        return {normalized, *alias_map.get(normalized, ())} if normalized else set()

    def _expected_type_allows_name_match(self, expected_type: str) -> bool:
        """Allow lexical label matching only for scalar-like answer types."""
        normalized = self._normalize_relation_text(expected_type)
        return normalized in {"date", "year", "number", "boolean"}

    def _metadata_type_text(self, metadata: Any) -> str:
        """Build a normalized type-only text view for one entity."""
        if metadata is None:
            return ""
        return self._normalize_relation_text(" ".join(str(value) for value in getattr(metadata, "types", ()) if value))

    def _metadata_name_text(self, metadata: Any) -> str:
        """Build a normalized label-only text view for one entity."""
        if metadata is None:
            return ""
        return self._normalize_relation_text(str(getattr(metadata, "name", "") or ""))

    @classmethod
    def _type_alias_matches_text(cls, aliases: set[str], text: str) -> bool:
        """Return whether any normalized expected-type alias matches one normalized text field."""
        normalized_text = cls._normalize_relation_text(text)
        if not aliases or not normalized_text:
            return False
        for alias in aliases:
            normalized_alias = cls._normalize_relation_text(alias)
            if not normalized_alias:
                continue
            if normalized_alias == normalized_text:
                return True
            if normalized_alias in normalized_text:
                return True
        return False

    def _type_filtering_enabled(self) -> bool:
        cfg = self.config.get("answer_filtering", {}).get("type_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", True))
        return bool(cfg)

    def _filter_subquestion_candidates_enabled(self) -> bool:
        cfg = self.config.get("answer_filtering", {}).get("type_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("filter_subquestion_candidates", True))
        return self._type_filtering_enabled()

    def _filter_final_candidates_enabled(self) -> bool:
        cfg = self.config.get("answer_filtering", {}).get("type_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("filter_final_candidates", True))
        return self._type_filtering_enabled()

    def _allow_soft_type_pass_through(self) -> bool:
        cfg = self.config.get("answer_filtering", {}).get("type_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("allow_soft_type_pass_through", True))
        return True

    def _entity_matches_expected_type(
        self,
        metadata: Any,
        expected_type: str,
    ) -> bool:
        if metadata is None:
            return False
        expected_aliases = self._expected_type_aliases(expected_type)
        if not expected_aliases:
            return True
        if self._type_alias_matches_text(expected_aliases, self._metadata_type_text(metadata)):
            return True
        if self._expected_type_allows_name_match(expected_type):
            return self._type_alias_matches_text(expected_aliases, self._metadata_name_text(metadata))
        return False

    def _filter_candidates_by_expected_type(
        self,
        graph_api: Any,
        expected_type: str,
        entity_ids: list[str],
        labels: list[str],
    ) -> tuple[list[str], list[str], dict[str, int]]:
        """Filter entity ids and labels so only expected-type candidates survive."""
        if not self._type_filtering_enabled():
            return self._deduplicate_strings(entity_ids), self._deduplicate_strings(labels), {
                "type_filter_before_count": len(entity_ids or labels),
                "type_filter_after_count": len(entity_ids or labels),
            }
        normalized_type = self._normalize_relation_text(expected_type)
        if not normalized_type or graph_api is None or not hasattr(graph_api, "get_nodes_metadata_batched"):
            dedup_labels = self._deduplicate_strings(labels)
            return self._deduplicate_strings(entity_ids), dedup_labels, {
                "type_filter_before_count": len(entity_ids) or len(dedup_labels),
                "type_filter_after_count": len(entity_ids) or len(dedup_labels),
            }

        candidate_ids = self._deduplicate_strings(list(entity_ids))
        if hasattr(graph_api, "resolve_entity_mentions"):
            unresolved_labels = [
                label for label in self._deduplicate_strings(labels)
                if self._is_propagatable_text(label)
            ]
            for label in unresolved_labels[:10]:
                candidate_ids.extend(graph_api.resolve_entity_mentions([label], top_k=3))
        candidate_ids = self._deduplicate_strings(candidate_ids)
        if not candidate_ids:
            filtered_labels = [
                label for label in self._deduplicate_strings(labels)
                if self._is_propagatable_text(label)
            ]
            return [], filtered_labels, {
                "type_filter_before_count": len(filtered_labels),
                "type_filter_after_count": len(filtered_labels),
            }

        metadata_by_id = graph_api.get_nodes_metadata_batched(candidate_ids)
        filtered_ids = [
            entity_id
            for entity_id in candidate_ids
            if self._entity_matches_expected_type(metadata_by_id.get(entity_id), normalized_type)
        ]
        filtered_labels = self._deduplicate_strings(
            [
                str((metadata_by_id.get(entity_id).name if metadata_by_id.get(entity_id) is not None else graph_api.get_entity_display_name(entity_id)) or "").strip()
                for entity_id in filtered_ids
                if str((metadata_by_id.get(entity_id).name if metadata_by_id.get(entity_id) is not None else graph_api.get_entity_display_name(entity_id)) or "").strip()
            ]
        )
        return filtered_ids, filtered_labels, {
            "type_filter_before_count": len(candidate_ids),
            "type_filter_after_count": len(filtered_ids),
        }

    def _backfill_subquestion_answer(
        self,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        step_summaries: list[dict[str, Any]],
        answer_result: AnswerResult,
    ) -> AnswerResult:
        """Derive a sub-question answer from summarized evidence when the LLM answer is empty."""
        if answer_result.predicted_answers and self._is_propagatable_text(answer_result.answer):
            return answer_result
        labels = self._extract_answer_like_labels(
            sub_question=sub_question,
            pruned_paths=pruned_paths,
            summaries=step_summaries,
        )
        labels = [label for label in labels if self._is_propagatable_text(label)]
        if not labels:
            return answer_result
        if not answer_result.predicted_answers:
            answer_result.predicted_answers = list(labels)
        if not self._is_propagatable_text(answer_result.answer):
            answer_result.answer = labels[0]
        answer_result.resolved_literals = self._deduplicate_strings(
            list(answer_result.resolved_literals) + list(labels)
        )
        answer_result.sufficient = True
        return answer_result

    def _is_propagatable_text(self, value: str) -> bool:
        """Reject empty or failure-placeholder text before dependency propagation."""
        normalized = str(value).strip()
        if not normalized:
            return False
        lowered = normalized.lower()
        blocked_markers = (
            "insufficient evidence",
            "insufficient",
            "unknown",
            "none",
            "null",
            "not enough evidence",
            "no evidence",
        )
        return not any(marker in lowered for marker in blocked_markers)

    def _agentic_step_to_sub_question(
        self,
        plan_step: AgenticPlanStep,
        topic_entities: list[str],
    ) -> SubQuestionSpec:
        """Convert one planner step into the sub-question structure reused by downstream tools."""
        topic_specs = list(plan_step.topic_entity_mentions)
        if not topic_specs:
            topic_specs = [
                EntityMentionSpec(name=name, role="topic_entity")
                for name in topic_entities
                if name
            ]
        return SubQuestionSpec(
            id=plan_step.step_id,
            question=plan_step.question,
            topic_entities=topic_specs,
            interested_nodes=[
                EntityMentionSpec(name=value, role="constraint_or_anchor")
                for value in plan_step.carryover_constraints
                if value
            ],
            interested_relations=list(plan_step.relation_hints),
            expected_answer_type=plan_step.expected_answer_type,
            expected_hop=1,
            depends_on=list(plan_step.depends_on_step_ids),
            solver_type="explore",
            solver_reason="planner step fallback",
        )

    def _resolve_agentic_seed_ids(
        self,
        run_state: AgenticRunState,
        resolved_seed_ids: list[str],
        fallback_seed_ids: list[str],
    ) -> list[str]:
        """Choose the current step seed ids, prioritizing the current step over older frontier state."""
        if resolved_seed_ids:
            return self._deduplicate_strings(resolved_seed_ids)
        if run_state.current_frontier_entities:
            return self._deduplicate_strings(run_state.current_frontier_entities)
        return self._deduplicate_strings(fallback_seed_ids)

    def _execute_agentic_retrieval_step(
        self,
        question: str,
        topic_entities: list[str],
        subgraph: KnowledgeGraph | None,
        graph_api: Any,
        plan_step: AgenticPlanStep,
        step_analysis: QuestionAnalysisResult,
        step_seed_ids: list[str],
        relation_ids: list[str],
        sub_question: SubQuestionSpec | None,
        resolved_sub_answers: dict[str, dict[str, Any]] | None,
        dmax: int,
        max_expand_steps: int,
        max_paths_per_pair: int,
        trace: list[SearchTraceStep],
        trace_prefix: str,
    ) -> list[ReasoningPath]:
        """Run one retrieval step, combining local and external graph access when available."""
        candidate_paths: list[ReasoningPath] = []
        focus_entities = [spec.name for spec in plan_step.topic_entity_mentions if spec.name] or list(topic_entities)

        if subgraph is not None and plan_step.strategy in {"auto", "local_subgraph", "supplement"}:
            local_paths = explore_topic_entity_paths(
                subgraph=subgraph,
                ordered_topic_entities=focus_entities,
                dpredict=min(step_analysis.predicted_depth, dmax),
                dmax=dmax,
                max_paths_per_pair=max_paths_per_pair,
            )
            if not local_paths and relation_ids:
                supplement_hints = SupplementHints(
                    entity_candidates=list(plan_step.carryover_constraints),
                    relation_candidates=[hint.name for hint in plan_step.relation_hints if hint.name],
                    linked_relations=list(relation_ids),
                )
                local_paths = explore_supplement_paths(
                    subgraph=subgraph,
                    topic_entities=focus_entities,
                    supplement_hints=supplement_hints,
                    dmax=dmax,
                    max_paths_per_pair=max_paths_per_pair,
                )
            if not local_paths and plan_step.strategy in {"auto", "node_expand"} and self.kg.triples:
                visited_nodes = {node for step in trace for node in []}
                local_paths = expand_nodes_from_paths(
                    current_paths=[],
                    kg=self.kg,
                    visited_nodes=visited_nodes,
                    max_expand_steps=max_expand_steps,
                )
            candidate_paths.extend(local_paths)

        if graph_api is not None and step_seed_ids:
            if self._retrieval_subquestion_explorer() == "relation_beam" and isinstance(graph_api, IndexedSQLiteGraphBackend):
                relation_beam_paths = self._expand_subquestion_with_relation_beam(
                    question=question,
                    topic_entities=focus_entities,
                    graph_api=graph_api,
                    plan_step=plan_step,
                    step_seed_ids=step_seed_ids,
                    sub_question=sub_question,
                    resolved_sub_answers=resolved_sub_answers or {},
                    trace=trace,
                    trace_prefix=trace_prefix,
                )
                candidate_paths.extend(relation_beam_paths)
                return self._sort_cwq_candidate_paths(deduplicate_paths(candidate_paths))
            external_paths = self._expand_external_paths_from_alignment(
                seed_node_ids=step_seed_ids,
                pruning_relation_ids=relation_ids,
                graph_api=graph_api,
                trace=trace,
                trace_prefix=trace_prefix,
            )
            candidate_paths.extend(external_paths)
        return self._sort_cwq_candidate_paths(deduplicate_paths(candidate_paths))

    def _expand_subquestion_with_relation_beam(
        self,
        question: str,
        topic_entities: list[str],
        graph_api: IndexedSQLiteGraphBackend,
        plan_step: AgenticPlanStep,
        step_seed_ids: list[str],
        sub_question: SubQuestionSpec | None,
        resolved_sub_answers: dict[str, dict[str, Any]],
        trace: list[SearchTraceStep],
        trace_prefix: str,
    ) -> list[ReasoningPath]:
        """Run the optional relation-beam explorer for one sub-question retrieval step."""
        anchor_mentions, relation_hint_names, dependency_context = self._relation_beam_subquestion_context(
            plan_step=plan_step,
            sub_question=sub_question,
            resolved_sub_answers=resolved_sub_answers,
        )
        beam_config = self._relation_beam_config()
        if sub_question is not None and int(sub_question.expected_hop or 0) > 0:
            beam_config = replace(
                beam_config,
                max_hops=max(1, min(beam_config.max_hops, int(sub_question.expected_hop))),
            )
        explorer = SubquestionRelationExplorer(
            graph=graph_api,
            llm=BaseLLMRelationBeamAdapter(
                self.llm,
                relation_batch_size=beam_config.llm_relation_batch_size,
            ),
            config=beam_config,
        )
        result = explorer.explore(
            original_question=question,
            subquestion_id=plan_step.step_id,
            subquestion_text=plan_step.question,
            input_entities=step_seed_ids,
            expected_answer_type=plan_step.expected_answer_type,
            anchor_mentions=anchor_mentions,
            relation_hint_names=relation_hint_names,
            resolved_dependencies=dependency_context,
        )
        trace.append(
            SearchTraceStep(
                stage=f"{trace_prefix}_relation_beam",
                depth=result.searched_hops,
                candidate_count=len(result.candidate_paths),
                pruned_count=len(result.answer_entities),
                note=(
                    f"sub_question_id={plan_step.step_id} seeds={step_seed_ids} "
                    f"expected_answer_type={plan_step.expected_answer_type or ''} "
                    f"expected_hop={(sub_question.expected_hop if sub_question is not None else beam_config.max_hops)} "
                    f"anchors={anchor_mentions[:5]} relation_hints={relation_hint_names[:5]} "
                    f"best_score={result.best_score:.4f} complete={result.complete}"
                ),
            )
        )
        LOGGER.info(
            "Sub-question relation beam sub_question_id=%s expected_hop=%d searched_hops=%d answer_entities=%s anchors=%s relation_hints=%s best_score=%.4f complete=%s",
            plan_step.step_id,
            beam_config.max_hops,
            result.searched_hops,
            result.answer_entities[:10],
            anchor_mentions[:5],
            relation_hint_names[:5],
            result.best_score,
            result.complete,
        )
        return result.candidate_paths

    def _relation_beam_subquestion_context(
        self,
        plan_step: AgenticPlanStep,
        sub_question: SubQuestionSpec | None,
        resolved_sub_answers: dict[str, dict[str, Any]],
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Build a compact, structured sub-question context for relation-beam planning."""
        anchor_mentions = self._deduplicate_strings(
            [
                *(entity.name for entity in plan_step.topic_entity_mentions if entity.name),
                *(value for value in plan_step.carryover_constraints if value),
                *(value for value in plan_step.downstream_filters if value),
                *(entity.name for entity in (sub_question.interested_nodes if sub_question else []) if entity.name),
            ]
        )
        relation_hint_names = self._deduplicate_strings(
            [
                *(hint.name for hint in plan_step.relation_hints if hint.name),
                *(alias for hint in plan_step.relation_hints for alias in hint.aliases if alias),
                *(hint.description for hint in plan_step.relation_hints if hint.description),
            ]
        )
        dependency_ids = list(sub_question.depends_on) if sub_question is not None else list(plan_step.depends_on_step_ids)
        resolved_dependencies = [
            {
                "sub_question_id": dependency_id,
                "answer": resolved_sub_answers.get(dependency_id, {}).get("answer", ""),
                "predicted_answers": resolved_sub_answers.get(dependency_id, {}).get("predicted_answers", []),
                "entity_ids": resolved_sub_answers.get(dependency_id, {}).get("entity_ids", []),
                "literals": resolved_sub_answers.get(dependency_id, {}).get("literals", []),
            }
            for dependency_id in dependency_ids
            if dependency_id in resolved_sub_answers
        ]
        predicted_entity_hints = self._predict_relation_beam_entity_hints(
            plan_step=plan_step,
            sub_question=sub_question,
            resolved_dependencies=resolved_dependencies,
        )
        anchor_mentions = self._deduplicate_strings(anchor_mentions + predicted_entity_hints)
        if predicted_entity_hints:
            LOGGER.info(
                "Relation beam entity hints sub_question_id=%s hints=%s",
                plan_step.step_id,
                predicted_entity_hints,
            )
        return anchor_mentions, relation_hint_names, resolved_dependencies

    def _predict_relation_beam_entity_hints(
        self,
        plan_step: AgenticPlanStep,
        sub_question: SubQuestionSpec | None,
        resolved_dependencies: list[dict[str, Any]],
    ) -> list[str]:
        """Predict weak entity-like hints for relation beam without turning them into hard answers."""
        if not resolved_dependencies:
            return []
        compact_dependencies: list[dict[str, Any]] = []
        for item in resolved_dependencies[:3]:
            compact_dependencies.append(
                {
                    "sub_question_id": str(item.get("sub_question_id") or "")[:24],
                    "answer": str(item.get("answer") or "")[:64],
                    "entity_ids": [str(value)[:24] for value in item.get("entity_ids", [])[:1]],
                    "predicted_answers": [str(value)[:48] for value in item.get("predicted_answers", [])[:2]],
                }
            )
        prompt = RELATION_BEAM_ENTITY_HINT_PROMPT.format(
            sub_question=(sub_question.question if sub_question is not None else plan_step.question),
            expected_answer_type=(sub_question.expected_answer_type if sub_question is not None else plan_step.expected_answer_type),
            resolved_dependencies=json.dumps(compact_dependencies, ensure_ascii=False, separators=(",", ":")),
        )
        LOGGER.info(
            "Relation beam entity hint prompt sub_question_id=%s dependency_count=%d prompt_chars=%d",
            plan_step.step_id,
            len(compact_dependencies),
            len(prompt),
        )
        try:
            raw = self.llm.generate(prompt, max_tokens=128)
        except Exception:
            return []
        parsed = robust_json_parse(raw, fallback={})
        if not isinstance(parsed, dict):
            return []
        raw_hints = [
            str(value).strip()
            for value in parsed.get("entity_hints", [])
            if str(value).strip()
        ]
        question_text = self._normalize_relation_text(sub_question.question if sub_question is not None else plan_step.question)
        answer_type_text = self._normalize_relation_text(sub_question.expected_answer_type if sub_question is not None else plan_step.expected_answer_type)
        filtered: list[str] = []
        seen: set[str] = set()
        for hint in raw_hints:
            normalized = self._normalize_relation_text(hint)
            if not normalized or normalized in seen:
                continue
            if normalized == answer_type_text:
                continue
            if len(normalized.split()) > 5:
                continue
            if normalized in question_text:
                filtered.append(hint[:48])
                seen.add(normalized)
                continue
            filtered.append(hint[:48])
            seen.add(normalized)
            if len(filtered) >= 3:
                break
        return filtered

    def _derive_agentic_carryover(
        self,
        graph_api: Any,
        step_summaries: list[dict[str, Any]],
        pruned_paths: list[ReasoningPath],
        candidate_answers: list[str],
    ) -> tuple[list[str], list[str]]:
        """Project current-step outputs into next-step entity frontier and text constraints."""
        texts: list[str] = []
        for item in step_summaries:
            for triple in item.get("key_triples", []):
                if not isinstance(triple, dict):
                    continue
                tail = str(triple.get("tail") or "").strip()
                if tail:
                    texts.append(tail)
        if not texts:
            for path in pruned_paths:
                if path.triples:
                    tail = str(path.triples[-1].tail).strip()
                    if tail:
                        texts.append(tail)
        texts.extend(str(value).strip() for value in candidate_answers if str(value).strip())
        texts = self._deduplicate_strings(texts)
        if graph_api is None or not texts:
            return [], texts[: self._agentic_max_carryover_text_values()]

        carryover_entities = graph_api.resolve_entity_mentions(texts, top_k=1)
        resolved_names = {graph_api.get_entity_display_name(entity_id) for entity_id in carryover_entities}
        carryover_text_values = [text for text in texts if text not in resolved_names]
        return (
            self._deduplicate_strings(carryover_entities)[: self._agentic_max_carryover_entities()],
            self._deduplicate_strings(carryover_text_values)[: self._agentic_max_carryover_text_values()],
        )

    def _run_sqlite_external_only(
        self,
        question: str,
        sample_id: str | None,
        gold_answers: list[str] | None,
        dmax: int,
        w1: int,
        wmax: int,
        max_expand_steps: int,
        entity_source: str,
        topic_entities_override: list[str] | None,
        trace_prefix: str,
    ) -> PipelineResult:
        """Run one dataset flow by aligning question/sub-question hints against SQLite only."""
        graph_api = self._require_sqlite_graph_api()

        analysis = analyze_question(
            question,
            self.llm,
            dmax=dmax,
            topic_entities_override=topic_entities_override if entity_source == "provided" else None,
        )
        analysis = self._attach_downstream_filters_to_analysis(analysis)
        topic_entities = (
            list(topic_entities_override)
            if entity_source == "provided" and topic_entities_override
            else list(analysis.topic_entities or analysis.ordered_topic_entities)
        )
        dpredict = min(analysis.predicted_depth, dmax)
        trace: list[SearchTraceStep] = []
        alignment_debug: list[dict[str, Any]] = []
        topic_entity_candidates = self._resolve_entity_specs_with_sqlite(
            specs=[EntityMentionSpec(name=name, role="topic_entity") for name in topic_entities],
            graph_api=graph_api,
            context_text=question,
        )
        topic_seed_ids = self._deduplicate_strings([candidate.id_or_mid for candidate in topic_entity_candidates])
        alignment_debug.append(
            {
                "stage": "question_topic_entity_alignment",
                "topic_entities": list(topic_entities),
                "resolved_topic_entity_candidates": [candidate.to_dict() for candidate in topic_entity_candidates],
                "seed_node_ids": topic_seed_ids,
            }
        )
        trace.append(
            SearchTraceStep(
                stage=f"{trace_prefix}_topic_entity_resolution",
                depth=min(dmax, 3),
                candidate_count=len(topic_entity_candidates),
                pruned_count=len(topic_seed_ids),
                note=f"topic_entities={topic_entities} seed_node_ids={topic_seed_ids}",
            )
        )

        resolved_sub_questions: list[ResolvedSubQuestion] = []
        for sub_question in analysis.sub_questions:
            resolved_sub_question = self._resolve_cwq_sub_question_with_sqlite(
                sub_question=sub_question,
                graph_api=graph_api,
            )
            resolved_sub_questions.append(resolved_sub_question)
            alignment_debug.append(
                {
                    "stage": "sub_question_alignment",
                    "sub_question_id": sub_question.id,
                    "question": sub_question.question,
                    "interested_nodes": [item.to_dict() for item in sub_question.interested_nodes],
                    "interested_relations": [item.to_dict() for item in sub_question.interested_relations],
                    "resolved": resolved_sub_question.to_dict(),
                }
            )
            trace.append(
                SearchTraceStep(
                    stage=f"{trace_prefix}_sub_question_alignment",
                    depth=min(dmax, 3),
                    candidate_count=(
                        len(resolved_sub_question.topic_entity_candidates)
                        + len(resolved_sub_question.interested_node_candidates)
                        + len(resolved_sub_question.relation_candidates)
                    ),
                    pruned_count=0,
                    note=(
                        f"{sub_question.id}: seeds={resolved_sub_question.seed_node_ids} "
                        f"selected_relations={resolved_sub_question.selected_relation_ids or resolved_sub_question.relation_ids} "
                        f"depends_on={resolved_sub_question.depends_on} debug={resolved_sub_question.resolution_debug}"
                    ),
                )
            )

        combined_seed_ids, combined_pruning_relation_ids = self._collect_cwq_alignment_inputs(
            topic_seed_ids=topic_seed_ids,
            resolved_sub_questions=resolved_sub_questions,
        )
        trace.append(
            SearchTraceStep(
                stage=f"{trace_prefix}_alignment_summary",
                depth=min(dmax, 2),
                candidate_count=len(combined_seed_ids) + len(combined_pruning_relation_ids),
                pruned_count=len(combined_seed_ids),
                note=(
                    f"topic_seed_ids={topic_seed_ids} combined_seed_ids={combined_seed_ids} "
                    f"combined_pruning_relation_ids={combined_pruning_relation_ids} "
                    "explore_relation_ids=[]"
                ),
            )
        )

        candidate_paths = self._expand_external_paths_from_alignment(
            seed_node_ids=combined_seed_ids,
            pruning_relation_ids=combined_pruning_relation_ids,
            graph_api=graph_api,
            trace=trace,
            trace_prefix=trace_prefix,
        )
        pruned_paths, summarized_paths, answer_result, sufficient = self._evaluate_paths(
            question=question,
            topic_entities=topic_entities,
            question_analysis=analysis,
            candidate_paths=candidate_paths,
            w1=w1,
            wmax=wmax,
            relation_hints=combined_pruning_relation_ids,
            sample_id=sample_id,
            trace=trace,
            dmax=dmax,
            depth=min(dmax, max(dpredict, 1)),
        )
        confidence = "high" if sufficient else "low"
        answer = answer_result.answer if answer_result.answer else "insufficient"
        return self._build_pipeline_result(
            question=question,
            topic_entities=topic_entities,
            question_analysis=analysis,
            dmax=dmax,
            dpredict=dpredict,
            candidate_paths=candidate_paths,
            pruned_paths=pruned_paths,
            summarized_paths=summarized_paths,
            answer=answer,
            sufficient=sufficient,
            confidence=confidence,
            sample_id=sample_id,
            gold_answers=gold_answers,
            predicted_answers=answer_result.predicted_answers,
            search_trace=trace,
            alignment_debug=alignment_debug,
        )

    def _require_sqlite_graph_api(self) -> Any:
        """Return the graph API object required for SQLite external-only dataset flows."""
        if self.graph_api is None:
            raise ValueError("SQLite external-only dataset flow requires graphapi to be enabled.")
        required_methods = (
            "resolve_entity_candidates",
            "resolve_relation_candidates",
            "resolve_entity_mentions",
            "resolve_relation_hints",
            "find_two_hop_extensions",
            "expand_paths",
        )
        missing = [name for name in required_methods if not hasattr(self.graph_api, name)]
        if missing:
            raise TypeError(
                "SQLite external-only dataset flow requires a SQLiteGraphAPI-style backend; "
                f"missing methods: {', '.join(missing)}"
            )
        return self.graph_api

    def _collect_cwq_alignment_inputs(
        self,
        topic_seed_ids: list[str],
        resolved_sub_questions: list[ResolvedSubQuestion],
    ) -> tuple[list[str], list[str]]:
        """Merge topic-entity seeds with aligned nodes and pruning-only relation ids."""
        combined_seed_ids = list(topic_seed_ids)
        combined_pruning_relation_ids: list[str] = []
        for resolved in resolved_sub_questions:
            combined_seed_ids.extend(resolved.seed_node_ids)
            combined_pruning_relation_ids.extend(resolved.selected_relation_ids or resolved.relation_ids)
        return (
            self._deduplicate_strings(combined_seed_ids),
            self._deduplicate_strings(combined_pruning_relation_ids),
        )

    def _expand_external_paths_from_alignment(
        self,
        seed_node_ids: list[str],
        pruning_relation_ids: list[str],
        graph_api: Any,
        trace: list[SearchTraceStep],
        trace_prefix: str,
    ) -> list[ReasoningPath]:
        """Expand candidate paths from unified dataset seeds using the retrieval toolkit."""
        explore_relation_ids: list[str] = []
        primary_paths: list[ReasoningPath] = []
        max_depth = self._graphapi_max_depth()
        if seed_node_ids and self.hybrid_searcher is not None and isinstance(graph_api, IndexedSQLiteGraphBackend):
            primary_paths = self.hybrid_searcher.search(
                SearchRequest(
                    seed_entity_ids=seed_node_ids,
                    target_entity_ids=[],
                    relation_hints=explore_relation_ids,
                    answer_type_hints=[],
                    max_depth=max_depth,
                    beam_width=self._retrieval_beam_width(),
                    max_expansions=self._retrieval_max_expansions(),
                    max_paths=self._retrieval_max_paths(),
                    literal_policy=self._retrieval_literal_policy(),
                    high_degree_policy=self._retrieval_high_degree_policy(),
                    strict_relation_filter=False,
                )
            )
            for path in primary_paths:
                path.source_stage = "external_graphapi_expansion"
            annotate_paths_with_relation_matches(primary_paths, pruning_relation_ids)
            primary_paths = self._sort_cwq_candidate_paths(primary_paths)
        elif seed_node_ids:
            primary_paths = graph_api.expand_paths(
                seed_nodes=seed_node_ids,
                relation_hints=explore_relation_ids,
                max_depth=max_depth,
                strict_relation_filter=False,
                max_paths=self._cwq_max_paths(),
                max_nodes=self._cwq_max_nodes(),
            )
            annotate_paths_with_relation_matches(primary_paths, pruning_relation_ids)
            primary_paths = self._sort_cwq_candidate_paths(primary_paths)
        trace.append(
            SearchTraceStep(
                stage=f"{trace_prefix}_graph_construction",
                depth=max_depth,
                candidate_count=len(primary_paths),
                pruned_count=0,
                note=(
                    f"seed_node_ids={seed_node_ids} explore_relation_ids={explore_relation_ids} "
                    f"combined_pruning_relation_ids={pruning_relation_ids} max_depth={max_depth} "
                    f"strict_relation_filter=False primary_paths={len(primary_paths)} "
                    f"search_strategy={self._retrieval_default_strategy()}"
                ),
            )
        )

        fallback_paths: list[ReasoningPath] = []
        if not self._graphapi_one_hop_only():
            fallback_paths = explore_graphapi_bootstrap_paths(
                aligned_entity_ids=seed_node_ids,
                aligned_relation_ids=explore_relation_ids,
                graph_api=graph_api,
                neighbor_limit=self._graphapi_neighbor_limit(),
            )
            if not fallback_paths and seed_node_ids:
                fallback_paths = graph_api.find_two_hop_extensions(
                    frontier_nodes=seed_node_ids,
                    relation_hints=explore_relation_ids,
                    limit=self._graphapi_neighbor_limit(),
                )
        annotate_paths_with_relation_matches(fallback_paths, pruning_relation_ids)
        fallback_paths = self._sort_cwq_candidate_paths(fallback_paths)
        merged_paths = deduplicate_paths(primary_paths + fallback_paths) if primary_paths else fallback_paths
        merged_paths = self._sort_cwq_candidate_paths(merged_paths)
        trace.append(
            SearchTraceStep(
                stage=f"{trace_prefix}_graph_fallback",
                depth=max_depth,
                candidate_count=len(fallback_paths),
                pruned_count=0,
                note=(
                    f"seed_node_ids={seed_node_ids} explore_relation_ids={explore_relation_ids} "
                    f"combined_pruning_relation_ids={pruning_relation_ids} "
                    f"one_hop_only={self._graphapi_one_hop_only()} "
                    f"primary_empty={not bool(primary_paths)} primary_paths={len(primary_paths)} "
                    f"fallback_paths={len(fallback_paths)} merged_paths={len(merged_paths)}"
                ),
            )
        )
        return merged_paths

    def _webqsp_uses_external_graph_only(self) -> bool:
        """Return whether WebQSP should bypass sample-local graphs and use SQLite only."""
        if self.graph_api is None:
            return False
        dataset_cfg = self.config.get("datasets", {}).get("webqsp", {})
        return bool(dataset_cfg.get("external_graph_only", True))

    def _analysis_for_sub_question(
        self,
        question_analysis: QuestionAnalysisResult,
        sub_question: SubQuestionSpec,
    ) -> QuestionAnalysisResult:
        """Build a focused analysis view for one CWQ sub-question."""
        ordered_topic_entities = [entity.name for entity in self._effective_sub_question_topics(sub_question) if entity.name]
        if not ordered_topic_entities:
            ordered_topic_entities = list(question_analysis.ordered_topic_entities)
        return QuestionAnalysisResult(
            split_questions=[sub_question.question],
            reasoning_indicator=sub_question.question or question_analysis.reasoning_indicator,
            ordered_topic_entities=ordered_topic_entities,
            predicted_depth=max(1, min(question_analysis.predicted_depth, 3)),
            topic_entities=ordered_topic_entities,
            sub_questions=[sub_question],
        )

    def _resolve_cwq_sub_question_with_sqlite(
        self,
        sub_question: SubQuestionSpec,
        graph_api: Any,
    ) -> ResolvedSubQuestion:
        """Resolve one CWQ sub-question into retrieval seeds only."""
        topic_entity_candidates = self._resolve_entity_specs_with_sqlite(
            specs=self._effective_sub_question_topics(sub_question),
            graph_api=graph_api,
            context_text=sub_question.question,
        )
        if topic_entity_candidates:
            seed_node_ids = self._deduplicate_strings([candidate.id_or_mid for candidate in topic_entity_candidates])
        else:
            seed_node_ids = []
        filtered_seed_node_ids, anchor_filter_debug = self._filter_subquestion_start_nodes(
            sub_question=sub_question,
            graph_api=graph_api,
            seed_node_ids=seed_node_ids,
        )
        LOGGER.info(
            "Sub-question start-node filtering sub_question_id=%s original_seed_node_ids=%s filtered_seed_node_ids=%s anchor_filter=%s",
            sub_question.id,
            seed_node_ids,
            filtered_seed_node_ids,
            anchor_filter_debug,
        )
        if filtered_seed_node_ids:
            seed_node_ids = filtered_seed_node_ids
        resolution_debug = {
            "interested_nodes_mode": "local_path_filter_only",
            "relation_hints_mode": "local_path_filter_only",
            "selected_relation_ids": [],
            "anchor_filter": anchor_filter_debug,
        }
        return ResolvedSubQuestion(
            id=sub_question.id,
            question=sub_question.question,
            topic_entity_candidates=topic_entity_candidates,
            interested_node_candidates=[],
            relation_candidates=[],
            seed_node_ids=seed_node_ids,
            relation_ids=[],
            selected_relation_ids=[],
            expected_answer_type=sub_question.expected_answer_type,
            depends_on=list(sub_question.depends_on),
            resolution_debug=resolution_debug,
        )

    def _resolve_entity_specs_with_sqlite(
        self,
        specs: list[Any],
        graph_api: Any,
        context_text: str = "",
    ) -> list[ResolvedCandidate]:
        """Resolve entity-like specs and keep the best top-k matches for each spec."""
        candidates: list[ResolvedCandidate] = []
        for spec in specs:
            names = self._build_entity_query_names(spec)
            recall_top_k = (
                self._sqlite_entity_candidate_typed_recall_top_k()
                if str(getattr(spec, "expected_type", "") or "").strip()
                else self._sqlite_entity_candidate_recall_top_k()
            )
            LOGGER.info(
                "Entity query expansion mention=%s expected_type=%s role=%s recall_top_k=%s query_names=%s",
                getattr(spec, "name", ""),
                getattr(spec, "expected_type", ""),
                getattr(spec, "role", ""),
                recall_top_k,
                names,
            )
            spec_rows: dict[str, ResolvedCandidate] = {}
            for name in names:
                for row in graph_api.resolve_entity_candidates([name], top_k=recall_top_k):
                    candidate_id = str(row.get("mid") or "").strip()
                    if not candidate_id:
                        continue
                    candidate = ResolvedCandidate(
                        id_or_mid=candidate_id,
                        label=str(row.get("name") or graph_api.get_entity_display_name(candidate_id)),
                        mention_or_hint=str(row.get("mention") or name),
                        match_type=str(row.get("match_type") or ""),
                        score=float(row.get("score") or 0.0),
                    )
                    current = spec_rows.get(candidate.id_or_mid)
                    if current is None or candidate.score > current.score:
                        spec_rows[candidate.id_or_mid] = candidate
            if spec_rows:
                LOGGER.info(
                    "Entity candidate recall mention=%s expected_type=%s role=%s raw_candidates=%s",
                    getattr(spec, "name", ""),
                    getattr(spec, "expected_type", ""),
                    getattr(spec, "role", ""),
                    [
                        {
                            "entity_id": item.id_or_mid,
                            "label": item.label,
                            "match_type": item.match_type,
                            "score": round(float(item.score), 3),
                        }
                        for item in sorted(spec_rows.values(), key=lambda value: (-value.score, value.id_or_mid))[:10]
                    ],
                )
            ranked = self._rerank_entity_candidates_with_sqlite_context(
                spec=spec,
                candidates=list(spec_rows.values()),
                graph_api=graph_api,
                context_text=context_text,
            )[: self._sqlite_entity_candidate_selected_top_k()]
            candidates.extend(ranked)
        return candidates

    def _build_entity_query_names(self, spec: Any) -> list[str]:
        """Expand entity queries with lightweight type-aware variants before reranking."""
        base_name = str(getattr(spec, "name", "") or "").strip()
        aliases = [str(value).strip() for value in getattr(spec, "aliases", []) if str(value).strip()]
        expected_type = self._normalize_relation_text(str(getattr(spec, "expected_type", "") or "").strip())
        names = self._deduplicate_strings([base_name, *aliases])
        if not base_name:
            return names

        type_query_suffixes = {
            "state": ["state", "us state"],
            "country": ["country", "nation"],
            "nation": ["country", "nation"],
            "city": ["city", "town"],
            "person": ["person"],
            "organization": ["organization"],
            "sports team": ["team", "sports team"],
            "educational institution": ["university", "college", "school"],
            "college": ["college", "university", "school"],
            "university": ["university", "college"],
            "school": ["school"],
            "religion": ["religion"],
            "language": ["language"],
            "film": ["film", "movie"],
            "book": ["book"],
            "tv series": ["tv series", "television series", "show"],
            "radio station": ["radio station"],
        }
        suffixes = type_query_suffixes.get(expected_type, [])
        expanded: list[str] = list(names)
        for suffix in suffixes:
            expanded.append(f"{base_name} {suffix}")
            expanded.append(f"{suffix} {base_name}")
            if expected_type in {"state", "country", "nation", "city"}:
                expanded.append(f"{base_name} {suffix} of")
                expanded.append(f"{suffix} of {base_name}")
        return self._deduplicate_strings(expanded)

    def _subquestion_anchor_filter_enabled(self) -> bool:
        cfg = self.config.get("retrieval", {}).get("subquestion_anchor_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", False))
        return bool(cfg)

    def _subquestion_anchor_filter_max_start_nodes(self) -> int:
        cfg = self.config.get("retrieval", {}).get("subquestion_anchor_filtering", {})
        if isinstance(cfg, dict):
            return max(1, int(cfg.get("max_start_nodes", 2)))
        return 2

    def _subquestion_anchor_filter_dependency_seeds(self) -> bool:
        cfg = self.config.get("retrieval", {}).get("subquestion_anchor_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("filter_dependency_seeds", False))
        return False

    def _subquestion_anchor_filter_use_llm(self) -> bool:
        cfg = self.config.get("retrieval", {}).get("subquestion_anchor_filtering", {})
        if isinstance(cfg, dict):
            return bool(cfg.get("use_llm", True))
        return True

    def _filter_subquestion_start_nodes(
        self,
        *,
        sub_question: SubQuestionSpec,
        graph_api: Any,
        seed_node_ids: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        """Filter aligned start nodes using expected type, local relations, and optional LLM judgment."""
        seed_node_ids = self._deduplicate_strings(seed_node_ids)
        if not self._subquestion_anchor_filter_enabled():
            return seed_node_ids, {"enabled": False}
        if not seed_node_ids or graph_api is None:
            return seed_node_ids, {"enabled": True, "skipped": "missing_graph_or_seeds"}
        if sub_question.depends_on and not self._subquestion_anchor_filter_dependency_seeds():
            return seed_node_ids, {"enabled": True, "skipped": "dependency_seed_filter_disabled"}
        if not hasattr(graph_api, "get_nodes_metadata_batched") or not hasattr(graph_api, "get_node_relations"):
            return seed_node_ids, {"enabled": True, "skipped": "missing_metadata_or_relations"}

        metadata_by_id = graph_api.get_nodes_metadata_batched(seed_node_ids)
        relation_terms = self._deduplicate_strings(
            [
                sub_question.question,
                *[spec.name for spec in self._effective_sub_question_topics(sub_question) if spec.name],
                *[spec.name for spec in sub_question.interested_nodes if spec.name],
                *[rel.name for rel in sub_question.interested_relations if rel.name],
                *[alias for rel in sub_question.interested_relations for alias in rel.aliases if alias],
            ]
        )
        expected_topic_types = self._deduplicate_strings(
            [
                str(spec.expected_type).strip()
                for spec in self._effective_sub_question_topics(sub_question)
                if str(spec.expected_type or "").strip()
            ]
        )

        scored_rows: list[dict[str, Any]] = []
        type_match_exists = False
        for entity_id in seed_node_ids:
            metadata = metadata_by_id.get(entity_id)
            relation_rows = graph_api.get_node_relations(
                entity_id,
                include_reverse=True,
                include_literals=False,
            )[:8]
            relation_text = self._normalize_relation_text(
                " ".join(
                    f"{item.relation} {item.relation_name} {' '.join(item.sample_target_types)} {' '.join(item.sample_target_names)}"
                    for item in relation_rows
                )
            )
            metadata_text = self._normalize_relation_text(
                " ".join(
                    [
                        str(metadata.name if metadata is not None else graph_api.get_entity_display_name(entity_id) or ""),
                        *list(metadata.types if metadata is not None else ()),
                    ]
                )
            )
            type_match = any(
                metadata is not None and self._entity_matches_expected_type(metadata, expected_type)
                for expected_type in expected_topic_types
            )
            if type_match:
                type_match_exists = True
            score = 0.0
            if type_match:
                score += 18.0
            score += 14.0 * max(
                [self._token_overlap_ratio(term, metadata_text) for term in relation_terms] or [0.0]
            )
            score += 12.0 * max(
                [self._token_overlap_ratio(term, relation_text) for term in relation_terms] or [0.0]
            )
            if metadata is not None and metadata.is_probable_cvt:
                score -= 4.0
            scored_rows.append(
                {
                    "entity_id": entity_id,
                    "label": str(metadata.name if metadata is not None else graph_api.get_entity_display_name(entity_id) or entity_id),
                    "types": list(metadata.types if metadata is not None else ()),
                    "top_relations": [item.relation for item in relation_rows[:5]],
                    "sample_targets": [name for item in relation_rows[:3] for name in item.sample_target_names[:1]][:3],
                    "type_match": type_match,
                    "heuristic_score": round(score, 3),
                }
            )

        if type_match_exists:
            scored_rows = [row for row in scored_rows if row["type_match"]]
        scored_rows.sort(key=lambda row: (-float(row["heuristic_score"]), row["entity_id"]))
        selected_ids = [row["entity_id"] for row in scored_rows[: self._subquestion_anchor_filter_max_start_nodes()]]

        llm_selected_ids: list[str] = []
        if self._subquestion_anchor_filter_use_llm() and len(scored_rows) > 1:
            compact_candidates = [
                {
                    "entity_id": row["entity_id"],
                    "label": row["label"][:64],
                    "types": row["types"][:5],
                    "top_relations": row["top_relations"][:5],
                    "sample_targets": [str(value)[:36] for value in row["sample_targets"][:3]],
                    "heuristic_score": row["heuristic_score"],
                }
                for row in scored_rows[: min(5, len(scored_rows))]
            ]
            prompt = ENTITY_CANDIDATE_DISAMBIGUATION_PROMPT.format(
                question=sub_question.question,
                mention=", ".join(spec.name for spec in self._effective_sub_question_topics(sub_question) if spec.name) or sub_question.question,
                expected_type=", ".join(expected_topic_types),
                role="subquestion_start_anchor",
                candidate_entities=json.dumps(compact_candidates, ensure_ascii=False, separators=(",", ":")),
            )
            try:
                raw = self.llm.generate(prompt, max_tokens=192)
                parsed = robust_json_parse(raw, fallback={})
                if isinstance(parsed, dict):
                    candidate_id_set = {item["entity_id"] for item in compact_candidates}
                    llm_selected_ids = self._deduplicate_strings(
                        [
                            str(value).strip()
                            for value in parsed.get("selected_entity_ids", [])
                            if str(value).strip() in candidate_id_set
                        ]
                    )[: self._subquestion_anchor_filter_max_start_nodes()]
            except Exception:
                llm_selected_ids = []
        if llm_selected_ids:
            selected_ids = llm_selected_ids

        return selected_ids, {
            "enabled": True,
            "selected_seed_node_ids": list(selected_ids),
            "topic_expected_types": expected_topic_types,
            "relation_terms": relation_terms[:8],
            "used_llm": bool(llm_selected_ids),
            "scored_candidates": scored_rows[:5],
        }

    def _resolve_relation_specs_with_sqlite(
        self,
        specs: list[Any],
        sub_question: SubQuestionSpec,
        graph_api: Any,
    ) -> list[ResolvedCandidate]:
        """Resolve relation specs and keep the best top-k matches for each spec."""
        candidates: list[ResolvedCandidate] = []
        for spec in specs:
            names = self._build_relation_query_names(spec=spec, sub_question=sub_question)
            spec_rows: dict[str, ResolvedCandidate] = {}
            for name in names:
                for row in graph_api.resolve_relation_candidates([name], top_k=3):
                    relation_id = str(row.get("relation") or "").strip()
                    if not relation_id:
                        continue
                    candidate = ResolvedCandidate(
                        id_or_mid=relation_id,
                        label=str(
                            row.get("relation_name")
                            or graph_api.get_relation_display_name(relation_id)
                        ),
                        mention_or_hint=str(row.get("hint") or name),
                        match_type=str(row.get("match_type") or ""),
                        score=float(row.get("score") or 0.0),
                    )
                    current = spec_rows.get(candidate.id_or_mid)
                    if current is None or candidate.score > current.score:
                        spec_rows[candidate.id_or_mid] = candidate
            ranked = sorted(
                spec_rows.values(),
                key=lambda item: (-item.score, item.id_or_mid),
            )[:3]
            candidates.extend(ranked)
        return candidates

    def _rerank_entity_candidates_with_sqlite_context(
        self,
        spec: Any,
        candidates: list[ResolvedCandidate],
        graph_api: Any,
        context_text: str,
    ) -> list[ResolvedCandidate]:
        """Rerank SQLite entity candidates using metadata, local relations, and optional LLM supervision."""
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates
        if not hasattr(graph_api, "get_nodes_metadata_batched") or not hasattr(graph_api, "get_node_relations"):
            return sorted(candidates, key=lambda item: (-item.score, item.id_or_mid))

        candidate_ids = [candidate.id_or_mid for candidate in candidates]
        metadata_by_id = graph_api.get_nodes_metadata_batched(candidate_ids)
        candidate_relation_rows: dict[str, list[Any]] = {}
        heuristic_ranked: list[tuple[float, ResolvedCandidate]] = []
        mention = str(getattr(spec, "name", "") or "").strip()
        expected_type = str(getattr(spec, "expected_type", "") or "").strip()
        role = str(getattr(spec, "role", "") or "").strip()
        normalized_context = self._normalize_relation_text(" ".join([context_text, mention, expected_type, role]))
        expected_type_aliases = self._expected_type_aliases(expected_type)
        preserve_multiple_candidates = role in {"topic_entity", "constraint_or_anchor"}
        cross_domain_prefixes = (
            "music.",
            "film.",
            "tv.",
            "book.",
            "comic_books.",
            "media_common.",
            "award.",
        )

        for candidate in candidates:
            metadata = metadata_by_id.get(candidate.id_or_mid)
            relation_rows = graph_api.get_node_relations(
                candidate.id_or_mid,
                include_reverse=True,
                include_literals=False,
            )[:6]
            candidate_relation_rows[candidate.id_or_mid] = relation_rows
            metadata_text = self._normalize_relation_text(
                " ".join(
                    [
                        candidate.label,
                        *(metadata.types if metadata is not None else ()),
                    ]
                )
            )
            relation_text = self._normalize_relation_text(
                " ".join(
                    f"{item.relation} {item.relation_name} {' '.join(item.sample_target_types)} {' '.join(item.sample_target_names)}"
                    for item in relation_rows
                )
            )
            raw_relation_ids = [str(item.relation or "") for item in relation_rows]
            raw_metadata_types = [str(value) for value in (metadata.types if metadata is not None else ())]
            score = float(candidate.score)
            if expected_type:
                score += 25.0 * self._token_overlap_ratio(expected_type, metadata_text)
                score += 10.0 * self._token_overlap_ratio(expected_type, relation_text)
                if metadata is not None and self._entity_matches_expected_type(metadata, expected_type):
                    score += 30.0
                elif expected_type_aliases and metadata is not None:
                    score -= 10.0
                if any(value.startswith(cross_domain_prefixes) for value in [*raw_relation_ids, *raw_metadata_types]):
                    score -= 20.0
            score += 12.0 * self._token_overlap_ratio(normalized_context, metadata_text)
            score += 14.0 * self._token_overlap_ratio(normalized_context, relation_text)
            if metadata is not None and metadata.is_probable_cvt:
                score -= 3.0
            heuristic_ranked.append((score, candidate))

        heuristic_ranked.sort(key=lambda item: (-item[0], item[1].id_or_mid))
        heuristic_debug_rows = [
            {
                "entity_id": candidate.id_or_mid,
                "label": candidate.label,
                "types": list((metadata_by_id.get(candidate.id_or_mid).types if metadata_by_id.get(candidate.id_or_mid) is not None else ()))[:5],
                "score": round(float(score), 3),
                "type_match": bool(
                    metadata_by_id.get(candidate.id_or_mid) is not None
                    and expected_type
                    and self._entity_matches_expected_type(metadata_by_id.get(candidate.id_or_mid), expected_type)
                ),
                "top_relations": [item.relation for item in candidate_relation_rows.get(candidate.id_or_mid, [])[:5]],
            }
            for score, candidate in heuristic_ranked[:10]
        ]
        LOGGER.info(
            "Entity candidate heuristic ranking mention=%s expected_type=%s role=%s ranked_candidates=%s",
            mention,
            expected_type,
            role,
            heuristic_debug_rows,
        )
        if expected_type and any(
            metadata_by_id.get(candidate.id_or_mid) is not None
            and self._entity_matches_expected_type(metadata_by_id.get(candidate.id_or_mid), expected_type)
            for _, candidate in heuristic_ranked
        ):
            heuristic_ranked = [
                (score, candidate)
                for score, candidate in heuristic_ranked
                if metadata_by_id.get(candidate.id_or_mid) is not None
                and self._entity_matches_expected_type(metadata_by_id.get(candidate.id_or_mid), expected_type)
            ]
        compact_candidates: list[dict[str, Any]] = []
        for score, candidate in heuristic_ranked[: self._sqlite_entity_candidate_recall_top_k()]:
            metadata = metadata_by_id.get(candidate.id_or_mid)
            relation_rows = candidate_relation_rows.get(candidate.id_or_mid, [])
            compact_candidates.append(
                {
                    "entity_id": candidate.id_or_mid,
                    "label": candidate.label[:64],
                    "types": list((metadata.types if metadata is not None else ())[:5]),
                    "degree": int(metadata.degree if metadata is not None else 0),
                    "top_relations": [item.relation for item in relation_rows[:5]],
                    "sample_targets": [
                        str(value)[:36]
                        for item in relation_rows[:3]
                        for value in item.sample_target_names[:1]
                    ][:3],
                    "heuristic_score": round(score, 3),
                }
            )

        prompt = ENTITY_CANDIDATE_DISAMBIGUATION_PROMPT.format(
            question=context_text or mention,
            mention=mention,
            expected_type=expected_type,
            role=role,
            candidate_entities=json.dumps(compact_candidates, ensure_ascii=False, separators=(",", ":")),
        )
        LOGGER.info(
            "Entity candidate disambiguation mention=%s candidate_count=%d prompt_chars=%d expected_type=%s",
            mention,
            len(compact_candidates),
            len(prompt),
            expected_type,
        )
        selected_ids: list[str] = []
        try:
            raw = self.llm.generate(prompt, max_tokens=192)
            parsed = robust_json_parse(raw, fallback={})
            if isinstance(parsed, dict):
                candidate_id_set = {item["entity_id"] for item in compact_candidates}
                selected_ids = self._deduplicate_strings(
                    [
                        str(value).strip()
                        for value in parsed.get("selected_entity_ids", [])
                        if str(value).strip() in candidate_id_set
                        ]
                    )
        except Exception:
            selected_ids = []
        LOGGER.info(
            "Entity candidate LLM selection mention=%s expected_type=%s role=%s selected_entity_ids=%s preserve_multiple_candidates=%s",
            mention,
            expected_type,
            role,
            selected_ids,
            preserve_multiple_candidates,
        )

        if not selected_ids:
            best_score = heuristic_ranked[0][0]
            keep_threshold = max(best_score * 0.9, best_score - 4.0)
            selected_ids = [
                candidate.id_or_mid
                for score, candidate in heuristic_ranked
                if score >= keep_threshold
            ][: self._sqlite_entity_candidate_selected_top_k()]
            LOGGER.info(
                "Entity candidate threshold fallback mention=%s expected_type=%s role=%s best_score=%.3f keep_threshold=%.3f threshold_selected=%s",
                mention,
                expected_type,
                role,
                float(best_score),
                float(keep_threshold),
                selected_ids,
            )
        elif preserve_multiple_candidates:
            # For start-anchor entities, keep a small pool alive for the downstream
            # anchor filter instead of collapsing to a single LLM-picked candidate.
            selected_id_set = set(selected_ids)
            for score, candidate in heuristic_ranked:
                if candidate.id_or_mid in selected_id_set:
                    continue
                selected_ids.append(candidate.id_or_mid)
                selected_id_set.add(candidate.id_or_mid)
                if len(selected_ids) >= self._sqlite_entity_candidate_selected_top_k():
                    break
            LOGGER.info(
                "Entity candidate preserve-multiple mention=%s expected_type=%s role=%s expanded_selected=%s",
                mention,
                expected_type,
                role,
                selected_ids,
            )

        candidate_by_id = {candidate.id_or_mid: candidate for candidate in candidates}
        reranked: list[ResolvedCandidate] = []
        selected_id_set = set(selected_ids)
        for score, candidate in heuristic_ranked:
            if candidate.id_or_mid not in selected_id_set:
                continue
            reranked.append(
                ResolvedCandidate(
                    id_or_mid=candidate.id_or_mid,
                    label=candidate.label,
                    mention_or_hint=candidate.mention_or_hint,
                    match_type=f"{candidate.match_type}|context_rerank",
                    score=score,
                )
            )
        dropped_ids = [candidate.id_or_mid for _, candidate in heuristic_ranked if candidate.id_or_mid not in selected_id_set]
        LOGGER.info(
            "Entity candidate final selection mention=%s expected_type=%s role=%s kept=%s dropped=%s",
            mention,
            expected_type,
            role,
            [
                {
                    "entity_id": item.id_or_mid,
                    "label": item.label,
                    "score": round(float(item.score), 3),
                    "match_type": item.match_type,
                }
                for item in reranked
            ],
            dropped_ids[:10],
        )
        if reranked:
            return reranked
        return [candidate_by_id[selected_ids[0]]] if selected_ids and selected_ids[0] in candidate_by_id else [heuristic_ranked[0][1]]

    def _build_relation_query_names(
        self,
        spec: Any,
        sub_question: SubQuestionSpec,
    ) -> list[str]:
        """Expand relation lookup strings using lightweight domain heuristics."""
        base_names = self._deduplicate_strings(
            [
                getattr(spec, "name", ""),
                *getattr(spec, "aliases", []),
                *getattr(spec, "freebase_like_ids", []),
            ]
        )
        normalized = " ".join(self._normalize_relation_text(name) for name in base_names).strip()
        expanded: list[str] = list(base_names)
        for key, aliases in CONTEXTUAL_RELATION_QUERY_EXPANSIONS.items():
            if key in normalized:
                expanded.extend(aliases)
        question_norm = self._normalize_relation_text(sub_question.question)
        if "anthem" in question_norm:
            expanded.extend(CONTEXTUAL_RELATION_QUERY_EXPANSIONS["national anthem"])
        if "mascot" in question_norm:
            expanded.extend(CONTEXTUAL_RELATION_QUERY_EXPANSIONS["mascot"])
        if "central america" in question_norm or "region" in question_norm:
            expanded.extend(CONTEXTUAL_RELATION_QUERY_EXPANSIONS["located in region"])
        if "department" in question_norm or "contains" in question_norm:
            expanded.extend(CONTEXTUAL_RELATION_QUERY_EXPANSIONS["contained by country"])
        return self._deduplicate_strings(expanded)

    def _select_relation_ids_for_sub_question(
        self,
        sub_question: SubQuestionSpec,
        relation_candidates: list[ResolvedCandidate],
        entity_candidates: list[ResolvedCandidate],
    ) -> tuple[list[str], dict[str, Any]]:
        """Context-rerank relation candidates and keep only the best 1-2 ids for strict expansion."""
        if not relation_candidates:
            return [], {"ranked_candidates": [], "selected_relation_ids": []}

        seed_text = " ".join(
            self._normalize_relation_text(candidate.label or candidate.mention_or_hint)
            for candidate in entity_candidates
        )
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        relation_hints = " ".join(
            self._normalize_relation_text(item)
            for relation in sub_question.interested_relations
            for item in [relation.name, *relation.aliases]
        )
        question_text = self._normalize_relation_text(sub_question.question)
        direction_hints = {
            self._normalize_relation_text(relation.direction)
            for relation in sub_question.interested_relations
            if relation.direction
        }
        scored_rows: list[tuple[float, ResolvedCandidate]] = []
        for candidate in relation_candidates:
            relation_text = self._normalize_relation_text(f"{candidate.id_or_mid} {candidate.label}")
            score = float(candidate.score)
            score += 15.0 * self._token_overlap_ratio(relation_hints, relation_text)
            score += 10.0 * self._token_overlap_ratio(expected_type, relation_text)
            score += 6.0 * self._token_overlap_ratio(question_text, relation_text)
            if expected_type and expected_type in relation_text:
                score += 8.0
            if any(direction and "answer" in direction and "topic entity" in direction for direction in direction_hints):
                if any(token in relation_text for token in ("parent", "type", "anthem", "mascot", "leader", "religion")):
                    score += 4.0
            if any(direction and "country" in direction for direction in direction_hints):
                if any(token in relation_text for token in ("country", "parent", "containedby", "administrative")):
                    score += 6.0
            if any(direction and "answer" in direction and "country" in direction for direction in direction_hints):
                if "region" in relation_text:
                    score += 5.0
            if seed_text and any(token in relation_text for token in seed_text.split()):
                score -= 6.0
            if any(token in relation_text for token in NOISE_RELATION_HINTS):
                score -= 30.0
            if "religion" in expected_type and "religion" in relation_text:
                score += 10.0
            if "government" in expected_type and "government" in relation_text:
                score += 10.0
            if "date" in expected_type and any(token in relation_text for token in ("date", "time", "last_win", "won")):
                score += 8.0
            if "country" in expected_type and any(token in relation_text for token in ("country", "containedby", "parent")):
                score += 8.0
            scored_rows.append((score, candidate))

        scored_rows.sort(key=lambda item: (-item[0], item[1].id_or_mid))
        selected: list[str] = []
        if scored_rows:
            top_score = scored_rows[0][0]
            for index, (score, candidate) in enumerate(scored_rows[:2]):
                if index == 0:
                    selected.append(candidate.id_or_mid)
                    continue
                if score >= top_score - 5.0 and score >= 80.0:
                    selected.append(candidate.id_or_mid)
        debug_rows = [
            {
                "relation_id": candidate.id_or_mid,
                "label": candidate.label,
                "mention_or_hint": candidate.mention_or_hint,
                "base_score": candidate.score,
                "reranked_score": round(score, 3),
            }
            for score, candidate in scored_rows
        ]
        return self._deduplicate_strings(selected), {
            "ranked_candidates": debug_rows,
            "selected_relation_ids": self._deduplicate_strings(selected),
        }

    def _rerank_candidate_paths_with_local_hints(
        self,
        sub_question: SubQuestionSpec,
        candidate_paths: list[ReasoningPath],
        trace: list[SearchTraceStep],
        depth: int,
    ) -> list[ReasoningPath]:
        """Use interested nodes and relation hints only as local path-level reranking signals."""
        if not candidate_paths:
            return []
        relation_terms = self._deduplicate_strings(
            [
                relation.name
                for relation in sub_question.interested_relations
                if relation.name
            ]
            + [
                alias
                for relation in sub_question.interested_relations
                for alias in relation.aliases
                if alias
            ]
            + list(sub_question.downstream_filters)
        )
        node_terms = self._deduplicate_strings(
            [node.name for node in sub_question.interested_nodes if node.name]
            + [entity.name for entity in self._effective_sub_question_topics(sub_question) if entity.name]
        )
        expected_type_terms = self._deduplicate_strings([sub_question.expected_answer_type])

        scored_paths: list[tuple[float, ReasoningPath]] = []
        for path in candidate_paths:
            relation_text = " ".join(
                self._normalize_relation_text(triple.relation)
                for triple in path.triples
            )
            node_text = " ".join(
                self._normalize_relation_text(value)
                for triple in path.triples
                for value in (triple.head, triple.tail)
            )
            terminal_text = self._normalize_relation_text(path.nodes[-1] if path.nodes else "")
            score = float(path.path_score)
            score += 8.0 * max(
                [self._token_overlap_ratio(term, relation_text) for term in relation_terms] or [0.0]
            )
            score += 5.0 * max(
                [self._token_overlap_ratio(term, node_text) for term in node_terms] or [0.0]
            )
            score += 4.0 * max(
                [self._token_overlap_ratio(term, terminal_text) for term in expected_type_terms] or [0.0]
            )
            path.path_score = score
            scored_paths.append((score, path))

        scored_paths.sort(key=lambda item: item[0], reverse=True)
        reranked = [path for _, path in scored_paths]
        trace.append(
            SearchTraceStep(
                stage="local_hint_rerank",
                depth=depth,
                candidate_count=len(candidate_paths),
                pruned_count=min(len(reranked), len(candidate_paths)),
                note=(
                    f"sub_question_id={sub_question.id} relation_terms={relation_terms[:4]} "
                    f"node_terms={node_terms[:4]} expected_type={sub_question.expected_answer_type}"
                ),
            )
        )
        return reranked

    def _select_local_relation_probe_ids(
        self,
        sub_question: SubQuestionSpec,
        step_seed_ids: list[str],
        graph_api: Any,
        candidate_paths: list[ReasoningPath],
        trace: list[SearchTraceStep],
        depth: int,
        attempt_index: int,
    ) -> list[str]:
        """Enumerate one-hop relations around current seeds and pick top local probe ids."""
        if graph_api is None or not step_seed_ids:
            return []

        relation_terms = self._deduplicate_strings(
            [
                sub_question.question,
                sub_question.expected_answer_type,
                *[relation.name for relation in sub_question.interested_relations if relation.name],
                *[
                    alias
                    for relation in sub_question.interested_relations
                    for alias in relation.aliases
                    if alias
                ],
                *[node.name for node in sub_question.interested_nodes if node.name],
                *list(sub_question.downstream_filters),
            ]
        )
        if not relation_terms:
            return []

        relation_rows: dict[str, dict[str, Any]] = {}
        per_seed_limit = self._graphapi_local_relation_candidate_limit()
        for seed_id in step_seed_ids[: self._agentic_max_carryover_entities()]:
            for edge in graph_api.get_neighbors(
                seed_id,
                include_reverse=True,
                relation_filter=None,
                limit=per_seed_limit,
                strict_relation_filter=False,
            ):
                relation_id = str(edge.triple.relation).strip()
                if not relation_id:
                    continue
                display_name = str(graph_api.get_relation_display_name(relation_id) or relation_id)
                target_label = (
                    edge.triple.head if edge.reversed else edge.triple.tail
                )
                row = relation_rows.setdefault(
                    relation_id,
                    {
                        "relation_id": relation_id,
                        "display_name": display_name,
                        "targets": [],
                    },
                )
                if target_label and target_label not in row["targets"]:
                    row["targets"].append(str(target_label))

        if not relation_rows:
            return []

        beam_relation_ids = self._deduplicate_strings(
            [
                triple.relation
                for path in candidate_paths
                if path.source_stage in {"relation_beam", CVT_BUNDLE_SOURCE_STAGE}
                for triple in path.triples
                if triple.relation
            ]
        )
        scored_rows: list[tuple[float, dict[str, Any]]] = []
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        question_text = self._normalize_relation_text(sub_question.question)
        generic_noise_domains = (
            "award.",
            "book.",
            "tv.",
            "music.",
            "media_common.",
            "comic_books.",
        )
        for row in relation_rows.values():
            relation_text = self._normalize_relation_text(
                f"{row['relation_id']} {row['display_name']}"
            )
            target_text = self._normalize_relation_text(" ".join(row["targets"][:3]))
            score = 0.0
            score += 12.0 * max(
                [self._token_overlap_ratio(term, relation_text) for term in relation_terms] or [0.0]
            )
            score += 5.0 * max(
                [self._token_overlap_ratio(term, target_text) for term in relation_terms] or [0.0]
            )
            score += 8.0 * self._token_overlap_ratio(question_text, relation_text)
            score += 2.0 * self._token_overlap_ratio(question_text, target_text)
            if row["relation_id"] in beam_relation_ids:
                score += 6.0
            if row["relation_id"].startswith(generic_noise_domains):
                score -= 4.0
            if expected_type and any(token in relation_text for token in ("date", "time", "year", "championship", "championships")):
                score += 4.0
            if "world series" in target_text:
                score += 8.0
            if "championship" in relation_text or "championships" in relation_text:
                score += 6.0
            scored_rows.append((score, row))

        scored_rows.sort(key=lambda item: (-item[0], item[1]["relation_id"]))
        top_k = min(self._agentic_local_relation_probe_top_k() + max(0, attempt_index - 1), len(scored_rows))
        top_rows = scored_rows[:top_k]
        selected_ids = self._supervise_local_relation_probe_ids_with_llm(
            sub_question=sub_question,
            scored_rows=top_rows,
            beam_suggested_ids=beam_relation_ids,
            trace=trace,
            depth=depth,
        )
        trace.append(
            SearchTraceStep(
                stage="local_relation_probe",
                depth=depth,
                candidate_count=len(scored_rows),
                pruned_count=len(selected_ids),
                note=(
                    f"sub_question_id={sub_question.id} selected_relation_ids={selected_ids} "
                    f"beam_relation_ids={beam_relation_ids[:5]} "
                    f"top_candidates={[{'relation_id': row['relation_id'], 'targets': row['targets'][:2], 'score': round(score, 3)} for score, row in scored_rows[:5]]}"
                ),
            )
        )
        return self._deduplicate_strings(selected_ids)

    def _filter_dependency_seed_ids_with_llm(
        self,
        sub_question: SubQuestionSpec,
        dependency_seed_ids: list[str],
        resolved_sub_answers: dict[str, dict[str, Any]],
        graph_api: Any,
        trace: list[SearchTraceStep],
        depth: int,
    ) -> list[str]:
        """Use the LLM as a supervisor to keep only dependency entities relevant to the current sub-question."""
        dependency_seed_ids = self._deduplicate_strings(dependency_seed_ids)
        if graph_api is None or len(dependency_seed_ids) <= 1 or not sub_question.depends_on:
            return dependency_seed_ids
        dependency_seed_ids = self._rank_dependency_seed_ids_by_provenance(
            sub_question=sub_question,
            dependency_seed_ids=dependency_seed_ids,
            resolved_sub_answers=resolved_sub_answers,
            graph_api=graph_api,
            trace=trace,
            depth=depth,
        )
        if len(dependency_seed_ids) <= 1:
            return dependency_seed_ids

        candidate_entities = [
            {
                "entity_id": entity_id,
                "label": str(graph_api.get_entity_display_name(entity_id) or entity_id),
                "source_sub_question_id": dependency_id,
                "source_answer": resolved_sub_answers.get(dependency_id, {}).get("answer", ""),
            }
            for dependency_id in sub_question.depends_on
            for entity_id in resolved_sub_answers.get(dependency_id, {}).get("entity_ids", [])
            if entity_id in dependency_seed_ids
        ]
        if len(candidate_entities) <= 1:
            return dependency_seed_ids

        resolved_dependencies = [
            {
                "sub_question_id": dependency_id,
                "answer": resolved_sub_answers.get(dependency_id, {}).get("answer", ""),
                "entity_ids": resolved_sub_answers.get(dependency_id, {}).get("entity_ids", []),
            }
            for dependency_id in sub_question.depends_on
        ]
        prompt = DEPENDENCY_ENTITY_FILTER_PROMPT.format(
            sub_question=sub_question.question,
            expected_answer_type=sub_question.expected_answer_type,
            resolved_dependencies=json.dumps(resolved_dependencies, ensure_ascii=False),
            candidate_entities=json.dumps(candidate_entities, ensure_ascii=False),
        )
        raw = self.llm.generate(prompt, max_tokens=192)
        parsed = robust_json_parse(raw, fallback={})
        selected_ids = self._deduplicate_strings(
            [item for item in parsed.get("selected_entity_ids", []) if item in dependency_seed_ids]
            if isinstance(parsed, dict)
            else []
        )
        if not selected_ids:
            selected_ids = dependency_seed_ids
        trace.append(
            SearchTraceStep(
                stage="dependency_entity_filter",
                depth=depth,
                candidate_count=len(dependency_seed_ids),
                pruned_count=len(selected_ids),
                note=(
                    f"sub_question_id={sub_question.id} input_entity_ids={dependency_seed_ids} "
                    f"selected_entity_ids={selected_ids}"
                ),
            )
        )
        return selected_ids

    def _collect_summary_relation_ids(self, summaries: list[dict[str, Any]]) -> list[str]:
        """Collect relation ids from summarized evidence for downstream provenance use."""
        relation_ids: list[str] = []
        for item in summaries:
            if not isinstance(item, dict):
                continue
            for triple in item.get("key_triples", []):
                if not isinstance(triple, dict):
                    continue
                relation = str(triple.get("relation") or "").strip()
                if relation:
                    relation_ids.append(relation)
        return self._deduplicate_strings(relation_ids)

    def _collect_summary_evidence_text(self, summaries: list[dict[str, Any]]) -> list[str]:
        """Collect summarized evidence text for downstream provenance use."""
        evidence_lines: list[str] = []
        for item in summaries:
            if not isinstance(item, dict):
                continue
            evidence_lines.extend(
                str(line).strip()
                for line in item.get("evidence", [])
                if str(line).strip()
            )
        return self._deduplicate_strings(evidence_lines)

    def _rank_dependency_seed_ids_by_provenance(
        self,
        sub_question: SubQuestionSpec,
        dependency_seed_ids: list[str],
        resolved_sub_answers: dict[str, dict[str, Any]],
        graph_api: Any,
        trace: list[SearchTraceStep],
        depth: int,
    ) -> list[str]:
        """Disambiguate dependency seeds using upstream evidence provenance and node metadata."""
        if (
            not dependency_seed_ids
            or not hasattr(graph_api, "get_nodes_metadata_batched")
            or not hasattr(graph_api, "get_node_relations")
        ):
            return dependency_seed_ids

        metadata_by_id = graph_api.get_nodes_metadata_batched(dependency_seed_ids)
        dependency_payloads = [
            resolved_sub_answers.get(dependency_id, {})
            for dependency_id in sub_question.depends_on
            if dependency_id in resolved_sub_answers
        ]
        provenance_chunks: list[str] = []
        for payload in dependency_payloads:
            provenance_chunks.append(str(payload.get("answer") or "").strip())
            provenance_chunks.extend(str(item).strip() for item in payload.get("predicted_answers", []) if str(item).strip())
            provenance_chunks.extend(str(item).strip() for item in payload.get("literals", []) if str(item).strip())
            provenance_chunks.extend(str(item).strip() for item in payload.get("evidence_relations", []) if str(item).strip())
            provenance_chunks.extend(str(item).strip() for item in payload.get("evidence_text", []) if str(item).strip())
            provenance_chunks.extend(str(item).strip() for item in payload.get("supporting_paths", []) if str(item).strip())
        provenance_text = self._normalize_relation_text(" ".join(provenance_chunks))
        current_context_text = self._normalize_relation_text(
            " ".join(
                [
                    sub_question.question,
                    sub_question.expected_answer_type,
                    *[item.name for item in sub_question.interested_nodes if item.name],
                    *[item.name for item in self._effective_sub_question_topics(sub_question) if item.name],
                    *[hint.name for hint in sub_question.interested_relations if hint.name],
                    *[
                        alias
                        for hint in sub_question.interested_relations
                        for alias in hint.aliases
                        if alias
                    ],
                    *list(sub_question.downstream_filters),
                ]
            )
        )

        scored_ids: list[tuple[float, str, dict[str, Any]]] = []
        for entity_id in dependency_seed_ids:
            metadata = metadata_by_id.get(entity_id)
            if metadata is None:
                continue
            relation_candidates = graph_api.get_node_relations(
                entity_id,
                include_reverse=True,
                include_literals=False,
            )
            relation_text = self._normalize_relation_text(
                " ".join(
                    f"{candidate.relation} {candidate.relation_name}"
                    for candidate in relation_candidates[:8]
                )
            )
            metadata_text = self._normalize_relation_text(" ".join([metadata.name, *metadata.types]))
            score = 0.0
            score += 10.0 * self._token_overlap_ratio(provenance_text, metadata_text)
            score += 14.0 * self._token_overlap_ratio(provenance_text, relation_text)
            score += 8.0 * self._token_overlap_ratio(current_context_text, metadata_text)
            score += 10.0 * self._token_overlap_ratio(current_context_text, relation_text)
            if metadata.is_probable_cvt:
                score -= 2.0
            score += min(2.0, metadata.degree / 2000.0)
            scored_ids.append(
                (
                    score,
                    entity_id,
                    {
                        "entity_id": entity_id,
                        "label": metadata.name,
                        "types": list(metadata.types[:4]),
                        "top_relations": [candidate.relation for candidate in relation_candidates[:5]],
                        "score": round(score, 3),
                    },
                )
            )

        if not scored_ids:
            return dependency_seed_ids
        scored_ids.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored_ids[0][0]
        keep_threshold = max(best_score * 0.85, best_score - 1.5)
        selected_ids = [entity_id for score, entity_id, _ in scored_ids if score >= keep_threshold]
        trace.append(
            SearchTraceStep(
                stage="dependency_seed_disambiguation",
                depth=depth,
                candidate_count=len(dependency_seed_ids),
                pruned_count=len(selected_ids),
                note=(
                    f"sub_question_id={sub_question.id} selected_seed_ids={selected_ids} "
                    f"best_score={round(best_score, 3)} "
                    f"top_candidates={[payload for _, _, payload in scored_ids[:5]]}"
                ),
            )
        )
        return self._deduplicate_strings(selected_ids or dependency_seed_ids)

    def _supervise_local_relation_probe_ids_with_llm(
        self,
        sub_question: SubQuestionSpec,
        scored_rows: list[tuple[float, dict[str, Any]]],
        beam_suggested_ids: list[str],
        trace: list[SearchTraceStep],
        depth: int,
    ) -> list[str]:
        """Use the LLM as the final selector over a heuristic-truncated candidate relation set."""
        beam_suggested_ids = self._deduplicate_strings(beam_suggested_ids)
        if not scored_rows:
            return beam_suggested_ids

        compact_candidate_relations = []
        candidate_relations = [
            {
                "relation_id": row["relation_id"],
                "display_name": row["display_name"],
                "sample_targets": row["targets"][:3],
                "heuristic_score": round(score, 3),
                "beam_suggested": row["relation_id"] in beam_suggested_ids,
            }
            for score, row in scored_rows
        ]
        for item in candidate_relations:
            compact_candidate_relations.append(
                {
                    "relation_id": str(item["relation_id"])[:96],
                    "display_name": str(item["display_name"])[:48],
                    "sample_targets": [str(value)[:40] for value in item["sample_targets"][:1]],
                    "heuristic_score": item["heuristic_score"],
                    "beam_suggested": item["beam_suggested"],
                }
            )
        resolved_dependencies = [
            {"sub_question_id": dependency_id, "note": "resolved dependency"}
            for dependency_id in sub_question.depends_on
        ]
        prompt = RELATION_SUPERVISION_PROMPT.format(
            sub_question=sub_question.question,
            expected_answer_type=sub_question.expected_answer_type,
            resolved_dependencies=json.dumps(resolved_dependencies, ensure_ascii=False),
            candidate_relations=json.dumps(compact_candidate_relations, ensure_ascii=False),
        )
        LOGGER.info(
            "Local relation supervision prompt sub_question_id=%s candidate_count=%d prompt_chars=%d beam_suggested=%d",
            sub_question.id,
            len(compact_candidate_relations),
            len(prompt),
            len(beam_suggested_ids),
        )
        raw = self.llm.generate(prompt, max_tokens=768)
        parsed = robust_json_parse(raw, fallback={})
        candidate_relation_ids = {item["relation_id"] for item in compact_candidate_relations}
        selected_ids: list[str] = []
        if isinstance(parsed, dict):
            selected_ids = self._deduplicate_strings(
                [
                    str(item).strip()
                    for item in parsed.get("selected_relation_ids", [])
                    if str(item).strip() in candidate_relation_ids
                ]
            )
        if not selected_ids:
            selected_ids = self._deduplicate_strings(beam_suggested_ids)[:2]
        if not selected_ids and scored_rows:
            selected_ids = [scored_rows[0][1]["relation_id"]]
        guard_applied = False
        if beam_suggested_ids and self._local_relation_probe_preserve_beam():
            overlapping_ids = [relation_id for relation_id in selected_ids if relation_id in beam_suggested_ids]
            if not overlapping_ids:
                guard_applied = True
                selected_ids = list(beam_suggested_ids)
                if self._local_relation_probe_allow_beam_supplement():
                    for relation_id in self._deduplicate_strings(
                        [*selected_ids, *[row["relation_id"] for _, row in scored_rows]]
                    ):
                        if relation_id in selected_ids:
                            continue
                        selected_ids.append(relation_id)
                        break
        trace.append(
            SearchTraceStep(
                stage="local_relation_supervision",
                depth=depth,
                candidate_count=len(compact_candidate_relations),
                pruned_count=len(selected_ids),
                note=(
                    f"sub_question_id={sub_question.id} beam_suggested_ids={beam_suggested_ids} "
                    f"selected_relation_ids={selected_ids} guard_applied={guard_applied}"
                ),
            )
        )
        return selected_ids

    def _probe_paths_with_local_relations(
        self,
        sub_question: SubQuestionSpec,
        step_seed_ids: list[str],
        relation_ids: list[str],
        graph_api: Any,
        trace: list[SearchTraceStep],
        depth: int,
    ) -> list[ReasoningPath]:
        """Run a strict local relation probe and merge it with a relation-filtered two-hop fallback."""
        if graph_api is None or not step_seed_ids or not relation_ids:
            return []
        max_depth = self._graphapi_max_depth()
        strict_paths = graph_api.expand_paths(
            seed_nodes=step_seed_ids,
            relation_hints=relation_ids,
            max_depth=max_depth,
            strict_relation_filter=True,
            max_paths=self._cwq_max_paths(),
            max_nodes=self._cwq_max_nodes(),
        )
        for path in strict_paths:
            path.source_stage = "external_graphapi_strict_probe"
        annotate_paths_with_relation_matches(strict_paths, relation_ids)

        fallback_paths: list[ReasoningPath] = []
        if not self._graphapi_one_hop_only():
            fallback_paths = graph_api.find_two_hop_extensions(
                frontier_nodes=step_seed_ids,
                relation_hints=relation_ids,
                limit=self._graphapi_neighbor_limit(),
            )
        for path in fallback_paths:
            path.source_stage = "external_graphapi_relation_probe"
        annotate_paths_with_relation_matches(fallback_paths, relation_ids)
        merged_paths = self._sort_cwq_candidate_paths(deduplicate_paths(strict_paths + fallback_paths))
        merged_paths = expand_cvt_bundle_paths(
            graph=graph_api,
            paths=merged_paths,
            subquestion=sub_question.question,
            expected_answer_type=sub_question.expected_answer_type,
            relation_hint_names=relation_ids,
            anchor_mentions=self._deduplicate_strings(
                [
                    sub_question.question,
                    *[relation.name for relation in sub_question.interested_relations if relation.name],
                    *[
                        alias
                        for relation in sub_question.interested_relations
                        for alias in relation.aliases
                        if alias
                    ],
                    *[node.name for node in sub_question.interested_nodes if node.name],
                ]
            ),
        )
        trace.append(
            SearchTraceStep(
                stage="local_relation_probe_search",
                depth=depth,
                candidate_count=len(strict_paths) + len(fallback_paths),
                pruned_count=len(merged_paths),
                note=(
                    f"sub_question_id={sub_question.id} seed_node_ids={step_seed_ids} "
                    f"relation_ids={relation_ids} max_depth={max_depth} "
                    f"one_hop_only={self._graphapi_one_hop_only()} "
                    f"strict_paths={len(strict_paths)} fallback_paths={len(fallback_paths)}"
                ),
            )
        )
        return merged_paths

    @staticmethod
    def _normalize_relation_text(text: str) -> str:
        """Normalize relation-like text for simple heuristic scoring."""
        return re.sub(r"[\s._/\-()|]+", " ", str(text).strip().lower()).strip()

    @classmethod
    def _token_overlap_ratio(cls, left: str, right: str) -> float:
        """Compute simple token overlap for heuristic reranking."""
        left_tokens = {token for token in cls._normalize_relation_text(left).split() if token}
        right_tokens = {token for token in cls._normalize_relation_text(right).split() if token}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(len(left_tokens), 1)

    def _dependency_seed_ids_for_sub_question(
        self,
        sub_question: SubQuestionSpec,
        dependency_seed_ids: dict[str, list[str]],
    ) -> list[str]:
        """Collect any seed node ids contributed by declared sub-question dependencies."""
        resolved: list[str] = []
        for dependency_id in sub_question.depends_on:
            resolved.extend(dependency_seed_ids.get(dependency_id, []))
        return self._deduplicate_strings(resolved)

    def _collect_dependency_seed_ids(
        self,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        summaries: list[dict[str, Any]],
        graph_api: Any,
    ) -> list[str]:
        """Resolve answer-like nodes only for downstream dependent sub-questions."""
        labels = self._extract_answer_like_labels(
            sub_question=sub_question,
            pruned_paths=[path for path in pruned_paths if path.pruning_status == "preserved"],
            summaries=summaries,
        )
        if not labels:
            return []
        return self._deduplicate_strings(graph_api.resolve_entity_mentions(labels[:3], top_k=1))

    def _extract_answer_like_labels(
        self,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        summaries: list[dict[str, Any]],
    ) -> list[str]:
        """Extract answer-like labels from summaries first, then preserved path endpoints."""
        labels: list[str] = []
        direction = " ".join(
            self._normalize_relation_text(relation.direction)
            for relation in sub_question.interested_relations
            if relation.direction
        )
        anchor_labels = {
            self._normalize_relation_text(entity.name)
            for entity in [*self._effective_sub_question_topics(sub_question), *sub_question.interested_nodes]
            if entity.name
        }

        for summary in summaries:
            for triple in summary.get("key_triples", []):
                if not isinstance(triple, dict):
                    continue
                head = str(triple.get("head") or "").strip()
                tail = str(triple.get("tail") or "").strip()
                label = self._select_answer_like_label(
                    head=head,
                    tail=tail,
                    direction=direction,
                    anchor_labels=anchor_labels,
                )
                if label:
                    labels.append(label)

        if labels:
            return self._deduplicate_strings(labels)[:3]

        for path in pruned_paths:
            if len(path.nodes) < 2:
                continue
            head = str(path.nodes[0]).strip()
            tail = str(path.nodes[-1]).strip()
            label = self._select_answer_like_label(
                head=head,
                tail=tail,
                direction=direction,
                anchor_labels=anchor_labels,
            )
            if label:
                labels.append(label)

        return self._deduplicate_strings(labels)[:3]

    def _select_answer_like_label(
        self,
        head: str,
        tail: str,
        direction: str,
        anchor_labels: set[str],
    ) -> str:
        """Choose which endpoint should be propagated as the answer-like node."""
        norm_head = self._normalize_relation_text(head)
        norm_tail = self._normalize_relation_text(tail)
        if "answer topic entity" in direction:
            if norm_head and norm_head not in anchor_labels:
                return head
            return tail if norm_tail and norm_tail not in anchor_labels else ""
        if "topic entity answer" in direction or "country answer" in direction:
            if norm_tail and norm_tail not in anchor_labels:
                return tail
            return head if norm_head and norm_head not in anchor_labels else ""
        if norm_tail and norm_tail not in anchor_labels:
            return tail
        if norm_head and norm_head not in anchor_labels:
            return head
        return ""

    def _expand_cwq_sub_question_paths(
        self,
        sub_question: SubQuestionSpec,
        resolved_sub_question: ResolvedSubQuestion,
        dependent_seed_ids: list[str],
        graph_api: Any,
        trace: list[SearchTraceStep],
    ) -> list[ReasoningPath]:
        """Expand one sub-question directly against SQLite, with per-subquestion fallback."""
        seed_node_ids = self._deduplicate_strings(resolved_sub_question.seed_node_ids + dependent_seed_ids)
        relation_ids = self._deduplicate_strings(
            resolved_sub_question.selected_relation_ids or resolved_sub_question.relation_ids
        )
        max_depth = self._graphapi_subquestion_max_depth()
        strict_paths = graph_api.expand_paths(
            seed_nodes=seed_node_ids,
            relation_hints=relation_ids,
            max_depth=max_depth,
            strict_relation_filter=True,
            max_paths=self._cwq_max_paths(),
            max_nodes=self._cwq_max_nodes(),
        )
        for path in strict_paths:
            path.source_stage = "external_graphapi_expansion"
        annotate_paths_with_relation_matches(strict_paths, relation_ids)
        strict_paths = self._sort_cwq_candidate_paths(strict_paths)
        trace.append(
            SearchTraceStep(
                stage="cwq_sub_question_graph_construction",
                depth=max_depth,
                candidate_count=len(strict_paths),
                pruned_count=0,
                note=(
                    f"{sub_question.id}: sqlite expansion seeds={seed_node_ids} "
                    f"selected_relations={relation_ids} max_depth={max_depth} "
                    f"strict_relation_filter=True strict_paths={len(strict_paths)}"
                ),
            )
        )
        fallback_paths: list[ReasoningPath] = []
        if not self._graphapi_one_hop_only():
            fallback_paths = explore_graphapi_bootstrap_paths(
                aligned_entity_ids=seed_node_ids,
                aligned_relation_ids=relation_ids,
                graph_api=graph_api,
                neighbor_limit=self._graphapi_neighbor_limit(),
            )
            if not fallback_paths and seed_node_ids:
                fallback_paths = graph_api.find_two_hop_extensions(
                    frontier_nodes=seed_node_ids,
                    relation_hints=relation_ids,
                    limit=self._graphapi_neighbor_limit(),
                )
        annotate_paths_with_relation_matches(fallback_paths, relation_ids)
        fallback_paths = self._sort_cwq_candidate_paths(fallback_paths)
        merged_paths = deduplicate_paths(strict_paths + fallback_paths) if strict_paths else fallback_paths
        merged_paths = self._sort_cwq_candidate_paths(merged_paths)
        trace.append(
            SearchTraceStep(
                stage="cwq_sub_question_fallback",
                depth=max_depth,
                candidate_count=len(fallback_paths),
                pruned_count=0,
                note=(
                    f"{sub_question.id}: fallback bootstrap/two-hop seeds={seed_node_ids} "
                    f"one_hop_only={self._graphapi_one_hop_only()} "
                    f"selected_relations={relation_ids} strict_empty={not bool(strict_paths)} "
                    f"strict_paths={len(strict_paths)} fallback_paths={len(fallback_paths)}"
                ),
            )
        )
        return merged_paths

    def _sort_cwq_candidate_paths(self, paths: list[ReasoningPath]) -> list[ReasoningPath]:
        """Sort strict matches ahead of fallback and preserve relation-aware ranking."""
        ranked = sort_paths_by_relation_bias(paths)
        ranked.sort(
            key=lambda path: (
                0 if path.source_stage == "external_graphapi_expansion" else 1,
                0 if path.matched_relations else 1,
                -len(path.matched_relations),
                len(path.triples),
                path.text,
            )
        )
        return ranked

    def _aggregate_subquestion_summaries(
        self,
        question: str,
        subquestion_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge sub-question summaries into one aggregate evidence object."""
        key_triples: list[dict[str, str]] = []
        evidence: list[str] = []
        seen_triples: set[tuple[str, str, str]] = set()
        seen_evidence: set[str] = set()
        sub_question_ids: list[str] = []
        answered_sub_question_ids: list[str] = []

        for item in subquestion_summaries:
            sub_question_id = str(item.get("sub_question_id") or "").strip()
            if sub_question_id and sub_question_id not in sub_question_ids:
                sub_question_ids.append(sub_question_id)
            if sub_question_id and (item.get("key_triples") or item.get("evidence")) and sub_question_id not in answered_sub_question_ids:
                answered_sub_question_ids.append(sub_question_id)
            for triple in item.get("key_triples", []):
                if not isinstance(triple, dict):
                    continue
                triple_key = (
                    str(triple.get("head") or ""),
                    str(triple.get("relation") or ""),
                    str(triple.get("tail") or ""),
                )
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)
                key_triples.append(
                    {
                        "head": triple_key[0],
                        "relation": triple_key[1],
                        "tail": triple_key[2],
                    }
                )
            for line in item.get("evidence", []):
                evidence_line = str(line).strip()
                if not evidence_line or evidence_line in seen_evidence:
                    continue
                seen_evidence.add(evidence_line)
                evidence.append(evidence_line)

        return {
            "question": question,
            "question_focus": question,
            "summary_type": "aggregate",
            "sub_question_ids": sub_question_ids,
            "answered_sub_question_ids": answered_sub_question_ids,
            "key_triples": key_triples,
            "evidence": evidence,
        }

    def _prune_paths_for_context(
        self,
        question: str,
        question_analysis: QuestionAnalysisResult,
        candidate_paths: list[ReasoningPath],
        w1: int,
        wmax: int,
        relation_hints: list[str],
        sample_id: str | None,
        trace: list[SearchTraceStep],
        depth: int,
        stage_note: str,
    ) -> list[ReasoningPath]:
        """Run pruning and append a trace entry for the current context."""
        LOGGER.info(
            "Pruning input sample_id=%s candidate_count=%d relation_hints=%s paths=%s",
            sample_id,
            len(candidate_paths),
            relation_hints,
            _path_preview_payload(candidate_paths),
        )
        beam_stages = {"relation_beam", CVT_BUNDLE_SOURCE_STAGE}
        relation_beam_paths = [path for path in candidate_paths if path.source_stage in beam_stages]
        non_beam_paths = [path for path in candidate_paths if path.source_stage not in beam_stages]
        use_relation_beam_consolidation = (
            self._retrieval_subquestion_explorer() == "relation_beam" and bool(relation_beam_paths)
        )
        if use_relation_beam_consolidation:
            preserved_supplement = deduplicate_paths(relation_beam_paths + non_beam_paths)
            pruned_paths = prune_relation_beam_paths(
                candidate_paths=preserved_supplement,
                wmax=wmax,
            )
            stage_note = (
                f"{stage_note} mode=relation_beam_evidence_consolidation "
                f"mixed_sources={bool(non_beam_paths)} relation_beam_paths={len(relation_beam_paths)} "
                f"fallback_paths={len(non_beam_paths)}"
            )
        else:
            pruned_paths = prune_paths(
                question=question,
                question_analysis=question_analysis,
                candidate_paths=candidate_paths,
                w1=w1,
                wmax=wmax,
                relation_hints=relation_hints,
                sample_id=sample_id,
                compact_precise_path_prompt=self._use_compact_precise_path_prompt(),
                llm=self.llm,
            )
        trace.append(
            SearchTraceStep(
                stage="path_pruning",
                depth=depth,
                candidate_count=len(candidate_paths),
                pruned_count=len(pruned_paths),
                note=stage_note,
            )
        )
        return pruned_paths

    def _summarize_paths_for_context(
        self,
        question: str,
        topic_entities: list[str],
        question_analysis: QuestionAnalysisResult,
        pruned_paths: list[ReasoningPath],
        trace: list[SearchTraceStep],
        depth: int,
        stage_note: str,
        sub_question: SubQuestionSpec | None = None,
        evidence_bank: list[dict[str, Any]] | None = None,
        plan_step_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Summarize pruned paths and append a trace entry for the current context."""
        summarized_paths = summarize_paths(
            question=question,
            topic_entities=topic_entities,
            question_analysis=question_analysis,
            pruned_paths=pruned_paths,
            llm=self.llm,
            sub_question=sub_question,
            evidence_bank=evidence_bank,
            plan_step_context=plan_step_context,
        )
        if (
            sub_question is not None
            and self._paths_need_conservative_summary(pruned_paths)
            and not self._summaries_have_fact_evidence(summarized_paths)
        ):
            summarized_paths = [
                {
                    "question": question,
                    "question_focus": question,
                    "summary_type": "sub_question",
                    "sub_question_id": sub_question.id,
                    "key_triples": [],
                    "evidence": [],
                }
            ]
        trace.append(
            SearchTraceStep(
                stage="subquestion_summarization" if sub_question is not None else "path_summarization",
                depth=depth,
                candidate_count=len(pruned_paths),
                pruned_count=len(summarized_paths),
                note=stage_note,
            )
        )
        return [item for item in summarized_paths if isinstance(item, dict)]

    def _paths_need_conservative_summary(self, pruned_paths: list[ReasoningPath]) -> bool:
        """Suppress summary evidence when preserved paths are disconnected from selected relations."""
        return bool(pruned_paths) and not any(path.matched_relations for path in pruned_paths)

    def _summaries_have_fact_evidence(self, summarized_paths: list[Any]) -> bool:
        """Return True when summarized outputs already contain concrete triples or evidence lines."""
        for item in summarized_paths:
            if not isinstance(item, dict):
                continue
            key_triples = item.get("key_triples", [])
            evidence = item.get("evidence", [])
            if key_triples or evidence:
                return True
        return False

    def _answer_from_summaries(
        self,
        question: str,
        topic_entities: list[str],
        question_analysis: QuestionAnalysisResult,
        dmax: int,
        depth: int,
        summarized_paths: list[dict[str, Any]],
        trace: list[SearchTraceStep],
        exploration_hints: ExplorationHints | None = None,
        agentic_state: AgenticRunState | None = None,
    ) -> tuple[object, bool]:
        """Run final sufficiency checking and answer generation over prepared summaries."""
        sufficiency = check_sufficiency(
            question=question,
            topic_entities=topic_entities,
            question_analysis=question_analysis,
            dmax=dmax,
            dpredict=depth,
            split_questions=question_analysis.split_questions,
            summarized_paths=summarized_paths,
            exploration_hints=exploration_hints,
            llm=self.llm,
            agentic_state=agentic_state,
        )
        sufficient = bool(sufficiency.get("sufficient", False))
        answer_result = generate_answer(
            question=question,
            topic_entities=topic_entities,
            question_analysis=question_analysis,
            dmax=dmax,
            dpredict=depth,
            split_questions=question_analysis.split_questions,
            summarized_paths=summarized_paths,
            exploration_hints=exploration_hints,
            llm=self.llm,
            agentic_state=agentic_state,
        )
        answer_result = self._filter_final_answer_result_by_type(
            graph_api=self.graph_api,
            question_analysis=question_analysis,
            answer_result=answer_result,
        )
        note = str(sufficiency.get("reason", ""))
        aggregate_summary = next(
            (
                item
                for item in summarized_paths
                if isinstance(item, dict) and item.get("summary_type") == "aggregate"
            ),
            {},
        )
        answered_sub_question_ids = aggregate_summary.get("answered_sub_question_ids", [])
        if (
            not sufficient
            and answer_result.predicted_answers
            and answered_sub_question_ids
            and len(answered_sub_question_ids) < len(aggregate_summary.get("sub_question_ids", []))
        ):
            note = f"{note} partial-answer from {answered_sub_question_ids}".strip()
        trace.append(
            SearchTraceStep(
                stage="question_answering",
                depth=depth,
                candidate_count=len(summarized_paths),
                pruned_count=len(summarized_paths),
                sufficient=sufficient,
                note=note,
            )
        )
        return answer_result, sufficient

    def _evaluate_paths(
        self,
        question: str,
        topic_entities: list[str],
        question_analysis: QuestionAnalysisResult,
        candidate_paths: list[ReasoningPath],
        w1: int,
        wmax: int,
        relation_hints: list[str],
        sample_id: str | None,
        trace: list[SearchTraceStep],
        dmax: int,
        depth: int,
        exploration_hints: ExplorationHints | None = None,
        agentic_state: AgenticRunState | None = None,
    ) -> tuple[list[ReasoningPath], list[dict], object, bool]:
        """Prune, summarize, and answer over the current candidate path set."""
        pruned_paths = self._prune_paths_for_context(
            question=question,
            question_analysis=question_analysis,
            candidate_paths=candidate_paths,
            w1=w1,
            wmax=wmax,
            relation_hints=relation_hints,
            sample_id=sample_id,
            trace=trace,
            depth=depth,
            stage_note="Embedding + LLM pruning.",
        )
        summarized_paths = self._summarize_paths_for_context(
            question=question,
            topic_entities=topic_entities,
            question_analysis=question_analysis,
            pruned_paths=pruned_paths,
            trace=trace,
            depth=depth,
            stage_note="Question-level path summarization.",
        )
        answer_result, sufficient = self._answer_from_summaries(
            question=question,
            topic_entities=topic_entities,
            question_analysis=question_analysis,
            dmax=dmax,
            depth=depth,
            summarized_paths=summarized_paths,
            trace=trace,
            exploration_hints=exploration_hints,
            agentic_state=agentic_state,
        )
        if not sufficient:
            answer_result = self._apply_best_effort_answer_selection(
                question_analysis=question_analysis,
                pruned_paths=pruned_paths,
                candidate_paths=candidate_paths,
                summarized_paths=summarized_paths,
                answer_result=answer_result,
                agentic_state=agentic_state,
                trace=trace,
                depth=depth,
            )
            answer_result = self._filter_final_answer_result_by_type(
                graph_api=self.graph_api,
                question_analysis=question_analysis,
                answer_result=answer_result,
            )
        return pruned_paths, summarized_paths, answer_result, sufficient

    def _build_pipeline_result(
        self,
        question: str,
        topic_entities: list[str],
        question_analysis: QuestionAnalysisResult,
        dmax: int,
        dpredict: int,
        candidate_paths: list[ReasoningPath],
        pruned_paths: list[ReasoningPath],
        summarized_paths: list[dict],
        answer: str,
        sufficient: bool,
        confidence: str,
        supplement_hints: SupplementHints | None = None,
        exploration_hints: ExplorationHints | None = None,
        sample_id: str | None = None,
        gold_answers: list[str] | None = None,
        predicted_answers: list[str] | None = None,
        search_trace: list[SearchTraceStep] | None = None,
        alignment_debug: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        """Build a PipelineResult with consistent defaults for optional hint structures."""
        return PipelineResult(
            question=question,
            topic_entities=topic_entities,
            question_analysis=question_analysis,
            dmax=dmax,
            dpredict=dpredict,
            candidate_paths=candidate_paths,
            pruned_paths=pruned_paths,
            summarized_paths=summarized_paths,
            answer=answer,
            sufficient=sufficient,
            confidence=confidence,
            supplement_hints=supplement_hints or SupplementHints(),
            exploration_hints=exploration_hints or ExplorationHints(),
            sample_id=sample_id,
            gold_answers=list(gold_answers or []),
            predicted_answers=list(predicted_answers or []),
            search_trace=list(search_trace or []),
            alignment_debug=list(alignment_debug or []),
        )

    def _build_graph_api(self) -> BaseGraphAPI | None:
        """Instantiate the optional external graph API if enabled."""
        graphapi_cfg = self.config.get("graphapi", {})
        if not bool(graphapi_cfg.get("enabled", False)):
            LOGGER.info("GraphAPI disabled by config.")
            return None
        backend = str(graphapi_cfg.get("backend", "sqlite")).strip().lower()
        if backend != "sqlite":
            raise ValueError(f"Unsupported graphapi.backend={backend!r}. Only 'sqlite' is supported.")
        db_path = str(graphapi_cfg.get("db_path", "")).strip()
        if not db_path:
            db_path = str(graphapi_cfg.get("fb2m_db_path", "")).strip()
        if not db_path:
            raise ValueError("graphapi.enabled=true requires graphapi.db_path or graphapi.fb2m_db_path")
        LOGGER.info("GraphAPI enabled backend=%s db_path=%s", backend, db_path)
        retrieval_cfg = self.config.get("retrieval", {})
        retrieval_backend = str(retrieval_cfg.get("backend", "indexed_sqlite")).strip().lower()
        if retrieval_backend == "indexed_sqlite":
            index_dir = str(retrieval_cfg.get("index_dir", "")).strip() or None
            return IndexedSQLiteGraphBackend(
                db_path=db_path,
                index_dir=index_dir,
                neighbor_limit=self._graphapi_neighbor_limit(),
            )
        return SQLiteGraphAPI(
            db_path=db_path,
            neighbor_limit=self._graphapi_neighbor_limit(),
        )

    def _graphapi_enabled_for_stage(self, stage: str) -> bool:
        """Return whether graphapi is enabled for one exploration stage."""
        if self.graph_api is None:
            return False
        graphapi_cfg = self.config.get("graphapi", {})
        stages = [str(item).strip() for item in graphapi_cfg.get("augment_stages", [])]
        return stage in stages

    def _graphapi_neighbor_limit(self) -> int:
        """Return the configured external graph neighbor limit."""
        graphapi_cfg = self.config.get("graphapi", {})
        return int(graphapi_cfg.get("neighbor_limit", 25))

    def _sqlite_entity_candidate_recall_top_k(self) -> int:
        """Return the candidate recall width for SQLite entity disambiguation."""
        retrieval_cfg = self.config.get("retrieval", {})
        return max(3, int(retrieval_cfg.get("entity_candidate_top_k", 10)))

    def _sqlite_entity_candidate_typed_recall_top_k(self) -> int:
        """Return a widened recall width for mentions with explicit expected types."""
        retrieval_cfg = self.config.get("retrieval", {})
        base_top_k = self._sqlite_entity_candidate_recall_top_k()
        configured = retrieval_cfg.get("entity_candidate_typed_top_k")
        if configured is None:
            return max(base_top_k, min(max(base_top_k * 6, 60), 150))
        return max(base_top_k, int(configured))

    def _sqlite_entity_candidate_selected_top_k(self) -> int:
        """Return the final number of SQLite entity candidates to keep after reranking."""
        retrieval_cfg = self.config.get("retrieval", {})
        return max(1, int(retrieval_cfg.get("entity_candidate_selected_top_k", 5)))

    def _graphapi_one_hop_only(self) -> bool:
        """Return whether external graph retrieval should stop after one hop."""
        graphapi_cfg = self.config.get("graphapi", {})
        return bool(graphapi_cfg.get("one_hop_only", False))

    def _graphapi_max_depth(self) -> int:
        """Return the default expansion depth for question-level external retrieval."""
        return 1 if self._graphapi_one_hop_only() else 2

    def _graphapi_subquestion_max_depth(self) -> int:
        """Return the expansion depth for strict sub-question retrieval."""
        return 1 if self._graphapi_one_hop_only() else 3

    def _graphapi_relation_bias_top_k(self) -> int:
        """Return the number of relation hints to bias external graph queries with."""
        graphapi_cfg = self.config.get("graphapi", {})
        return max(1, int(graphapi_cfg.get("relation_bias_top_k", 5)))

    def _graphapi_local_relation_candidate_limit(self) -> int:
        """Return the number of local one-hop edges inspected for relation probing."""
        graphapi_cfg = self.config.get("graphapi", {})
        return max(10, int(graphapi_cfg.get("local_relation_candidate_limit", self._graphapi_neighbor_limit() * 4)))

    def _cwq_max_paths(self) -> int:
        """Return the maximum number of CWQ graphapi paths to expand before pruning."""
        graphapi_cfg = self.config.get("graphapi", {})
        return max(1, int(graphapi_cfg.get("cwq_max_paths", self._graphapi_neighbor_limit())))

    def _cwq_max_nodes(self) -> int:
        """Return the maximum number of distinct CWQ graph nodes to visit during expansion."""
        graphapi_cfg = self.config.get("graphapi", {})
        return max(1, int(graphapi_cfg.get("cwq_max_nodes", self._graphapi_neighbor_limit())))

    def _retrieval_default_strategy(self) -> str:
        """Return the configured retrieval strategy name."""
        retrieval_cfg = self.config.get("retrieval", {})
        search_cfg = retrieval_cfg.get("search", {})
        return str(search_cfg.get("default_strategy", "hybrid")).strip().lower()

    def _retrieval_subquestion_explorer(self) -> str:
        """Return which sub-question explorer implementation should run."""
        retrieval_cfg = self.config.get("retrieval", {})
        search_cfg = retrieval_cfg.get("search", {})
        return str(search_cfg.get("subquestion_explorer", "legacy")).strip().lower()

    def _relation_beam_config(self) -> RelationBeamConfig:
        """Build the optional relation-beam configuration from runtime config."""
        retrieval_cfg = self.config.get("retrieval", {})
        beam_cfg = retrieval_cfg.get("relation_beam", {})
        return RelationBeamConfig(
            max_hops=max(1, int(beam_cfg.get("max_hops", 3))),
            beam_width=max(1, int(beam_cfg.get("beam_width", 6))),
            relations_per_state=max(1, int(beam_cfg.get("relations_per_state", 3))),
            relation_retrieval_top_k=max(1, int(beam_cfg.get("relation_retrieval_top_k", 20))),
            neighbors_per_source=max(1, int(beam_cfg.get("neighbors_per_source", 40))),
            max_nodes_per_path=max(1, int(beam_cfg.get("max_nodes_per_path", 50))),
            llm_path_rerank_top_k=max(1, int(beam_cfg.get("llm_path_rerank_top_k", 8))),
            answer_threshold=float(beam_cfg.get("answer_threshold", 0.82)),
            continue_threshold=float(beam_cfg.get("continue_threshold", 0.45)),
            llm_relation_batch_size=max(1, int(beam_cfg.get("llm_relation_batch_size", 100))),
        )

    def _retrieval_beam_width(self) -> int:
        """Return the configured retrieval beam width."""
        retrieval_cfg = self.config.get("retrieval", {})
        beam_cfg = retrieval_cfg.get("beam", {})
        return int(beam_cfg.get("width", self._graphapi_neighbor_limit()))

    def _retrieval_max_expansions(self) -> int:
        """Return the configured retrieval expansion cap."""
        retrieval_cfg = self.config.get("retrieval", {})
        search_cfg = retrieval_cfg.get("search", {})
        return max(1, int(search_cfg.get("max_expansions", self._graphapi_neighbor_limit() * 20)))

    def _retrieval_max_paths(self) -> int:
        """Return the configured retrieval path cap."""
        retrieval_cfg = self.config.get("retrieval", {})
        search_cfg = retrieval_cfg.get("search", {})
        return int(search_cfg.get("max_paths", self._cwq_max_paths()))

    def _retrieval_literal_policy(self) -> str:
        """Return the configured literal expansion policy."""
        retrieval_cfg = self.config.get("retrieval", {})
        search_cfg = retrieval_cfg.get("search", {})
        return str(search_cfg.get("literal_policy", "stop")).strip().lower()

    def _retrieval_high_degree_policy(self) -> str:
        """Return the configured high-degree node policy."""
        retrieval_cfg = self.config.get("retrieval", {})
        search_cfg = retrieval_cfg.get("search", {})
        return str(search_cfg.get("high_degree_policy", "penalize")).strip().lower()

    def _agentic_enabled(self) -> bool:
        """Return whether the agentic step loop should be the default execution mode."""
        agentic_cfg = self.config.get("agentic", {})
        return bool(agentic_cfg.get("enabled", True))

    def _agentic_max_loops(self) -> int:
        """Return the maximum number of planner-retrieval iterations."""
        agentic_cfg = self.config.get("agentic", {})
        return max(1, int(agentic_cfg.get("max_loops", 5)))

    def _agentic_max_carryover_entities(self) -> int:
        """Return the carryover entity frontier cap between steps."""
        agentic_cfg = self.config.get("agentic", {})
        return max(1, int(agentic_cfg.get("max_carryover_entities", 5)))

    def _agentic_max_carryover_text_values(self) -> int:
        """Return the carryover text-constraint cap between steps."""
        agentic_cfg = self.config.get("agentic", {})
        return max(1, int(agentic_cfg.get("max_carryover_text_values", 5)))

    def _agentic_stop_on_no_new_evidence_rounds(self) -> int:
        """Return the number of no-progress rounds allowed before stopping."""
        agentic_cfg = self.config.get("agentic", {})
        return max(1, int(agentic_cfg.get("stop_on_no_new_evidence_rounds", 2)))

    def _agentic_local_relation_probe_top_k(self) -> int:
        """Return the number of local relation ids selected for strict probing."""
        agentic_cfg = self.config.get("agentic", {})
        return max(1, int(agentic_cfg.get("local_relation_probe_top_k", 2)))

    def _agentic_branching_path_threshold(self) -> int:
        """Return the candidate-path count that triggers branching-aware retry logic."""
        agentic_cfg = self.config.get("agentic", {})
        return max(4, int(agentic_cfg.get("branching_path_threshold", 12)))

    def _agentic_large_dependency_seed_threshold(self) -> int:
        """Return the dependency-seed count that triggers downstream-filter pushdown retry."""
        agentic_cfg = self.config.get("agentic", {})
        return max(2, int(agentic_cfg.get("large_dependency_seed_threshold", 4)))

    def _local_relation_probe_preserve_beam(self) -> bool:
        """Keep beam-selected relations when they already provide executable evidence."""
        agentic_cfg = self.config.get("agentic", {})
        probe_cfg = agentic_cfg.get("local_relation_probe", {})
        if isinstance(probe_cfg, dict):
            return bool(probe_cfg.get("preserve_beam_when_available", True))
        return True

    def _local_relation_probe_allow_beam_supplement(self) -> bool:
        """Allow non-beam local probe relations to supplement, but not replace, beam relations."""
        agentic_cfg = self.config.get("agentic", {})
        probe_cfg = agentic_cfg.get("local_relation_probe", {})
        if isinstance(probe_cfg, dict):
            return bool(probe_cfg.get("allow_supplemental_non_beam_relations", True))
        return True

    @staticmethod
    def _supplement_relation_hints(supplement_hints: SupplementHints) -> list[str]:
        """Prefer linked relation hints, but fall back to raw candidate strings when needed."""
        return supplement_hints.linked_relations or supplement_hints.relation_candidates

    def _use_compact_precise_path_prompt(self) -> bool:
        """Return whether precise path selection should use compact path strings."""
        pruning_cfg = self.config.get("pruning", {})
        return bool(pruning_cfg.get("compact_precise_path_prompt", False))

    def _format_aligned_entities(self, entity_ids: list[str]) -> list[str]:
        """Return readable aligned entity strings for logs and output payloads."""
        if self.graph_api is None:
            return []
        return [
            f"{self.graph_api.get_entity_display_name(entity_id)} [{entity_id}]"
            for entity_id in entity_ids
        ]

    def _format_aligned_relations(self, relation_ids: list[str]) -> list[str]:
        """Return readable aligned relation strings for logs and output payloads."""
        if self.graph_api is None:
            return list(relation_ids)
        return [
            f"{self.graph_api.get_relation_display_name(relation_id)} [{relation_id}]"
            for relation_id in relation_ids
        ]

    @staticmethod
    def _deduplicate_strings(values: list[str]) -> list[str]:
        """Deduplicate strings while preserving order."""
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped
