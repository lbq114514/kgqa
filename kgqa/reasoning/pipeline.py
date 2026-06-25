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
    AnswerGrounding,
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


@dataclass(frozen=True)
class ConstraintPlan:
    """Path-level constraints derived from interested nodes."""

    target_entity_ids: tuple[str, ...] = ()
    target_texts: tuple[str, ...] = ()
    relation_markers: tuple[str, ...] = ()


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
                final_grounding=self._grounding_to_dict(answer_result.grounding),
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
                final_grounding=self._grounding_to_dict(answer_result.grounding),
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
            final_grounding=self._grounding_to_dict(answer_result.grounding),
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
                                key: self._dependency_payload_seed_entity_ids(
                                    payload=value,
                                    graph_api=graph_api,
                                )
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
                        constraint_target_ids=(
                            list(resolved_sub_question.constraint_target_ids)
                            if resolved_sub_question is not None
                            else []
                        ),
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
                            constraint_target_ids=(
                                list(resolved_sub_question.constraint_target_ids)
                                if resolved_sub_question is not None
                                else []
                            ),
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
                    candidate_paths = self._normalize_terminal_cvt_paths(
                        graph_api=graph_api,
                        candidate_paths=candidate_paths,
                        subquestion_text=sub_question.question,
                        expected_answer_type=sub_question.expected_answer_type,
                        relation_hint_names=effective_relation_ids,
                        anchor_mentions=self._deduplicate_strings(
                            [
                                sub_question.question,
                                *[entity.name for entity in self._effective_sub_question_topics(sub_question) if entity.name],
                                *[node.name for node in sub_question.interested_nodes if node.name],
                                *[relation.name for relation in sub_question.interested_relations if relation.name],
                                *[
                                    alias
                                    for relation in sub_question.interested_relations
                                    for alias in relation.aliases
                                    if alias
                                ],
                            ]
                        ),
                        trace=trace,
                        depth=min(dmax, loop_index),
                        stage_note=f"sub_question_id={sub_question.id}",
                    )
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
                    sub_answer_result = self._apply_sufficiency_answer_hints(
                        sub_answer_result,
                        sub_sufficiency,
                    )
                    sub_answer_result = self._backfill_subquestion_answer(
                        sub_question=sub_question,
                        pruned_paths=pruned_paths,
                        step_summaries=step_summaries,
                        answer_result=sub_answer_result,
                    )
                    grounding, type_filter_stats = self._canonicalize_subquestion_outputs(
                        graph_api=graph_api,
                        sub_question=sub_question,
                        pruned_paths=pruned_paths,
                        step_summaries=step_summaries,
                        answer_result=sub_answer_result,
                        structured_outputs=solver_result.structured_outputs,
                        primary_entity_ids=solver_result.primary_entity_ids,
                        primary_literals=solver_result.primary_literals,
                    )
                    self._project_grounding_to_answer_result(sub_answer_result, grounding)
                    output_entities, output_literals, projected_answer = self._project_grounding_to_payload(
                        grounding
                    )
                    projected_predicted_answers = self._grounding_candidate_answers(grounding)
                    has_grounded_payload = self._grounding_has_answer_payload(grounding)
                    if self._is_propagatable_text(projected_answer):
                        sub_answer_result.answer = projected_answer
                    sub_answer_result.reason = str(sub_sufficiency.get("reason", ""))
                    type_eval = self._evaluate_subquestion_output_type(
                        graph_api=graph_api,
                        sub_question=sub_question,
                        entity_ids=output_entities,
                        literals=output_literals,
                        answer_result=sub_answer_result,
                        step_summaries=step_summaries,
                    )
                    output_type_ok = bool(type_eval.get("passed", False))
                    soft_type_ok = bool(type_eval.get("soft_pass", False))
                    boolean_type_bypass = self._should_bypass_boolean_subquestion_type_check(
                        sub_question=sub_question,
                        answer_result=sub_answer_result,
                        sufficiency=sub_sufficiency,
                    )
                    if boolean_type_bypass and not output_type_ok:
                        output_type_ok = True
                        soft_type_ok = True
                        type_eval = {
                            **dict(type_eval),
                            "passed": True,
                            "soft_pass": True,
                            "mode": "boolean_sufficiency_bypass",
                        }
                    LOGGER.info(
                        "Sub-question type check sub_question_id=%s expected_type=%s passed=%s mode=%s "
                        "hard_match=%s relation_score=%.3f embedding_score=%.3f candidate_ids=%s labels=%s",
                        sub_question.id,
                        sub_question.expected_answer_type,
                        output_type_ok,
                        type_eval.get("mode", ""),
                        bool(type_eval.get("hard_match", False)),
                        float(type_eval.get("relation_score", 0.0)),
                        float(type_eval.get("embedding_score", 0.0)),
                        output_entities[:3],
                        output_literals[:3],
                    )

                    sub_resolved = output_type_ok and (
                        (
                            bool(sub_sufficiency.get("sufficient", False))
                            and has_grounded_payload
                        )
                        or (
                            has_grounded_payload
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
                                    f"{sub_answer_result.reason} Soft type pass-through accepted for expected_answer_type={sub_question.expected_answer_type}. "
                                    f"type_mode={type_eval.get('mode', '')} relation_score={float(type_eval.get('relation_score', 0.0)):.3f} "
                                    f"embedding_score={float(type_eval.get('embedding_score', 0.0)):.3f}."
                                    if soft_type_ok
                                    else f"{sub_answer_result.reason} Type check failed for expected_answer_type={sub_question.expected_answer_type}. "
                                    f"type_mode={type_eval.get('mode', '')} relation_score={float(type_eval.get('relation_score', 0.0)):.3f} "
                                    f"embedding_score={float(type_eval.get('embedding_score', 0.0)):.3f}."
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
                        candidate_answers=list(projected_predicted_answers),
                        sub_answer=sub_answer_result.answer,
                        sub_answer_entities=list(output_entities),
                        sub_answer_literals=list(output_literals),
                        sub_answer_grounding=self._grounding_to_dict(grounding),
                        status=step_status,
                        attempt_count=attempt_index,
                        depends_on_step_ids=list(sub_question.depends_on),
                        failure_reason="" if sub_resolved else sub_answer_result.reason,
                        carryover_entities=list(output_entities),
                        carryover_text_values=list(output_literals),
                        is_step_resolved=sub_resolved,
                        solver_type=solver_result.solver_type,
                        solver_debug={**dict(solver_result.solver_debug), **type_filter_stats, "type_check": dict(type_eval)},
                    )
                    run_state.step_history.append(step_result)

                    if sub_resolved:
                        step_summaries = [
                            {**item, "grounding": self._grounding_to_dict(grounding)}
                            if isinstance(item, dict)
                            else item
                            for item in step_summaries
                        ]
                        run_state.evidence_bank.extend(step_summaries)
                        run_state.resolved_sub_answers[sub_question.id] = {
                            "answer": sub_answer_result.answer,
                            "predicted_answers": list(projected_predicted_answers),
                            "entity_ids": list(output_entities),
                            "literals": list(output_literals),
                            "grounding": self._grounding_to_dict(grounding),
                            "supporting_paths": list(sub_answer_result.supporting_paths),
                            "evidence_relations": self._collect_summary_relation_ids(step_summaries),
                            "evidence_text": self._collect_summary_evidence_text(step_summaries),
                            "depends_on": list(sub_question.depends_on),
                            "solver_type": solver_result.solver_type,
                            "solver_reason": sub_question.solver_reason,
                            "solver_debug": {**dict(solver_result.solver_debug), "type_check": dict(type_eval)},
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
            can_finalize_partial = False

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
                final_grounding=self._grounding_to_dict(final_answer_result.grounding),
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
            grounding=self._empty_answer_grounding(),
        )

    @staticmethod
    def _empty_answer_grounding() -> AnswerGrounding:
        """Return an empty canonical answer-grounding object."""
        return AnswerGrounding()

    def _grounding_to_dict(self, grounding: AnswerGrounding | None) -> dict[str, Any]:
        """Serialize grounding to a stable dictionary for JSON outputs."""
        if grounding is None:
            return {}
        return grounding.to_dict()

    def _grounding_from_payload(self, payload: dict[str, Any] | None) -> AnswerGrounding:
        """Read grounding from a resolved-sub-answer payload when available."""
        if not isinstance(payload, dict):
            return self._empty_answer_grounding()
        raw = payload.get("grounding")
        if not isinstance(raw, dict):
            return self._empty_answer_grounding()
        return AnswerGrounding(
            answer_texts=[str(item).strip() for item in raw.get("answer_texts", []) if str(item).strip()],
            entity_ids=[str(item).strip() for item in raw.get("entity_ids", []) if str(item).strip()],
            entity_labels=[str(item).strip() for item in raw.get("entity_labels", []) if str(item).strip()],
            literal_values=[str(item).strip() for item in raw.get("literal_values", []) if str(item).strip()],
            primary_answer_text=str(raw.get("primary_answer_text") or "").strip(),
            primary_entity_id=str(raw.get("primary_entity_id") or "").strip(),
            answer_view_triples=[
                {
                    "head": str(item.get("head") or "").strip(),
                    "relation": str(item.get("relation") or "").strip(),
                    "tail": str(item.get("tail") or "").strip(),
                }
                for item in raw.get("answer_view_triples", [])
                if isinstance(item, dict)
            ],
            answer_view_facts=[
                {
                    "head_id": str(item.get("head_id") or "").strip(),
                    "head_label": str(item.get("head_label") or "").strip(),
                    "relation_id": str(item.get("relation_id") or "").strip(),
                    "tail_id": str(item.get("tail_id") or "").strip(),
                    "tail_label": str(item.get("tail_label") or "").strip(),
                    "tail_kind": str(item.get("tail_kind") or "id").strip(),
                }
                for item in raw.get("answer_view_facts", [])
                if isinstance(item, dict)
            ],
            supporting_relation_ids=[
                str(item).strip() for item in raw.get("supporting_relation_ids", []) if str(item).strip()
            ],
            source_mode=str(raw.get("source_mode") or "").strip(),
            confidence=str(raw.get("confidence") or "").strip(),
            debug=dict(raw.get("debug", {})) if isinstance(raw.get("debug", {}), dict) else {},
        )

    def _project_grounding_to_answer_result(
        self,
        answer_result: AnswerResult,
        grounding: AnswerGrounding | None,
    ) -> AnswerResult:
        """Project canonical grounding back to legacy AnswerResult fields."""
        grounding = grounding or self._empty_answer_grounding()
        answer_result.grounding = grounding
        answer_texts = self._deduplicate_strings(
            [value for value in grounding.answer_texts if self._is_propagatable_text(value)]
        )
        literal_values = self._deduplicate_strings(
            [value for value in grounding.literal_values if self._is_propagatable_text(value)]
        )
        entity_labels = self._deduplicate_strings(
            [value for value in grounding.entity_labels if self._is_propagatable_text(value)]
        )
        answer_result.predicted_answers = list(answer_texts)
        answer_result.resolved_entity_mentions = list(entity_labels or answer_texts)
        answer_result.resolved_literals = self._deduplicate_strings(answer_texts + literal_values)
        if self._is_propagatable_text(grounding.primary_answer_text):
            answer_result.answer = grounding.primary_answer_text
        elif answer_texts:
            answer_result.answer = answer_texts[0]
        elif literal_values:
            answer_result.answer = literal_values[0]
        elif not self._is_propagatable_text(answer_result.answer):
            answer_result.answer = "insufficient"
        return answer_result

    def _project_grounding_to_payload(
        self,
        grounding: AnswerGrounding | None,
    ) -> tuple[list[str], list[str], str]:
        """Return legacy entity_ids/literals/answer view from grounding."""
        grounding = grounding or self._empty_answer_grounding()
        entity_ids = self._deduplicate_strings(grounding.entity_ids)
        literals = self._deduplicate_strings(
            [*grounding.answer_texts, *grounding.literal_values]
        )
        answer = grounding.primary_answer_text or (literals[0] if literals else "")
        return entity_ids, literals, answer

    def _grounding_candidate_answers(
        self,
        grounding: AnswerGrounding | None,
    ) -> list[str]:
        """Return answer-facing candidate texts derived only from canonical grounding."""
        grounding = grounding or self._empty_answer_grounding()
        return self._deduplicate_strings(
            [
                *[
                    value
                    for value in grounding.answer_texts
                    if self._is_propagatable_text(value)
                ],
                *[
                    value
                    for value in grounding.literal_values
                    if self._is_propagatable_text(value)
                ],
                *[
                    value
                    for value in grounding.entity_labels
                    if self._is_propagatable_text(value)
                ],
            ]
        )

    def _answer_constraint_texts_for_grounding(
        self,
        answer_result: AnswerResult,
        primary_literals: list[str] | None = None,
    ) -> list[str]:
        """Return answer-facing texts that downstream grounding must align to."""
        values = [
            str(answer_result.answer or "").strip(),
            *[str(value).strip() for value in getattr(answer_result, "predicted_answers", []) if str(value).strip()],
            *[str(value).strip() for value in getattr(answer_result, "resolved_literals", []) if str(value).strip()],
            *[str(value).strip() for value in getattr(answer_result, "resolved_entity_mentions", []) if str(value).strip()],
            *[str(value).strip() for value in (primary_literals or []) if str(value).strip()],
        ]
        return self._sanitize_answer_candidate_texts(
            [value for value in values if self._is_propagatable_text(value)]
        )

    def _filter_grounding_pairs_by_answer_texts(
        self,
        ordered_pairs: list[tuple[str, str]],
        answer_constraint_texts: list[str],
    ) -> list[tuple[str, str]]:
        """Keep resolved entity candidates whose labels match the sub-answer text."""
        if not ordered_pairs or not answer_constraint_texts:
            return ordered_pairs
        normalized_answers = {
            self._normalize_relation_text(value)
            for value in answer_constraint_texts
            if self._normalize_relation_text(value)
        }
        if not normalized_answers:
            return ordered_pairs
        filtered = [
            (entity_id, label)
            for entity_id, label in ordered_pairs
            if self._normalize_relation_text(label) in normalized_answers
        ]
        return filtered or ordered_pairs

    def _filter_grounding_literals_by_answer_texts(
        self,
        values: list[str],
        answer_constraint_texts: list[str],
    ) -> list[str]:
        """Keep scalar grounding values aligned with the sub-answer text."""
        if not values or not answer_constraint_texts:
            return values
        normalized_answers = {
            self._normalize_relation_text(value)
            for value in answer_constraint_texts
            if self._normalize_relation_text(value)
        }
        if not normalized_answers:
            return values
        filtered = [
            value
            for value in values
            if self._normalize_relation_text(value) in normalized_answers
        ]
        return filtered or values

    def _grounding_has_answer_payload(
        self,
        grounding: AnswerGrounding | None,
    ) -> bool:
        """Return whether canonical grounding contains any answer-carrying payload."""
        grounding = grounding or self._empty_answer_grounding()
        return bool(
            grounding.entity_ids
            or grounding.answer_texts
            or grounding.literal_values
            or grounding.entity_labels
            or (grounding.primary_answer_text and self._is_propagatable_text(grounding.primary_answer_text))
        )

    def _apply_sufficiency_answer_hints(
        self,
        answer_result: AnswerResult,
        sufficiency: dict[str, Any] | None,
    ) -> AnswerResult:
        """Inject structured answer hints emitted by sufficiency checking."""
        if not isinstance(sufficiency, dict):
            return answer_result
        primary_answer = str(sufficiency.get("primary_answer") or "").strip()
        answer_candidates = self._deduplicate_strings(
            [
                primary_answer,
                *[
                    str(value).strip()
                    for value in sufficiency.get("answer_candidates", [])
                    if str(value).strip()
                ],
            ]
        )
        answer_candidates = [
            value for value in answer_candidates if self._is_propagatable_text(value)
        ]
        if not answer_candidates:
            return answer_result
        answer_result.predicted_answers = self._deduplicate_strings(
            [*answer_candidates, *answer_result.predicted_answers]
        )
        answer_result.resolved_literals = self._deduplicate_strings(
            [*answer_candidates, *answer_result.resolved_literals]
        )
        if self._is_propagatable_text(primary_answer):
            answer_result.answer = primary_answer
        elif answer_candidates and not self._is_propagatable_text(answer_result.answer):
            answer_result.answer = answer_candidates[0]
        return answer_result

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
            payload = run_state.resolved_sub_answers.get(dependency_id, {})
            grounding = self._grounding_from_payload(payload)
            dependency_literals.extend(grounding.answer_texts)
            dependency_literals.extend(grounding.literal_values)
            if not grounding.answer_texts and not grounding.literal_values:
                dependency_literals.extend(payload.get("literals", []))
                dependency_literals.extend(payload.get("predicted_answers", []))
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
            execution_hints={
                **dict(sub_question.execution_hints),
                "question_interested_nodes": [
                    item.to_dict() for item in sub_question.interested_nodes if item.name
                ],
            },
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
        constraint_target_ids: list[str],
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
            constraint_target_ids=list(constraint_target_ids),
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
        return self._sanitize_answer_candidate_texts(
            [str(label).strip() for label in labels if str(label).strip()]
        )

    def _collect_answer_view_triples(
        self,
        step_summaries: list[dict[str, Any]],
        pruned_paths: list[ReasoningPath],
    ) -> list[dict[str, str]]:
        """Collect answer-facing triples, preferring summary-provided answer-view triples."""
        triples: list[dict[str, str]] = []
        for item in step_summaries:
            if not isinstance(item, dict):
                continue
            for triple in self._summary_fact_triples(item):
                if not isinstance(triple, dict):
                    continue
                head = str(triple.get("head") or "").strip()
                relation = str(triple.get("relation") or "").strip()
                tail = str(triple.get("tail") or "").strip()
                if head and relation and tail:
                    triples.append({"head": head, "relation": relation, "tail": tail})
        if triples:
            deduped: list[dict[str, str]] = []
            seen: set[tuple[str, str, str]] = set()
            for triple in triples:
                key = (triple["head"], triple["relation"], triple["tail"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(triple)
            return deduped

        for path in pruned_paths:
            for triple in path.triples:
                head = str(triple.head or "").strip()
                relation = str(triple.relation or "").strip()
                tail = str(triple.tail or "").strip()
                if head and relation and tail:
                    triples.append({"head": head, "relation": relation, "tail": tail})
        deduped = []
        seen = set()
        for triple in triples:
            key = (triple["head"], triple["relation"], triple["tail"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(triple)
        return deduped

    def _collect_answer_view_facts(
        self,
        step_summaries: list[dict[str, Any]],
        pruned_paths: list[ReasoningPath],
    ) -> list[dict[str, str]]:
        """Collect answer-facing facts with stable ids, preferring summary-provided fact views."""
        facts: list[dict[str, str]] = []
        for item in step_summaries:
            if not isinstance(item, dict):
                continue
            raw_facts = item.get("answer_view_facts")
            if not isinstance(raw_facts, list):
                continue
            for fact in raw_facts:
                if not isinstance(fact, dict):
                    continue
                facts.append(
                    {
                        "head_id": str(fact.get("head_id") or "").strip(),
                        "head_label": str(fact.get("head_label") or "").strip(),
                        "relation_id": str(fact.get("relation_id") or "").strip(),
                        "tail_id": str(fact.get("tail_id") or "").strip(),
                        "tail_label": str(fact.get("tail_label") or "").strip(),
                        "tail_kind": str(fact.get("tail_kind") or "id").strip(),
                    }
                )
        if not facts:
            for path in pruned_paths:
                path_facts = getattr(path, "triple_facts", [])
                if path_facts:
                    for fact in path_facts:
                        facts.append(
                            {
                                "head_id": str(getattr(fact, "head_id", "") or "").strip(),
                                "head_label": str(getattr(fact, "head_label", "") or "").strip(),
                                "relation_id": str(getattr(fact, "relation_id", "") or "").strip(),
                                "tail_id": str(getattr(fact, "tail_id", "") or "").strip(),
                                "tail_label": str(getattr(fact, "tail_label", "") or "").strip(),
                                "tail_kind": str(getattr(fact, "tail_kind", "id") or "id").strip(),
                            }
                        )
                    continue
                for index, edge_id in enumerate(getattr(path, "edge_ids", [])):
                    parts = str(edge_id or "").split(":", 2)
                    if len(parts) != 3:
                        continue
                    head_id, relation_id, tail_id = (part.strip() for part in parts)
                    head_label = str(path.nodes[index] if index < len(path.nodes) else "").strip()
                    tail_label = str(path.nodes[index + 1] if index + 1 < len(path.nodes) else "").strip()
                    tail_kind = "id" if tail_id.startswith(("m.", "g.")) else "literal"
                    facts.append(
                        {
                            "head_id": head_id if head_id.startswith(("m.", "g.")) else "",
                            "head_label": head_label,
                            "relation_id": relation_id,
                            "tail_id": tail_id if tail_kind == "id" else "",
                            "tail_label": tail_label,
                            "tail_kind": tail_kind,
                        }
                    )
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for fact in facts:
            key = (fact["head_id"], fact["relation_id"], fact["tail_id"], fact["tail_label"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    def _resolve_grounding_entity_candidates(
        self,
        graph_api: Any,
        candidate_labels: list[str],
        candidate_entity_ids: list[str],
        expected_type: str,
    ) -> tuple[list[str], list[str], dict[str, int]]:
        """Resolve and type-filter entity candidates for one grounding object."""
        type_filter_stats = {
            "type_filter_before_count": len(candidate_entity_ids) or len(candidate_labels),
            "type_filter_after_count": len(candidate_entity_ids) or len(candidate_labels),
        }
        entity_ids = self._deduplicate_strings(candidate_entity_ids)
        labels = self._deduplicate_strings(candidate_labels)
        if self._filter_subquestion_candidates_enabled():
            entity_ids, labels, type_filter_stats = self._filter_candidates_by_expected_type(
                graph_api=graph_api,
                expected_type=expected_type,
                entity_ids=entity_ids,
                labels=labels,
            )
        return entity_ids, labels, type_filter_stats

    def _candidate_values_from_answer_view_facts(
        self,
        answer_view_facts: list[dict[str, str]],
    ) -> tuple[list[str], list[str], list[str]]:
        """Extract candidate entity ids, labels, and literals from fact tails."""
        entity_ids: list[str] = []
        entity_labels: list[str] = []
        literal_values: list[str] = []
        for fact in answer_view_facts:
            if not isinstance(fact, dict):
                continue
            tail_id = str(fact.get("tail_id") or "").strip()
            tail_label = str(fact.get("tail_label") or "").strip()
            tail_kind = str(fact.get("tail_kind") or "id").strip().lower()
            if tail_kind == "id" and tail_id:
                entity_ids.append(tail_id)
                if self._is_propagatable_text(tail_label):
                    entity_labels.append(tail_label)
            elif self._is_propagatable_text(tail_label):
                literal_values.append(tail_label)
        return (
            self._deduplicate_strings(entity_ids),
            self._deduplicate_strings(entity_labels),
            self._deduplicate_strings(literal_values),
        )

    def _build_subquestion_grounding(
        self,
        *,
        graph_api: Any,
        sub_question: SubQuestionSpec,
        pruned_paths: list[ReasoningPath],
        step_summaries: list[dict[str, Any]],
        answer_result: AnswerResult,
        structured_outputs: dict[str, Any] | None = None,
        primary_entity_ids: list[str] | None = None,
        primary_literals: list[str] | None = None,
    ) -> tuple[AnswerGrounding, dict[str, int]]:
        """Construct canonical answer grounding for one sub-question."""
        structured_outputs = structured_outputs or {}
        answer_view_triples = self._collect_answer_view_triples(step_summaries, pruned_paths)
        answer_view_facts = self._collect_answer_view_facts(step_summaries, pruned_paths)
        fact_entity_ids, fact_entity_labels, fact_literal_values = self._candidate_values_from_answer_view_facts(
            answer_view_facts
        )
        labels = self._collect_subquestion_output_labels(
            sub_question=sub_question,
            pruned_paths=pruned_paths,
            step_summaries=step_summaries,
            answer_result=answer_result,
            structured_outputs=structured_outputs,
            primary_literals=primary_literals,
        )
        answer_constraint_texts = self._answer_constraint_texts_for_grounding(
            answer_result=answer_result,
            primary_literals=primary_literals,
        )
        summary_tail_labels = self._deduplicate_strings(
            [
                str(triple.get("tail") or "").strip()
                for triple in answer_view_triples
                if isinstance(triple, dict) and self._is_propagatable_text(str(triple.get("tail") or "").strip())
            ]
        )
        candidate_labels = self._sanitize_answer_candidate_texts(fact_entity_labels + fact_literal_values + summary_tail_labels + labels)
        evidence_entity_ids = self._extract_subquestion_entity_ids_from_evidence(
            graph_api=graph_api,
            sub_question=sub_question,
            pruned_paths=pruned_paths,
            step_summaries=step_summaries,
            answer_result=answer_result,
            literals=candidate_labels,
        )
        candidate_entity_ids = self._deduplicate_strings([*fact_entity_ids, *evidence_entity_ids, *(primary_entity_ids or [])])
        canonical_entity_ids, canonical_labels, type_filter_stats = self._resolve_grounding_entity_candidates(
            graph_api=graph_api,
            candidate_labels=candidate_labels,
            candidate_entity_ids=candidate_entity_ids,
            expected_type=sub_question.expected_answer_type,
        )
        canonical_labels = self._filter_grounding_literals_by_answer_texts(
            values=canonical_labels,
            answer_constraint_texts=answer_constraint_texts,
        )

        entity_labels: list[str] = []
        primary_entity_id = ""
        if canonical_entity_ids and graph_api is not None and hasattr(graph_api, "get_nodes_metadata_batched"):
            metadata_by_id = graph_api.get_nodes_metadata_batched(canonical_entity_ids)
            ordered_pairs = self._order_canonical_entity_candidates(
                graph_api=graph_api,
                entity_ids=canonical_entity_ids,
                metadata_by_id=metadata_by_id,
                step_summaries=step_summaries,
            )
            ordered_pairs = self._filter_grounding_pairs_by_answer_texts(
                ordered_pairs=ordered_pairs,
                answer_constraint_texts=answer_constraint_texts,
            )
            canonical_entity_ids = [entity_id for entity_id, _label in ordered_pairs]
            entity_labels = self._deduplicate_strings(
                [
                    label
                    for _entity_id, label in ordered_pairs
                    if self._is_propagatable_text(label) and not self._looks_like_mid_or_cvt_id(label)
                ]
            )
            if not entity_labels:
                entity_labels = self._deduplicate_strings(
                    [label for _entity_id, label in ordered_pairs if self._is_propagatable_text(label)]
                )
            if entity_labels:
                normalized_labels = {self._normalize_relation_text(label) for label in entity_labels}
                canonical_entity_ids = [
                    entity_id
                    for entity_id in canonical_entity_ids
                    if self._normalize_relation_text(
                        str(
                            (
                                metadata_by_id.get(entity_id).name
                                if metadata_by_id.get(entity_id) is not None
                                else graph_api.get_entity_display_name(entity_id)
                            )
                            or entity_id
                        )
                    )
                    in normalized_labels
                ] or canonical_entity_ids
            if canonical_entity_ids:
                primary_entity_id = canonical_entity_ids[0]

        answer_texts = self._sanitize_answer_candidate_texts(
            [*entity_labels, *canonical_labels]
        )
        literal_values = self._sanitize_answer_candidate_texts(
            [
                str(value).strip()
                for value in [
                    *fact_literal_values,
                    *(primary_literals or []),
                    *getattr(answer_result, "resolved_literals", []),
                    *canonical_labels,
                ]
                if str(value).strip()
            ]
        )
        literal_values = self._filter_grounding_literals_by_answer_texts(
            values=literal_values,
            answer_constraint_texts=answer_constraint_texts,
        )
        if entity_labels:
            literal_values = self._deduplicate_strings(
                [value for value in literal_values if self._normalize_relation_text(value) not in {
                    self._normalize_relation_text(label) for label in entity_labels
                }]
            )
        primary_answer_text = (
            answer_texts[0]
            if answer_texts
            else (literal_values[0] if literal_values else str(answer_result.answer or "").strip())
        )
        source_mode = "summary_tail_resolution" if answer_view_triples else "path_terminal"
        grounding = AnswerGrounding(
            answer_texts=list(answer_texts),
            entity_ids=list(canonical_entity_ids),
            entity_labels=list(entity_labels or answer_texts),
            literal_values=list(literal_values),
            primary_answer_text=primary_answer_text if self._is_propagatable_text(primary_answer_text) else "",
            primary_entity_id=primary_entity_id,
            answer_view_triples=list(answer_view_triples),
            answer_view_facts=list(answer_view_facts),
            supporting_relation_ids=self._collect_summary_relation_ids(step_summaries),
            source_mode=source_mode,
            confidence="high" if bool(answer_texts or canonical_entity_ids or literal_values) else "low",
            debug={
                "summary_tail_labels": list(summary_tail_labels[:8]),
                "candidate_fact_entity_ids": list(fact_entity_ids[:8]),
                "candidate_fact_labels": list((fact_entity_labels + fact_literal_values)[:8]),
                "candidate_labels": list(candidate_labels[:8]),
                "candidate_entity_ids": list(candidate_entity_ids[:8]),
                "answer_constraint_texts": list(answer_constraint_texts[:8]),
                "grounding_entity_ids_after_answer_filter": list(canonical_entity_ids[:8]),
                "used_label_fallback": not bool(fact_entity_ids),
                "type_filter_stats": dict(type_filter_stats),
            },
        )
        return grounding, type_filter_stats

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
    ) -> tuple[AnswerGrounding, dict[str, int]]:
        """Resolve one canonical answer grounding before state propagation."""
        grounding, type_filter_stats = self._build_subquestion_grounding(
            graph_api=graph_api,
            sub_question=sub_question,
            pruned_paths=pruned_paths,
            step_summaries=step_summaries,
            answer_result=answer_result,
            structured_outputs=structured_outputs,
            primary_entity_ids=primary_entity_ids,
            primary_literals=primary_literals,
        )
        self._project_grounding_to_answer_result(answer_result, grounding)
        return grounding, type_filter_stats

    def _order_canonical_entity_candidates(
        self,
        *,
        graph_api: Any,
        entity_ids: list[str],
        metadata_by_id: dict[str, Any],
        step_summaries: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Prefer leaf answer nodes over bridge/CVT-like intermediate nodes."""
        head_labels: set[str] = set()
        tail_labels: set[str] = set()
        for item in step_summaries:
            if not isinstance(item, dict):
                continue
            for triple in self._summary_fact_triples(item):
                if not isinstance(triple, dict):
                    continue
                head = self._normalize_relation_text(str(triple.get("head") or ""))
                tail = self._normalize_relation_text(str(triple.get("tail") or ""))
                if head:
                    head_labels.add(head)
                if tail:
                    tail_labels.add(tail)

        scored: list[tuple[float, str, str]] = []
        for entity_id in entity_ids:
            metadata = metadata_by_id.get(entity_id)
            label = str(
                (
                    metadata.name
                    if metadata is not None and str(getattr(metadata, "name", "") or "").strip()
                    else graph_api.get_entity_display_name(entity_id)
                )
                or entity_id
            ).strip()
            normalized_label = self._normalize_relation_text(label)
            score = 0.0
            if normalized_label in tail_labels:
                score += 2.0
            if normalized_label in head_labels:
                score -= 1.5
            if not self._looks_like_mid_or_cvt_id(label):
                score += 1.0
            if self._looks_like_mid_or_cvt_id(label):
                score -= 1.0
            scored.append((score, entity_id, label))
        scored.sort(key=lambda item: (-item[0], item[2]))
        return [(entity_id, label) for _score, entity_id, label in scored]

    @staticmethod
    def _looks_like_mid_or_cvt_id(value: str) -> bool:
        """Return whether a label still looks like a raw Freebase id."""
        normalized = str(value).strip()
        return normalized.startswith("m.") or normalized.startswith("g.")

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
        answer_view_facts = self._collect_answer_view_facts(step_summaries, pruned_paths)
        fact_entity_ids = self._deduplicate_strings(
            [
                str(fact.get("tail_id") or "").strip()
                for fact in answer_view_facts
                if isinstance(fact, dict)
                and str(fact.get("tail_kind") or "id").strip().lower() == "id"
                and str(fact.get("tail_id") or "").strip()
            ]
        )
        if fact_entity_ids:
            return fact_entity_ids
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
            for triple in self._summary_fact_triples(item):
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

    def _question_expected_answer_type(self, question_analysis: QuestionAnalysisResult) -> str:
        """Approximate the final expected answer type from the last planned sub-question."""
        if not question_analysis.sub_questions:
            return ""
        for sub_question in reversed(question_analysis.sub_questions):
            expected_type = str(sub_question.expected_answer_type or "").strip()
            if expected_type:
                return expected_type
        return ""

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

        grounding = answer_result.grounding or self._empty_answer_grounding()
        entity_ids = self._deduplicate_strings(list(grounding.entity_ids) + list(entity_ids))
        literals = self._deduplicate_strings(list(grounding.answer_texts) + list(grounding.literal_values) + list(literals))

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

    def _evaluate_subquestion_output_type(
        self,
        *,
        graph_api: Any,
        sub_question: SubQuestionSpec,
        entity_ids: list[str],
        literals: list[str],
        answer_result: AnswerResult,
        step_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a structured type-check result for sub-question outputs."""
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        if not expected_type:
            return {
                "passed": True,
                "soft_pass": False,
                "mode": "disabled",
                "hard_match": True,
                "relation_score": 0.0,
                "embedding_score": 0.0,
            }

        grounding = answer_result.grounding or self._empty_answer_grounding()
        entity_ids = self._deduplicate_strings(list(grounding.entity_ids) + list(entity_ids))
        literals = self._deduplicate_strings(list(grounding.answer_texts) + list(grounding.literal_values) + list(literals))

        hard_match = self._subquestion_outputs_match_expected_type(
            graph_api=graph_api,
            sub_question=sub_question,
            entity_ids=entity_ids,
            literals=literals,
            answer_result=answer_result,
        )
        relation_texts = self._collect_summary_relation_texts(step_summaries)
        candidate_texts = self._deduplicate_strings(
            [
                *entity_ids,
                *literals,
                *grounding.entity_labels,
                *answer_result.predicted_answers,
                *answer_result.resolved_literals,
                *answer_result.resolved_entity_mentions,
            ]
        )
        candidate_texts = self._sanitize_answer_candidate_texts(candidate_texts)
        relation_score = self._relation_type_soft_score(expected_type, relation_texts)
        embedding_score = self._embedding_type_soft_score(
            graph_api=graph_api,
            expected_type=expected_type,
            entity_ids=entity_ids,
            candidate_texts=candidate_texts,
            relation_texts=relation_texts,
        )
        soft_pass = False
        mode = "hard_fail"
        if hard_match:
            mode = "hard_match"
        else:
            soft_pass = self._subquestion_outputs_soft_match_expected_type(
                graph_api=graph_api,
                sub_question=sub_question,
                entity_ids=entity_ids,
                literals=literals,
                answer_result=answer_result,
                step_summaries=step_summaries,
            )
            if soft_pass:
                if relation_score >= 0.72 and embedding_score >= 0.32:
                    mode = "relation_embedding_soft"
                elif embedding_score >= 0.58:
                    mode = "embedding_soft"
                else:
                    mode = "alias_soft"
        return {
            "passed": bool(hard_match or soft_pass),
            "soft_pass": bool(soft_pass),
            "mode": mode,
            "hard_match": bool(hard_match),
            "relation_score": float(relation_score),
            "embedding_score": float(embedding_score),
        }

    def _should_bypass_boolean_subquestion_type_check(
        self,
        *,
        sub_question: SubQuestionSpec,
        answer_result: AnswerResult,
        sufficiency: dict[str, Any],
    ) -> bool:
        """Allow explicit yes/no verify answers through when evidence already proved the boolean claim."""
        expected_type = self._normalize_relation_text(sub_question.expected_answer_type)
        if expected_type != "boolean":
            return False
        if not bool(sufficiency.get("sufficient", False)):
            return False
        candidates = self._deduplicate_strings(
            [
                str(answer_result.answer or "").strip(),
                *[str(value).strip() for value in answer_result.predicted_answers if str(value).strip()],
                *[str(value).strip() for value in answer_result.resolved_literals if str(value).strip()],
                str(sufficiency.get("primary_answer") or "").strip(),
                *[str(value).strip() for value in sufficiency.get("answer_candidates", []) if str(value).strip()],
            ]
        )
        normalized = {self._normalize_relation_text(value) for value in candidates if value}
        return bool(normalized & {"yes", "no", "true", "false"})

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

        grounding = answer_result.grounding or self._empty_answer_grounding()
        entity_ids = self._deduplicate_strings(list(grounding.entity_ids) + list(entity_ids))
        literals = self._deduplicate_strings(list(grounding.answer_texts) + list(grounding.literal_values) + list(literals))

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
        relation_texts = self._collect_summary_relation_texts(step_summaries)

        candidate_texts = self._deduplicate_strings(
            [
                *entity_ids,
                *literals,
                *grounding.entity_labels,
                *answer_result.predicted_answers,
                *answer_result.resolved_literals,
                *answer_result.resolved_entity_mentions,
            ]
        )
        candidate_texts = self._sanitize_answer_candidate_texts(candidate_texts)
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
        for relation_text in relation_texts:
            if any(token in relation_text for token in alias_tokens):
                return True

        for value in candidate_texts:
            haystack = self._normalize_relation_text(value)
            if any(token in haystack for token in alias_tokens):
                return True
        relation_score = self._relation_type_soft_score(expected_type, relation_texts)
        embedding_score = self._embedding_type_soft_score(
            graph_api=graph_api,
            expected_type=expected_type,
            entity_ids=entity_ids,
            candidate_texts=candidate_texts,
            relation_texts=relation_texts,
        )
        if relation_score >= 0.72 and embedding_score >= 0.32:
            return True
        if embedding_score >= 0.58:
            return True
        return False

    def _collect_summary_relation_texts(
        self,
        step_summaries: list[dict[str, Any]],
    ) -> list[str]:
        """Collect normalized relation strings from summarized evidence."""
        relation_texts: list[str] = []
        for item in step_summaries:
            if not isinstance(item, dict):
                continue
            for triple in self._summary_fact_triples(item):
                if not isinstance(triple, dict):
                    continue
                relation = str(triple.get("relation") or "").strip()
                if not relation:
                    continue
                relation_texts.append(self._normalize_relation_text(relation.replace(".", " ").replace("_", " ")))
        return self._deduplicate_strings([text for text in relation_texts if text])

    def _relation_type_soft_score(
        self,
        expected_type: str,
        relation_texts: list[str],
    ) -> float:
        """Estimate whether relation semantics imply the expected answer type."""
        if not expected_type or not relation_texts:
            return 0.0
        aliases = self._expected_type_aliases(expected_type)
        best = 0.0
        for relation_text in relation_texts:
            normalized_relation = self._normalize_relation_text(relation_text)
            for alias in aliases:
                normalized_alias = self._normalize_relation_text(alias)
                if not normalized_alias:
                    continue
                if normalized_alias in normalized_relation:
                    best = max(best, 1.0)
                alias_tokens = [token for token in normalized_alias.split() if len(token) > 2]
                overlap = 0.0
                if alias_tokens:
                    relation_tokens = set(normalized_relation.split())
                    overlap = len(relation_tokens & set(alias_tokens)) / max(len(alias_tokens), 1)
                best = max(best, overlap)
        return best

    def _embedding_type_soft_score(
        self,
        *,
        graph_api: Any,
        expected_type: str,
        entity_ids: list[str],
        candidate_texts: list[str],
        relation_texts: list[str],
    ) -> float:
        """Use the shared embedding model to score candidate/type semantic compatibility."""
        if not expected_type:
            return 0.0
        evidence_texts = self._deduplicate_strings(
            [
                *relation_texts,
                *[text for text in candidate_texts if self._is_propagatable_text(text)],
            ]
        )
        if graph_api is not None and entity_ids and hasattr(graph_api, "get_nodes_metadata_batched"):
            metadata_by_id = graph_api.get_nodes_metadata_batched(entity_ids[:5])
            for entity_id in entity_ids[:5]:
                metadata = metadata_by_id.get(entity_id)
                if metadata is None:
                    continue
                evidence_texts.append(
                    self._normalize_relation_text(
                        " ".join(
                            [
                                str(getattr(metadata, "name", "") or ""),
                                *[str(value) for value in getattr(metadata, "types", ()) if value],
                            ]
                        )
                    )
                )
        evidence_texts = self._deduplicate_strings([text for text in evidence_texts if text])[:8]
        if not evidence_texts:
            return 0.0
        type_queries = self._deduplicate_strings(
            [
                self._normalize_relation_text(expected_type),
                *[self._normalize_relation_text(alias) for alias in self._expected_type_aliases(expected_type)],
            ]
        )[:8]
        if not type_queries:
            return 0.0
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return 0.0
        try:
            model = get_entity_embedding_model(self.config["embedding"]["model"])
            query_embeddings = np.asarray(self._encode_embedding_texts(model, type_queries))
            evidence_embeddings = np.asarray(self._encode_embedding_texts(model, evidence_texts))
            similarities = cosine_similarity(query_embeddings, evidence_embeddings)
            return float(similarities.max()) if similarities.size else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _encode_embedding_texts(model: object, texts: list[str]) -> Any:
        """Encode texts without progress bars when the backend supports it."""
        try:
            return model.encode(texts, show_progress_bar=False)
        except TypeError:
            return model.encode(texts)

    def _expected_type_aliases(self, expected_type: str) -> set[str]:
        normalized = self._normalize_relation_text(expected_type)
        alias_map = {
            "country": (
                "location country",
                "country",
                "sovereign state",
                "nation",
                "location administrative division",
                "constituent country",
                "administrative division",
                "first level administrative division",
                "dependent territory",
                "region",
            ),
            "nation": (
                "location country",
                "country",
                "sovereign state",
                "nation",
                "location administrative division",
                "constituent country",
                "administrative division",
                "first level administrative division",
                "dependent territory",
                "region",
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
            "i am sorry",
            "the provided evidence",
            "does not contain the name",
            "machine id",
        )
        return not any(marker in lowered for marker in blocked_markers)

    def _sanitize_answer_candidate_texts(self, values: list[str]) -> list[str]:
        """Drop failure-like explanations so they cannot be mistaken for answer candidates."""
        return self._deduplicate_strings(
            [value for value in values if self._is_propagatable_text(str(value).strip())]
        )

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
            graph_api=graph_api,
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
        graph_api: Any,
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
                "answer": (
                    self._grounding_from_payload(resolved_sub_answers.get(dependency_id, {})).primary_answer_text
                    or resolved_sub_answers.get(dependency_id, {}).get("answer", "")
                ),
                "predicted_answers": (
                    self._grounding_from_payload(resolved_sub_answers.get(dependency_id, {})).answer_texts
                    or resolved_sub_answers.get(dependency_id, {}).get("predicted_answers", [])
                ),
                "entity_ids": self._dependency_payload_seed_entity_ids(
                    payload=resolved_sub_answers.get(dependency_id, {}),
                    graph_api=graph_api,
                ),
                "literals": (
                    self._grounding_from_payload(resolved_sub_answers.get(dependency_id, {})).literal_values
                    or resolved_sub_answers.get(dependency_id, {}).get("literals", [])
                ),
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
            for triple in self._summary_fact_triples(item):
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
            final_grounding=self._grounding_to_dict(answer_result.grounding),
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
        question_interested_node_specs = [
            EntityMentionSpec(
                name=str(item.get("name") or "").strip(),
                aliases=[str(value).strip() for value in item.get("aliases", []) if str(value).strip()],
                expected_type=str(item.get("expected_type") or "").strip(),
                role=str(item.get("role") or "constraint_or_anchor").strip(),
            )
            for item in (sub_question.execution_hints.get("question_interested_nodes", []) or [])
            if str(item.get("name") or "").strip()
        ]
        topic_entity_candidates = self._resolve_entity_specs_with_sqlite(
            specs=self._effective_sub_question_topics(sub_question),
            graph_api=graph_api,
            context_text=sub_question.question,
        )
        interested_node_candidates = self._resolve_entity_specs_with_sqlite(
            specs=question_interested_node_specs,
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
            "interested_nodes_mode": "question_only_constraint_targets_plus_local_path_filter",
            "relation_hints_mode": "local_path_filter_only",
            "selected_relation_ids": [],
            "anchor_filter": anchor_filter_debug,
            "question_constraint_target_ids": [candidate.id_or_mid for candidate in interested_node_candidates],
        }
        return ResolvedSubQuestion(
            id=sub_question.id,
            question=sub_question.question,
            topic_entity_candidates=topic_entity_candidates,
            interested_node_candidates=interested_node_candidates,
            relation_candidates=[],
            seed_node_ids=seed_node_ids,
            constraint_target_ids=self._deduplicate_strings(
                [candidate.id_or_mid for candidate in interested_node_candidates]
            ),
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
            names = self._build_entity_query_names(spec, context_text=context_text)
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

    def _compound_entity_query_names_from_context(
        self,
        base_name: str,
        context_text: str,
        expected_type: str,
    ) -> list[str]:
        """Return explicit context spans such as "Vienna, Austria" for location mentions."""
        if not base_name or not context_text:
            return []
        location_types = {
            "administrative division",
            "city",
            "citytown",
            "country",
            "location",
            "place",
            "region",
            "state",
            "town",
        }
        normalized_type = self._normalize_relation_text(expected_type)
        if not normalized_type or not any(location_type in normalized_type for location_type in location_types):
            return []

        pattern = re.compile(rf"(?i)(?<!\w){re.escape(base_name)}\s*,\s*([^,;?]+)")
        stop_tokens = {
            "and",
            "are",
            "for",
            "from",
            "has",
            "have",
            "in",
            "is",
            "of",
            "or",
            "that",
            "the",
            "to",
            "was",
            "were",
            "what",
            "where",
            "which",
            "who",
            "with",
        }
        spans: list[str] = []
        for match in pattern.finditer(context_text):
            raw_qualifier = match.group(1).strip()
            qualifier_tokens: list[str] = []
            for token in raw_qualifier.split():
                cleaned = token.strip(" \t\r\n\"'()[]{}")
                if not cleaned:
                    continue
                if cleaned.lower().strip(".") in stop_tokens:
                    break
                qualifier_tokens.append(cleaned)
                if len(qualifier_tokens) >= 4:
                    break
            qualifier = " ".join(qualifier_tokens).strip(" ,")
            if not qualifier:
                continue
            spans.append(f"{base_name}, {qualifier}")
            spans.append(f"{base_name} {qualifier}")
        return self._deduplicate_strings(spans)

    def _build_entity_query_names(self, spec: Any, context_text: str = "") -> list[str]:
        """Expand entity queries with lightweight type-aware variants before reranking."""
        base_name = str(getattr(spec, "name", "") or "").strip()
        aliases = [str(value).strip() for value in getattr(spec, "aliases", []) if str(value).strip()]
        expected_type = self._normalize_relation_text(str(getattr(spec, "expected_type", "") or "").strip())
        context_names = self._compound_entity_query_names_from_context(
            base_name=base_name,
            context_text=context_text,
            expected_type=expected_type,
        )
        names = self._deduplicate_strings([*context_names, base_name, *aliases])
        if not base_name:
            return names
        typo_variants: list[str] = []
        if expected_type in {"character", "person"} or "character" in expected_type:
            tokens = base_name.split()
            for index, token in enumerate(tokens):
                cleaned = token.strip()
                if len(cleaned) < 5:
                    continue
                if cleaned.lower().endswith("d"):
                    variant_tokens = list(tokens)
                    variant_tokens[index] = cleaned[:-1]
                    typo_variants.append(" ".join(variant_tokens))
                if cleaned.lower().endswith("nd"):
                    variant_tokens = list(tokens)
                    variant_tokens[index] = f"{cleaned[:-2]}n"
                    typo_variants.append(" ".join(variant_tokens))

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
            "character": ["character", "fictional character"],
        }
        suffixes = type_query_suffixes.get(expected_type, [])
        expanded: list[str] = self._deduplicate_strings([*names, *typo_variants])
        for query_name in [base_name, *typo_variants]:
            for suffix in suffixes:
                expanded.append(f"{query_name} {suffix}")
                expanded.append(f"{suffix} {query_name}")
                if expected_type in {"state", "country", "nation", "city"}:
                    expanded.append(f"{query_name} {suffix} of")
                    expanded.append(f"{suffix} of {query_name}")
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
        team_championship_context = self._is_team_championship_relation_context(sub_question)
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
            if team_championship_context:
                if candidate.id_or_mid == "sports.sports_team.championships":
                    score += 30.0
                elif candidate.id_or_mid.startswith("award."):
                    score -= 8.0
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

    def _is_team_championship_relation_context(self, sub_question: SubQuestionSpec) -> bool:
        """Return whether relation selection is asking for a team's championships."""
        text = self._normalize_relation_text(
            " ".join(
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
                    *[relation.description for relation in sub_question.interested_relations if relation.description],
                    *[node.name for node in sub_question.interested_nodes if node.name],
                    *list(sub_question.downstream_filters),
                ]
            )
        )
        has_team_context = any(
            token in text
            for token in (
                "team",
                "basketball",
                "baseball",
                "football",
                "hockey",
                "soccer",
                "nba",
                "nfl",
                "mlb",
                "nhl",
            )
        )
        has_championship_context = any(
            token in text
            for token in (
                "championship",
                "championships",
                "champion",
                "won",
                "win",
                "wins",
                "world series",
                "nba finals",
                "finals",
            )
        )
        return has_team_context and has_championship_context

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
        constraint_plan = self._constraint_plan_for_subquestion(sub_question)

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
            if self._path_matches_constraint_plan(path, constraint_plan):
                score += 8.0
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

    def _constraint_plan_for_subquestion(
        self,
        sub_question: SubQuestionSpec,
        constraint_target_ids: list[str] | None = None,
    ) -> ConstraintPlan:
        """Build reusable path constraints from interested nodes and resolved ids."""
        target_texts = self._deduplicate_strings(
            [
                node.name
                for node in sub_question.interested_nodes
                if node.name
            ]
        )
        expected_types = {
            self._normalize_relation_text(node.expected_type)
            for node in sub_question.interested_nodes
            if node.expected_type
        }
        marker_map = {
            "character": ("character", "role", "performance", "cast", "portray"),
            "person": ("person", "people", "actor", "cast", "performer"),
            "team": ("team", "roster", "franchise"),
            "sports team": ("team", "roster", "franchise"),
            "country": ("country", "location", "containedby", "nation"),
            "location": ("location", "containedby", "contains", "place"),
            "government": ("government", "office", "position"),
            "government position": ("government", "office", "position"),
        }
        markers: list[str] = []
        for expected_type in expected_types:
            markers.extend(marker_map.get(expected_type, ()))
            if expected_type:
                markers.append(expected_type)
        for hint in sub_question.interested_relations:
            markers.append(hint.name)
            markers.extend(hint.aliases)
        return ConstraintPlan(
            target_entity_ids=tuple(self._deduplicate_strings(list(constraint_target_ids or []))),
            target_texts=tuple(target_texts),
            relation_markers=tuple(self._deduplicate_strings(markers)),
        )

    def _path_matches_constraint_plan(self, path: ReasoningPath, plan: ConstraintPlan) -> bool:
        if not plan.target_entity_ids and not plan.target_texts:
            return True
        target_id_set = set(plan.target_entity_ids)
        if target_id_set:
            if str(path.terminal_node_id or "").strip() in target_id_set:
                return True
            for fact in path.triple_facts:
                if str(fact.head_id or "").strip() in target_id_set:
                    return True
                if str(fact.tail_id or "").strip() in target_id_set:
                    return True
            for edge_id in path.edge_ids:
                if any(target_id in str(edge_id) for target_id in target_id_set):
                    return True
        if plan.target_texts:
            path_text = self._normalize_relation_text(
                " ".join(
                    [
                        path.text,
                        *path.nodes,
                        *[
                            " ".join([triple.head, triple.relation, triple.tail])
                            for triple in path.triples
                        ],
                    ]
                )
            )
            if any(self._token_overlap_ratio(term, path_text) >= 0.5 for term in plan.target_texts):
                return True
        return False

    def _constraint_relation_bonus(self, relation_text: str, plan: ConstraintPlan) -> float:
        if not plan.relation_markers:
            return 0.0
        return 8.0 * max(
            (self._token_overlap_ratio(marker, relation_text) for marker in plan.relation_markers),
            default=0.0,
        )

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

        constraint_plan = self._constraint_plan_for_subquestion(sub_question)
        relation_rows: dict[str, dict[str, Any]] = {}
        per_seed_limit = self._graphapi_local_relation_candidate_limit()
        for seed_id in step_seed_ids:
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
        team_championship_context = self._is_team_championship_relation_context(sub_question)
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
            score += self._constraint_relation_bonus(relation_text, constraint_plan)
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
            if team_championship_context:
                if row["relation_id"] == "sports.sports_team.championships":
                    score += 30.0
                elif row["relation_id"].startswith("award."):
                    score -= 8.0
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
        if constraint_plan.target_texts or constraint_plan.target_entity_ids:
            for score, row in scored_rows:
                relation_text = self._normalize_relation_text(f"{row['relation_id']} {row['display_name']}")
                if self._constraint_relation_bonus(relation_text, constraint_plan) <= 0.0:
                    continue
                relation_id = row["relation_id"]
                if relation_id not in selected_ids:
                    selected_ids.append(relation_id)
                if len(selected_ids) >= max(3, self._agentic_local_relation_probe_top_k() + 1):
                    break
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
        """Order dependency entities for the current sub-question without dropping candidates."""
        dependency_seed_ids = self._deduplicate_strings(dependency_seed_ids)
        if graph_api is None or len(dependency_seed_ids) <= 1 or not sub_question.depends_on:
            return dependency_seed_ids
        ranked_dependency_seed_ids = self._rank_dependency_seed_ids_by_provenance(
            sub_question=sub_question,
            dependency_seed_ids=dependency_seed_ids,
            resolved_sub_answers=resolved_sub_answers,
            graph_api=graph_api,
            trace=trace,
            depth=depth,
        )
        if len(ranked_dependency_seed_ids) <= 1:
            return ranked_dependency_seed_ids

        candidate_entities = [
            {
                "entity_id": entity_id,
                "label": str(graph_api.get_entity_display_name(entity_id) or entity_id),
                "source_sub_question_id": dependency_id,
                "source_answer": self._grounding_from_payload(
                    resolved_sub_answers.get(dependency_id, {})
                ).primary_answer_text
                or resolved_sub_answers.get(dependency_id, {}).get("answer", ""),
            }
            for dependency_id in sub_question.depends_on
            for entity_id in self._dependency_payload_seed_entity_ids(
                payload=resolved_sub_answers.get(dependency_id, {}),
                graph_api=graph_api,
            )
            if entity_id in ranked_dependency_seed_ids
        ]
        if len(candidate_entities) <= 1:
            return ranked_dependency_seed_ids

        resolved_dependencies = [
            {
                "sub_question_id": dependency_id,
                "answer": (
                    self._grounding_from_payload(resolved_sub_answers.get(dependency_id, {})).primary_answer_text
                    or resolved_sub_answers.get(dependency_id, {}).get("answer", "")
                ),
                "entity_ids": self._dependency_payload_seed_entity_ids(
                    payload=resolved_sub_answers.get(dependency_id, {}),
                    graph_api=graph_api,
                ),
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
        preferred_ids = self._deduplicate_strings(
            [item for item in parsed.get("selected_entity_ids", []) if item in ranked_dependency_seed_ids]
            if isinstance(parsed, dict)
            else []
        )
        selected_ids = self._deduplicate_strings(
            [
                *preferred_ids,
                *[entity_id for entity_id in ranked_dependency_seed_ids if entity_id not in set(preferred_ids)],
            ]
        )
        trace.append(
            SearchTraceStep(
                stage="dependency_entity_filter",
                depth=depth,
                candidate_count=len(ranked_dependency_seed_ids),
                pruned_count=len(selected_ids),
                note=(
                    f"sub_question_id={sub_question.id} input_entity_ids={ranked_dependency_seed_ids} "
                    f"preferred_entity_ids={preferred_ids} ordered_entity_ids={selected_ids}"
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
            for triple in self._summary_fact_triples(item):
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
        return self._compress_evidence_lines(self._deduplicate_strings(evidence_lines))

    def _compress_evidence_lines(self, evidence_lines: list[str]) -> list[str]:
        """Prefer one whole CVT bundle line over repeated atomic lines it already subsumes."""
        deduped = self._deduplicate_strings([str(line).strip() for line in evidence_lines if str(line).strip()])
        if not deduped:
            return []

        bundle_heads: set[str] = set()
        compressed: list[str] = []
        for line in deduped:
            if "{" in line and ";" in line:
                prefix = line.split("{", 1)[0].strip()
                if ";" in prefix:
                    bundle_head = prefix.rsplit(";", 1)[-1].strip()
                    if bundle_head:
                        bundle_heads.add(bundle_head)
                compressed.append(line)

        for line in deduped:
            if line in compressed:
                continue
            line_head = line.split("->", 1)[0].strip() if "->" in line else ""
            if line_head and line_head in bundle_heads:
                continue
            compressed.append(line)
        return compressed

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
            grounding = self._grounding_from_payload(payload)
            provenance_chunks.append(grounding.primary_answer_text or str(payload.get("answer") or "").strip())
            provenance_chunks.extend(str(item).strip() for item in grounding.answer_texts if str(item).strip())
            provenance_chunks.extend(str(item).strip() for item in grounding.literal_values if str(item).strip())
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
        preferred_ids = [entity_id for score, entity_id, _ in scored_ids if score >= keep_threshold]
        selected_ids = self._deduplicate_strings(
            [
                *preferred_ids,
                *[entity_id for _score, entity_id, _payload in scored_ids if entity_id not in set(preferred_ids)],
            ]
        )
        trace.append(
            SearchTraceStep(
                stage="dependency_seed_disambiguation",
                depth=depth,
                candidate_count=len(dependency_seed_ids),
                pruned_count=len(selected_ids),
                note=(
                    f"sub_question_id={sub_question.id} preferred_seed_ids={preferred_ids} "
                    f"ordered_seed_ids={selected_ids} "
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
        constraint_target_ids: list[str],
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
        constraint_plan = self._constraint_plan_for_subquestion(
            sub_question=sub_question,
            constraint_target_ids=constraint_target_ids,
        )
        target_filtered_count = 0
        if constraint_plan.target_entity_ids or constraint_plan.target_texts:
            original_count = len(strict_paths) + len(fallback_paths)
            strict_paths = [
                path for path in strict_paths
                if self._path_matches_constraint_plan(path, constraint_plan)
            ]
            fallback_paths = [
                path for path in fallback_paths
                if self._path_matches_constraint_plan(path, constraint_plan)
            ]
            filtered_count = len(strict_paths) + len(fallback_paths)
            if filtered_count > 0:
                target_filtered_count = original_count - filtered_count
        merged_paths = self._sort_cwq_candidate_paths(deduplicate_paths(strict_paths + fallback_paths))
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
                    f"strict_paths={len(strict_paths)} fallback_paths={len(fallback_paths)} "
                    f"constraint_target_ids={constraint_target_ids[:5]} target_filtered={target_filtered_count}"
                ),
            )
        )
        return merged_paths

    def _question_level_cvt_anchor_mentions(
        self,
        question: str,
        question_analysis: QuestionAnalysisResult,
    ) -> list[str]:
        """Build compact anchor mentions for question-level CVT normalization."""
        return self._deduplicate_strings(
            [
                question,
                *list(question_analysis.ordered_topic_entities),
                *[
                    entity.name
                    for sub_question in question_analysis.sub_questions
                    for entity in [*sub_question.topic_entities, *sub_question.local_topic_entities, *sub_question.interested_nodes]
                    if entity.name
                ],
                *[
                    relation.name
                    for sub_question in question_analysis.sub_questions
                    for relation in sub_question.interested_relations
                    if relation.name
                ],
                *[
                    alias
                    for sub_question in question_analysis.sub_questions
                    for relation in sub_question.interested_relations
                    for alias in relation.aliases
                    if alias
                ],
            ]
        )

    def _normalize_terminal_cvt_paths(
        self,
        *,
        graph_api: Any,
        candidate_paths: list[ReasoningPath],
        subquestion_text: str,
        expected_answer_type: str,
        relation_hint_names: list[str],
        anchor_mentions: list[str],
        trace: list[SearchTraceStep] | None = None,
        depth: int | None = None,
        stage_note: str = "",
    ) -> list[ReasoningPath]:
        """Normalize terminal CVT paths into answer-facing bundles before pruning/summarization."""
        if graph_api is None or not candidate_paths:
            return candidate_paths
        normalized_paths = expand_cvt_bundle_paths(
            graph=graph_api,
            paths=candidate_paths,
            subquestion=subquestion_text,
            expected_answer_type=expected_answer_type,
            relation_hint_names=relation_hint_names,
            anchor_mentions=anchor_mentions,
        )
        if trace is not None and depth is not None:
            before_bundle = sum(1 for path in candidate_paths if path.source_stage == CVT_BUNDLE_SOURCE_STAGE)
            after_bundle = sum(1 for path in normalized_paths if path.source_stage == CVT_BUNDLE_SOURCE_STAGE)
            trace.append(
                SearchTraceStep(
                    stage="terminal_cvt_normalization",
                    depth=depth,
                    candidate_count=len(candidate_paths),
                    pruned_count=len(normalized_paths),
                    note=(
                        f"{stage_note} before_bundle_paths={before_bundle} "
                        f"after_bundle_paths={after_bundle} expanded={max(0, after_bundle - before_bundle)}"
                    ).strip(),
                )
            )
        return normalized_paths

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

    def _dependency_payload_seed_entity_ids(
        self,
        *,
        payload: dict[str, Any],
        graph_api: Any,
    ) -> list[str]:
        """Resolve dependency seed ids from canonical grounding first, then legacy payloads."""
        grounding = self._grounding_from_payload(payload)
        entity_ids = self._deduplicate_strings(
            [
                *([grounding.primary_entity_id] if grounding.primary_entity_id else []),
                *grounding.entity_ids,
            ]
        )
        if entity_ids:
            return entity_ids
        fact_entity_ids = self._deduplicate_strings(
            [
                *[
                    str(item.get("tail_id") or "").strip()
                    for item in grounding.answer_view_facts
                    if isinstance(item, dict)
                    and str(item.get("tail_kind") or "id").strip().lower() == "id"
                    and str(item.get("tail_id") or "").strip()
                ],
                *[
                    str(item.get("tail_id") or "").strip()
                    for item in payload.get("answer_view_facts", [])
                    if isinstance(item, dict)
                    and str(item.get("tail_kind") or "id").strip().lower() == "id"
                    and str(item.get("tail_id") or "").strip()
                ],
                *[
                    str(item.get("tail_id") or "").strip()
                    for item in payload.get("key_triple_facts", [])
                    if isinstance(item, dict)
                    and str(item.get("tail_kind") or "id").strip().lower() == "id"
                    and str(item.get("tail_id") or "").strip()
                ],
            ]
        )
        if fact_entity_ids:
            return fact_entity_ids
        answer_texts = self._deduplicate_strings(
            [*grounding.answer_texts, *grounding.literal_values]
        )
        if not answer_texts:
            answer_texts = self._deduplicate_strings(
                [
                    *[str(item).strip() for item in payload.get("predicted_answers", []) if str(item).strip()],
                    *[str(item).strip() for item in payload.get("literals", []) if str(item).strip()],
                    str(payload.get("answer") or "").strip(),
                ]
            )
        if answer_texts and graph_api is not None and hasattr(graph_api, "resolve_entity_mentions"):
            resolved_ids: list[str] = []
            for label in answer_texts[:5]:
                resolved_ids.extend(graph_api.resolve_entity_mentions([label], top_k=3))
            return self._deduplicate_strings(resolved_ids)
        return self._deduplicate_strings(payload.get("entity_ids", []))

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
            for triple in self._summary_fact_triples(summary):
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
        key_triple_facts: list[dict[str, str]] = []
        evidence: list[str] = []
        seen_triples: set[tuple[str, str, str]] = set()
        seen_facts: set[tuple[str, str, str, str]] = set()
        seen_evidence: set[str] = set()
        sub_question_ids: list[str] = []
        answered_sub_question_ids: list[str] = []

        for item in subquestion_summaries:
            sub_question_id = str(item.get("sub_question_id") or "").strip()
            if sub_question_id and sub_question_id not in sub_question_ids:
                sub_question_ids.append(sub_question_id)
            if sub_question_id and (item.get("key_triples") or item.get("evidence")) and sub_question_id not in answered_sub_question_ids:
                answered_sub_question_ids.append(sub_question_id)
            for triple in self._summary_fact_triples(item):
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
            for fact in self._summary_fact_facts(item):
                if not isinstance(fact, dict):
                    continue
                fact_key = (
                    str(fact.get("head_id") or ""),
                    str(fact.get("relation_id") or ""),
                    str(fact.get("tail_id") or ""),
                    str(fact.get("tail_label") or ""),
                )
                if fact_key in seen_facts:
                    continue
                seen_facts.add(fact_key)
                key_triple_facts.append(
                    {
                        "head_id": fact_key[0],
                        "head_label": str(fact.get("head_label") or ""),
                        "relation_id": fact_key[1],
                        "tail_id": fact_key[2],
                        "tail_label": fact_key[3],
                        "tail_kind": str(fact.get("tail_kind") or "id"),
                    }
                )
            for line in item.get("evidence", []):
                evidence_line = str(line).strip()
                if not evidence_line or evidence_line in seen_evidence:
                    continue
                seen_evidence.add(evidence_line)
                evidence.append(evidence_line)
        evidence = self._compress_evidence_lines(evidence)

        return {
            "question": question,
            "question_focus": question,
            "summary_type": "aggregate",
            "sub_question_ids": sub_question_ids,
            "answered_sub_question_ids": answered_sub_question_ids,
            "key_triples": key_triples,
            "key_triple_facts": key_triple_facts,
            "answer_view_facts": [],
            "evidence": evidence,
        }

    def _summary_fact_triples(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Prefer answer-facing summary triples when available."""
        triples = item.get("answer_view_triples")
        if isinstance(triples, list) and triples:
            return [triple for triple in triples if isinstance(triple, dict)]
        triples = item.get("key_triples", [])
        if isinstance(triples, list):
            return [triple for triple in triples if isinstance(triple, dict)]
        return []

    def _summary_fact_facts(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Prefer answer-facing summary facts with ids when available."""
        facts = item.get("answer_view_facts")
        if isinstance(facts, list) and facts:
            return [fact for fact in facts if isinstance(fact, dict)]
        facts = item.get("key_triple_facts", [])
        if isinstance(facts, list):
            return [fact for fact in facts if isinstance(fact, dict)]
        return []

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
        candidate_paths = self._normalize_terminal_cvt_paths(
            graph_api=self.graph_api,
            candidate_paths=candidate_paths,
            subquestion_text=question,
            expected_answer_type=self._question_expected_answer_type(question_analysis),
            relation_hint_names=relation_hints,
            anchor_mentions=self._question_level_cvt_anchor_mentions(question, question_analysis),
            trace=trace,
            depth=depth,
            stage_note="question_level",
        )
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
        final_grounding: dict[str, Any] | None = None,
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
            final_grounding=dict(final_grounding or {}),
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
        rerank_cfg = beam_cfg.get("sql_candidate_rerank", {})
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
            sql_candidate_rerank_enabled=bool(rerank_cfg.get("enabled", True)),
            sql_candidate_rerank_overfetch_factor=max(1, int(rerank_cfg.get("overfetch_factor", 3))),
            sql_candidate_rerank_max_overfetch=max(1, int(rerank_cfg.get("max_overfetch", 200))),
            sql_candidate_rerank_min_candidates=max(1, int(rerank_cfg.get("min_candidates_for_rerank", 50))),
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
