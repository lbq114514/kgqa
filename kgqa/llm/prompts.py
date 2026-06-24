"""Prompt templates for the training-free KGQA pipeline."""

ENTITY_EXTRACTION_PROMPT = """You extract entity mentions from a question.
Question: {question}

Example 1:
Question: What country bordering France contains an airport that serves Nijmegen?
Output:
["France", "Nijmegen"]

Example 2:
Question: what time zone am i in cleveland ohio
Output:
["Cleveland", "Ohio"]

Return only a JSON list of entity strings.
Extract entity mentions only. Do not output relation phrases, property names, or answer types.
Do not output markdown.
Do not explain anything.
"""

QUESTION_ANALYSIS_PROMPT = """You analyze a KGQA question and return structured JSON.
Question: {question}
Maximum depth allowed: {dmax}
Topic entities override: {topic_entities_override}

Example 1:
Question: What country bordering France contains an airport that serves Nijmegen?
Maximum depth allowed: 3
Topic entities override: []
Output:
{{
  "topic_entities": ["Nijmegen", "France"],
  "split_questions": [
    "What airport serves Nijmegen?",
    "What country contains that airport?",
    "Which of those countries borders France?"
  ],
  "sub_questions": [
    {{
      "id": "sq1",
      "question": "What airport serves Nijmegen?",
      "topic_entities": [
        {{
          "name": "Nijmegen",
          "aliases": [],
          "expected_type": "city",
          "role": "topic_entity"
        }}
      ],
      "interested_nodes": [],
      "interested_relations": [
        {{
          "name": "served by airport",
          "aliases": ["airport serving"],
          "freebase_like_ids": [],
          "direction": "topic_entity -> answer",
          "description": "airport associated with Nijmegen"
        }}
      ],
      "expected_answer_type": "airport",
      "expected_hop": 1,
      "depends_on": []
    }},
    {{
      "id": "sq2",
      "question": "What country contains that airport?",
      "topic_entities": [],
      "interested_nodes": [
        {{
          "name": "Nijmegen",
          "aliases": [],
          "expected_type": "city",
          "role": "constraint_or_anchor"
        }}
      ],
      "interested_relations": [
        {{
          "name": "contained by country",
          "aliases": [],
          "freebase_like_ids": [],
          "direction": "airport -> answer",
          "description": "country containing the airport"
        }}
      ],
      "expected_answer_type": "country",
      "expected_hop": 1,
      "depends_on": ["sq1"]
    }}
  ],
  "reasoning_indicator": "Find the airport serving Nijmegen, then the country containing it, then check whether that country borders France.",
  "ordered_topic_entities": ["Nijmegen", "France"],
  "predicted_depth": 3
}}

Example 2:
Question: what time zone am i in cleveland ohio
Maximum depth allowed: 3
Topic entities override: []
Output:
{{
  "topic_entities": ["Cleveland", "Ohio"],
  "split_questions": [
    "What time zone is Cleveland in?"
  ],
  "sub_questions": [
    {{
      "id": "sq1",
      "question": "What time zone is Cleveland in?",
      "topic_entities": [
        {{
          "name": "Cleveland",
          "aliases": ["Cleveland, Ohio"],
          "expected_type": "city",
          "role": "topic_entity"
        }}
      ],
      "interested_nodes": [
        {{
          "name": "Ohio",
          "aliases": [],
          "expected_type": "state",
          "role": "constraint_or_anchor"
        }}
      ],
      "interested_relations": [
        {{
          "name": "time zone",
          "aliases": ["timezone"],
          "freebase_like_ids": [],
          "direction": "topic_entity -> answer",
          "description": "time zone associated with Cleveland"
        }}
      ],
      "expected_answer_type": "time zone",
      "expected_hop": 1,
      "depends_on": []
    }}
  ],
  "reasoning_indicator": "Find the time zone associated with Cleveland, Ohio.",
  "ordered_topic_entities": ["Cleveland", "Ohio"],
  "predicted_depth": 1
}}

Return only JSON with this schema:
{{
  "topic_entities": ["..."],
  "split_questions": ["..."],
  "sub_questions": [
    {{
      "id": "sq1",
      "question": "...",
      "topic_entities": [{{"name": "...", "aliases": [], "expected_type": "...", "role": "topic_entity"}}],
      "local_topic_entities": [{{"name": "...", "aliases": [], "expected_type": "...", "role": "topic_entity"}}],
      "interested_nodes": [{{"name": "...", "aliases": [], "expected_type": "...", "role": "constraint_or_anchor"}}],
      "interested_relations": [
        {{
          "name": "...",
          "aliases": [],
          "freebase_like_ids": [],
          "direction": "topic_entity -> answer",
          "description": "..."
        }}
      ],
      "expected_answer_type": "...",
      "expected_hop": 1,
      "depends_on": [],
      "solver_type": "explore",
      "solver_reason": "..."
    }}
  ],
  "reasoning_indicator": "...",
  "ordered_topic_entities": ["..."],
  "predicted_depth": 2
}}

Rules:
- If topic_entities_override is non-empty, treat it as the canonical question-level topic entity list and keep it unchanged.
- Otherwise, first extract question-level topic_entities for the whole question, then decompose the question.
- Keep the old fields populated even when you return sub_questions.
- If the question is simple, return exactly one sub-question.
- topic_entities must contain the whole-question primary entities and must be concrete, linkable names.
- local_topic_entities should contain the current sub-question's own executable seed entities when they differ from the whole-question topic_entities.
- topic_entities must be concrete, linkable entity names, not vague categories.
- For quoted titles, named values, anthem names, award names, or event names, prefer putting the literal string into local_topic_entities or interested_nodes instead of forcing it to be a country/person topic entity.
- interested_nodes may be concrete entities, anchors, constraints, or likely intermediate nodes.
- interested_relations must be short relation phrases or Freebase-like relation ids.
- Do not use vague relation names such as "related to".
- Include aliases and direction whenever possible.
- expected_hop must estimate how many KG hops this sub-question itself should need, between 1 and {dmax}.
- For validation/filter follow-up questions, make the follow-up explicitly verify whether the prior answer satisfies the condition, rather than independently searching for a new entity.
- solver_type must be one of: explore, verify, aggregate.
- Solver taxonomy:
  - explore: use when the sub-question still needs bridge/path finding to an unknown target entity, or when a local anchor exists but the final answer entity is not yet known.
  - verify: use only when there is already a known candidate answer/entity and the task is mainly checking whether it satisfies an extra condition or relation.
  - aggregate: use only for compare/max/min/intersect/count style follow-up steps over candidate sets or structured rows.
- If the sub-question is mainly driven by a constraint anchor, quoted title, named anthem, award, event, book, film, song, or team and the final answer is still unknown, prefer explore.
- If the sub-question refers to "that country/person/team/institution" and depends on a prior step, use verify only when the prior step already provides the candidate being checked; otherwise prefer explore.
- solver_reason should be one short sentence and should explicitly mention one of these structural cues when applicable: known seed, unknown target, verify constraint, set operation.
"""

AGENTIC_STEP_PLANNING_PROMPT = """You are a KGQA planner operating in a step-by-step agent loop.
Question: {question}
Maximum depth allowed: {dmax}
Loop index: {loop_index}
Remaining loop budget: {remaining_loops}
Question-level analysis:
{question_analysis}
Current agent state:
{agent_state}

Return only JSON with this schema:
{{
  "step_id": "step_1",
  "question": "...",
  "goal": "...",
  "depends_on_step_ids": ["..."],
  "topic_entity_mentions": [
    {{"name": "...", "aliases": [], "expected_type": "...", "role": "topic_entity"}}
  ],
  "relation_hints": [
    {{
      "name": "...",
      "aliases": [],
      "freebase_like_ids": [],
      "direction": "topic_entity -> answer",
      "description": "..."
    }}
  ],
  "expected_answer_type": "...",
  "carryover_constraints": ["..."],
  "stop_if_answered": false,
  "strategy": "auto"
}}

Rules:
- Produce exactly one current step, not a full multi-step decomposition.
- If previous evidence suggests a bridge entity or intermediate answer, make that the key entity for this step.
- topic_entity_mentions should focus on the current step only.
- relation_hints should be short phrases or Freebase-like relation ids relevant to this step only.
- carryover_constraints may include literal values, dates, numbers, or textual constraints from earlier steps.
- Set stop_if_answered=true only if this step would complete the original question when enough evidence is found.
- strategy may be one of auto, local_subgraph, external_graph, supplement, or node_expand.
- Do not output markdown or explanations.
"""

SUPPLEMENT_ENTITY_PROMPT = """You propose bridge entities and useful relation names for KG path exploration.
Question: {question}
Topic entities: {topic_entities}
Current candidate paths:
{current_paths}

Return only JSON with this schema:
{{
  "entities": ["..."],
  "relations": ["..."]
}}

The relation strings may be natural-language descriptions or KG-style relation names.
Do not output markdown or explanations.
"""

CWQ_BOOTSTRAP_EXPLORATION_PROMPT = """You plan the first round of external knowledge-graph exploration for a CWQ-style question.
Question: {question}
Topic entities from the question: {topic_entities}
Question analysis:
{question_analysis}

Return only JSON with this schema:
{{
  "focus_entities": ["..."],
  "answer_type_hints": ["..."],
  "relation_name_hints": ["..."],
  "reasoning_focus": "..."
}}

Rules:
- focus_entities must be concrete entity or node names that are worth resolving in the graph.
- Do not put generic answer categories such as country, government, person, place, city, school, language into focus_entities unless they are actual named entities in the question.
- Put generic target categories such as government type, country, airport, religion, language into answer_type_hints instead.
- relation_name_hints should be short phrases close to Freebase relation-name style, such as administrative parent, major field of study, served by airport, government type, official language.
- relation_name_hints should not be full sentences.
- reasoning_focus should briefly describe the intended first retrieval direction.
- Do not output markdown or explanations.
"""

PRECISE_PATH_SELECTION_PROMPT = """You select the most useful candidate reasoning paths.
Question: {question}
Question analysis: {question_analysis}
Supplement relation hints: {supplement_relations}
Candidate paths:
{candidate_paths}
Select at most {wmax} path indices that best support the answer.
Candidate path strings may be abbreviated to start entity, relation sequence, and end entity.
Prefer paths that both support the answer and align with the relation hints when possible.
Relation alignment is a preference, not a hard constraint.
If a path without relation matches is more directly useful for answering, you may still keep it.

Return only a JSON list of integer indices.
"""

PATH_SUMMARIZATION_PROMPT = """You summarize reasoning paths into concise evidence.
Question to answer: {question}
Topic entities: {topic_entities}
Question analysis:
{question_analysis}
Existing evidence bank:
{evidence_bank}
Current plan step:
{plan_step_context}
Sub-question context:
{sub_question_context}
Pruned paths:
{pruned_paths}

Example:
Question to answer: what is nina dobrev nationality
Pruned paths:
[
  {{
    "triples": [
      {{"head": "Nina Dobrev", "relation": "people.person.nationality", "tail": "Canada"}},
      {{"head": "Nina Dobrev", "relation": "people.person.nationality", "tail": "Bulgaria"}}
    ]
  }}
]
Output:
{{
  "question": "what is nina dobrev nationality",
  "question_focus": "nationality of Nina Dobrev",
  "key_triples": [
    {{"head": "Nina Dobrev", "relation": "people.person.nationality", "tail": "Canada"}},
    {{"head": "Nina Dobrev", "relation": "people.person.nationality", "tail": "Bulgaria"}}
  ],
  "evidence": [
    "Nina Dobrev -> people.person.nationality -> Canada",
    "Nina Dobrev -> people.person.nationality -> Bulgaria"
  ]
}}

Summarize strictly for answering the question above.
Only keep triples that are directly useful for answering the question.
Preserve entity and value surface forms exactly as they appear in the input triples.
Do not rewrite dates, numbers, entities, or literal values.
Do not add background knowledge or inferred facts not explicitly supported by the input triples.
The evidence field must stay question-focused and must mention the original entity/value names from the triples.

Return only JSON with this schema:
{{
  "question": "{question}",
  "question_focus": "...",
  "summary_type": "sub_question",
  "sub_question_id": "...",
  "key_triples": [{{"head": "...", "relation": "...", "tail": "..."}}],
  "evidence": ["..."]
}}
"""

SUFFICIENCY_PROMPT = """You judge whether the provided evidence is sufficient to answer the question.
Question: {question}
Topic entities: {topic_entities}
Task analysis:
{task_analysis}
Agentic context:
{agentic_context}
Search context:
{search_context}
Split questions: {split_questions}
Provided evidence:
{summarized_paths}

How to use the context:
- Treat Task analysis, Search context, and Split questions as task guidance only.
- Treat Provided evidence as the only factual basis for the decision.
- Provided evidence may contain sub-question summaries and one aggregate summary.
- Prefer the aggregate summary when it is present, but you may cross-check against sub-question summaries.
- Use predicted/current depth only to estimate whether the evidence is appropriately direct or requires more hops; depth is not evidence.
- If the evidence does not directly support an answer, return insufficient.

Return only JSON:
{{
  "sufficient": true,
  "reason": "...",
  "primary_answer": "...",
  "answer_candidates": ["..."]
}}

Rules for answer extraction:
- If sufficient is true, set primary_answer to the single best direct answer supported by the evidence when one exists.
- answer_candidates should contain only direct answer candidates, not supporting entities, bridge nodes, CVT fillers, dates, countries, or co-participants unless the question explicitly asks for them.
- For bundle-like evidence with multiple slots, choose the slot that best matches the question target. For example, if the question asks "for what event", prefer the event value rather than medal, country, or teammate names.
- If sufficient is false, return an empty primary_answer and an empty answer_candidates list.
"""

ANSWERING_PROMPT = """You answer a question using only the provided evidence.
Question: {question}
Topic entities: {topic_entities}
Task analysis:
{task_analysis}
Agentic context:
{agentic_context}
Search context:
{search_context}
Split questions: {split_questions}
Provided evidence:
{summarized_paths}

Example:
Question: what is ava stone nationality
Topic entities: ["Ava Stone"]
Task analysis:
{{
  "reasoning_indicator": "Find the nationality values directly attached to Ava Stone.",
  "ordered_topic_entities": ["Ava Stone"],
  "predicted_depth": 1
}}
Search context:
{{
  "dmax": 3,
  "current_evaluation_depth": 1
}}
Split questions: ["what is ava stone nationality"]
Provided evidence:
[
  {{
    "question": "what is ava stone nationality",
    "question_focus": "nationality of Ava Stone",
    "key_triples": [
      {{"head": "Ava Stone", "relation": "people.person.nationality", "tail": "Canada"}},
      {{"head": "Ava Stone", "relation": "people.person.nationality", "tail": "Bulgaria"}}
    ],
    "evidence": [
      "Ava Stone -> people.person.nationality -> Canada",
      "Ava Stone -> people.person.nationality -> Bulgaria"
    ]
  }}
]
Output:
{{
  "predicted_answers": ["Canada", "Bulgaria"],
  "answer": "Canada, Bulgaria",
  "supporting_paths": [
    "Ava Stone -> people.person.nationality -> Canada",
    "Ava Stone -> people.person.nationality -> Bulgaria"
  ]
}}

How to use the context:
- Use Topic entities, Task analysis, Search context, and Split questions only to understand what the question is asking.
- Use Provided evidence as the only factual basis for the answer.
- Provided evidence may contain sub-question summaries and one aggregate summary.
- Prefer answers supported by the aggregate summary when it is present, and use sub-question summaries for supporting detail.
- Prefer answers that match the intended reasoning path and granularity suggested by the task analysis.
- Do not turn predicted depth or current evaluation depth into facts; they only describe the search process.

Do not fabricate facts. If evidence is insufficient, say so.
Return atomic answers only, not conversational sentences.
If there are multiple answers, return each answer as a separate list element.
Do not include explanatory text inside the answer list.
Place the answers that best match the question first.
Prefer answers whose semantic type, specificity, and granularity best fit what the question is asking for.
If several evidence-supported candidates are related, keep the most direct answer before broader, narrower, or less relevant ones.
If an answer is supported by a head or tail value in the provided evidence, prefer that value directly.
You may make a minimal surface-form adjustment to a head or tail string only when necessary to better match the question's expected answer form while preserving the same entity or value.
Do not invent unsupported aliases, new facts, or paraphrases that change the underlying entity or value.
Do not change dates, numbers, literals, or unsupported entity identities.
Return only JSON:
{{
  "predicted_answers": ["..."],
  "answer": "...",
  "supporting_paths": ["..."]
}}
"""
