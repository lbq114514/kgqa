# KGQA 框架梳理：工具包 + 智能体自主规划 Pipeline

## 1. 一句话定义

当前这个 `kgqa` 不是单纯的“LLM 一次性问答”，而是一个 **LLM 驱动的多阶段 KGQA agent**：

- `LLM` 负责理解问题、生成搜索提示、判断证据是否足够、总结证据、生成答案
- `KG / Retrieval Toolkit` 负责实体解析、关系解析、图检索、路径扩展、候选路径排序
- `Pipeline Orchestrator` 负责把这些工具串成一个可回退、可继续扩展的搜索决策流程

可以把它抽象成：

```text
User Question
-> Planner Agent
-> Tool Invocation Loop
-> Evidence Accumulation
-> Sufficiency Check
-> Final Answer
```

## 2. 当前代码里的真实模块分层

### 2.1 Orchestrator 层

- `main.py`
  - CLI 入口
  - 加载配置、数据集、LLM、KG、Pipeline
  - 区分 `single / validate`、`webqsp / cwq`
- `kgqa/reasoning/pipeline.py`
  - 系统总控
  - 决定走哪条执行路径
  - 协调 question analysis、graph exploration、path pruning、summarization、answering

### 2.2 Agent Cognition 层

- `kgqa/reasoning/question_analysis.py`
  - 解析问题
  - 输出 topic entities、ordered topic entities、predicted depth、sub-questions
- `kgqa/reasoning/answering.py`
  - 判断当前证据是否足够
  - 基于证据生成最终答案
- `kgqa/reasoning/summarization.py`
  - 把路径压缩成可供回答的结构化证据

### 2.3 Retrieval / Search Toolkit 层

- `kgqa/kg/entity_linking.py`
  - LLM 抽实体 mention
  - embedding 做 entity / relation linking
- `kgqa/kg/subgraph.py`
  - 构建本地问题子图
  - 进行 topic connectivity pruning
- `kgqa/reasoning/exploration.py`
  - topic entity path exploration
  - supplement path exploration
  - node expansion
- `kgqa/reasoning/pruning.py`
  - embedding fuzzy recall
  - LLM precise path selection
  - branch reduction
- `kgqa/retrieval/search.py`
  - 两跳扩展、BFS、beam search、hybrid search
- `kgqa/retrieval/ranking.py`
  - 路径打分、重排、去分支冗余

### 2.4 Graph Backend 层

- `kgqa/kg/graph.py`
  - 小图 / demo 的内存图结构
- `kgqa/kg/sqlite_graph_api.py`
  - 大图 SQLite 图接口
  - 实体候选解析、关系候选解析、邻居查询、路径扩展
- `kgqa/retrieval/backend.py`
  - `IndexedSQLiteGraphBackend`
  - 在 SQLite 真值图之上叠加 runtime index

### 2.5 Model / Infra 层

- `kgqa/llm/vllm_client.py`
  - 对接本地 vLLM OpenAI-compatible 接口
- `kgqa/utils/types.py`
  - 全链路状态对象
  - 例如 `QuestionAnalysisResult`、`ReasoningPath`、`PipelineResult`

## 3. 当前系统的两条真实执行路径

### 3.1 路径 A：Demo / 小图模式

适用：

- 单题 demo
- 有本地小型三元组图 `sample_kg.json`

流程：

```text
Question
-> LLM Entity Extraction
-> Embedding Entity Linking
-> Build Question Subgraph
-> LLM Question Analysis
-> Topic Entity Path Exploration
-> Path Pruning
-> Path Summarization
-> Sufficiency Check
-> if insufficient:
     LLM Supplement Hint Generation
     -> Supplement Path Exploration
     -> Path Re-evaluation
-> if still insufficient:
     Node Expand Exploration
     -> Path Re-evaluation
-> Final Answer
```

特点：

- 先在局部子图里做路径搜索
- 搜索深度由 `predicted_depth` 起步，最多到 `dmax`
- 有明显的“先尝试、再补充、再扩展”的多轮决策结构

### 3.2 路径 B：WebQSP / CWQ 外部图模式

适用：

- WebQSP / CWQ
- 大规模 Freebase SQLite 图

流程：

```text
Question
-> LLM Question Analysis
-> Topic Entity / Interested Node / Relation Hint Extraction
-> SQLite Entity Resolution
-> SQLite Relation Resolution
-> Build Seed Nodes + Pruning Relation Hints
-> External Path Expansion (Hybrid Search over SQLite backend)
-> Path Pruning
-> Path Summarization
-> Sufficiency Check
-> Final Answer
```

特点：

- 这里默认不走本地小图 `subgraph`
- 核心不是“先构图再找路径”，而是“先对齐问题语义，再直接调外部检索工具”
- `CWQ` 更像一个 **sub-question driven retrieval agent**

## 4. 如果改写成“工具包 + Agent 自主规划”，推荐这样抽象

## 4.1 Agent 角色定义

把 `KGQAPipeline` 看成一个单智能体系统，内部包含 4 类能力：

### A. Planner

负责：

- 分析问题复杂度
- 分解子问题
- 预测起始搜索深度
- 决定下一步调用哪个工具

对应当前代码：

- `question_analysis.analyze_question`
- `pipeline.py` 里的阶段调度逻辑

### B. Resolver

负责：

- 实体 mention 对齐到 KG entity
- 关系提示对齐到 KG relation
- 形成后续检索的 seed nodes / relation constraints

对应当前代码：

- `entity_linking.py`
- `sqlite_graph_api.py` 里的 `resolve_entity_candidates` / `resolve_relation_candidates`

### C. Retriever

负责：

- 子图构建
- 多跳路径搜索
- supplement path 搜索
- external graph expansion
- node expansion

对应当前代码：

- `subgraph.py`
- `exploration.py`
- `retrieval/search.py`
- `retrieval/backend.py`

### D. Verifier / Answerer

负责：

- 路径裁剪
- 证据总结
- sufficiency check
- 最终答案生成

对应当前代码：

- `pruning.py`
- `summarization.py`
- `answering.py`

## 4.2 推荐的 Toolkits 拆分

建议画图时把系统拆成下面 6 个工具包：

### Toolkit 1: Question Understanding Toolkit

工具：

- `analyze_question`
- `extract_entities_with_llm`

输入：

- question

输出：

- topic entities
- ordered topic entities
- predicted depth
- sub-questions
- reasoning indicator

### Toolkit 2: Entity / Relation Resolution Toolkit

工具：

- `link_entities_to_kg`
- `link_relations_to_kg`
- `resolve_entity_candidates`
- `resolve_relation_candidates`

输入：

- entity mentions
- relation hints

输出：

- resolved entity ids
- resolved relation ids
- seed nodes

### Toolkit 3: Graph Access Toolkit

工具：

- `KnowledgeGraph.find_paths`
- `KnowledgeGraph.multi_hop_neighbors`
- `SQLiteGraphAPI.get_neighbors`
- `IndexedSQLiteGraphBackend.get_neighbors`

输入：

- seed nodes
- relation filters
- depth / beam / expansion budget

输出：

- raw neighbors
- candidate path fragments

### Toolkit 4: Retrieval Strategy Toolkit

工具：

- `TwoHopExpansionSearcher`
- `ConstrainedBFSSearcher`
- `BeamSearchSearcher`
- `HybridSearcher`
- `explore_topic_entity_paths`
- `explore_supplement_paths`
- `expand_nodes_with_graph_api`

输入：

- seeds
- relation hints
- answer type hints

输出：

- candidate reasoning paths

### Toolkit 5: Evidence Selection Toolkit

工具：

- `fuzzy_select_paths`
- `precise_select_paths_with_llm`
- `branch_reduced_selection`
- `summarize_paths`

输入：

- candidate paths
- question analysis

输出：

- pruned paths
- summarized evidence

### Toolkit 6: Answer Decision Toolkit

工具：

- `check_sufficiency`
- `generate_answer`

输入：

- summarized evidence
- task analysis

输出：

- sufficient / insufficient
- final answer

## 5. 推荐画成的 Agentic Pipeline

下面这个版本最适合交给 GPT 画架构图。

```text
[User Question]
    ->
[Planner Agent]
    - analyze question
    - decompose sub-questions
    - predict search depth
    - decide next tool
    ->
[Resolution Toolkit]
    - entity extraction
    - entity linking
    - relation linking
    - seed construction
    ->
[Retrieval Toolkit]
    - local subgraph search OR external SQLite search
    - topic path exploration
    - supplement path exploration
    - node expansion / beam / bfs / hybrid search
    ->
[Evidence Selection Toolkit]
    - embedding recall pruning
    - LLM precise pruning
    - branch reduction
    - path summarization
    ->
[Verifier Agent]
    - sufficiency check
    ->
if insufficient:
    go back to [Planner Agent]
    - request supplement hints
    - choose deeper / broader search
else:
    ->
[Answer Agent]
    - generate final answer
    ->
[Structured Output]
    - answer
    - predicted_answers
    - summarized_paths
    - search_trace
```

## 6. 更贴近当前代码实现的循环控制逻辑

当前代码本质上是一个固定策略的 agent loop：

```text
Step 1. Analyze question
Step 2. Retrieve initial evidence
Step 3. Prune and summarize evidence
Step 4. Check sufficiency
Step 5. If insufficient, ask LLM for supplement hints
Step 6. Retrieve supplement evidence
Step 7. Re-check sufficiency
Step 8. If still insufficient, expand frontier nodes
Step 9. Re-check sufficiency
Step 10. Generate answer
```

也就是说，它还不是完全开放式 ReAct agent，但已经具备：

- `planning`
- `tool calling`
- `stateful evidence accumulation`
- `iterative verification`

因此可以定义为：

```text
Semi-Agentic KGQA Pipeline
```

或者：

```text
Planner-Guided Retrieval-Augmented KGQA Agent
```

## 7. 画图时建议强调的状态对象

建议把下面几个对象单独画成中间状态框，因为它们是这个系统真正的“工作记忆”。

- `QuestionAnalysisResult`
  - topic entities
  - ordered topic entities
  - predicted depth
  - sub-questions
- `ResolvedCandidate / ResolvedSubQuestion`
  - resolved entity ids / relation ids
- `ReasoningPath`
  - triples
  - nodes
  - source stage
  - path score
- `PipelineResult`
  - answer
  - summarized_paths
  - predicted_answers
  - search_trace

## 8. 一张图里建议怎么分区

建议把图分成 5 个横向泳道：

```text
Lane 1: User / Dataset Input
Lane 2: Planner Agent
Lane 3: Toolkits
Lane 4: Evidence Memory / Search State
Lane 5: Answer Output / Evaluation
```

其中：

- `Planner Agent` 负责决策
- `Toolkits` 负责执行
- `Evidence Memory` 负责保存 `question_analysis / candidate_paths / summarized_paths / search_trace`

## 9. 可直接交给 GPT 画图的 prompt

下面这段可以直接丢给画图模型：

```text
请画一个 KGQA agent pipeline 架构图，风格要求清晰、学术化、模块边界明确。

系统名称：Planner-Guided KGQA Agent

请把系统拆成 5 层：
1. Input Layer
2. Planner Agent Layer
3. Toolkit Layer
4. Evidence Memory Layer
5. Answer Layer

Input Layer 包含：
- User Question
- Dataset Sample (optional: WebQSP / CWQ)

Planner Agent Layer 包含：
- Question Analyzer
- Sub-question Decomposer
- Search Depth Predictor
- Tool Scheduler

Toolkit Layer 包含 6 个工具包：
- Question Understanding Toolkit
- Entity / Relation Resolution Toolkit
- Graph Access Toolkit
- Retrieval Strategy Toolkit
- Evidence Selection Toolkit
- Answer Decision Toolkit

Graph Access Toolkit 要显示两种 backend：
- Local In-Memory Knowledge Graph
- External SQLite Freebase Graph with Runtime Index

Retrieval Strategy Toolkit 要显示：
- Topic Entity Path Exploration
- Supplement Path Exploration
- Node Expansion
- Beam Search
- BFS Search
- Hybrid Search

Evidence Memory Layer 包含：
- QuestionAnalysisResult
- ResolvedSubQuestion State
- Candidate Paths
- Pruned Paths
- Summarized Evidence
- Search Trace

Answer Layer 包含：
- Sufficiency Checker
- Final Answer Generator
- Structured Output JSON

请把主流程画成一个闭环：
Question -> Planning -> Resolution -> Retrieval -> Evidence Selection -> Sufficiency Check
如果 insufficient，则返回 Planner Agent 继续调用 supplement retrieval 或 deeper expansion
如果 sufficient，则进入 Final Answer Generator

请额外标注两个 execution modes：
- Demo Mode: local subgraph search
- Dataset Mode: external SQLite retrieval

图中请强调：
- LLM 负责分析、补充提示、裁剪、总结、作答
- KG / Retrieval backend 负责查图和扩展路径
- Pipeline orchestrator 负责多轮控制
```

## 10. 最后给你的命名建议

如果你要在图标题里写得更像论文/项目，建议用下面任一名称：

- `Planner-Guided KGQA Agent`
- `Tool-Augmented Knowledge Graph Question Answering Pipeline`
- `Agentic KGQA with Iterative Retrieval and Evidence Verification`
- `Semi-Agentic KGQA over Local and External Knowledge Graph Backends`

