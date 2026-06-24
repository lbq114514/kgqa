# KGQA

一个 PoG-style 的 training-free 知识图谱问答系统。系统输入自然语言问题和三元组知识图谱，输出答案、支撑推理路径，以及中间搜索过程。

## 特性

- 仅支持本地 vLLM 的 OpenAI-compatible 服务端
- 不使用 OpenAI API
- 不训练任何模型，不做 fine-tune / gradient update
- 使用预训练 embedding model、图搜索与 prompt inference
- 模块化设计，便于替换 KG、embedding model 和 pruning 策略

## Pipeline

```text
Question
-> Topic Entity Recognition
-> Question Subgraph Detection
-> Question Analysis
-> Topic Entity Path Exploration
-> LLM Supplement Path Exploration
-> Node Expand Exploration
-> Path Pruning
-> Path Summarization
-> Question Answering
```

## 关键概念

- `Dmax`：人工设定的最大搜索深度，是 question subgraph 和后续搜索的硬上限。
- `Dpredict`：LLM 在 Question Analysis 阶段基于问题复杂度预测的推理深度。
- 搜索从 `min(Dpredict, Dmax)` 开始，最多扩展到 `Dmax`。
- LLM Supplement Path Exploration 不训练模型，只让 LLM 提出潜在 bridge entities，再由 KG 路径搜索验证。
- Path Pruning 使用预训练 embedding model 和本地 vLLM prompt，不更新参数。

## 启动本地 vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

默认配置使用：

- model: `Qwen/Qwen2.5-7B-Instruct`
- base URL: `http://localhost:8000/v1`

## 安装依赖

建议在 `kgqa` conda 环境中安装：

```bash
pip install -r requirements.txt
```

## GitHub 上传前后的配置方式

仓库默认提交的是示例配置，不提交你本机的绝对路径配置。

首次使用时先复制：

```bash
cp config.example.yaml config.yaml
cp cwq_validate_relation_beam_guarded.example.yaml cwq_validate_relation_beam_guarded.yaml
```

然后把本地模型路径、SQLite 数据库路径、runtime index 路径和数据集路径改成你自己的环境。

## 运行 Demo

```bash
python examples/run_demo.py
```

或直接运行主程序：

```bash
python main.py \
  --question "What country bordering France contains an airport that serves Nijmegen?" \
  --config config.yaml
```

## WebQSP Validation

支持接入本地 WebQSP `validation.jsonl`，默认直接使用外部 SQLite Freebase 图数据库运行。

- 默认是 end-to-end LLM entity extraction，不使用数据集自带的 `q_entity` 作为推理输入
- 默认配置下 `WebQSP` 和 `CWQ` 都统一走外部 SQLite 图数据库
- 评测输出包含 `exact match`、`hit@1` 和集合级 `precision / recall / F1`
- 逐题结果会写到 `outputs/webqsp_validation/sample_predictions/`
- 汇总指标会写到 `outputs/webqsp_validation/metrics.json`

单题运行：

```bash
python main.py \
  --dataset webqsp \
  --mode single \
  --split validation \
  --index 0 \
  --config config.yaml
```

批量 validation：

```bash
python main.py \
  --dataset webqsp \
  --mode validate \
  --split validation \
  --limit 10 \
  --config config.yaml
```

## Optional SQLite GraphAPI

可以把外部 Freebase SQLite 图后端作为可选增强，只作用在：

- `LLM Supplement Path Exploration`
- `Node Expand Exploration`

默认关闭，不影响原始 pipeline。

FB2M / FB5M 风格 CSV 可直接构建：

```bash
python examples/build_fb2m_sqlite.py \
  --entities-csv /home/ubuntu/research/hdd01/QADatasets/freebase/processed_fb2m/entities.csv \
  --triples-csv /home/ubuntu/research/hdd01/QADatasets/freebase/processed_fb2m/triples.csv \
  --db-path /home/ubuntu/research/hdd01/QADatasets/freebase/processed_fb2m/fb2m.sqlite
```

当前推荐直接用 `processed_nobase` 产物构建统一数据库：

```bash
python examples/build_freebase_processed_sqlite.py \
  --entities-csv /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/entities.csv \
  --triples-csv /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/triplets.csv \
  --db-path /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/freebase_processed.sqlite
```

然后构建 runtime retrieval index：

```bash
python examples/build_freebase_runtime_index.py \
  --db-path /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/freebase_processed.sqlite \
  --index-dir /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/runtime_index
```

这份 `processed_nobase/triplets.csv` 已经包含：

- literal 属性边
- CVT 展开后的复合 relation
- 边表自带的 `head_name` / `tail_name`

运行时推荐在 `config.yaml` 中使用：

```yaml
graphapi:
  enabled: true
  backend: sqlite
  db_path: /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/freebase_processed.sqlite

retrieval:
  backend: indexed_sqlite
  index_dir: /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/runtime_index
```

`graphapi.fb2m_db_path` 仍兼容旧配置，但新配置建议统一使用 `graphapi.db_path`。

默认配置已经把 `WebQSP` 和 `CWQ` 指到这份数据库。

单题 WebQSP：

```bash
python main.py \
  --dataset webqsp \
  --mode single \
  --split validation \
  --index 7 \
  --config config.yaml
```

其中 `index 7` 对应当前本地 `validation.jsonl` 中的 `WebQTrn-143`。

批量 WebQSP validation：

```bash
python main.py \
  --dataset webqsp \
  --mode validate \
  --split validation \
  --limit 100 \
  --qa-concurrency 6 \
  --config config.yaml
```

单题 CWQ：

```bash
python main.py \
  --dataset cwq \
  --mode single \
  --split test \
  --index 0 \
  --config config.yaml
```

批量 CWQ validation：

```bash
python main.py \
  --dataset cwq \
  --mode validate \
  --split test \
  --limit 100 \
  --qa-concurrency 6 \
  --config config.yaml
```

如果更习惯脚本方式，也可以直接运行：

```bash
bash scripts/run_webqsp_validate_graphapi.sh --limit 100
bash scripts/run_cwq_validate_graphapi.sh --limit 100
```

## Retrieval Toolkit

当前仓库还提供一组独立的 KGQA 检索工具：

```bash
python -m kgqa.tools.build_sqlite --help
python -m kgqa.tools.build_index --help
python -m kgqa.tools.inspect_entity --help
python -m kgqa.tools.search_paths --help
```

例如直接搜索路径：

```bash
python -m kgqa.tools.search_paths \
  --db-path /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/freebase_processed.sqlite \
  --index-dir /home/ubuntu/research/hdd01/QADatasets/freebase/freebase_full/processed_nobase/runtime_index \
  --seed "Barack Obama" \
  --relation-hint "place of birth" \
  --strategy hybrid
```

## 输出

系统输出完整 JSON，包括：

- `question`
- `topic_entities`
- `dmax`
- `dpredict`
- `candidate_paths`
- `pruned_paths`
- `summarized_paths`
- `answer`
- `sufficient`
- `confidence`
- `search_trace`

WebQSP prediction 还会额外包含：

- `sample_id`
- `gold_answers`
- `predicted_answers`
- `metrics`

## 扩展方向

- 替换 `data/` 中的 triples 文件即可接入新的 KG
- 替换 `embedding.model` 可切换 embedding model
- 在 `reasoning/pruning.py` 中替换 pruning 策略
- 当前结构已接入 WebQSP validation，并为 CWQ 继续预留了 loader 和 pipeline 扩展接口
