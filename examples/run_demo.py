"""Run the sample KGQA demo question."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgqa.kg.loader import load_knowledge_graph
from kgqa.llm.vllm_client import VLLMLLM
from kgqa.reasoning.pipeline import KGQAPipeline
from kgqa.utils.logging import configure_logging


def main() -> None:
    """Run the demo pipeline against the bundled sample question."""
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    with (PROJECT_ROOT / config["demo"]["question_path"]).open("r", encoding="utf-8") as handle:
        question_payload = json.load(handle)[0]

    kg = load_knowledge_graph(PROJECT_ROOT / config["kg"]["path"])
    llm = VLLMLLM(
        model=config["llm"]["model"],
        base_url=config["llm"]["base_url"],
        api_key=config["llm"].get("api_key", "EMPTY"),
        temperature=config["llm"].get("temperature", 0.0),
        max_tokens=config["llm"].get("max_tokens", 1024),
    )
    pipeline = KGQAPipeline(kg=kg, llm=llm, config=config)
    result = pipeline.run(question_payload["question"])
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
