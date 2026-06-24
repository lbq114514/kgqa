"""Local vLLM OpenAI-compatible chat client."""

from __future__ import annotations

from typing import Any

import requests

from kgqa.llm.base import BaseLLM
from kgqa.utils.logging import get_logger


class VLLMLLM(BaseLLM):
    """Thin wrapper around a local vLLM chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = f"{self.base_url}/chat/completions"
        self.logger = get_logger(__name__)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the local vLLM endpoint and return the first message content."""
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.logger.debug("Sending prompt to vLLM endpoint %s", self.endpoint)
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
        except requests.RequestException as exc:
            response_text = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    response_text = exc.response.text[:2000]
                except Exception:
                    response_text = ""
            raise RuntimeError(
                f"Failed to reach local vLLM endpoint {self.endpoint!r}: {exc}"
                f"{' | response=' + response_text if response_text else ''}"
            ) from exc
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Malformed vLLM response without choices: {data}")
        return choices[0].get("message", {}).get("content", "").strip()
