"""Abstract LLM interface used by every reasoning stage."""

from __future__ import annotations


class BaseLLM:
    """Minimal text generation interface."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
