"""Logging configuration for the KGQA project."""

from __future__ import annotations

import logging


class _GraphAPILogFilter(logging.Filter):
    """Drop verbose GraphAPI log lines while keeping other project logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        noisy_logger_prefixes = (
            "kgqa.kg.sqlite_graph_api",
        )
        return not any(record.name.startswith(prefix) for prefix in noisy_logger_prefixes)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the CLI and examples."""
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    graphapi_filter = _GraphAPILogFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(graphapi_filter)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""
    return logging.getLogger(name)
