"""Structured logging configuration.

JSON-by-default so logs are machine-parseable in CI / containers; switch to
``console`` via ``LOG_FORMAT=console`` for readable local development output.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog

from churn.config import get_settings


def configure_logging() -> None:
    """Idempotently configure structlog. Safe to call multiple times."""
    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    # ``add_logger_name`` is intentionally omitted: it reads ``logger.name``,
    # which exists on stdlib loggers but not on structlog's ``PrintLogger``.
    # Callers that want a logger name in the event can ``log.bind(logger="...")``.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger; call ``configure_logging`` once at process startup first."""
    # ``structlog.get_logger`` is typed as returning ``Any`` in upstream stubs.
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
