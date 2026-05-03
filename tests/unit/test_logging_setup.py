from __future__ import annotations

import structlog

from churn.logging_setup import configure_logging, get_logger


def test_configure_logging_is_idempotent_and_emits():
    configure_logging()
    configure_logging()  # second call must not raise
    log = get_logger("test")
    log.info("hello", k="v")  # smoke test — no exception is the contract here


def test_get_logger_returns_a_structlog_bound_logger():
    log = get_logger("test")
    # structlog's bound loggers expose .bind() and the standard level methods
    assert hasattr(log, "bind")
    assert hasattr(log, "info")
    assert hasattr(log, "error")
    bound = log.bind(request_id="abc")
    assert isinstance(bound, structlog.types.BindableLogger)
