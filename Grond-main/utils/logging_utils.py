"""Centralized, JSON-formatted logging utilities (singleton configuration).

Exports:
- configure_logging(): install a single JSON StreamHandler on the root logger.
- get_logger(): module logger that uses the root’s handler (no duplicates).
- write_status(): convenience logger that attributes to the caller (stacklevel=2).
- set_level(): adjust root log level dynamically.
- REST_CALLS: Counter(service, endpoint, method, status) — outbound REST calls.
- REST_LATENCY: Histogram(service, endpoint, method) — REST latency seconds.
- REST_429: Counter(service, endpoint) — HTTP 429 rate-limit events.

Design goals:
- Avoid duplicate handlers / double-logging across modules.
- Never crash if prometheus_client is unavailable (no-op shims).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

__all__ = [
    "configure_logging",
    "get_logger",
    "write_status",
    "set_level",
    "REST_CALLS",
    "REST_LATENCY",
    "REST_429",
]

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus metrics (safe no-op shims if prometheus_client is unavailable)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Histogram  # type: ignore
except Exception:
    class _NoopMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "._NoopMetric":  # type: ignore
            return self
        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass
        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass
    def Counter(*args: Any, **kwargs: Any) -> _NoopMetric:  # type: ignore
        return _NoopMetric()
    def Histogram(*args: Any, **kwargs: Any) -> _NoopMetric:  # type: ignore
        return _NoopMetric()

# Outbound REST call counter (service/endpoint/method/status labeled)
REST_CALLS = Counter(
    "rest_calls_total",
    "Count of outbound REST API calls",
    ["service", "endpoint", "method", "status"],
)

# Outbound REST call latency (in seconds)
REST_LATENCY = Histogram(
    "rest_call_latency_seconds",
    "Latency of outbound REST API calls in seconds",
    ["service", "endpoint", "method"],
)

# Rate-limit events (HTTP 429)
# Labels: REST_429.labels(service, endpoint).inc()
REST_429 = Counter(
    "rest_429_total",
    "Count of HTTP 429 (rate-limited) responses from outbound REST calls",
    ["service", "endpoint"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────

_CONFIGURED = False

class JsonFormatter(logging.Formatter):
    """JSON line formatter with timestamp, level, module, message."""
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S,%f"
        )[:-3]
        payload: Dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "module": record.name or record.module,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)

def configure_logging(level: int = logging.INFO) -> None:
    """Install a single JSON StreamHandler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Remove pre-existing handlers to prevent duplicate lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    logging.captureWarnings(True)
    _CONFIGURED = True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger that uses the root’s single handler."""
    return logging.getLogger(name if name else __name__)

def write_status(msg: str, level: int = logging.INFO) -> None:
    """Log a status line, attributed to the caller (stacklevel=2 on 3.8+)."""
    logger = logging.getLogger(__name__)
    try:
        logger.log(level, msg, stacklevel=2)
    except TypeError:
        logger.log(level, msg)

def set_level(level: int) -> None:
    """Dynamically adjust root log level."""
    logging.getLogger().setLevel(level)
