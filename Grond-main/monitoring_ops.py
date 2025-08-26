"""Monitoring operations for Prometheus and Flask endpoints.

This module configures structured logging and exposes Prometheus counters,
histograms, and gauges. It also starts a metrics server and a Flask app
that provides `/metrics` and `/health` endpoints.
"""

from __future__ import annotations

import errno
import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, request
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from config import METRICS_PORT, HTTP_PORT, SERVICE_NAME

# Module-level flags to enforce idempotency
_metrics_server_started = False
_http_server_started   = False


def setup_logging() -> None:
    """Set up structured JSON logging for the entire application."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            (
                '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
                '"module":"%(module)s","message":"%(message)s"}'
            )
        )
    )
    root.handlers.clear()
    root.addHandler(handler)


# ── Prometheus Metrics ─────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter(
    f"{SERVICE_NAME}_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint"],
)
REQUEST_LATENCY = Histogram(
    f"{SERVICE_NAME}_http_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
)
IN_PROGRESS     = Gauge(
    f"{SERVICE_NAME}_inprogress_requests",
    "In-flight HTTP requests",
)

# ── Flask App ───────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.before_request
def before_request() -> None:
    """Increment in-progress gauge and record start time."""
    IN_PROGRESS.inc()
    request._start_time = datetime.utcnow()


@app.after_request
def after_request(response):
    """Record latency, count the request, and decrement in-progress gauge."""
    elapsed = (datetime.utcnow() - request._start_time).total_seconds()
    REQUEST_LATENCY.labels(endpoint=request.path).observe(elapsed)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path).inc()
    IN_PROGRESS.dec()
    return response


@app.route("/metrics")
def metrics():
    """Expose Prometheus metrics in text format."""
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health")
def health():
    """Simple JSON health‐check endpoint."""
    return jsonify(status="ok", time=datetime.utcnow().isoformat() + "Z"), 200


def start_monitoring_server() -> None:
    """
    Launch Prometheus on METRICS_PORT (idempotent, swallow EADDRINUSE),
    then launch Flask on HTTP_PORT (idempotent, swallow EADDRINUSE).
    """
    global _metrics_server_started, _http_server_started

    setup_logging()
    logger = logging.getLogger("monitoring_ops")

    # ── Start Prometheus metrics endpoint ──────────────────────────────────
    if not _metrics_server_started:
        logger.info("Starting Prometheus metrics server on port %s", METRICS_PORT)
        try:
            start_http_server(METRICS_PORT)
            _metrics_server_started = True
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EADDRINUSE:
                logger.warning(
                    "Metrics port %s already in use; skipping metrics server start.",
                    METRICS_PORT,
                )
                _metrics_server_started = True
            else:
                logger.error(
                    "Failed to start Prometheus server on port %s: %r",
                    METRICS_PORT,
                    exc,
                )
                raise
    else:
        logger.debug("Prometheus metrics server already started; skipping.")

    # ── Start Flask HTTP server ───────────────────────────────────────────
    if not _http_server_started:
        logger.info("Starting Flask health & metrics server on port %s", HTTP_PORT)
        def _run_flask() -> None:
            try:
                app.run(host="0.0.0.0", port=HTTP_PORT, debug=False, use_reloader=False)
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.EADDRINUSE:
                    logging.getLogger("monitoring_ops").warning(
                        "HTTP port %s already in use; skipping Flask server start.",
                        HTTP_PORT,
                    )
                else:
                    raise

        thread = threading.Thread(target=_run_flask, daemon=True)
        thread.start()
        _http_server_started = True
    else:
        logger.debug("Flask HTTP server already started; skipping.")
