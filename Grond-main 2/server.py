# server.py
# Uses your existing FastAPI app from serve_mlclassifier AND starts the Grond
# orchestrator in a background thread. Also downloads the model before import.
from __future__ import annotations

import os
import sys
import threading
import logging
import traceback
from typing import Any

# Quieter matplotlib in containers
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        )
    else:
        root.setLevel(LOG_LEVEL)


_configure_logging()
LOG = logging.getLogger("server")


def _ensure_model_early() -> None:
    """Download/verify model to /app/models/* using MODEL_PRESIGNED_URL (if set)."""
    try:
        from utils.model_fetch import ensure_model  # type: ignore
        ensure_model()
        LOG.info("Model bootstrap: ensure_model() completed.")
    except Exception as e:
        LOG.warning("Model bootstrap skipped/failed: %s", e)


# Pull the model BEFORE importing serve_mlclassifier (it loads at import time)
_ensure_model_early()


def _attach_ops_endpoints(app_obj) -> None:
    """Add /health and /ready if they aren't present."""
    have_health = any(getattr(r, "path", None) == "/health" for r in getattr(app_obj.router, "routes", []))
    have_ready = any(getattr(r, "path", None) == "/ready" for r in getattr(app_obj.router, "routes", []))
    if not have_health:
        @app_obj.get("/health", tags=["ops"])
        def _health() -> dict[str, Any]:
            return {"status": "ok"}
    if not have_ready:
        @app_obj.get("/ready", tags=["ops"])
        def _ready() -> dict[str, Any]:
            return {"ready": True}


def _attach_orchestrator_startup(app_obj) -> None:
    """Start GrondOrchestrator in a daemon thread on FastAPI startup.
       Stores references on app.state to avoid globals."""
    try:
        from grond_orchestrator import GrondOrchestrator  # type: ignore
    except Exception as e:
        LOG.error("Cannot import GrondOrchestrator: %s", e)
        return

    # Init state holders once
    if not hasattr(app_obj.state, "grond_worker"):
        app_obj.state.grond_worker = None
    if not hasattr(app_obj.state, "grond_orchestrator"):
        app_obj.state.grond_orchestrator = None

    def _ensure_model_late() -> None:
        try:
            from utils.model_fetch import ensure_model  # type: ignore
            ensure_model()
        except Exception as ee:
            LOG.warning("Late model bootstrap skipped/failed: %s", ee)

    def _run_orchestrator() -> None:
        try:
            LOG.info("Bootstrap (late): ensuring model presence.")
            _ensure_model_late()
            LOG.info("Starting GrondOrchestrator background thread.")
            orch = GrondOrchestrator()
            app_obj.state.grond_orchestrator = orch
            orch.run()  # blocking loop
        except SystemExit:
            LOG.info("GrondOrchestrator requested exit.")
        except Exception as ex:
            LOG.exception("Fatal error in orchestrator: %s", ex)
        finally:
            LOG.info("Background orchestrator thread exiting.")

    @app_obj.on_event("startup")
    def _on_startup() -> None:
        # Gate via env for API-only runs
        if os.getenv("START_ORCHESTRATOR", "1").lower() in {"0", "false", "no"}:
            LOG.info("START_ORCHESTRATOR disabled; not starting orchestrator.")
            return
        worker = getattr(app_obj.state, "grond_worker", None)
        if worker is None or not worker.is_alive():
            worker = threading.Thread(target=_run_orchestrator, name="grond-worker", daemon=True)
            app_obj.state.grond_worker = worker
            worker.start()
            LOG.info("Background worker started (orchestrator).")


# Try to use your existing FastAPI app from serve_mlclassifier
try:
    from serve_mlclassifier import app as _ml_app  # FastAPI instance
except Exception as e:
    print(f"[SERVER IMPORT ERROR] serve_mlclassifier import failed: {e}", file=sys.stderr)
    traceback.print_exc()

    # Fallback: minimal app that still tries to run the orchestrator
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Grond Service (fallback)", version=os.getenv("SERVICE_VERSION", "1.0.0"))  # type: ignore
    _attach_orchestrator_startup(app)
    _attach_ops_endpoints(app)

    @app.get("/", tags=["ops"])  # type: ignore
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "service": "grond",
                "version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "warning": "serve_mlclassifier import failed; see logs.",
            }
        )
else:
    # Success: keep your original app AND start the orchestrator on startup
    app = _ml_app  # noqa: F401
    _attach_orchestrator_startup(app)
    _attach_ops_endpoints(app)
    LOG.info("serve_mlclassifier.app loaded; orchestrator will start on startup.")
