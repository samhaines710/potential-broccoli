# server.py
# Runs your ML API (serve_mlclassifier.app) AND starts the Grond orchestrator in the same
# process. It also downloads the model BEFORE importing serve_mlclassifier so import-time
# model loads succeed. Designed for Render with:
#   APP_SERVER=uvicorn
#   APP_MODULE=server:app
#   WEB_CONCURRENCY=1
from __future__ import annotations

import os
import sys
import threading
import logging
import traceback
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Environment / logging basics
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")  # quiet matplotlib in containers
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def _configure_logging() -> None:
    # idempotent root logger setup
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


# ──────────────────────────────────────────────────────────────────────────────
# Model bootstrap BEFORE importing anything that loads the model at import time
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_model_early() -> None:
    """Download/verify the model to /app/models/* using MODEL_PRESIGNED_URL if set."""
    try:
        from utils.model_fetch import ensure_model  # local helper (safe if missing)
        ensure_model()
        LOG.info("Model bootstrap: ensure_model() completed.")
    except Exception as e:
        # Do not crash: some endpoints may not need the model; log loudly.
        LOG.warning("Model bootstrap skipped/failed: %s", e)


_ensure_model_early()


# ──────────────────────────────────────────────────────────────────────────────
# Try to import your existing FastAPI app (serve_mlclassifier.app)
# If that fails, fall back to a minimal app and still try to run the orchestrator.
# ──────────────────────────────────────────────────────────────────────────────
app = None  # will be assigned below


def _start_orchestrator_on(app_obj) -> None:
    """Attach a startup hook to start GrondOrchestrator in a background thread."""
    try:
        from grond_orchestrator import GrondOrchestrator  # type: ignore
    except Exception as e:
        LOG.error("Cannot import GrondOrchestrator: %s", e)
        return

    # module-level globals so handlers can mutate
    global _worker, _orchestrator
    _worker: Optional[threading.Thread] = None
    _orchestrator: Optional[GrondOrchestrator] = None

    def _ensure_model_late() -> None:
        try:
            from utils.model_fetch import ensure_model  # type: ignore
            ensure_model()
        except Exception as ee:
            LOG.warning("Late model bootstrap skipped/failed: %s", ee)

    def _run_orchestrator() -> None:
        global _orchestrator
        try:
            LOG.info("Bootstrap (late): ensuring model presence.")
            _ensure_model_late()
            LOG.info("Starting GrondOrchestrator background thread.")
            _orchestrator = GrondOrchestrator()
            _orchestrator.run()  # blocking loop
        except SystemExit:
            LOG.info("GrondOrchestrator requested exit.")
        except Exception as ex:
            LOG.exception("Fatal error in orchestrator: %s", ex)
        finally:
            LOG.info("Background orchestrator thread exiting.")

    @app_obj.on_event("startup")
    def _on_startup() -> None:
        # Gate via env if you ever need API-only mode
        if os.getenv("START_ORCHESTRATOR", "1").lower() in {"0", "false", "no"}:
            LOG.info("START_ORCHESTRATOR disabled; not starting orchestrator.")
            return
        global _worker
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run_orchestrator, name="grond-worker", daemon=True)
            _worker.start()
            LOG.info("Background worker started (orchestrator).")


def _attach_ops_endpoints(app_obj) -> None:
    """Add /health and /ready if they don't already exist."""
    have_health = any(getattr(r, "path", None) == "/health" for r in getattr(app_obj.router, "routes", []))
    have_ready = any(getattr(r, "path", None) == "/ready" for r in getattr(app_obj.router, "routes", []))
    if not have_health:
        @app_obj.get("/health", tags=["ops"])
        def _health() -> dict:
            return {"status": "ok"}
    if not have_ready:
        @app_obj.get("/ready", tags=["ops"])
        def _ready() -> dict:
            return {"ready": True}


# Attempt the fast path: import the existing ML API app
try:
    from serve_mlclassifier import app as _ml_app  # FastAPI instance provided by your module
except Exception as e:
    # Surface the real reason; keep the service alive with a diagnostic app
    print(f"[SERVER IMPORT ERROR] serve_mlclassifier import failed: {e}", file=sys.stderr)
    traceback.print_exc()

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Grond Service (fallback)", version=os.getenv("SERVICE_VERSION", "1.0.0"))  # type: ignore

    # Try to start orchestrator even in fallback mode
    _start_orchestrator_on(app)
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
    # Success: use your ML API app AND start the orchestrator on startup
    app = _ml_app  # use the app from serve_mlclassifier
    _start_orchestrator_on(app)
    _attach_ops_endpoints(app)
    LOG.info("serve_mlclassifier.app loaded; orchestrator will start on startup.")
