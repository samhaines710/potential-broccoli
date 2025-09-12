# server.py
# Purpose:
# 1) Make uvicorn imports bullet-proof by re-exporting a FastAPI `app` if your
#    existing module (serve_mlclassifier.py) already defines it.
# 2) If that import fails for any reason (path/package/import error), fall back
#    to a self-contained FastAPI app that starts your Grond orchestrator in a
#    background thread and exposes /health, /ready, and / endpoints.
#
# Usage with uvicorn (Render):
#   APP_SERVER=uvicorn
#   APP_MODULE=server:app
#   WEB_CONCURRENCY=1

import os
import sys
import threading
import logging
import traceback
from typing import Optional

# ---------- Try the simple path first: re-export an existing FastAPI app ----------
try:
    # If your file 'serve_mlclassifier.py' exists next to this file and defines
    #   app = FastAPI(...)
    # this import will succeed and we simply re-export `app` for uvicorn.
    from serve_mlclassifier import app as app  # noqa: F401
    # If we got here, uvicorn will use this `app` and the rest of the file won’t matter.
    # Keeping the remainder loaded is harmless.
except Exception as import_err:
    # Print full traceback so Render logs show the root cause (missing package, etc.)
    print(f"[SERVER IMPORT ERROR] Failed to import serve_mlclassifier: {import_err}", file=sys.stderr)
    traceback.print_exc()

    # ---------- Robust fallback: construct a FastAPI app and start orchestrator ----------
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    # Optional model prefetch helper. If not present, we continue without it.
    def _ensure_model_if_available() -> None:
        try:
            from utils.model_fetch import ensure_model  # type: ignore
        except Exception as e:
            logging.getLogger("server").warning("Model bootstrap helper not available: %s", e)
            return
        try:
            ensure_model()
        except Exception as e:
            logging.getLogger("server").warning("Model download/verify failed: %s", e)

    # Import your orchestrator (the trading loop)
    try:
        from grond_orchestrator import GrondOrchestrator  # type: ignore
    except Exception as e:
        # If even the orchestrator can't be imported, expose an app that reports the error.
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
        _tmp_app = FastAPI(title="Grond Service (import error)", version="0")

        @_tmp_app.get("/health")
        def _health_err() -> JSONResponse:
            return JSONResponse({"status": "degraded", "error": f"orchestrator import failed: {e}"}, status_code=500)

        @_tmp_app.get("/")
        def _root_err() -> JSONResponse:
            return JSONResponse({"service": "grond", "error": f"orchestrator import failed: {e}"}, status_code=500)

        app = _tmp_app  # noqa: F401
    else:
        # Normal fallback path: run orchestrator in a background thread and expose ops endpoints.
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        )
        LOG = logging.getLogger("server")

        app = FastAPI(title="Grond Service", version=os.getenv("SERVICE_VERSION", "1.0.0"))  # noqa: F401

        _worker: Optional[threading.Thread] = None
        _orchestrator: Optional[GrondOrchestrator] = None

        def _run_orchestrator() -> None:
            """Blocking trading loop in a daemon thread."""
            nonlocal _orchestrator
            try:
                LOG.info("Bootstrap: ensuring model presence (if configured).")
                _ensure_model_if_available()

                LOG.info("Starting GrondOrchestrator background thread.")
                _orchestrator = GrondOrchestrator()
                _orchestrator.run()  # blocking loop
            except SystemExit:
                LOG.info("GrondOrchestrator requested exit.")
            except Exception as e:
                LOG.exception("Fatal error in orchestrator: %s", e)
            finally:
                LOG.info("Background orchestrator thread exiting.")

        @app.on_event("startup")
        def _on_startup() -> None:
            nonlocal _worker
            if _worker is None or not _worker.is_alive():
                # Important: set WEB_CONCURRENCY=1 in Render so only one process runs this.
                _worker = threading.Thread(target=_run_orchestrator, name="grond-worker", daemon=True)
                _worker.start()
                LOG.info("Background worker started.")

        @app.get("/health", tags=["ops"])
        def health() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        @app.get("/ready", tags=["ops"])
        def ready() -> JSONResponse:
            # Optionally verify model presence here:
            # ready = os.path.exists("/app/models/xgb_classifier.pipeline.joblib")
            return JSONResponse({"ready": True})

        @app.get("/", tags=["ops"])
        def root() -> JSONResponse:
            return JSONResponse({"service": "grond", "version": os.getenv("SERVICE_VERSION", "1.0.0")})
