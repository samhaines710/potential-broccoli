import os
import sys
import threading
import logging
import traceback
from typing import Optional

# First, try to re-export the FastAPI app from serve_mlclassifier.py if it exists.
try:
    from serve_mlclassifier import app as app  # noqa: F401
except Exception as e:
    # Surface the actual import error in logs, then fall back to a minimal ASGI app.
    print(f"[SERVER IMPORT ERROR] serve_mlclassifier import failed: {e}", file=sys.stderr)
    traceback.print_exc()

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    def _ensure_model() -> None:
        """Optional model bootstrap via presigned URL; safe to no-op if helper missing."""
        try:
            from utils.model_fetch import ensure_model  # type: ignore
            ensure_model()
        except Exception as ee:
            logging.getLogger("server").warning("Model bootstrap skipped/failed: %s", ee)

    try:
        from grond_orchestrator import GrondOrchestrator  # type: ignore
    except Exception as ee:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
        app = FastAPI(title="Grond Service (import error)", version="0")  # noqa: F401

        @app.get("/health")
        def _health_err() -> JSONResponse:
            return JSONResponse({"status": "degraded", "error": f"orchestrator import failed: {ee}"}, status_code=500)

        @app.get("/")
        def _root_err() -> JSONResponse:
            return JSONResponse({"service": "grond", "error": f"orchestrator import failed: {ee}"}, status_code=500)
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        )
        LOG = logging.getLogger("server")
        app = FastAPI(title="Grond Service", version=os.getenv("SERVICE_VERSION", "1.0.0"))  # noqa: F401

        _worker: Optional[threading.Thread] = None
        _orchestrator: Optional[GrondOrchestrator] = None

        def _run_orchestrator() -> None:
            nonlocal _orchestrator
            try:
                LOG.info("Bootstrap: ensuring model presence.")
                _ensure_model()
                LOG.info("Starting GrondOrchestrator background thread.")
                _orchestrator = GrondOrchestrator()
                _orchestrator.run()  # blocking loop
            except SystemExit:
                LOG.info("GrondOrchestrator requested exit.")
            except Exception as ex:
                LOG.exception("Fatal error in orchestrator: %s", ex)
            finally:
                LOG.info("Background orchestrator thread exiting.")

        @app.on_event("startup")
        def _on_startup() -> None:
            nonlocal _worker
            if _worker is None or not _worker.is_alive():
                # IMPORTANT: keep WEB_CONCURRENCY=1 in Render
                _worker = threading.Thread(target=_run_orchestrator, name="grond-worker", daemon=True)
                _worker.start()
                LOG.info("Background worker started.")

        @app.get("/health", tags=["ops"])
        def health() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        @app.get("/ready", tags=["ops"])
        def ready() -> JSONResponse:
            return JSONResponse({"ready": True})

        @app.get("/", tags=["ops"])
        def root() -> JSONResponse:
            return JSONResponse({"service": "grond", "version": os.getenv("SERVICE_VERSION", "1.0.0")})
