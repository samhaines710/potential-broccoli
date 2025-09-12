import os
import threading
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# your existing orchestrator (the file you showed)
from grond_orchestrator import GrondOrchestrator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
LOG = logging.getLogger("server")

app = FastAPI(title="Grond Service", version=os.getenv("SERVICE_VERSION", "1.0.0"))

_worker: Optional[threading.Thread] = None
_orchestrator: Optional[GrondOrchestrator] = None


def _run_orchestrator() -> None:
    global _orchestrator
    try:
        LOG.info("Starting GrondOrchestrator background thread.")
        _orchestrator = GrondOrchestrator()
        _orchestrator.run()  # blocking loop inside the class
    except SystemExit:
        LOG.info("GrondOrchestrator requested exit.")
    except Exception as e:
        LOG.exception("Fatal error in orchestrator: %s", e)
    finally:
        LOG.info("Background orchestrator thread exiting.")


@app.on_event("startup")
def on_startup() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
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