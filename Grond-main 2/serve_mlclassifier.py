"""
FastAPI app exposing inference endpoints backed by MLClassifier (binary).

Endpoints:
- GET  /          -> service/version
- GET  /health    -> liveness
- GET  /ready     -> readiness
- POST /predict   -> { features: {...} } → { p_up, movement_type, top_contributions }
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_classifier import MLClassifier
from utils.feature_sanitizer import sanitize_features

LOG = logging.getLogger("serve_mlclassifier")

SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")

app = FastAPI(title="Grond ML API", version=SERVICE_VERSION)

# Load classifier once
try:
    classifier = MLClassifier(os.getenv("ML_MODEL_PATH"))
    LOG.info("MLClassifier loaded successfully.")
except Exception as e:
    LOG.error("Failed to initialize MLClassifier: %r", e)
    raise


class PredictIn(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "grond-ml", "version": SERVICE_VERSION}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    return {"ready": True}


@app.post("/predict")
def predict(body: PredictIn) -> Dict[str, Any]:
    try:
        clean = sanitize_features(body.features)
        out = classifier.classify(clean)
        return out
    except Exception as e:
        LOG.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
