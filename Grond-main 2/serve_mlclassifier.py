#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_classifier import HCBCClassifier

BUNDLE_PATH = os.environ.get("ML_MODEL_PATH", "Resources/xgb_hcbc.bundle.joblib")

app = FastAPI(title="HCBC Inference API", version="1.0")

class ScoreRequest(BaseModel):
    H: int
    features: Dict[str, Any]

class ScoreResponse(BaseModel):
    p_up: float
    decision: str
    H: int
    thresholds: Dict[str, float]

@app.on_event("startup")
def _load_model():
    global clf
    try:
        clf = HCBCClassifier(BUNDLE_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if req.H <= 0:
        raise HTTPException(status_code=422, detail="H must be positive integer")
    try:
        r = clf.score(req.features, int(req.H))
        return ScoreResponse(p_up=r.p_up, decision=r.decision, H=r.H, thresholds=r.thresholds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
def health():
    return {"status": "ok"}