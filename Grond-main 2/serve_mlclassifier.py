#!/usr/bin/env python3
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ml_classifier import MLClassifier

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment-driven settings
MODEL_PATH = os.getenv(
    "ML_MODEL_PATH", "/app/models/xgb_classifier.pipeline.joblib"
)
API_PORT = int(os.getenv("API_PORT", "9000"))

# Load classifier once at startup
try:
    logger.info(f"Loading MLClassifier from {MODEL_PATH}")
    classifier = MLClassifier(MODEL_PATH)
    logger.info("MLClassifier loaded successfully.")
except Exception as e:
    logger.error(f"Failed to initialize MLClassifier: {e!r}")
    raise

# Build FastAPI app
app = FastAPI(title="Movement Classification Service")


class PredictRequest(BaseModel):
    data: dict


class PredictResponse(BaseModel):
    movement_type: str
    expected_move_pct: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Accepts JSON { "data": { feature_name: value, ... } }
    Returns predicted movement_type and expected_move_pct.
    """
    try:
        result = classifier.classify(req.data)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e!r}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
