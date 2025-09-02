# tests/test_ml_classifier.py

import os
import pytest
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from ml_classifier import MLClassifier

MODEL_PATH = "models/xgb_classifier.pipeline.joblib"

@pytest.fixture(scope="session", autouse=True)
def ensure_model_exists():
    # Create a dummy pipeline if none exists
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", RandomForestClassifier(n_estimators=1, random_state=42))
        ])
        pipe.feature_names = ["delta", "gamma", "rsi"]
        joblib.dump(pipe, MODEL_PATH)
    yield

def test_ml_classifier_loads():
    clf = MLClassifier(model_path=MODEL_PATH)
    assert hasattr(clf, "pipeline")
    assert isinstance(clf.feature_names, list)
    assert len(clf.feature_names) > 0

def test_ml_classifier_classify():
    clf = MLClassifier(model_path=MODEL_PATH)
    sample = {fn: 0.0 for fn in clf.feature_names}
    out = clf.classify(sample)
    assert "movement_type" in out
    assert "expected_move_pct" in out
    assert isinstance(out["movement_type"], str)
    assert isinstance(out["expected_move_pct"], float)
