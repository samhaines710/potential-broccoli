#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML Classifier for movement_type prediction.

Key hardening for NumPy 2.0 / scikit-learn OneHotEncoder:
- Safe np.isnan shim: returns False for non-numeric/object arrays instead of raising TypeError.
- Strict column alignment to pipeline.feature_names_in_ (if present) without boolean-evaluating arrays.
- Deterministic filling for missing inputs (incl. categorical 'source').
- Returns both 'movement_type' and class 'probs' in [CALL, PUT, NEUTRAL] order.

Environment:
- MODEL_URI (e.g., s3://bucket/path/model.joblib)
- AWS creds via standard env if MODEL_URI is s3://
"""

from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd

from utils.logging_utils import get_logger, write_status, configure_logging

# Ensure logging once
configure_logging()
logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# NumPy isnan shim (NumPy 2.0 object/string arrays raise TypeError)
# ──────────────────────────────────────────────────────────────────────────────

try:
    _orig_isnan = np.isnan  # type: ignore[attr-defined]
except Exception:
    _orig_isnan = None  # type: ignore


def _isnan_safe(x: Any):
    """Return isnan(x) when x is numeric; for object/string inputs return False/zeros."""
    if _orig_isnan is None:
        if isinstance(x, np.ndarray):
            return np.zeros(x.shape, dtype=bool)
        return False
    try:
        return _orig_isnan(x)  # works for numeric arrays and scalars
    except TypeError:
        if isinstance(x, np.ndarray):
            return np.zeros(x.shape, dtype=bool)
        return False


# Monkey-patch numpy.isnan in-process (safe for our runtime)
np.isnan = _isnan_safe  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────────────────────────

CALL_LABEL = "CALL"
PUT_LABEL = "PUT"
NEU_LABEL = "NEUTRAL"
ORDERED_LABELS = [CALL_LABEL, PUT_LABEL, NEU_LABEL]

# ──────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    m = re.match(r"^s3://([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return m.group(1), m.group(2)


def _download_from_s3(s3_uri: str) -> bytes:
    bucket, key = _parse_s3_uri(s3_uri)
    write_status(f"Downloading model from {s3_uri}")
    s3 = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    write_status("Model downloaded successfully.")
    return buf.read()


def _load_pipeline(model_uri: str):
    write_status(f"Loading ML pipeline from {model_uri}")
    if model_uri.startswith("s3://"):
        data = _download_from_s3(model_uri)
        pipe = joblib.load(io.BytesIO(data))
    else:
        pipe = joblib.load(model_uri)
    write_status("ML pipeline loaded successfully.")
    return pipe


# ──────────────────────────────────────────────────────────────────────────────
# Feature-frame utilities
# ──────────────────────────────────────────────────────────────────────────────


def _coerce_val(v: Any) -> Any:
    """Try to coerce numerics to float; leave strings as str; fallback safe."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, (bool, np.bool_)):
        return float(bool(v))
    return str(v)


def _ensure_frame(features: Dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame with coerced values."""
    row = {k: _coerce_val(v) for k, v in features.items()}
    return pd.DataFrame([row])


def _expected_features_for_pipeline(pipeline, df_cols) -> List[str]:
    """
    Determine the expected feature order for the pipeline without evaluating
    any ndarray in a boolean context (avoids ambiguous truth-value errors).
    """
    # 1) Try pipeline-level attribute
    feat = getattr(pipeline, "feature_names_in_", None)
    if feat is not None:
        try:
            return list(feat)
        except Exception:
            pass

    # 2) Try step-level attributes (e.g., preprocessor/ColumnTransformer)
    try:
        steps = getattr(pipeline, "steps", None)
        if steps:
            for _, step in steps:
                step_feat = getattr(step, "feature_names_in_", None)
                if step_feat is not None:
                    try:
                        return list(step_feat)
                    except Exception:
                        continue
    except Exception:
        pass

    # 3) Fallback: current DataFrame columns
    return list(df_cols)


def _align_columns_for_pipeline(df: pd.DataFrame, pipeline) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reindex df to match pipeline expected features.
    Fill known missing fields deterministically.
    Returns (df_aligned, filled_cols_list).
    """
    filled: List[str] = []
    expected = _expected_features_for_pipeline(pipeline, df.columns)

    DEFAULTS: Dict[str, Any] = {
        # categorical
        "source": "fallback",
        "time_of_day": "MIDDAY",
        # numeric defaults
        "breakout_prob": 0.0,
        "recent_move_pct": 0.0,
        "volume_ratio": 1.0,
        "rsi": 50.0,
        "corr_dev": 0.0,
        "skew_ratio": 1.0,
        "yield_spike_2year": 0.0,
        "yield_spike_10year": 0.0,
        "yield_spike_30year": 0.0,
        "delta": 0.0,
        "gamma": 0.0,
        "theta": 0.0,
        "vega": 0.0,
        "rho": 0.0,
        "vanna": 0.0,
        "vomma": 0.0,
        "charm": 0.0,
        "veta": 0.0,
        "speed": 0.0,
        "zomma": 0.0,
        "color": 0.0,
        "implied_volatility": 0.0,
        "theta_day": 0.0,
        "theta_5m": 0.0,
        # microstructure fields that may be absent in some payloads
        "ms_spread_mean": 0.0,
        "ms_depth_imbalance_mean": 0.0,
        "ms_ofi_sum": 0.0,
        "ms_signed_volume_sum": 0.0,
        "ms_vpin": 0.0,
    }

    for col in expected:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, 0.0)
            filled.append(col)

    df = df.reindex(columns=expected)

    # Ensure categorical dtype→str to avoid encoder issues
    for cat_col in ("source", "time_of_day"):
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)

    if filled:
        write_status(f"Filled missing model inputs with defaults: {filled}")

    return df, filled


# ──────────────────────────────────────────────────────────────────────────────
# Class/label utilities
# ──────────────────────────────────────────────────────────────────────────────


def _map_class_index_to_label(classes: np.ndarray, idx: int) -> str:
    """Map estimator.classes_ value at index idx to canonical label."""
    val = classes[idx]
    try:
        ival = int(val)
        mapping = {0: CALL_LABEL, 1: PUT_LABEL, 2: NEU_LABEL}
        if ival in mapping:
            return mapping[ival]
    except Exception:
        pass
    sval = str(val).upper()
    if sval in {CALL_LABEL, PUT_LABEL, NEU_LABEL}:
        return sval
    return NEU_LABEL


def _probs_canonical(classes: np.ndarray, proba_row: np.ndarray) -> List[float]:
    """
    Map model probs → [CALL, PUT, NEUTRAL] in a robust way.

    - If the model is tri-class, return the three probabilities in canonical order.
    - If the model is binary (missing NEUTRAL), set p_neutral=0.0 and
      renormalize [p_call, p_put] to sum to 1.0 to avoid leaking mass.
    - If a class is missing entirely, its probability is 0.0.
    """
    classes = np.asarray(classes)
    idx = {str(c).upper(): i for i, c in enumerate(classes)}

    p_call = float(proba_row[idx["CALL"]]) if "CALL" in idx else 0.0
    p_put = float(proba_row[idx["PUT"]]) if "PUT" in idx else 0.0
    p_neu = float(proba_row[idx["NEUTRAL"]]) if "NEUTRAL" in idx else 0.0

    # If NEUTRAL is missing (typical binary model), renormalize CALL/PUT.
    if "NEUTRAL" not in idx:
        s2 = p_call + p_put
        if s2 > 0:
            p_call, p_put = p_call / s2, p_put / s2
        p_neu = 0.0
        return [p_call, p_put, p_neu]

    # Tri-class path: soft-normalize against numeric drift
    s = p_call + p_put + p_neu
    if s > 0:
        return [p_call / s, p_put / s, p_neu / s]
    return [p_call, p_put, p_neu]


# ──────────────────────────────────────────────────────────────────────────────
# Public classifier
# ──────────────────────────────────────────────────────────────────────────────


class MLClassifier:
    def __init__(self) -> None:
        model_uri = os.getenv("MODEL_URI", "s3://bucketbuggypie/models/xgb_classifier.pipeline.joblib")
        self.pipeline = _load_pipeline(model_uri)

    def classify(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single feature dict.
        Returns:
            {
              "movement_type": "CALL"|"PUT"|"NEUTRAL",
              "probs": [p_call, p_put, p_neutral]
            }
        """
        df = _ensure_frame(features)
        df, _ = _align_columns_for_pipeline(df, self.pipeline)

        # Predict probabilities
        proba = self.pipeline.predict_proba(df)[0]

        # Obtain classes_ from pipeline or final estimator
        classes = getattr(self.pipeline, "classes_", None)
        if classes is None:
            final_est = getattr(self.pipeline, "steps", [[None, None]])[-1][1]
            classes = getattr(final_est, "classes_", np.array(["CALL", "PUT", "NEUTRAL"]))

        # Canonicalize probabilities into [CALL, PUT, NEUTRAL],
        # supporting both tri-class and binary models
        probs = _probs_canonical(np.asarray(classes), np.asarray(proba))

        # Argmax on canonical order yields the final label
        label_idx = int(np.argmax(probs))
        label = ["CALL", "PUT", "NEUTRAL"][label_idx]

        return {"movement_type": label, "probs": probs}
