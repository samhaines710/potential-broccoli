"""
Binary ML classifier wrapper for live inference.

- Expects a persisted sklearn Pipeline with an XGBClassifier as the final step.
- Computes p_up and maps to CALL/PUT/NEUTRAL via thresholds (policy).
- Logs model summary and per-instance top feature contributions via XGBoost TreeSHAP.

Env:
  ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib
  TAU_BUY=0.60
  TAU_SELL=0.40
  LOG_FEATURE_CONTRIBS_TOP_N=8
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("ml_classifier")

MODEL_PATH = os.getenv("ML_MODEL_PATH", "/app/models/xgb_classifier.pipeline.joblib")

# Threshold policy (binary → tri-action)
TAU_BUY  = float(os.getenv("TAU_BUY",  "0.60"))
TAU_SELL = float(os.getenv("TAU_SELL", "0.40"))

TOP_N = int(os.getenv("LOG_FEATURE_CONTRIBS_TOP_N", "8"))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_get_final_estimator(pipeline) -> Any:
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(pipeline, Pipeline):
            return pipeline.steps[-1][1]
        return pipeline
    except Exception:
        return pipeline


def _get_preprocessor(pipeline):
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(pipeline, Pipeline):
            if len(pipeline.steps) >= 2:
                return Pipeline(pipeline.steps[:-1])
    except Exception:
        pass
    return None


def _get_feature_names(pre, X_one: pd.DataFrame) -> List[str]:
    try:
        # sklearn >=1.0
        names = pre.get_feature_names_out()
        return list(names)
    except Exception:
        # fallback to original columns if preprocessor does nothing
        return list(X_one.columns)


class MLClassifier:
    def __init__(self, model_path: str | None = None) -> None:
        path = model_path or MODEL_PATH
        LOG.info("Loading ML pipeline from %s", path)
        self.pipeline = joblib.load(path)  # persisted sklearn Pipeline (preprocess + XGBClassifier)
        LOG.info("ML pipeline loaded successfully.")

        self.pre = _get_preprocessor(self.pipeline)
        self.final_est = _safe_get_final_estimator(self.pipeline)

        # Summarize model
        try:
            final_name = self.final_est.__class__.__name__
        except Exception:
            final_name = str(type(self.final_est))
        LOG.info("Model summary: final_estimator=%s", final_name)

        # Warn if not using XGBoost
        try:
            import xgboost as xgb  # noqa: F401
            have_xgb = True
        except Exception:
            have_xgb = False
        if not have_xgb:
            LOG.warning("XGBoost not importable; ensure xgboost is installed.")
        if have_xgb and "XGB" not in final_name:
            LOG.warning("Final estimator is not XGB* (%s). Consider retraining with XGBClassifier.", final_name)

    # ---- binary decision policy ------------------------------------------------

    @staticmethod
    def _movement_from_p_up(p_up: float) -> str:
        if p_up >= TAU_BUY:
            return "CALL"
        if p_up <= TAU_SELL:
            return "PUT"
        return "NEUTRAL"

    # ---- per-instance feature contributions -----------------------------------

    def _instance_contribs(self, X_proc: np.ndarray, feat_names: List[str]) -> List[Tuple[str, float]]:
        """Return top-N absolute contributions (name, value) excluding bias term."""
        try:
            import xgboost as xgb
        except Exception:
            return []

        try:
            # Prefer sklearn API direct pred_contribs if supported
            if hasattr(self.final_est, "predict"):
                try:
                    contribs = self.final_est.predict(X_proc, pred_contribs=True)
                except TypeError:
                    # Use booster with DMatrix
                    booster = self.final_est.get_booster()
                    dmat = xgb.DMatrix(X_proc)
                    contribs = booster.predict(dmat, pred_contribs=True)
            else:
                booster = self.final_est.get_booster()
                dmat = xgb.DMatrix(X_proc)
                contribs = booster.predict(dmat, pred_contribs=True)

            # contribs shape: (n_samples, n_features + 1 bias)
            row = contribs[0]
            if len(row) == len(feat_names) + 1:
                vals = row[:-1]  # drop bias term
            else:
                vals = row

            pairs = list(zip(feat_names, vals))
            pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
            return pairs[:max(1, TOP_N)]
        except Exception as e:
            LOG.warning("Could not compute per-instance contributions: %s", e)
            # fallback: use feature_importances_ if available (global, not per-instance)
            try:
                importances = getattr(self.final_est, "feature_importances_", None)
                if importances is not None and len(importances) == X_proc.shape[1]:
                    pairs = list(zip(feat_names, importances))
                    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
                    return pairs[:max(1, TOP_N)]
            except Exception:
                pass
            return []

    # ---- public classify API ---------------------------------------------------

    def classify(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Return dict with p_up, movement_type, and top feature contributions."""
        # Build single-row DataFrame
        df = pd.DataFrame([features])

        # Transform through preprocessor (if any)
        if self.pre is not None:
            X_proc = self.pre.transform(df)
            feat_names = _get_feature_names(self.pre, df)
        else:
            # Pipeline is just estimator on raw columns
            X_proc = df.values
            feat_names = list(df.columns)

        # Predict probability of UP
        p: float
        try:
            # Try standard predict_proba
            proba = self.pipeline.predict_proba(df)  # pipeline handles pre + est
            # Expect binary: [:,1] is UP
            p = float(proba[0, 1])
        except Exception:
            # Some wrappers expose decision_function or raw score
            try:
                score = float(self.pipeline.decision_function(df)[0])
                p = float(_sigmoid(np.array([score]))[0])
            except Exception as e:
                LOG.warning("Fallback scoring path: %s", e)
                # Last resort: est.predict returns {0,1}
                pred = int(self.pipeline.predict(df)[0])
                p = 0.9 if pred == 1 else 0.1  # heuristic

        movement = MLClassifier._movement_from_p_up(p)

        # Per-instance contributions (needs processed matrix)
        contribs = self._instance_contribs(
            X_proc=X_proc if isinstance(X_proc, np.ndarray) else X_proc.toarray(),
            feat_names=feat_names,
        )
        # Log a clear trading decision line
        LOG.info(
            "Decision: p_up=%.4f → movement=%s (tau_buy=%.2f, tau_sell=%.2f) | top=%s",
            p, movement, TAU_BUY, TAU_SELL,
            ", ".join(f"{k}:{v:+.3f}" for k, v in contribs),
        )

        return {
            "p_up": p,
            "movement_type": movement,
            "top_contributions": [{"feature": k, "contribution": float(v)} for k, v in contribs],
        }
