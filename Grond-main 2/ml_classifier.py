#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizon-aware binary classifier wrapper for live inference.

Loads a bundle produced by train_hcbc_xgb_optuna.py and exposes a simple
API for single-row scoring with an explicit lookahead horizon H.

Bundle contract (joblib):
{
  "model": sklearn.Pipeline(pre -> XGBClassifier),
  "feature_columns": [ ... ordered feature names incl. 'H' ... ],
  "label_col": "label_up",
  "h_key": "H",
  "calibrators": { int(H): IsotonicRegression, ... },
  "thresholds": { int(H): {"buy": float, "sell": float}, ... },
  "cv_report": {...}
}
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

@dataclass
class ScoreResult:
    p_up: float
    decision: str
    H: int
    thresholds: Dict[str, float]

class HCBCClassifier:
    def __init__(self, bundle_path: str):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
        self.bundle = joblib.load(bundle_path)
        self.pipeline = self.bundle["model"]
        self.features: List[str] = list(self.bundle["feature_columns"])
        self.h_key: str = self.bundle.get("h_key", "H")
        self.calibrators: Dict[int, Any] = self.bundle.get("calibrators", {})
        self.thresholds: Dict[int, Dict[str, float]] = self.bundle.get("thresholds", {})
        if self.h_key not in self.features:
            raise ValueError(f"h_key '{self.h_key}' not in feature_columns; got {self.features}")

    def _calibrate(self, p_raw: float, H: int) -> float:
        iso = self.calibrators.get(int(H))
        if iso is None:
            return float(p_raw)
        return float(iso.transform(np.array([p_raw]))[0])

    def _thresholds_for(self, H: int) -> Dict[str, float]:
        th = self.thresholds.get(int(H))
        if th is None:
            return {"buy": 0.6, "sell": 0.4}
        return th

    def score(self, features: Dict[str, Any], H: int) -> ScoreResult:
        row = {k: features.get(k, np.nan) for k in self.features}
        row[self.h_key] = int(H)
        df = pd.DataFrame([row], columns=self.features)

        try:
            p_raw = float(self.pipeline.predict_proba(df)[:, 1][0])
        except Exception:
            from scipy.special import expit
            score = float(self.pipeline.decision_function(df)[0])
            p_raw = float(expit(score))

        p = self._calibrate(p_raw, H)
        th = self._thresholds_for(H)

        if p >= th["buy"]:
            decision = "LONG"
        elif p <= th["sell"]:
            decision = "SHORT"
        else:
            decision = "FLAT"

        return ScoreResult(p_up=p, decision=decision, H=int(H), thresholds=th)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to xgb_hcbc.bundle.joblib")
    ap.add_argument("--H", type=int, required=True, help="Lookahead horizon (bars)")
    ap.add_argument("--features-json", required=True, help="JSON object with feature_name: value")
    args = ap.parse_args()

    clf = HCBCClassifier(args.bundle)
    feats = json.loads(args.features_json)
    res = clf.score(feats, args.H)
    print(json.dumps({"p_up": res.p_up, "decision": res.decision, "H": res.H, "thresholds": res.thresholds}))