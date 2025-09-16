#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from ml_classifier import HCBCClassifier

def evaluate(data_csv: str, bundle_path: str) -> Dict[str, Any]:
    if not os.path.exists(data_csv):
        raise FileNotFoundError(data_csv)
    df = pd.read_csv(data_csv)
    if "label_up" not in df.columns or "H" not in df.columns:
        raise ValueError("Expected columns 'label_up' and 'H' in CSV")
    clf = HCBCClassifier(bundle_path)

    metrics: Dict[int, Dict[str, Any]] = {}
    for h in sorted(df["H"].unique()):
        chunk = df[df["H"] == h]
        y = chunk["label_up"].astype(int).values

        feats_cols = [c for c in clf.features if c != clf.h_key]
        X_feats = chunk[feats_cols].to_dict(orient="records")

        p_list = []
        y_hat = []
        for rec in X_feats:
            r = clf.score(rec, int(h))
            p_list.append(r.p_up)
            y_hat.append(1 if r.decision == "LONG" else 0 if r.decision == "SHORT" else -1)

        p = np.array(p_list)
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = float("nan")

        mask = np.where(np.array(y_hat) >= 0)[0]
        if len(mask):
            y_eff = y[mask]
            yhat_eff = np.array(y_hat)[mask]
            pr, rc, f1, _ = precision_recall_fscore_support(y_eff, yhat_eff, average="binary", zero_division=0)
        else:
            pr = rc = f1 = float("nan")

        metrics[int(h)] = {"n": int(len(chunk)), "auc": float(auc), "precision": float(pr), "recall": float(rc), "f1": float(f1)}

    vals = [m for m in metrics.values() if m["n"] > 0]
    if vals:
        metrics["_macro"] = {
            "auc": float(np.nanmean([m["auc"] for m in vals])),
            "precision": float(np.nanmean([m["precision"] for m in vals])),
            "recall": float(np.nanmean([m["recall"] for m in vals])),
            "f1": float(np.nanmean([m["f1"] for m in vals])),
        }
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate HCBC model per-H")
    ap.add_argument("--data", required=True, help="CSV with features + H + label_up")
    ap.add_argument("--bundle", default="Resources/xgb_hcbc.bundle.joblib", help="Model bundle path")
    ap.add_argument("--out", default="", help="Optional JSON report path")
    args = ap.parse_args()

    res = evaluate(args.data, args.bundle)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))