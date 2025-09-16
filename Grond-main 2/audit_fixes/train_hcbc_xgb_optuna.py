#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizon-Conditioned Binary Classifier (HCBC) Trainer with Optuna + XGBoost
See docstring in file for full details.
"""
import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer

import optuna
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class TrainConfig:
    data_csv: str
    out_path: str
    n_trials: int = 40
    n_splits: int = 6
    embargo_rows: int = 200
    random_state: int = 42

def log(msg: str) -> None:
    print(json.dumps({"ts": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "msg": msg}))

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df

def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in {"label_up", "timestamp"}]
    if "H" not in df.columns:
        raise ValueError("Expected column 'H' (lookahead bars).")
    cats = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    nums = [c for c in cols if c not in cats]
    if "H" not in nums:
        nums.append("H")
        if "H" in cats:
            cats.remove("H")
    return nums, cats

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

def walk_forward_indices(n: int, n_splits: int, embargo_rows: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    sizes = np.full(n_splits, n // n_splits, dtype=int); sizes[: n % n_splits] += 1
    bounds = np.cumsum(sizes); starts = np.concatenate([[0], bounds[:-1]])
    parts = list(zip(starts, bounds)); indices = np.arange(n); splits = []
    for i in range(1, len(parts)):
        tr_end = max(0, parts[i][0] - embargo_rows)
        tr_idx = indices[:tr_end]; va_idx = indices[parts[i][0]:parts[i][1]]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            splits.append((tr_idx, va_idx))
    return splits

def xgb_from_trial(trial: optuna.Trial, seed: int) -> XGBClassifier:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 6.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1e-2),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "tree_method": "hist", "objective": "binary:logistic", "eval_metric": "logloss",
        "n_jobs": -1, "random_state": seed,
    }
    return XGBClassifier(**params)

def optuna_objective(trial: optuna.Trial, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: TrainConfig) -> float:
    pre = make_preprocessor(num_cols, cat_cols)
    model = xgb_from_trial(trial, seed=cfg.random_state)
    pipe = Pipeline([("pre", pre), ("xgb", model)])
    splits = walk_forward_indices(len(df), cfg.n_splits, cfg.embargo_rows)
    aucs = []
    for (tr_idx, va_idx) in splits:
        X_tr = df.iloc[tr_idx][num_cols + cat_cols]; y_tr = df.iloc[tr_idx]["label_up"].astype(int).values
        X_va = df.iloc[va_idx][num_cols + cat_cols]; y_va = df.iloc[va_idx]["label_up"].astype(int).values
        pipe.fit(X_tr, y_tr)
        try:
            p = pipe.predict_proba(X_va)[:, 1]
        except Exception:
            from scipy.special import expit
            p = expit(pipe.decision_function(X_va))
        aucs.append(roc_auc_score(y_va, p))
        trial.report(float(np.mean(aucs)), len(aucs))
        if trial.should_prune(): raise optuna.TrialPruned()
    return float(np.mean(aucs)) if aucs else 0.0

from sklearn.metrics import roc_auc_score

def fit_oof_calibration_and_thresholds(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], best_params: Dict[str, Any], cfg: TrainConfig):
    pre = make_preprocessor(num_cols, cat_cols)
    model = XGBClassifier(**best_params)
    pipe = Pipeline([("pre", pre), ("xgb", model)])
    splits = walk_forward_indices(len(df), cfg.n_splits, cfg.embargo_rows)
    oof = []
    for (tr_idx, va_idx) in splits:
        X_tr = df.iloc[tr_idx][num_cols + cat_cols]; y_tr = df.iloc[tr_idx]["label_up"].astype(int).values
        X_va = df.iloc[va_idx][num_cols + cat_cols]; y_va = df.iloc[va_idx]["label_up"].astype(int).values
        H_va = df.iloc[va_idx]["H"].astype(int).values
        pipe.fit(X_tr, y_tr)
        p_raw = pipe.predict_proba(X_va)[:, 1]
        oof.append(pd.DataFrame({"H": H_va, "y": y_va, "p_raw": p_raw}))
    oof_df = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame(columns=["H","y","p_raw"])
    calibrators, thresholds, cv_report = {}, {}, {}
    for h in sorted(oof_df["H"].unique()):
        chunk = oof_df[oof_df["H"] == h]
        if len(chunk) < 100: continue
        iso = IsotonicRegression(out_of_bounds="clip")
        p_cal = iso.fit_transform(chunk["p_raw"].values, chunk["y"].values)
        calibrators[int(h)] = iso
        grid = np.linspace(0.2, 0.8, 25)
        best_f1, best_tau = -1.0, 0.5
        for tau in grid:
            y_hat = (p_cal >= tau).astype(int)
            f1 = f1_score(chunk["y"].values, y_hat, zero_division=0)
            if f1 > best_f1: best_f1, best_tau = f1, float(tau)
        thresholds[int(h)] = {"buy": best_tau, "sell": 1.0 - best_tau}
        try: auc = roc_auc_score(chunk["y"].values, p_cal)
        except Exception: auc = float("nan")
        cv_report[int(h)] = {"n": int(len(chunk)), "auc_cal": float(auc), "f1@tau": float(best_f1), "tau": best_tau}
    return calibrators, thresholds, cv_report

def train(cfg: TrainConfig) -> None:
    df = load_data(cfg.data_csv)
    if "label_up" not in df.columns: raise ValueError("Expected 'label_up' column (0/1).")
    if "H" not in df.columns: raise ValueError("Expected 'H' column.")
    num_cols, cat_cols = infer_columns(df)
    log(f"Numeric columns: {num_cols}"); log(f"Categorical columns: {cat_cols}")
    log("Starting Optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: optuna_objective(t, df, num_cols, cat_cols, cfg), n_trials=cfg.n_trials)
    best_params = study.best_trial.params
    best_params.update({"objective":"binary:logistic","eval_metric":"logloss","tree_method":"hist","n_jobs":-1,"random_state":cfg.random_state})
    log(f"Best params: {best_params}")
    log("Fitting OOF calibrators + thresholds")
    calibrators, thresholds, cv_report = fit_oof_calibration_and_thresholds(df, num_cols, cat_cols, best_params, cfg)
    log("Fit final model")
    pre = make_preprocessor(num_cols, cat_cols)
    final_model = Pipeline([("pre", pre), ("xgb", XGBClassifier(**best_params))])
    final_model.fit(df[num_cols + cat_cols], df["label_up"].astype(int).values)
    out_dir = os.path.dirname(cfg.out_path); os.makedirs(out_dir, exist_ok=True)
    bundle = {
        "model": final_model,
        "feature_columns": num_cols + cat_cols,
        "label_col": "label_up",
        "h_key": "H",
        "calibrators": calibrators,
        "thresholds": thresholds,
        "cv_report": cv_report,
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
        "optuna_best_params": best_params,
    }
    joblib.dump(bundle, cfg.out_path)
    with open(os.path.join(out_dir, "hcbc_cv_report.json"), "w") as f: json.dump(cv_report, f, indent=2)
    with open(os.path.join(out_dir, "hcbc_thresholds.json"), "w") as f: json.dump(thresholds, f, indent=2)
    with open(os.path.join(out_dir, "hcbc_features.json"), "w") as f: json.dump({"numeric": num_cols, "categorical": cat_cols}, f, indent=2)
    log(f"Saved bundle → {cfg.out_path}")

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train horizon-conditioned binary XGB with Optuna")
    ap.add_argument("--data", required=True, help="Path to CSV with features + H + label_up")
    ap.add_argument("--out", default="Resources/xgb_hcbc.bundle.joblib", help="Output bundle path")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--embargo-rows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    return TrainConfig(data_csv=args.data, out_path=args.out, n_trials=args.n_trials, n_splits=args.folds, embargo_rows=args.embargo_rows, random_state=args.seed)

if __name__ == "__main__":
    cfg = parse_args(); train(cfg)
