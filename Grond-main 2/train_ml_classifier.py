#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


NUMERIC_COLS = [
    "breakout_prob", "recent_move_pct", "volume_ratio", "rsi", "corr_dev", "skew_ratio",
    "yield_spike_2year", "yield_spike_10year", "yield_spike_30year",
    "delta", "gamma", "theta", "vega", "rho", "vanna", "vomma", "charm", "veta",
    "speed", "zomma", "color", "implied_volatility", "theta_day", "theta_5m",
    "ms_spread_mean", "ms_depth_imbalance_mean", "ms_ofi_sum",
    "ms_signed_volume_sum", "ms_vpin"
]

CATEGORICAL_COLS = ["symbol", "time_of_day", "source"]

TARGET_COL = os.environ.get("LABEL_COL_DEFAULT", "movement_class")  # change if your column name differs


@dataclass
class TrainConfig:
    data_csv: str
    out_path: str
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


def _log(msg: str) -> None:
    print(json.dumps({
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
        "level": "INFO",
        "module": "train_ml_classifier",
        "message": msg
    }))

def load_data(path: str) -> pd.DataFrame:
    _log(f"Loading training data from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at: {path}")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    # coerce numeric columns safely
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            _log(f"Warning: expected numeric column missing in data: {c}")

    # basic sanitization
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_pipeline() -> Pipeline:
    _log(f"Numeric cols: {NUMERIC_COLS}")
    _log(f"Categorical cols: {CATEGORICAL_COLS}")

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, [c for c in NUMERIC_COLS if c]),
            ("cat", categorical_tf, [c for c in CATEGORICAL_COLS if c and c != TARGET_COL]),
        ],
        remainder="drop",
        n_jobs=None,
        verbose_feature_names_out=False
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preproc", preproc),
        ("clf", clf)
    ])
    return pipeline


def stratified_split(X: pd.DataFrame, y: np.ndarray, cfg: TrainConfig):
    stratify = y if cfg.stratify else None
    class_counts = {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()}
    _log(f"Stratified split: {'yes' if stratify is not None else 'no'} (classes={class_counts})")
    return train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify
    )


def fit_and_validate(pipeline: Pipeline, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    _log("Fitting pipeline on training data")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    _log(f"Validation Accuracy: {acc:.4f}")

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    # Try to log top importances if available (RandomForest has feature_importances_)
    try:
        clf = pipeline.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            preproc = pipeline.named_steps["preproc"]
            feature_names = preproc.get_feature_names_out()
            fi = sorted(
                zip(feature_names, importances),
                key=lambda t: t[1],
                reverse=True
            )[:30]
            metrics["top_feature_importances"] = fi
    except Exception as e:
        _log(f"Feature importance unavailable: {e}")

    return metrics


def save_artifact(out_path: str, pipeline: Pipeline, label_encoder: LabelEncoder, metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bundle = {
        "model": pipeline,
        "classes": pipeline.named_steps["clf"].classes_.tolist(),
        "label_encoder_classes": label_encoder.classes_.tolist(),
        "sklearn_version": __import__("sklearn").__version__,
        "metrics": metrics
    }
    joblib.dump(bundle, out_path)
    _log(f"Saved model bundle to {out_path}")


def train(cfg: TrainConfig) -> None:
    df = load_data(cfg.data_csv)

    missing = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y_raw = df[TARGET_COL].astype(str).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()

    X_train, X_val, y_train, y_val = stratified_split(X, y, cfg)

    pipeline = make_pipeline()
    metrics = fit_and_validate(pipeline, X_train, y_train, X_val, y_val)

    save_artifact(cfg.out_path, pipeline, label_encoder, metrics)

    print("\n=== VALIDATION METRICS ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(np.array(metrics["confusion_matrix"]))


def parse_args(argv: List[str]) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train movement classifier")
    # New-style
    p.add_argument("--data", default="data/movement_training_data.csv", help="Path to CSV")
    p.add_argument("--out", default="Resources/xgb_classifier.pipeline.joblib", help="Output model bundle path")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify", action="store_true")
    # Legacy CI aliases
    p.add_argument("--train-csv", dest="legacy_data", default=None, help="(legacy) training CSV path")
    p.add_argument("--label-col", dest="label_col", default=None, help="(legacy) label column name")
    p.add_argument("--model-dir", dest="model_dir", default=None, help="(legacy) output directory")
    p.add_argument("--model-filename", dest="model_filename", default=None, help="(legacy) output filename")
    args = p.parse_args(argv)

    data_path = args.data if args.legacy_data is None else args.legacy_data
    out_path = args.out
    if args.model_dir or args.model_filename:
        md = args.model_dir or ""
        mf = args.model_filename or "xgb_classifier.pipeline.joblib"
        out_path = os.path.join(md, mf)

    # Allow overriding label column for backward compatibility
    global TARGET_COL
    if args.label_col:
        TARGET_COL = args.label_col

    return TrainConfig(
        data_csv=data_path,
        out_path=out_path,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify
    )
        data_csv=args.data,
        out_path=args.out,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify
    )


if __name__ == "__main__":
    cfg = parse_args(sys.argv[1:])
    train(cfg)
