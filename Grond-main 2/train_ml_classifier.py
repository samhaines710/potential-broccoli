#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ml_classifier.py

Reads data/movement_training_data.csv, preprocesses numeric + categorical
(with robust imputation), label-encodes the target, fits an XGBoost
multi-class pipeline, evaluates, and saves the trained pipeline.

Hardening:
- Converts inf/-inf -> NaN and imputes numerics with 0.0 (avoids scaler warnings).
- Imputes categoricals with 'fallback' and OHE(handle_unknown='ignore').
- Safe OHE construction across sklearn versions (sparse_output vs sparse).
- Safe feature-name extraction across sklearn versions.
- Stratified split when possible; falls back safely if classes are imbalanced.
- Attaches feature_names and label_encoder to the saved pipeline.

Usage:
  python train_ml_classifier.py \
      --train-csv data/movement_training_data.csv \
      --label-col movement_type \
      --model-dir models \
      --model-filename xgb_classifier.pipeline.joblib
"""

from __future__ import annotations

import os
import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}'
    ))
    root.handlers.clear()
    root.addHandler(h)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

ORDERED_LABELS = ["CALL", "PUT", "NEUTRAL"]

def _finite_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    """Convert non-categorical columns to finite (inf→NaN)."""
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            df[col] = s
    return df

def _ohe_ctor() -> OneHotEncoder:
    """
    Construct OneHotEncoder compatibly across sklearn versions.
    - sklearn >= 1.2: OneHotEncoder(..., sparse_output=False)
    - older:           OneHotEncoder(..., sparse=False)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _ohe_feature_names(ohe: OneHotEncoder, cat_cols: List[str]) -> List[str]:
    """Get OHE output feature names across sklearn versions."""
    try:
        return list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        return list(ohe.get_feature_names(cat_cols))  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    train_csv: str,
    label_col: str,
    model_dir: str,
    model_filename: str,
    test_size: float,
    random_state: int,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading training data from {train_csv}")

    df = pd.read_csv(train_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {train_csv}")

    # Optional: Drop columns that should never be features
    drop_cols = []
    # If symbol is overly sparse/high-cardinality for your use-case, drop it:
    # drop_cols.append("symbol")
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure finite numerics before we compute dtypes and split
    df = _finite_numeric(df, exclude=[label_col])

    # Encode target with stable ordering if present; otherwise default LE order
    y_raw = df[label_col].astype(str)
    if all(lbl in set(y_raw.unique()) for lbl in ORDERED_LABELS):
        # map to ordered integers explicitly
        mapping = {k: i for i, k in enumerate(ORDERED_LABELS)}
        y_enc = y_raw.map(mapping).to_numpy()
        classes_out = ORDERED_LABELS
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)
        classes_out = list(le.classes_)

    X = df.drop(columns=[label_col])

    # Identify numeric vs. categorical columns after coercion
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(f"Numeric cols: {num_cols}")
    logger.info(f"Categorical cols: {cat_cols}")

    # Preprocessor: numeric -> impute(0) -> scale ; categorical -> impute('fallback') -> OHE
    numeric_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="constant", fill_value="fallback")),
        ("ohe", _ohe_ctor()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    # XGBoost multi-class classifier
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,            # CALL, PUT, NEUTRAL
        tree_method="hist",
        random_state=random_state,
    )

    pipeline = Pipeline(steps=[
        ("pre", preprocessor),
        ("xgb", clf),
    ])

    # Stratified split if possible; otherwise fallback
    unique, counts = np.unique(y_enc, return_counts=True)
    can_stratify = (len(unique) >= 2) and np.all(counts >= 2)
    logger.info(f"Stratified split: {'yes' if can_stratify else 'no'} "
                f"(classes={dict(zip(unique.tolist(), counts.tolist()))})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc if can_stratify else None,
    )

    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_enc = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred_enc)
    logger.info(f"Validation Accuracy: {acc:.4f}")

    # Attach metadata: feature names + label encoder semantics
    # Extract numeric names (as-is) and categorical OHE names
    num_names = num_cols
    ohe = pipeline.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
    cat_names = _ohe_feature_names(ohe, cat_cols) if cat_cols else []
    feature_names = num_names + cat_names

    # Attach for downstream consumers
    pipeline.feature_names = feature_names                 # type: ignore[attr-defined]
    pipeline.classes_ = np.array(classes_out)              # type: ignore[attr-defined]
    # For compatibility with your runtime classifier, also include a LabelEncoder-like object when applicable
    try:
        le = LabelEncoder()
        le.fit(classes_out)
        pipeline.label_encoder = le                        # type: ignore[attr-defined]
    except Exception:
        pass

    logger.info(f"Attached {len(feature_names)} feature names + label encoder/classes")

    # Save
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, model_filename)
    joblib.dump(pipeline, out_path)
    logger.info(f"✅ Trained pipeline saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv",      default="data/movement_training_data.csv")
    p.add_argument("--label-col",      default="movement_type")
    p.add_argument("--model-dir",      default="models")
    p.add_argument("--model-filename", default="xgb_classifier.pipeline.joblib")
    p.add_argument("--test-size",      type=float, default=0.25)
    p.add_argument("--random-state",   type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    train(
        train_csv=args.train_csv,
        label_col=args.label_col,
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        test_size=args.test_size,
        random_state=args.random_state,
    )
