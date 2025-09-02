  #!/usr/bin/env python3
"""
train_ml_classifier.py

Reads data/movement_training_data.csv, preprocesses numeric + categorical,
label-encodes the target, fits an XGBClassifier pipeline, evaluates, and
saves the pipeline (with feature_names + label_encoder attached)
"""

import os
import argparse
import logging

import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
    ))
    root.handlers.clear()
    root.addHandler(h)

def train(
    train_csv: str,
    label_col: str,
    model_dir: str,
    model_filename: str,
    test_size: float,
    random_state: int
):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading training data from {train_csv}")
    df = pd.read_csv(train_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {train_csv}")

    # drop symbol if present
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # encode target
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info(f"Classes mapped: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

    # split
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, stratify=y_enc, test_size=test_size, random_state=random_state
    )

    # identify numeric vs categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Numeric cols: {num_cols}")
    logger.info(f"Categorical cols: {cat_cols}")

    # build preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("xgb", xgb.XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state
        ))
    ])

    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_train, y_train)

    # evaluate
    preds_enc = pipeline.predict(X_test)
    preds     = le.inverse_transform(preds_enc)
    y_true    = le.inverse_transform(y_test)
    acc = accuracy_score(y_true, preds)
    logger.info(f"Validation Accuracy: {acc:.4f}")

    # attach metadata
    feature_names = num_cols + \
      list(pipeline.named_steps["pre"]
                   .named_transformers_["cat"]
                   .get_feature_names_out(cat_cols))
    pipeline.feature_names = feature_names
    pipeline.label_encoder = le
    logger.info(f"Attached {len(feature_names)} feature names + label encoder")

    # save
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)
    joblib.dump(pipeline, path)
    logger.info(f"✅ Trained pipeline saved to {path}")

def parse_args():
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
        train_csv      = args.train_csv,
        label_col      = args.label_col,
        model_dir      = args.model_dir,
        model_filename = args.model_filename,
        test_size      = args.test_size,
        random_state   = args.random_state
    )
