#!/usr/bin/env python3
"""Command-line script to evaluate the MLClassifier on a hold-out dataset.

This utility loads a CSV file with features and true labels, restores an ML
classifier, aligns feature columns, generates predictions, calculates ROC AUC
for binary classification, prints a classification report and confusion matrix,
and optionally writes a report to disk.

Usage:
    python evaluate_ml_classifier.py --data data.csv --label-col label
"""

from __future__ import annotations

import argparse
import logging


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from ml_classifier import MLClassifier


def main() -> None:
    """Entry point for command-line evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate MLClassifier on a labeled hold-out dataset."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV file containing features and true labels.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the column in the CSV that holds the true labels.",
    )
    parser.add_argument(
        "--report-output",
        help=(
            "Optional path to write a text report "
            "(ROC AUC, classification report, confusion matrix)."
        ),
    )
    args = parser.parse_args()

    # Configure logging across multiple lines to respect line-length limits
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading evaluation data from {args.data}")
    try:
        df = pd.read_csv(args.data)
    except Exception as exc:
        logger.error(f"Failed to read {args.data}: {exc!r}")
        raise SystemExit(1)

    if args.label_col not in df.columns:
        logger.error(f"Label column '{args.label_col}' not found in data.")
        raise SystemExit(1)

    X = df.drop(columns=[args.label_col])
    y_true = df[args.label_col]

    logger.info("Initializing MLClassifier")
    clf = MLClassifier()

    # Align columns exactly to the trained pipeline
    X = X[clf.feature_names]

    logger.info("Generating predictions and probabilities")
    y_pred = clf.pipeline.predict(X)
    try:
        # Only compute ROC AUC for binary targets
        if len(clf.pipeline.classes_) == 2:
            y_prob = clf.pipeline.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
            logger.info(f"ROC AUC: {auc:.4f}")
        else:
            logger.info("Skipping ROC AUC (multi-class problem).")
            auc = None
    except Exception as exc:
        logger.warning(f"Could not compute ROC AUC: {exc!r}")
        auc = None

    report = classification_report(y_true, y_pred, digits=4)
    logger.info("Classification Report:\n%s", report)

    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n%s", np.array2string(cm))

    if args.report_output:
        try:
            with open(args.report_output, "w") as f:
                if auc is not None:
                    f.write(f"ROC AUC: {auc:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report + "\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(cm) + "\n")
            logger.info(f"Saved evaluation report to {args.report_output}")
        except Exception as exc:
            logger.error(
                f"Failed to write report to {args.report_output}: {exc!r}"
            )


if __name__ == "__main__":
    main()
