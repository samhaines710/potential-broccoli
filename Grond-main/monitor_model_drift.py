#!/usr/bin/env python3
"""Compute population stability index (PSI) to monitor model drift.

This script compares a reference dataset with a recent production dataset,
calculates the PSI for each feature, logs detailed results, and alerts if
the overall PSI exceeds a specified threshold.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict

import numpy as np
import pandas as pd

from ml_classifier import MLClassifier


def population_stability_index(
    expected: pd.Series, actual: pd.Series, buckets: int = 10
) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    Parameters
    ----------
    expected : pd.Series
        The distribution of the training dataset.
    actual : pd.Series
        The distribution of the recent dataset.
    buckets : int, optional
        Number of quantile bins to use.

    Returns
    -------
    float
        The PSI value.
    """
    expected_bins = pd.qcut(expected, buckets, duplicates="drop")
    actual_bins = pd.qcut(actual, buckets, duplicates="drop")

    exp_counts = expected_bins.value_counts(normalize=True).sort_index()
    act_counts = actual_bins.value_counts(normalize=True).sort_index()

    idx = exp_counts.index.union(act_counts.index)
    exp = exp_counts.reindex(idx, fill_value=0)
    act = act_counts.reindex(idx, fill_value=0)

    psi = ((exp - act) * np.log(exp / act)).sum()
    return float(psi)


def main() -> None:
    """Parse arguments, compute PSI per feature, and report drift."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSI for recent vs. reference data to detect model drift."
        )
    )
    parser.add_argument(
        "--ref-data",
        required=True,
        help="Path to CSV of reference (training) features.",
    )
    parser.add_argument(
        "--recent-data",
        required=True,
        help="Path to CSV of recent production features.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="PSI threshold above which an alert is logged.",
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to write logs (defaults to stdout).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading reference dataset from %s", args.ref_data)
    ref_df = pd.read_csv(args.ref_data)

    logger.info("Loading recent dataset from %s", args.recent_data)
    recent_df = pd.read_csv(args.recent_data)

    logger.info("Initializing MLClassifier for feature extraction")
    clf = MLClassifier()
    pipeline = clf.pipeline
    feature_names = clf.feature_names

    # Assume first step in pipeline is the scaler
    scaler = pipeline.steps[0][1]

    logger.info("Transforming reference and recent features")
    ref_vals = scaler.transform(ref_df[feature_names])
    recent_vals = scaler.transform(recent_df[feature_names])

    # Compute PSI per feature
    psis: Dict[str, float] = {}
    for idx, feat in enumerate(feature_names):
        psi_val = population_stability_index(
            pd.Series(ref_vals[:, idx]),
            pd.Series(recent_vals[:, idx]),
        )
        psis[feat] = psi_val
        logger.debug("PSI for %s: %.4f", feat, psi_val)

    overall_psi = float(np.mean(list(psis.values())))
    logger.info(
        "Overall PSI across %d features: %.4f",
        len(feature_names),
        overall_psi,
    )

    if overall_psi > args.threshold:
        logger.warning(
            "Model drift detected: PSI %.4f exceeds threshold %.4f",
            overall_psi,
            args.threshold,
        )
    else:
        logger.info(
            "No significant drift: PSI %.4f within threshold %.4f",
            overall_psi,
            args.threshold,
        )


if __name__ == "__main__":
    main()
