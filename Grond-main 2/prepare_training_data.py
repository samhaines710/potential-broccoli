 #!/usr/bin/env python3
"""Prepare training data for movement classification.

This script fetches historical OHLC bars, treasury yields, and option Greeks,
computes a sliding window of features and labels, and writes the result to
`data/movement_training_data.csv`.

Changes:
- Uses centralized JSON logging via utils.logging_utils.configure_logging()
  (no basicConfig; no duplicate handlers).
- Keeps /fed/v1/treasury-yields with limit=1&sort=date.desc for latest record.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import logging
import pandas as pd
import requests

from config import TICKERS, tz
from data_ingestion import HistoricalDataLoader
from utils.logging_utils import configure_logging, get_logger
from utils import (
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_time_of_day,
    calculate_volume_ratio,
    compute_rsi,
    compute_corr_deviation,
    compute_skew_ratio,
    fetch_option_greeks,
)

# Configure logging once (JSON, singleton)
configure_logging()
logger = get_logger(__name__)

# Fetch Treasury Yields once
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
YIELDS_ENDPOINT = "https://api.polygon.io/fed/v1/treasury-yields"

def fetch_treasury_yields(date: str | None = None) -> Dict[str, Any]:
    """Fetch treasury yields for a specific date (or latest) from Polygon."""
    params: Dict[str, Any] = {"apiKey": POLYGON_API_KEY}
    if date:
        params["date"] = date
    else:
        params["limit"] = 1
        params["sort"] = "date.desc"

    resp = requests.get(YIELDS_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        logger.warning('No yield data returned for date=%s', date)
        return {}
    record = results[0]
    logger.info('Fetched treasury yields for date=%s', record.get("date"))
    return record

# Cache yields once
YIELDS = fetch_treasury_yields()

# Output Configuration
OUTPUT_DIR = "data"
OUTPUT_FILE = "movement_training_data.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# Parameters (CI / env override)
HIST_DAYS = int(os.getenv("HIST_DAYS", "90"))
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "18"))
LOOKAHEAD_BARS = int(os.getenv("LOOKAHEAD_BARS", "2"))

def extract_features_and_label(symbol: str) -> pd.DataFrame:
    """
    Fetch historical bars and compute sliding-window features and labels for a symbol.

    Returns a DataFrame of features (including yields and Greeks) plus movement_type.
    """
    end = datetime.now(tz)
    start = end - timedelta(days=HIST_DAYS)
    loader = HistoricalDataLoader()
    raw_bars = loader.fetch_bars(symbol, start, end)
    logger.info('Fetched %d bars for %s over %d days', len(raw_bars), symbol, HIST_DAYS)

    df = pd.DataFrame(
        [
            {
                "timestamp": b["t"],
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
            }
            for b in raw_bars
        ]
    )
    if df.empty:
        logger.warning("No bars for %s; returning empty frame", symbol)
        return pd.DataFrame(columns=[
            "symbol","breakout_prob","recent_move_pct","time_of_day","volume_ratio",
            "rsi","corr_dev","skew_ratio","yield_spike_2year","yield_spike_10year",
            "yield_spike_30year","delta","gamma","theta","vega","rho","vanna",
            "vomma","movement_type","theta_day","theta_5m"
        ])

    df["dt"] = (
        pd.to_datetime(df["timestamp"], unit="ms")
        .dt.tz_localize("UTC")
        .dt.tz_convert(tz)
    )
    df.set_index("dt", inplace=True)

    ys2 = float(YIELDS.get("yield_2_year", 0.0))
    ys10 = float(YIELDS.get("yield_10_year", 0.0))
    ys30 = float(YIELDS.get("yield_30_year", 0.0))
    greeks = fetch_option_greeks(symbol)
    logger.info(
        "Using yields (2y=%s,10y=%s,30y=%s) and Greeks for %s",
        ys2, ys10, ys30, symbol,
    )

    records: List[Dict[str, Any]] = []
    for i in range(LOOKBACK_BARS, len(df) - LOOKAHEAD_BARS):
        window = df.iloc[i - LOOKBACK_BARS:i]
        current = df.iloc[i]
        candles = window.reset_index().to_dict("records")

        feat: Dict[str, Any] = {
            "symbol": symbol,
            "breakout_prob": calculate_breakout_prob(candles),
            "recent_move_pct": calculate_recent_move_pct(candles),
            "time_of_day": calculate_time_of_day(current.name),
            "volume_ratio": calculate_volume_ratio(candles),
            "rsi": compute_rsi(candles),
            "corr_dev": compute_corr_deviation(symbol),
            "skew_ratio": compute_skew_ratio(symbol),
            "yield_spike_2year": ys2,
            "yield_spike_10year": ys10,
            "yield_spike_30year": ys30,
            **greeks,
        }

        next_bar = df.iloc[i + LOOKAHEAD_BARS]
        delta = (next_bar["close"] - current["close"]) / current["close"]
        if delta > 0:
            feat["movement_type"] = "CALL"
        elif delta < 0:
            feat["movement_type"] = "PUT"
        else:
            feat["movement_type"] = "NEUTRAL"

        theta_raw = float(greeks.get("theta", 0.0))
        feat["theta_day"] = theta_raw
        feat["theta_5m"] = theta_raw / 78.0

        records.append(feat)

    return pd.DataFrame(records)

def main() -> None:
    """Generate training data for all tickers and write to CSV."""
    out_frames: List[pd.DataFrame] = []
    for t in TICKERS:
        logger.info('Generating data for %s', t)
        out_frames.append(extract_features_and_label(t))

    if not out_frames:
        logger.error("No frames generated.")
        return

    full = pd.concat(out_frames, ignore_index=True) if len(out_frames) > 1 else out_frames[0]
    if full.empty:
        logger.warning("No rows generated; nothing to write.")
        return
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full.to_csv(OUTPUT_PATH, index=False)
    logger.info('✅ Saved %d rows to %s', len(full), OUTPUT_PATH)

if __name__ == "__main__":
    main()
