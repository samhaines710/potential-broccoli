#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Prepare training data for movement classification.

This script fetches historical OHLC bars, treasury yields, option Greeks,
and microstructure features, computes a sliding window of features and labels,
and writes the result to `data/movement_training_data.csv`.

Hardenings:
- No network calls at import time.
- Treasury yields fetch is lazy, tolerant of missing/invalid POLYGON_API_KEY,
  and never raises; falls back to zeros.
- Optional kill-switch via DISABLE_YIELDS=1 to skip yields in CI.
- Timezone-safe alignment for microstructure features.

Logging:
- Uses centralized JSON logging via utils.logging_utils.configure_logging()
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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
from utils.micro_features import build_microstructure_features

# ── Configure logging once (JSON, singleton) ───────────────────────────────────
configure_logging()
logger = get_logger(__name__)

# ── Polygon Treasury Yields endpoint ───────────────────────────────────────────
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
YIELDS_ENDPOINT = "https://api.polygon.io/fed/v1/treasury-yields"


def fetch_treasury_yields(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch treasury yields for a specific date (or latest) from Polygon.

    - Never raises: returns {} on any error.
    - If POLYGON_API_KEY is missing or invalid, logs a warning and returns {}.
    """
    if not POLYGON_API_KEY:
        logger.warning("POLYGON_API_KEY missing; skipping treasury yields.")
        return {}

    params: Dict[str, Any] = {"apiKey": POLYGON_API_KEY}
    if date:
        params["date"] = date
    else:
        params["limit"] = 1
        params["sort"] = "date.desc"

    try:
        resp = requests.get(YIELDS_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            logger.warning("No treasury yield data returned (date=%s).", date)
            return {}
        record = results[0]
        logger.info("Fetched treasury yields for date=%s", record.get("date"))
        return record
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        logger.warning("Treasury yields fetch failed: HTTP %s", status)
        return {}
    except Exception as e:
        logger.warning("Treasury yields fetch error: %s", e)
        return {}


# ── Output configuration ───────────────────────────────────────────────────────
OUTPUT_DIR = "data"
OUTPUT_FILE = "movement_training_data.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# ── Parameters (can be overridden via env) ─────────────────────────────────────
HIST_DAYS = int(os.getenv("HIST_DAYS", "90"))
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "18"))
LOOKAHEAD_BARS = int(os.getenv("LOOKAHEAD_BARS", "2"))


def _asof_micro(
    ms_df: pd.DataFrame,
    ts: pd.Timestamp,
    fields: List[str],
) -> Dict[str, float]:
    """
    Return most recent microstructure values at or before `ts` for listed fields.
    Expects ms_df indexed by UTC timestamps. Converts `ts` to UTC.
    """
    if ms_df.empty:
        return {f: float("nan") for f in fields}

    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    try:
        row = ms_df.loc[:ts_utc].iloc[-1]
        return {f: float(row.get(f, float("nan"))) for f in fields}
    except Exception:
        return {f: float("nan") for f in fields}


def extract_features_and_label(
    symbol: str,
    *,
    yields_rec: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Fetch historical bars and compute sliding-window features, Greeks, yields,
    plus microstructure features, and assign movement_type labels.
    """
    end = datetime.now(tz)
    start = end - timedelta(days=HIST_DAYS)

    # Historical bars
    loader = HistoricalDataLoader()
    raw_bars = loader.fetch_bars(symbol, start, end)
    logger.info("Fetched %d bars for %s over %d days", len(raw_bars), symbol, HIST_DAYS)

    if not raw_bars:
        return pd.DataFrame(
            columns=[
                "symbol",
                "breakout_prob",
                "recent_move_pct",
                "time_of_day",
                "volume_ratio",
                "rsi",
                "corr_dev",
                "skew_ratio",
                "yield_spike_2year",
                "yield_spike_10year",
                "yield_spike_30year",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
                "vanna",
                "vomma",
                "movement_type",
                "theta_day",
                "theta_5m",
                "ms_spread_mean",
                "ms_depth_imbalance_mean",
                "ms_ofi_sum",
                "ms_signed_volume_sum",
                "ms_vpin",
            ]
        )

    df = pd.DataFrame(
        [
            {"timestamp": b["t"], "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"], "volume": b["v"]}
            for b in raw_bars
        ]
    )
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(tz)
    df.set_index("dt", inplace=True)

    # Treasury yields (tolerant defaults)
    y = yields_rec or {}
    ys2 = float(y.get("yield_2_year", 0.0))
    ys10 = float(y.get("yield_10_year", 0.0))
    ys30 = float(y.get("yield_30_year", 0.0))

    # Option Greeks snapshot(s)
    greeks = fetch_option_greeks(symbol)
    logger.info("Using yields (2y=%s,10y=%s,30y=%s) and Greeks for %s", ys2, ys10, ys30, symbol)

    # Build microstructure features on the same horizon, bucketed to 5m
    ms_df = build_microstructure_features(
        symbol,
        start=df.index[0].to_pydatetime(),
        end=df.index[-1].to_pydatetime(),
        bucket="5min",
        include_trades=True,
    )

    records: List[Dict[str, Any]] = []
    for i in range(LOOKBACK_BARS, len(df) - LOOKAHEAD_BARS):
        window = df.iloc[i - LOOKBACK_BARS : i]
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

        # Label based on lookahead close vs current close
        next_bar = df.iloc[i + LOOKAHEAD_BARS]
        delta = (next_bar["close"] - current["close"]) / current["close"] if current["close"] else 0.0
        if delta > 0:
            feat["movement_type"] = "CALL"
        elif delta < 0:
            feat["movement_type"] = "PUT"
        else:
            feat["movement_type"] = "NEUTRAL"

        theta_raw = float(greeks.get("theta", 0.0))
        feat["theta_day"] = theta_raw
        feat["theta_5m"] = theta_raw / 78.0  # 6.5h * 60 / 5 = 78 bars in RTH

        # Attach microstructure features as-of the current bar end
        ms_fields = [
            "ms_spread_mean",
            "ms_depth_imbalance_mean",
            "ms_ofi_sum",
            "ms_signed_volume_sum",
            "ms_vpin",
        ]
        feat.update(_asof_micro(ms_df, current.name.tz_convert("UTC"), ms_fields))

        records.append(feat)

    return pd.DataFrame(records)


def main() -> None:
    """Generate training data for all tickers and write to CSV."""
    # Optional kill-switch for CI runs (set DISABLE_YIELDS=1 to skip)
    if os.getenv("DISABLE_YIELDS", "0") == "1":
        yields_rec: Dict[str, Any] = {}
        logger.info("DISABLE_YIELDS=1 set; proceeding without treasury yields.")
    else:
        yields_rec = fetch_treasury_yields()  # safe; may be {}

    out_frames: List[pd.DataFrame] = []
    for t in TICKERS:
        logger.info("Generating data for %s", t)
        out_frames.append(extract_features_and_label(t, yields_rec=yields_rec))

    if not out_frames:
        logger.error("No frames generated.")
        return

    full = pd.concat(out_frames, ignore_index=True) if len(out_frames) > 1 else out_frames[0]
    if full.empty:
        logger.warning("No rows generated; nothing to write.")
        return

    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full.to_csv(OUTPUT_PATH, index=False)
    logger.info("✅ Saved %d rows to %s", len(full), OUTPUT_PATH)


if __name__ == "__main__":
    main()
