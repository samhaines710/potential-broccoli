#!/usr/bin/env python3
"""
Prepare training data for movement classification.

- Fetches historical OHLC bars (Polygon), treasury yields (if available),
  per-bar option Greeks (Polygon snapshot -> near-ATM CALL; BS backfill if needed),
  and microstructure features.
- Computes sliding-window features + binary label and writes
  data/movement_training_data.csv.

Hardening:
- No network calls at import.
- Greeks are fetched per-bar; zeros are never injected. The job can hard-fail on
  poor Greeks coverage via REQUIRE_GREEKS=1 and MIN_GREEKS_COVERAGE (default 0.50).
- Yields fetch tolerant; DISABLE_YIELDS=1 to skip in CI.
- Timezone-safe for microstructure alignment.

Logging:
- Centralized JSON logging via utils.logging_utils.configure_logging().
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
)
from utils.micro_features import build_microstructure_features
from utils.greeks_polygon import fetch_greeks_polygon_asof  # NEW: per-bar Greeks

# ── Configure logging ───────────────────────────────────────────────────────────
configure_logging()
logger = get_logger(__name__)

# ── Polygon Treasury Yields endpoint ───────────────────────────────────────────
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
YIELDS_ENDPOINT = "https://api.polygon.io/fed/v1/treasury-yields"

def fetch_treasury_yields(date: Optional[str] = None) -> Dict[str, Any]:
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

# Greeks requirements
REQUIRE_GREEKS = os.getenv("REQUIRE_GREEKS", "1") == "1"
MIN_GREEKS_COVERAGE = float(os.getenv("MIN_GREEKS_COVERAGE", "0.50"))  # 50%

def _asof_micro(ms_df: pd.DataFrame, ts: pd.Timestamp, fields: List[str]) -> Dict[str, float]:
    if ms_df.empty:
        return {f: float("nan") for f in fields}
    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    try:
        row = ms_df.loc[:ts_utc].iloc[-1]
        return {f: float(row.get(f, float("nan"))) for f in fields}
    except Exception:
        return {f: float("nan") for f in fields}

def extract_features_and_label(symbol: str, *, yields_rec: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    end = datetime.now(tz)
    start = end - timedelta(days=HIST_DAYS)

    loader = HistoricalDataLoader()
    raw_bars = loader.fetch_bars(symbol, start, end)
    logger.info("Fetched %d bars for %s over %d days", len(raw_bars), symbol, HIST_DAYS)

    if not raw_bars:
        return pd.DataFrame(columns=[
            "symbol","breakout_prob","recent_move_pct","time_of_day","volume_ratio","rsi","corr_dev","skew_ratio",
            "yield_spike_2year","yield_spike_10year","yield_spike_30year",
            "delta","gamma","theta","vega","rho","implied_volatility",
            "movement_type","theta_day","theta_5m",
            "ms_spread_mean","ms_depth_imbalance_mean","ms_ofi_sum","ms_signed_volume_sum","ms_vpin",
        ])

    df = pd.DataFrame([{"timestamp": b["t"], "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"], "volume": b["v"]} for b in raw_bars])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(tz)
    df.set_index("dt", inplace=True)

    # Yields (tolerant)
    y = yields_rec or {}
    ys2 = float(y.get("yield_2_year", 0.0))
    ys10 = float(y.get("yield_10_year", 0.0))
    ys30 = float(y.get("yield_30_year", 0.0))

    # Microstructure horizon, bucketed 5m
    ms_df = build_microstructure_features(
        symbol,
        start=df.index[0].to_pydatetime(),
        end=df.index[-1].to_pydatetime(),
        bucket="5min",
        include_trades=True,
    )

    records: List[Dict[str, Any]] = []
    greeks_rows = 0
    for i in range(LOOKBACK_BARS, len(df) - LOOKAHEAD_BARS):
        window = df.iloc[i - LOOKBACK_BARS : i]
        current = df.iloc[i]
        current_ts = df.index[i]
        epoch_sec = int(current_ts.tz_convert("UTC").timestamp())
        spot = float(current["close"])

        # ── Real per-bar Greeks from Polygon (hard errors propagate)
        greeks: Dict[str, Any] = {}
        try:
            greeks = fetch_greeks_polygon_asof(symbol, epoch_sec, spot)
            # coverage check counter (>3 present counts as "has greeks")
            present = sum(1 for k in ("delta","gamma","theta","vega","rho","implied_volatility")
                          if (k in greeks) and pd.notna(greeks[k]))
            if present >= 3:
                greeks_rows += 1
        except Exception as e:
            logger.warning("Greeks missing for %s @ %s: %s", symbol, current_ts.isoformat(), e)

        candles = window.reset_index().to_dict("records")

        feat: Dict[str, Any] = {
            "symbol": symbol,
            "breakout_prob": calculate_breakout_prob(candles),
            "recent_move_pct": calculate_recent_move_pct(candles),
            "time_of_day": calculate_time_of_day(current_ts),
            "volume_ratio": calculate_volume_ratio(candles),
            "rsi": compute_rsi(candles),
            "corr_dev": compute_corr_deviation(symbol),
            "skew_ratio": compute_skew_ratio(symbol),
            "yield_spike_2year": ys2,
            "yield_spike_10year": ys10,
            "yield_spike_30year": ys30,
            **greeks,
        }

        # Label
        next_bar = df.iloc[i + LOOKAHEAD_BARS]
        delta_price = (next_bar["close"] - current["close"]) / current["close"] if current["close"] else 0.0
        feat["movement_type"] = "CALL" if delta_price > 0 else ("PUT" if delta_price < 0 else "NEUTRAL")

        # Theta conversions (if available, theta is annualized)
        theta_raw = feat.get("theta", float("nan"))
        try:
            theta_day = float(theta_raw) / 252.0
            theta_5m  = theta_day / 78.0  # 6.5h * 60 / 5 = 78 bars
        except Exception:
            theta_day = float("nan")
            theta_5m  = float("nan")
        feat["theta_day"] = theta_day
        feat["theta_5m"] = theta_5m

        # Microstructure as-of
        ms_fields = ["ms_spread_mean","ms_depth_imbalance_mean","ms_ofi_sum","ms_signed_volume_sum","ms_vpin"]
        feat.update(_asof_micro(ms_df, current_ts, ms_fields))

        records.append(feat)

    out = pd.DataFrame(records)
    # Hard coverage enforcement per symbol
    if REQUIRE_GREEKS and len(out) > 0:
        cov = float((out[["delta","gamma","theta","vega","rho","implied_volatility"]].notna().sum(axis=1) >= 3).mean())
        logger.info("Greeks coverage for %s rows with ≥3 fields present: %.1f%%", symbol, 100.0*cov)
        if cov < MIN_GREEKS_COVERAGE:
            raise SystemExit(f"Insufficient Greeks coverage for {symbol}: {cov:.2%} < {MIN_GREEKS_COVERAGE:.0%}")
    return out

def main() -> None:
    # Yields
    if os.getenv("DISABLE_YIELDS", "0") == "1":
        yields_rec: Dict[str, Any] = {}
        logger.info("DISABLE_YIELDS=1 set; proceeding without treasury yields.")
    else:
        yields_rec = fetch_treasury_yields()

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

    # Global coverage guard (post-concat)
    if REQUIRE_GREEKS:
        gcols = ["delta","gamma","theta","vega","rho","implied_volatility"]
        coverage = float((full[gcols].notna().sum(axis=1) >= 3).mean())
        logger.info("Greeks coverage across all rows (≥3 fields): %.1f%%", 100.0*coverage)
        if coverage < MIN_GREEKS_COVERAGE:
            raise SystemExit(f"Insufficient Greeks coverage overall: {coverage:.2%} < {MIN_GREEKS_COVERAGE:.0%}")

    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full.to_csv(OUTPUT_PATH, index=False)
    logger.info("✅ Saved %d rows to %s", len(full), OUTPUT_PATH)

if __name__ == "__main__":
    main()