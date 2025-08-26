"""Analytical utilities for evaluating market data and signals.

This module contains a collection of functions used to calculate
probabilities, ratios, and statistical indicators based on price and
volume data. It also includes helpers to fetch snapshots from the
Polygon API and compute derived metrics like skew ratio, correlation
deviation, RSI, and yield spikes.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, time as dtime
from typing import List, Tuple, Optional

from config import LOOKBACK_BREAKOUT, LOOKBACK_RISK_REWARD, tz, POLYGON_API_KEY
from utils.http_client import safe_fetch_polygon_data, fetch_option_greeks


# ──────────────────────────────────────────────────────────────────────────────
# Core feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def calculate_breakout_prob(
    candles: List[dict],
    lookback: int = LOOKBACK_BREAKOUT,
) -> float:
    """
    Estimate the probability of a breakout given a list of candles.

    The function looks back over the specified number of bars and computes
    a score based on price movement, volume, and trend direction. The
    resulting value is a percentage between 0 and 100.
    """
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves: List[float] = []
    vols: List[float] = []
    for bar in recent:
        open_price = bar.get("open", 0) or bar.get("o", 0)
        close_price = bar.get("close", 0) or bar.get("c", 0)
        if open_price:
            moves.append((close_price - open_price) / open_price)
            vols.append(bar.get("volume", 0) or bar.get("v", 0))
    if not moves:
        return 0.0
    avg_move = sum(abs(m) for m in moves) / len(moves)
    avg_vol = max(sum(vols) / len(vols), 1e-6)
    last = recent[-1]
    last_open = last.get("open", 0) or last.get("o", 0)
    last_close = last.get("close", 0) or last.get("c", 0)
    cm = abs(last_close - last_open) / last_open if last_open else 0.0
    cv = last.get("volume", 0) or last.get("v", 0)
    trend = sum(1.0 if m > 0 else -1.0 for m in moves) / len(moves)
    score = (
        50
        * math.log1p(cm / max(avg_move, 1e-6))
        * math.log1p(cv / avg_vol)
        * (1 + trend)
    )
    return round(max(0.0, min(100.0, score)), 2)


def calculate_recent_move_pct(candles: List[dict]) -> float:
    """
    Compute the percentage change from the first bar's open to the last bar's close.
    """
    if len(candles) < 2:
        return 0.0
    first = candles[0]
    last = candles[-1]
    first_open = first.get("open", 0) or first.get("o", 0)
    last_close = last.get("close", 0) or last.get("c", 0)
    if not first_open:
        return 0.0
    pct = (last_close - first_open) / first_open
    return round(pct, 4)


def calculate_signal_persistence(
    candles: List[dict],
    lookback: int = LOOKBACK_BREAKOUT,
) -> float:
    """
    Estimate how persistent a signal is by comparing
    the last bar's move to average moves.
    """
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves = [
        abs(((bar.get("close", 0) or bar.get("c", 0)) - (bar.get("open", 0) or bar.get("o", 0))) / (bar.get("open", 0) or bar.get("o", 0)))
        for bar in recent
        if (bar.get("open", 0) or bar.get("o", 0))
    ]
    if not moves:
        return 0.0
    avg_move = sum(moves) / len(moves)
    last_move = moves[-1]
    return round(100.0 * (1 - min(last_move / max(avg_move, 1e-6), 1.0)), 2)


def calculate_reversal_and_scope(
    candles: List[dict],
    oi_bias: str,
) -> Tuple[bool, bool, float]:
    """
    Determine if there is a reversal and whether the magnitude (scope) is significant.

    Returns a tuple ``(is_reversal, is_scope, move_pct*100)`` where ``is_reversal``
    indicates whether the price move conflicts with the open interest bias and
    ``is_scope`` flags whether the absolute move is above a small threshold.
    """
    if len(candles) < 2:
        return False, False, 0.0
    prev, last = candles[-2], candles[-1]
    prev_close = prev.get("close", 0) or prev.get("c", 0)
    last_close = last.get("close", 0) or last.get("c", 0)
    mv = (last_close - prev_close) / prev_close if prev_close else 0.0
    rev = (
        (oi_bias == "CALL_DOMINANT" and mv < -0.01)
        or (oi_bias == "PUT_DOMINANT" and mv > 0.01)
    )
    sc = abs(mv) > 0.005
    return rev, sc, round(mv * 100.0, 4)


def calculate_risk_reward(candles: List[dict]) -> float:
    """
    Compute a simple risk–reward ratio based on high–low ranges over a lookback window.
    """
    if len(candles) < LOOKBACK_RISK_REWARD:
        return 1.0
    sample = candles[-LOOKBACK_RISK_REWARD:]
    # Use ATR-like average range as a proxy
    hl_ranges = []
    for bar in sample:
        high = bar.get("high", 0) or bar.get("h", 0)
        low = bar.get("low", 0) or bar.get("l", 0)
        hl_ranges.append(max(high - low, 0.0))
    atr = sum(hl_ranges) / max(len(hl_ranges), 1)
    # Define reward as 2*ATR and risk as 0.5*ATR, clipping denominators
    return round((atr * 2.0) / max(atr * 0.5, 1e-6), 2)


def calculate_time_of_day(as_of: Optional[datetime] = None) -> str:
    """
    Categorize the current time into trading session labels.
    """
    now_time = (as_of or datetime.now(tz)).astimezone(tz).time()
    if dtime(4, 0) <= now_time < dtime(9, 30):
        return "PRE_MARKET"
    if dtime(9, 30) <= now_time < dtime(11, 0):
        return "MORNING"
    if dtime(11, 0) <= now_time < dtime(14, 0):
        return "MIDDAY"
    if dtime(14, 0) <= now_time < dtime(16, 0):
        return "AFTERNOON"
    if dtime(16, 0) <= now_time < dtime(20, 0):
        return "AFTER_HOURS"
    return "OFF_HOURS"


def calculate_volume_ratio(candles: List[dict]) -> float:
    """
    Compute the ratio of the last bar's volume to the average volume.
    """
    if not candles:
        return 1.0
    vols = [(bar.get("volume", 0) or bar.get("v", 0)) for bar in candles]
    avg_vol = max(sum(vols) / len(vols), 1.0)
    last_vol = candles[-1].get("volume", 0) or candles[-1].get("v", 0)
    return round(last_vol / avg_vol, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Technical indicators & market context features
# ──────────────────────────────────────────────────────────────────────────────

def compute_rsi(candles: List[dict], period: int = 14) -> float:
    """
    Compute RSI over 'period' using close prices from `candles`.
    """
    if len(candles) < period + 1:
        return 50.0
    closes = [(bar.get("close", 0) or bar.get("c", 0)) for bar in candles]
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(abs(min(diff, 0.0)))
    look = min(period, len(gains))
    avg_g = (sum(gains[-look:]) / look) if look else 0.0
    avg_l = (sum(losses[-look:]) / look) if look else 0.0
    avg_l = max(avg_l, 1e-6)
    rs = avg_g / avg_l
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _fetch_daily_aggs(symbol: str, days: int = 90) -> List[dict]:
    """Internal helper: fetch daily aggregates for `symbol` over the last `days`."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 5)  # pad for non-trading days
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    data = safe_fetch_polygon_data(url, symbol)
    return data.get("results", [])


def compute_corr_deviation(ticker: str, window: int = 20) -> float:
    """
    Compute the short-horizon correlation between `ticker` and SPY and return
    1 - |corr| as a simple 'deviation' score (higher => more idiosyncratic).
    """
    try:
        bars_a = _fetch_daily_aggs(ticker, days=max(window + 10, 60))
        bars_b = _fetch_daily_aggs("SPY", days=max(window + 10, 60))
        closes_a = [b["c"] for b in bars_a][- (window + 1):]
        closes_b = [b["c"] for b in bars_b][- (window + 1):]
        if len(closes_a) < window + 1 or len(closes_b) < window + 1:
            return 0.5
        rets_a = [(closes_a[i] - closes_a[i - 1]) / closes_a[i - 1] for i in range(1, len(closes_a))]
        rets_b = [(closes_b[i] - closes_b[i - 1]) / closes_b[i - 1] for i in range(1, len(closes_b))]
        # align lengths
        n = min(len(rets_a), len(rets_b), window)
        if n <= 2:
            return 0.5
        ma = sum(rets_a[-n:]) / n
        mb = sum(rets_b[-n:]) / n
        num = sum((rets_a[-n + i] - ma) * (rets_b[-n + i] - mb) for i in range(n))
        den_a = math.sqrt(sum((rets_a[-n + i] - ma) ** 2 for i in range(n)))
        den_b = math.sqrt(sum((rets_b[-n + i] - mb) ** 2 for i in range(n)))
        if den_a == 0 or den_b == 0:
            return 0.5
        corr = max(min(num / (den_a * den_b), 1.0), -1.0)
        return round(1.0 - abs(corr), 4)
    except Exception:
        # On any error, return neutral value
        return 0.5


def compute_skew_ratio(ticker: str) -> float:
    """
    Compute the skew ratio of implied volatilities between call and put options
    using Polygon's options snapshot. Returns (avg_call_iv / avg_put_iv).
    """
    try:
        # Use the same endpoint as fetch_option_greeks for consistency
        url = (
            f"https://api.polygon.io/v3/snapshot/options/{ticker}"
            f"?apiKey={POLYGON_API_KEY}"
        )
        data = safe_fetch_polygon_data(url, ticker)
        results = data.get("results", [])
        if not results:
            # fall back: use any implied_volatility from fetch_option_greeks
            g = fetch_option_greeks(ticker)
            iv = float(g.get("implied_volatility", 0.0))
            return 1.0 if iv == 0.0 else 1.0  # neutral if unknown
        call_ivs: List[float] = []
        put_ivs: List[float] = []
        for opt in results:
            details = opt.get("details", {})
            greeks = opt.get("greeks", {})
            iv = float(greeks.get("implied_volatility", 0.0))
            if iv <= 0:
                continue
            ctype = details.get("contract_type", "").lower()
            if ctype == "call":
                call_ivs.append(iv)
            elif ctype == "put":
                put_ivs.append(iv)
        if not call_ivs or not put_ivs:
            return 1.0
        avg_call = sum(call_ivs) / len(call_ivs)
        avg_put = sum(put_ivs) / len(put_ivs)
        if avg_put <= 0:
            return 1.0
        return round(avg_call / avg_put, 4)
    except Exception:
        return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Yield spike detection (updated for /fed/v1/treasury-yields)
# ──────────────────────────────────────────────────────────────────────────────

def detect_yield_spike(tenor: str = "10year", spike_pct: float = 0.25) -> bool:
    """
    Detect large jumps in U.S. Treasury yields relative to the previous value.

    Uses Polygon's `/fed/v1/treasury-yields` endpoint, which returns records
    containing fields like `yield_10_year`, `yield_30_year`, etc. We fetch
    the most recent record, persist the last seen value locally, and flag
    a spike when the relative change exceeds `spike_pct`.
    """
    field_map = {
        "1month": "yield_1_month",
        "3month": "yield_3_month",
        "6month": "yield_6_month",
        "1year":  "yield_1_year",
        "2year":  "yield_2_year",
        "3year":  "yield_3_year",
        "5year":  "yield_5_year",
        "7year":  "yield_7_year",
        "10year": "yield_10_year",
        "20year": "yield_20_year",
        "30year": "yield_30_year",
    }
    key = field_map.get(tenor.lower())
    if not key:
        return False

    fname = f"last_yield_{tenor}.json"
    try:
        url = (
            "https://api.polygon.io/fed/v1/treasury-yields"
            f"?limit=1&sort=date.desc&apiKey={POLYGON_API_KEY}"
        )
        data = safe_fetch_polygon_data(url, tenor)
        results = data.get("results", [])
        if not results:
            return False
        record = results[0]
        curr_raw = record.get(key)
        if curr_raw in (None, "", 0, 0.0):
            return False
        curr = float(curr_raw)
        prev: Optional[float] = None
        if os.path.exists(fname):
            try:
                with open(fname) as f:
                    prev = json.load(f).get("value")
            except Exception:
                prev = None
        with open(fname, "w") as f:
            json.dump({"value": curr}, f)
        return prev is not None and prev != 0 and abs(curr - prev) / abs(prev) > spike_pct
    except Exception:
        return False
