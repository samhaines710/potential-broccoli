"""
polygon_nbbo.py

Helpers for retrieving NBBO quote and trade data from Polygon.

All functions assume Polygon timestamps are **milliseconds since epoch** and
convert them to pandas datetime indices (UTC). If your data uses nanosecond
resolution, adjust the converters accordingly.

Dependencies
------------
- pandas
- numpy
- utils.http_client.safe_fetch_polygon_data
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.http_client import safe_fetch_polygon_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------

def _normalize_timestamp(ts: int | float | datetime) -> int:
    """
    Normalize a timestamp into **milliseconds** since the UNIX epoch.

    Accepts a timezone-aware datetime (assumed UTC if tz-naive), a Unix
    timestamp in seconds, or an integer/float milliseconds value.
    """
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, (int, float)):
        # if already ms (>= 10^12) just cast; else treat as seconds
        return int(ts if ts >= 1_000_000_000_000 else ts * 1000)
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _paginate_polygon(url: str, ticker: str) -> List[Dict[str, Any]]:
    """
    Walk a paginated Polygon v3 endpoint following `next_url` until exhausted.
    """
    results: List[Dict[str, Any]] = []
    next_url: Optional[str] = url
    while next_url:
        try:
            data = safe_fetch_polygon_data(next_url, ticker)
        except Exception as e:
            logger.error("Failed to fetch Polygon data for %s: %s", ticker, e)
            raise
        page = data.get("results") or []
        results.extend(page)
        next_url = data.get("next_url")
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_nbbo_quotes(
    ticker: str,
    *,
    start_timestamp: int | float | datetime,
    limit: int = 50_000,
) -> pd.DataFrame:
    """
    Retrieve historical NBBO quotes for `ticker` from /v3/quotes/{ticker}.

    Returns a DataFrame indexed by UTC timestamp with columns:
    ['sip_timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size'].
    """
    start_ms = _normalize_timestamp(start_timestamp)
    url = (
        f"https://api.polygon.io/v3/quotes/{ticker}"
        f"?timestamp={start_ms}&order=asc&limit={limit}&sort=timestamp"
    )
    raw = _paginate_polygon(url, ticker)
    if not raw:
        return pd.DataFrame(
            columns=["sip_timestamp", "bid_price", "ask_price", "bid_size", "ask_size"]
        ).set_index(pd.DatetimeIndex([], name="timestamp"))

    df = pd.DataFrame(raw)

    # Choose best available time field (SIP preferred)
    ts_col: Optional[str] = None
    for c in ("sip_timestamp", "participant_timestamp", "timestamp"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise KeyError("No timestamp column found in quote data")

    df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    for field in ("bid_price", "ask_price", "bid_size", "ask_size"):
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
        else:
            df[field] = np.nan

    cols = ["sip_timestamp", "bid_price", "ask_price", "bid_size", "ask_size"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def fetch_trades(
    ticker: str,
    *,
    start_timestamp: int | float | datetime,
    limit: int = 50_000,
) -> pd.DataFrame:
    """
    Retrieve historical trade prints for `ticker` from /v3/trades/{ticker}.

    Returns a DataFrame indexed by UTC timestamp with columns:
    ['sip_timestamp', 'price', 'size', 'conditions'].
    """
    start_ms = _normalize_timestamp(start_timestamp)
    url = (
        f"https://api.polygon.io/v3/trades/{ticker}"
        f"?timestamp={start_ms}&order=asc&limit={limit}&sort=timestamp"
    )
    raw = _paginate_polygon(url, ticker)
    if not raw:
        return pd.DataFrame(
            columns=["sip_timestamp", "price", "size", "conditions"]
        ).set_index(pd.DatetimeIndex([], name="timestamp"))

    df = pd.DataFrame(raw)

    ts_col: Optional[str] = None
    for c in ("sip_timestamp", "participant_timestamp", "timestamp"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise KeyError("No timestamp column found in trade data")

    df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    df["price"] = pd.to_numeric(df.get("price", np.nan), errors="coerce")
    df["size"] = pd.to_numeric(df.get("size", np.nan), errors="coerce")
    if "conditions" not in df.columns:
        df["conditions"] = [[] for _ in range(len(df))]

    cols = ["sip_timestamp", "price", "size", "conditions"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c != "conditions" else [[] for _ in range(len(df))]
    return df[cols]
