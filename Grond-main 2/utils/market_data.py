import time
from datetime import datetime, timedelta, date
from collections import deque

from config import POLYGON_API_KEY, TICKERS, DEFAULT_CANDLE_LIMIT
from utils.logging_utils import write_status
from utils.http_client import safe_fetch_polygon_data, rate_limited

# Thread-safe, in-memory 5-minute bar store for real-time feeds
REALTIME_CANDLES = { t: deque(maxlen=200) for t in TICKERS }

@rate_limited
def _get_json(url: str, ticker: str = "") -> dict:
    """
    Wrapper around safe_fetch_polygon_data to handle rate-limiting.
    """
    return safe_fetch_polygon_data(url, ticker)

def reformat_candles(raw: list) -> list:
    """
    Turn raw Polygon bar dicts into uniform OHLCV dicts.
    """
    return [
        {
            "timestamp": c.get("t", c.get("timestamp")),
            "open":      c.get("o", c.get("open")),
            "high":      c.get("h", c.get("high")),
            "low":       c.get("l", c.get("low")),
            "close":     c.get("c", c.get("close")),
            "volume":    c.get("v", c.get("volume", 0)),
        }
        for c in (raw or [])
    ]

def fetch_historical_5m_candles(
    ticker: str,
    days:    int | None       = None,
    date_obj: date | None    = None,
    limit:   int             = DEFAULT_CANDLE_LIMIT,
    as_of:   datetime | None = None
) -> list:
    """
    Backfill all 5-min bars for `ticker` between start and end dates.
    If `date_obj` is provided, fetch just that day; otherwise use last `days`.
    """
    now = datetime.now()
    if date_obj:
        start_str = date_obj.strftime("%Y-%m-%d")
        end_str   = date_obj.strftime("%Y-%m-%d")
    else:
        d = min(days or 3650, 3650)
        start_str = (now - timedelta(days=d)).strftime("%Y-%m-%d")
        end_str   = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    all_bars, next_url = [], None
    while True:
        if next_url:
            data = _get_json(next_url, ticker)
        else:
            url  = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/"
                f"{start_str}/{end_str}"
                f"?adjusted=true&sort=asc&limit={limit}&apiKey={POLYGON_API_KEY}"
            )
            data = _get_json(url, ticker)
        res = data.get("results", [])
        if not res:
            break
        all_bars.extend(res)
        next_url = data.get("next_url")
        if not next_url or len(res) < limit:
            break

    write_status(f"Fetched {len(all_bars)} historical 5m bars for {ticker}")
    if as_of:
        cutoff = int(as_of.timestamp() * 1000)
        return [b for b in all_bars if b["t"] <= cutoff]
    return all_bars

def fetch_today_5m_candles(
    ticker: str,
    limit:  int             = DEFAULT_CANDLE_LIMIT,
    as_of:  datetime | None = None
) -> list:
    """
    Fetch all 5-min bars for `ticker` from yesterday until now.
    """
    now      = as_of or datetime.now()
    today    = now.date()
    start_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    end_str   = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    all_bars, next_url = [], None
    while True:
        if next_url:
            data = _get_json(next_url, ticker)
        else:
            url  = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/"
                f"{start_str}/{end_str}"
                f"?adjusted=true&sort=asc&limit={limit}&apiKey={POLYGON_API_KEY}"
            )
            data = _get_json(url, ticker)
        res = data.get("results", [])
        if not res:
            break
        all_bars.extend(res)
        next_url = data.get("next_url")
        if not next_url or len(res) < limit:
            break

    cutoff = int(now.timestamp() * 1000)
    bars   = [b for b in all_bars if b["t"] <= cutoff]
    write_status(f"Fetched {len(bars)} today’s 5m bars for {ticker}")
    return bars

def fetch_latest_5m_candle(
    ticker: str,
    as_of:  datetime | None = None
) -> dict:
    """
    Get the single most‐recent 5-min bar for `ticker`.
    """
    now      = as_of or datetime.now()
    today    = now.date()
    start_str = today.strftime("%Y-%m-%d")
    end_str   = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    url  = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/"
        f"{start_str}/{end_str}"
        f"?adjusted=true&sort=desc&limit=5&apiKey={POLYGON_API_KEY}"
    )
    data = _get_json(url, ticker)
    for c in data.get("results", []):
        ts   = datetime.fromtimestamp(c["t"] / 1000)
        if ts <= now:
            return {
                "timestamp": c["t"],
                "open":      c["o"],
                "high":      c["h"],
                "low":       c["l"],
                "close":     c["c"],
                "volume":    c["v"],
            }
    return {}

def fetch_premarket_early_data(
    ticker: str,
    as_of:  datetime | None = None
) -> list:
    """
    Backfill off‐hours / premarket bars for `ticker` before market open.
    """
    now      = as_of or datetime.now()
    today    = now.date()
    start_str = today.strftime("%Y-%m-%d")
    end_str   = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    url  = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/"
        f"{start_str}/{end_str}"
        f"?adjusted=true&sort=asc&limit={DEFAULT_CANDLE_LIMIT}&apiKey={POLYGON_API_KEY}"
    )
    data = _get_json(url, ticker)

    bars = []
    for c in data.get("results", []):
        bar_time = datetime.fromtimestamp(c["t"] / 1000)
        if bar_time <= now:
            bars.append(c)
    write_status(f"Fetched {len(bars)} pre-market bars for {ticker}")
    return bars
