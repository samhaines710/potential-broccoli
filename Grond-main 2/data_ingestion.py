"""
Data ingestion utilities for historical and real-time market data.

- HistoricalDataLoader: fetches 5-minute bars from Polygon REST with pagination.
- RealTimeDataStreamer: connects to Polygon WebSocket, aggregates into 5-minute OHLCV.

This version is robust to status/control WS messages (no KeyError 't') and will
use the delayed socket by default unless USE_REALTIME_WEBSOCKET=1 is set.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode

import pytz
from websocket import WebSocketApp

from config import POLYGON_API_KEY, TICKERS, tz
from utils.http_client import safe_fetch_polygon_data, rate_limited
from utils.logging_utils import write_status

# ── In-memory stores for real-time candles ──────────────────────────────────────
REALTIME_CANDLES: Dict[str, deque] = {symbol: deque(maxlen=200) for symbol in TICKERS}
REALTIME_LOCK = threading.Lock()


# ── Historical REST Loader ──────────────────────────────────────────────────────
class HistoricalDataLoader:
    """Load historical 5-minute bar data from Polygon's REST API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or POLYGON_API_KEY
        self.base_url: str = "https://api.polygon.io"

    @rate_limited
    def _get(self, path_or_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to perform a REST GET with rate limiting.

        Accepts either a relative `path` (we append base_url & params)
        or a fully-qualified `next_url` returned by Polygon.
        """
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            url = path_or_url
        else:
            q = params.copy()
            q["apiKey"] = self.api_key
            url = f"{self.base_url}{path_or_url}?{urlencode(q)}"

        data: Dict[str, Any] = safe_fetch_polygon_data(url, ticker=path_or_url)
        return data or {}

    def fetch_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Return a list of 5-minute bars between `start` and `end` (inclusive)."""
        all_bars: List[Dict[str, Any]] = []
        next_url: Optional[str] = None
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        while True:
            if next_url:
                data = self._get(next_url, {})
            else:
                day0 = start.date().strftime("%Y-%m-%d")
                day1 = end.date().strftime("%Y-%m-%d")
                path = f"/v2/aggs/ticker/{ticker}/range/5/minute/{day0}/{day1}"
                params: Dict[str, Any] = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": limit,
                }
                data = self._get(path, params)

            bars = data.get("results", []) or []
            if not bars:
                break

            for bar in bars:
                t = bar.get("t")
                if t is not None and start_ts <= t <= end_ts:
                    all_bars.append(bar)

            next_url = data.get("next_url")
            if not next_url or len(bars) < limit:
                break

        write_status(f"Fetched {len(all_bars)} historical bars for {ticker}")
        return all_bars


# ── Real-Time WebSocket Streamer ────────────────────────────────────────────────
class RealTimeDataStreamer:
    """
    Subscribe to Polygon's WebSocket and aggregate minute events into 5-minute bars
    stored in REALTIME_CANDLES.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or POLYGON_API_KEY

        # Use delayed feed by default to avoid policy violation (1008) on accounts
        # without real-time entitlements. Set USE_REALTIME_WEBSOCKET=1 to force RT.
        if os.getenv("USE_REALTIME_WEBSOCKET", "0").lower() in ("1", "true", "yes"):
            self.ws_url: str = "wss://socket.polygon.io/stocks"
        else:
            self.ws_url: str = "wss://delayed.polygon.io/stocks"

        self._ws_lock = threading.Lock()
        self._ws: Optional[WebSocketApp] = None

        # We subscribe to minute aggregates explicitly
        self._agg_channel_prefix = "AM"  # minute aggregates

    def on_open(self, ws: WebSocketApp) -> None:
        """Authenticate and subscribe on WebSocket open."""
        write_status("RT WS opened; authenticating…")
        ws.send(json.dumps({"action": "auth", "params": self.api_key}))
        for ticker in TICKERS:
            ws.send(json.dumps({"action": "subscribe", "params": f"{self._agg_channel_prefix}.{ticker}"}))

    def _append_5m_bar(self, sym: str, ts_ms: int, o: float, h: float, l: float, c: float, v: float) -> None:
        """Bucket Minute agg into 5-min OHLCV in a threadsafe way."""
        # tz-aware bucketing
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=pytz.UTC).astimezone(tz)
        minute = (dt.minute // 5) * 5
        bucket = dt.replace(minute=minute, second=0, microsecond=0)
        bar_ts = int(bucket.timestamp() * 1000)

        rec = {
            "timestamp": bar_ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        }

        with REALTIME_LOCK:
            dq = REALTIME_CANDLES.setdefault(sym, deque(maxlen=200))
            if dq and dq[-1]["timestamp"] == bar_ts:
                prev = dq[-1]
                prev["high"] = max(prev["high"], rec["high"])
                prev["low"] = min(prev["low"], rec["low"])
                prev["close"] = rec["close"]
                prev["volume"] += rec["volume"]
            else:
                dq.append(rec)

    def on_message(self, ws: WebSocketApp, message: str) -> None:
        """Handle incoming WS messages; skip non-aggregate/status frames safely."""
        try:
            payload = json.loads(message)
            items = payload if isinstance(payload, list) else [payload]
            for itm in items:
                if not isinstance(itm, dict):
                    continue

                ev = itm.get("ev")
                # only handle minute aggregates
                if ev != "AM":
                    # status/heartbeat/etc come through here — ignore quietly
                    continue

                # AM frames use 's' (start) / 'e' (end) timestamps
                ts_ms = itm.get("s") or itm.get("e")
                o = itm.get("o")
                h = itm.get("h")
                l = itm.get("l")
                c = itm.get("c")
                v = itm.get("v")
                sym = itm.get("sym")

                if None in (ts_ms, o, h, l, c, v) or not sym:
                    # malformed aggregate; skip
                    continue

                self._append_5m_bar(sym, ts_ms, o, h, l, c, v)

        except Exception as exc:
            # Keep this quiet-ish to avoid spam; you can make this a debug if you want
            write_status(f"RT on_message error: {exc!s}")

    def on_error(self, ws: WebSocketApp, err: Exception) -> None:
        write_status(f"RT WS error: {err}")

    def on_close(self, ws: WebSocketApp, code: int, msg: str) -> None:
        write_status(f"RT WS closed: {code}/{msg}; reconnecting in 5s")
        time.sleep(5)
        self.start()

    def start(self) -> None:
        """Start the WebSocket streaming in a background daemon thread."""

        def _run() -> None:
            ws = WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            with self._ws_lock:
                self._ws = ws
            ws.run_forever()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        write_status("RealTimeDataStreamer thread started.")