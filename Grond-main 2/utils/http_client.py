#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HTTP client helpers with metrics, retries, rate limiting, and Polygon utilities.

Exports
-------
- rate_limited([calls_per_sec: float]) -> decorator
- safe_fetch_polygon_data(url, params=None, *, timeout=None, max_retries=None,
                          session=None, ticker=None) -> dict
- fetch_option_greeks(ticker: str) -> dict
- fetch_polygon_aggs_chunked(ticker, start, end, *, multiplier=1, timespan="day",
                             adjusted=True, sort="asc", limit=5000, chunk_days=None) -> list[dict]

Design goals
------------
- **Metrics**: Always emit REST_LATENCY(service, endpoint, method), REST_CALLS(..., status),
  count 429s via REST_429(service, endpoint).
- **Resilience**: Session pooling, gzip, retries with `Retry-After` honor, exponential backoff.
- **Compatibility**: Keep function names and signatures used elsewhere; `rate_limited` works
  with or without an argument.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, TypeVar, cast, List
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

import requests
from requests.adapters import HTTPAdapter

try:
    # urllib3 v2
    from urllib3.util import Retry  # type: ignore
except Exception:  # pragma: no cover
    # urllib3 v1 fallback
    from urllib3.util.retry import Retry  # type: ignore

from utils.logging_utils import (
    REST_CALLS,
    REST_LATENCY,
    REST_429,
    get_logger,
)

T = TypeVar("T")
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Configuration (env)
# -----------------------------------------------------------------------------
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "25"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "6"))
HTTP_BACKOFF_BASE = float(os.getenv("HTTP_BACKOFF_BASE", "0.6"))
HTTP_BACKOFF_MAX = float(os.getenv("HTTP_BACKOFF_MAX", "15.0"))
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Client-side soft rate limit (calls/sec) for decorated functions
DEFAULT_CALLS_PER_SEC = float(os.getenv("HTTP_CALLS_PER_SEC", "4.0"))

# Chunking for Polygon /v2/aggs range windows (days per request)
POLYGON_CHUNK_DAYS = int(os.getenv("POLYGON_CHUNK_DAYS", "30"))

_DEFAULT_HEADERS = {
    "User-Agent": "grond-http/1.1 (+metrics;retries;gzip)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# -----------------------------------------------------------------------------
# Service/endpoint helpers for metrics labeling
# -----------------------------------------------------------------------------
def _service_from_url(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if not netloc:
        return "unknown"
    # normalize polygon
    if "api.polygon.io" in netloc:
        return "polygon"
    return netloc


def _endpoint_from_url(url: str) -> str:
    p = urlparse(url)
    segs = [s for s in p.path.split("/") if s]
    if not segs:
        return "/"
    # Limit cardinality
    return "/" + "/".join(segs[:4])


def _append_polygon_key(url: str) -> str:
    """
    Ensure apiKey is present for Polygon endpoints (without overwriting).
    """
    if "api.polygon.io" not in url:
        return url
    if not POLYGON_API_KEY:
        return url
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "apiKey" not in qs:
        qs["apiKey"] = POLYGON_API_KEY
    return urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))


# -----------------------------------------------------------------------------
# Backoff helpers
# -----------------------------------------------------------------------------
def _sleep_backoff(attempt: int) -> None:
    """
    Exponential backoff with jitter; attempt starts at 1.
    """
    base = HTTP_BACKOFF_BASE * (2 ** max(attempt - 1, 0))
    delay = min(base * (0.7 + 0.6 * random.random()), HTTP_BACKOFF_MAX)
    time.sleep(delay)


# -----------------------------------------------------------------------------
# rate_limited decorator (compatible: with or without cps argument)
# -----------------------------------------------------------------------------
def rate_limited(func: Optional[Callable[..., T]] = None, calls_per_sec: Optional[float] = None):
    """
    Usage:
      @rate_limited                # uses DEFAULT_CALLS_PER_SEC
      def foo(...): ...

      @rate_limited(calls_per_sec=2.5)
      def bar(...): ...

    Implements **soft client-side pacing** (not retries). Retries are handled by HTTP stack.
    """
    cps = DEFAULT_CALLS_PER_SEC if calls_per_sec is None else float(calls_per_sec)
    interval = 1.0 / max(0.0001, cps)
    state = {"last": 0.0}

    def _decorate(f: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            dt = time.monotonic() - state["last"]
            if dt < interval:
                time.sleep(interval - dt)
            try:
                return f(*args, **kwargs)
            finally:
                state["last"] = time.monotonic()
        return wrapped

    # If used as @rate_limited without parentheses
    if callable(func):
        return _decorate(func)
    return _decorate


# -----------------------------------------------------------------------------
# Session with retries (429/5xx/read/connect timeouts), keep-alive pooling
# -----------------------------------------------------------------------------
def _build_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(_DEFAULT_HEADERS)
    retry = Retry(
        total=HTTP_MAX_RETRIES,
        connect=HTTP_MAX_RETRIES,
        read=HTTP_MAX_RETRIES,
        status=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF_BASE,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


_SESSION = _build_session()


# -----------------------------------------------------------------------------
# Core HTTP with metrics
# -----------------------------------------------------------------------------
def _request_with_metrics(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    """
    Perform an HTTP request and emit metrics:
      REST_LATENCY.labels(service, endpoint, method).observe(seconds)
      REST_CALLS.labels(service, endpoint, method, status).inc()
      (429 counting is done by callers that inspect the response code)
    """
    service = _service_from_url(url)
    endpoint = _endpoint_from_url(url)
    m = method.upper()

    start = time.monotonic()
    resp: Optional[requests.Response] = None
    try:
        hdrs = dict(_DEFAULT_HEADERS)
        if headers:
            hdrs.update(headers)
        sess = session or _SESSION
        resp = sess.request(m, url, params=params, headers=hdrs, timeout=timeout or HTTP_TIMEOUT)
        return resp
    finally:
        elapsed = max(time.monotonic() - start, 0.0)
        # metrics: always labeled
        REST_LATENCY.labels(service, endpoint, m).observe(elapsed)
        status_label = str(resp.status_code) if resp is not None else "NO_RESP"
        REST_CALLS.labels(service, endpoint, m, status_label).inc()


@rate_limited  # soft pacing across callers
def safe_fetch_polygon_data(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,   # kept for signature compatibility (handled by session retry)
    session: Optional[requests.Session] = None,
    ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """
    GET JSON with retries (via Session/Retry), metrics, and error handling.
    Automatically appends POLYGON_API_KEY for Polygon endpoints.

    Returns empty dict {} on non-retryable errors for resilience.
    """
    url = _append_polygon_key(url)
    resp = None
    try:
        resp = _request_with_metrics("GET", url, params=params, timeout=timeout, session=session)
        if resp.status_code == 429:
            REST_429.labels(_service_from_url(url), _endpoint_from_url(url)).inc()
        resp.raise_for_status()
        return cast(Dict[str, Any], resp.json())
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", "ERR")
        logger.warning("HTTP %s for %s (%s): %s", status, ticker or _endpoint_from_url(url), url, e)
        # return {} to avoid exploding upstream; your feature builders can choose to log/fallback
        return {}
    except (requests.ConnectionError, requests.Timeout) as e:
        logger.warning("Network error for %s (%s): %s", ticker or _endpoint_from_url(url), url, e)
        return {}
    except json.JSONDecodeError as e:
        logger.warning("Bad JSON for %s (%s): %s", ticker or _endpoint_from_url(url), url, e)
        return {}
    except Exception as e:
        logger.exception("Unexpected error for %s (%s): %s", ticker or _endpoint_from_url(url), url, e)
        return {}


# -----------------------------------------------------------------------------
# Polygon Greeks (snapshot)
# -----------------------------------------------------------------------------
_GREEK_KEYS = (
    "delta", "gamma", "theta", "vega", "rho",
    "vanna", "vomma", "charm", "veta", "speed", "zomma", "color",
    "implied_volatility",
)

def _zeros_greeks(source: str) -> Dict[str, float | str]:
    out: Dict[str, float | str] = {k: 0.0 for k in _GREEK_KEYS}
    out["source"] = source
    return out


def _avg(vals: List[float]) -> float:
    return (sum(vals) / float(len(vals))) if vals else 0.0


def fetch_option_greeks(ticker: str) -> Dict[str, float | str]:
    """
    Aggregate Greeks for the underlying via Polygon snapshot API.
    Averages available contracts; falls back to zeros if missing or failing.
    """
    if not POLYGON_API_KEY:
        return _zeros_greeks("fallback")

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    data = safe_fetch_polygon_data(url, ticker=ticker)
    results = data.get("results") if isinstance(data, dict) else None
    if not results or not isinstance(results, list):
        return _zeros_greeks("fallback")

    buckets: Dict[str, List[float]] = {k: [] for k in _GREEK_KEYS}
    for opt in results:
        greeks = opt.get("greeks") or {}
        for k in _GREEK_KEYS:
            try:
                v = float(greeks.get(k))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                buckets[k].append(v)

    out: Dict[str, float | str] = {k: _avg(v) for k, v in buckets.items()}
    out["source"] = "polygon"
    return out


# -----------------------------------------------------------------------------
# Polygon aggs: chunked fetch for reliability
# -----------------------------------------------------------------------------
def _to_datestr(x: str | datetime) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, datetime):
        # daily endpoint accepts YYYY-MM-DD; normalize to UTC date
        return x.astimezone(timezone.utc).date().isoformat()
    raise TypeError(f"Unsupported date type: {type(x)}")


def fetch_polygon_aggs_chunked(
    ticker: str,
    start: str | datetime,
    end: str | datetime,
    *,
    multiplier: int = 1,
    timespan: str = "day",
    adjusted: bool = True,
    sort: str = "asc",
    limit: int = 5000,
    chunk_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}
    in windows of `chunk_days` to avoid server timeouts.

    Returns: flat list of Polygon bar dicts (each includes 't','o','h','l','c','v', ...).
    """
    s = _to_datestr(start)
    e = _to_datestr(end)
    cd = int(POLYGON_CHUNK_DAYS if chunk_days is None else chunk_days)

    out: List[Dict[str, Any]] = []
    s_dt = datetime.fromisoformat(s).replace(tzinfo=None)
    e_dt = datetime.fromisoformat(e).replace(tzinfo=None)
    cur = s_dt

    while cur <= e_dt:
        win_end = min(e_dt, cur + timedelta(days=cd - 1))
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{cur.date().isoformat()}/{win_end.date().isoformat()}"
        )
        params = {"adjusted": str(adjusted).lower(), "sort": sort, "limit": limit}
        data = safe_fetch_polygon_data(url, params=params, ticker=ticker)
        results = data.get("results") if isinstance(data, dict) else None
        if results:
            out.extend(results)
        else:
            logger.warning("No results for %s window %s -> %s", ticker, cur.date(), win_end.date())
        cur = win_end + timedelta(days=1)

    out.sort(key=lambda r: r.get("t", 0))
    return out


__all__ = [
    "rate_limited",
    "safe_fetch_polygon_data",
    "fetch_option_greeks",
    "fetch_polygon_aggs_chunked",
]
