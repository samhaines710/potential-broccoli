"""
HTTP client helpers with metrics, retries, and Polygon utilities.

Key fixes:
- Always provide labels to Prometheus metrics (no missing label values).
- Record latency via REST_LATENCY with (service, endpoint, method) labels.
- Count 429s via REST_429 and back off with jittered retries.
- Expose safe_fetch_polygon_data() and fetch_option_greeks().
- Provide a rate_limited decorator compatible with existing imports.

Environment variables:
- HTTP_TIMEOUT_SECONDS (default: 10)
- HTTP_MAX_RETRIES (default: 3)
- HTTP_BACKOFF_BASE (seconds, default: 0.6)
- HTTP_BACKOFF_MAX (seconds, default: 5.0)
- POLYGON_API_KEY (required for Polygon endpoints)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, cast
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

import requests

from utils.logging_utils import (
    REST_CALLS,
    REST_LATENCY,
    REST_429,
    get_logger,
)

T = TypeVar("T")

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_BACKOFF_BASE = float(os.getenv("HTTP_BACKOFF_BASE", "0.6"))
HTTP_BACKOFF_MAX = float(os.getenv("HTTP_BACKOFF_MAX", "5.0"))
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

_DEFAULT_HEADERS = {
    "User-Agent": "grond-http/1.0 (+training; metrics; retries)",
    "Accept": "application/json",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _service_from_url(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if "api.polygon.io" in netloc:
        return "polygon"
    return netloc or "unknown"


def _endpoint_from_url(url: str) -> str:
    p = urlparse(url)
    parts = [seg for seg in p.path.split("/") if seg]
    if not parts:
        return "/"
    # keep first few segments for cardinality control
    return "/" + "/".join(parts[:4])


def _sleep_backoff(attempt: int) -> None:
    """
    Exponential backoff with jitter, capped by HTTP_BACKOFF_MAX.
    attempt starts at 1.
    """
    base = HTTP_BACKOFF_BASE * (2 ** max(attempt - 1, 0))
    delay = min(base * (0.7 + 0.6 * random.random()), HTTP_BACKOFF_MAX)
    time.sleep(delay)


def _append_polygon_key(url: str) -> str:
    """
    Ensure apiKey query param is present for Polygon endpoints.
    Does not overwrite if key already set.
    """
    if "api.polygon.io" not in url:
        return url
    if not POLYGON_API_KEY:
        # Let caller fail explicitly if required; do not mutate URL.
        return url
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "apiKey" not in q:
        q["apiKey"] = POLYGON_API_KEY
    return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))


# -----------------------------------------------------------------------------
# Decorator: rate_limited
# -----------------------------------------------------------------------------

def rate_limited(func: Callable[..., T]) -> Callable[..., T]:
    """
    Retry wrapper with backoff for transient HTTP failures (incl. 429).

    Compatible with existing imports and usage:
      - Retries on requests.ConnectionError / Timeout
      - Retries on HTTPError for 429 and 5xx
      - Increments REST_429 metric on 429
    """
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exc: Optional[Exception] = None
        # infer URL (positional or kw) for metrics labeling on 429
        def _infer_url() -> str:
            if "url" in kwargs:
                return str(kwargs["url"])
            if args:
                # many callers pass (method, url, ...) or (url, ...) as first arg
                return str(args[0]) if isinstance(args[0], str) else str(args[1] if len(args) > 1 else "")
            return ""
        for attempt in range(1, HTTP_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except requests.HTTPError as e:
                last_exc = e
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    url = _infer_url()
                    service = _service_from_url(url)
                    endpoint = _endpoint_from_url(url)
                    REST_429.labels(service, endpoint).inc()
                # do not retry 4xx except 429
                if status and status not in (429,) and status < 500:
                    break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
            if attempt < HTTP_MAX_RETRIES:
                _sleep_backoff(attempt + 1)
        assert last_exc is not None
        raise last_exc
    return wrapper


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
) -> requests.Response:
    """
    Perform an HTTP request, emitting labeled Prometheus metrics.

    Emits:
      REST_LATENCY.labels(service, endpoint, method).observe(seconds)
      REST_CALLS.labels(service, endpoint, method, status).inc()
      REST_429.labels(service, endpoint).inc() (in caller on 429)
    """
    service = _service_from_url(url)
    endpoint = _endpoint_from_url(url)
    method_up = method.upper()

    start = time.monotonic()
    resp: Optional[requests.Response] = None
    try:
        # Merge default headers with caller headers
        hdrs = dict(_DEFAULT_HEADERS)
        if headers:
            hdrs.update(headers)
        resp = requests.request(
            method_up,
            url,
            params=params,
            headers=hdrs,
            timeout=timeout or HTTP_TIMEOUT,
        )
        return resp
    finally:
        elapsed = max(time.monotonic() - start, 0.0)
        REST_LATENCY.labels(service, endpoint, method_up).observe(elapsed)
        # Record status if we have a response, else 'NO_RESP'
        status_label = str(resp.status_code) if resp is not None else "NO_RESP"
        REST_CALLS.labels(service, endpoint, method_up, status_label).inc()


@rate_limited
def safe_fetch_polygon_data(url: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    """
    GET JSON with retries, metrics, and error handling for Polygon endpoints.
    Automatically appends POLYGON_API_KEY if missing.
    """
    url = _append_polygon_key(url)
    service = _service_from_url(url)
    endpoint = _endpoint_from_url(url)
    method = "GET"

    try:
        resp = _request_with_metrics(method, url)
        status = resp.status_code
        if status == 429:
            REST_429.labels(service, endpoint).inc()
        resp.raise_for_status()
        return cast(Dict[str, Any], resp.json())
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", "ERR")
        REST_CALLS.labels(service, endpoint, method, str(status)).inc()
        logger.warning("HTTP %s for %s (%s): %s", status, ticker or endpoint, url, e)
        raise
    except (requests.ConnectionError, requests.Timeout) as e:
        REST_CALLS.labels(service, endpoint, method, "NETWORK").inc()
        logger.warning("Network error for %s (%s): %s", ticker or endpoint, url, e)
        raise
    except json.JSONDecodeError as e:
        REST_CALLS.labels(service, endpoint, method, "BAD_JSON").inc()
        logger.warning("Bad JSON for %s (%s): %s", ticker or endpoint, url, e)
        raise
    except Exception as e:
        REST_CALLS.labels(service, endpoint, method, "EXC").inc()
        logger.exception("Unexpected error for %s (%s): %s", ticker or endpoint, url, e)
        raise


# -----------------------------------------------------------------------------
# Greeks fetcher (Polygon snapshot)
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

def _avg(nums: list[float]) -> float:
    return (sum(nums) / float(len(nums))) if nums else 0.0


def fetch_option_greeks(ticker: str) -> Dict[str, float | str]:
    """
    Fetch a crude aggregate of option Greeks for the underlying via Polygon.
    Averages available contracts in the snapshot response; falls back to zeros.
    """
    if not POLYGON_API_KEY:
        return _zeros_greeks("fallback")

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    try:
        data = safe_fetch_polygon_data(url, ticker=ticker)
        results = data.get("results") or []
        if not isinstance(results, list) or not results:
            return _zeros_greeks("fallback")

        buckets: Dict[str, list[float]] = {k: [] for k in _GREEK_KEYS}
        for opt in results:
            greeks = opt.get("greeks") or {}
            for k in _GREEK_KEYS:
                val = greeks.get(k)
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    buckets[k].append(fv)

        out: Dict[str, float | str] = {k: _avg(v) for k, v in buckets.items()}
        out["source"] = "polygon"
        return out
    except Exception as e:
        logger.warning("Greeks fetch fallback for %s due to: %s", ticker, e)
        return _zeros_greeks("fallback")


__all__ = [
    "safe_fetch_polygon_data",
    "fetch_option_greeks",
    "rate_limited",
]
