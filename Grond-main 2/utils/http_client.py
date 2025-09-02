"""HTTP client helpers with metrics, retries, and Polygon utilities.

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

import os
import time
import math
import random
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, cast
from urllib.parse import urlparse

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
    return "/" + "/".join(parts[:4])

def _sleep_backoff(attempt: int) -> None:
    base = HTTP_BACKOFF_BASE * (2 ** max(attempt - 1, 0))
    delay = min(base * (0.7 + 0.6 * random.random()), HTTP_BACKOFF_MAX)
    time.sleep(delay)

# -----------------------------------------------------------------------------
# Decorator: rate_limited
# -----------------------------------------------------------------------------

def rate_limited(func: Callable[..., T]) -> Callable[..., T]:
    """Retry wrapper with backoff for transient HTTP failures (incl. 429)."""
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exc: Optional[Exception] = None
        for attempt in range(1, HTTP_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except requests.HTTPError as e:
                last_exc = e
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    url = kwargs.get("url") or (args[0] if args else "")
                    service = _service_from_url(str(url))
                    endpoint = _endpoint_from_url(str(url))
                    REST_429.labels(service, endpoint).inc()
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
    """Perform an HTTP request, emitting labeled Prometheus metrics."""
    service = _service_from_url(url)
    endpoint = _endpoint_from_url(url)
    method_up = method.upper()

    start = time.monotonic()
    try:
        resp = requests.request(
            method_up,
            url,
            params=params,
            headers=headers,
            timeout=timeout or HTTP_TIMEOUT,
        )
        return resp
    finally:
        elapsed = max(time.monotonic() - start, 0.0)
        REST_LATENCY.labels(service, endpoint, method_up).observe(elapsed)

@rate_limited
def safe_fetch_polygon_data(url: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    """GET JSON with retries, metrics, and error handling for Polygon endpoints."""
    if "apiKey=" not in url and POLYGON_API_KEY:
        sep = "&" if ("?" in url) else "?"
        url = f"{url}{sep}apiKey={POLYGON_API_KEY}"

    service = _service_from_url(url)
    endpoint = _endpoint_from_url(url)
    method = "GET"

    resp: requests.Response
    try:
        resp = _request_with_metrics(method, url)
        status = resp.status_code
        REST_CALLS.labels(service, endpoint, method, str(status)).inc()
        if status == 429:
            REST_429.labels(service, endpoint).inc()
        resp.raise_for_status()
        data = cast(Dict[str, Any], resp.json())
        return data
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", "ERR")
        REST_CALLS.labels(service, endpoint, method, str(status)).inc()
        logger.info("Request error for %s: %s", ticker or endpoint, f"{status} {str(e)}")
        raise
    except (requests.ConnectionError, requests.Timeout) as e:
        REST_CALLS.labels(service, endpoint, method, "NETWORK").inc()
        logger.info("Network error for %s: %s", ticker or endpoint, str(e))
        raise
    except Exception as e:
        REST_CALLS.labels(service, endpoint, method, "EXC").inc()
        logger.info("Unexpected error for %s: %s", ticker or endpoint, str(e))
        raise

# -----------------------------------------------------------------------------
# Greeks fetcher
# -----------------------------------------------------------------------------

_GREK_KEYS = (
    "delta", "gamma", "theta", "vega", "rho",
    "vanna", "vomma", "charm", "veta", "speed", "zomma", "color",
    "implied_volatility",
)

def _zeros_greeks(source: str) -> Dict[str, float | str]:
    g: Dict[str, float | str] = {k: 0.0 for k in _GREK_KEYS}
    g["source"] = source
    return g

def _avg(nums: list[float]) -> float:
    if not nums:
        return 0.0
    return sum(nums) / float(len(nums))

def fetch_option_greeks(ticker: str) -> Dict[str, float | str]:
    """
    Fetch an aggregated view of option Greeks for the underlying from Polygon.
    We average over available contracts in the snapshot (crude but stable).
    Fallback returns zeros and 'source'='fallback'.
    """
    if not POLYGON_API_KEY:
        return _zeros_greeks("fallback")

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    try:
        data = safe_fetch_polygon_data(url, ticker=ticker)
        results = data.get("results") or []
        if not isinstance(results, list) or not results:
            return _zeros_greeks("fallback")

        buckets: Dict[str, list[float]] = {k: [] for k in _GREK_KEYS}
        for opt in results:
            greeks = opt.get("greeks") or {}
            for k in _GREK_KEYS:
                val = greeks.get(k)
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    buckets[k].append(fv)

        out: Dict[str, float | str] = {k: _avg(buckets[k]) for k in _GREK_KEYS}
        out["source"] = "polygon"
        return out
    except Exception:
        return _zeros_greeks("fallback")
