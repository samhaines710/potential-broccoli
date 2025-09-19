#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Config
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_GREEKS_URL = os.getenv(
    "POLYGON_GREEKS_URL",
    "https://api.polygon.io/v3/snapshot/options/{underlying}"
)
BUCKET_SECONDS = int(os.getenv("GREEKS_BUCKET_SECONDS", "300"))  # 5-minute buckets

# ── Black–Scholes helpers (for backfilling specific greeks if IV present) ─────────
SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

@dataclass
class BSArgs:
    s: float      # underlying
    k: float      # strike
    t: float      # time to expiry in years
    r: float      # risk-free (annual)
    q: float      # dividend yield (annual)
    iv: float     # implied vol (annual)
    is_call: bool = True

def _d1_d2(a: BSArgs) -> Tuple[float, float]:
    if a.s <= 0 or a.k <= 0 or a.t <= 0 or a.iv <= 0:
        return float("nan"), float("nan")
    num = math.log(a.s / a.k) + (a.r - a.q + 0.5 * a.iv * a.iv) * a.t
    den = a.iv * math.sqrt(a.t)
    d1 = num / den
    d2 = d1 - a.iv * math.sqrt(a.t)
    return d1, d2

def bs_greeks(a: BSArgs) -> Dict[str, float]:
    d1, d2 = _d1_d2(a)
    if math.isnan(d1) or math.isnan(d2):
        return {}
    pdf_d1 = _norm_pdf(d1)
    # Delta
    if a.is_call:
        delta = math.exp(-a.q * a.t) * _norm_cdf(d1)
    else:
        delta = -math.exp(-a.q * a.t) * _norm_cdf(-d1)
    # Gamma
    gamma = (math.exp(-a.q * a.t) * pdf_d1) / (a.s * a.iv * math.sqrt(a.t))
    # Theta (per year)
    if a.is_call:
        theta = (- (a.s * math.exp(-a.q * a.t) * pdf_d1 * a.iv) / (2 * math.sqrt(a.t))
                 - a.r * a.k * math.exp(-a.r * a.t) * _norm_cdf(d2)
                 + a.q * a.s * math.exp(-a.q * a.t) * _norm_cdf(d1))
    else:
        theta = (- (a.s * math.exp(-a.q * a.t) * pdf_d1 * a.iv) / (2 * math.sqrt(a.t))
                 + a.r * a.k * math.exp(-a.r * a.t) * _norm_cdf(-d2)
                 - a.q * a.s * math.exp(-a.q * a.t) * _norm_cdf(-d1))
    # Vega
    vega = a.s * math.exp(-a.q * a.t) * pdf_d1 * math.sqrt(a.t)
    # Rho
    if a.is_call:
        rho = a.k * a.t * math.exp(-a.r * a.t) * _norm_cdf(d2)
    else:
        rho = -a.k * a.t * math.exp(-a.r * a.t) * _norm_cdf(-d2)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),  # annualized
        "vega": float(vega),
        "rho": float(rho),
    }

# ── Polygon fetch with per-bucket cache ───────────────────────────────────────────
_cache: Dict[Tuple[str, int], Dict[str, float]] = {}

def _bucket_ts(epoch_sec: int) -> int:
    return (epoch_sec // BUCKET_SECONDS) * BUCKET_SECONDS

def _pick_atm_contract(snap_json: dict, spot: float) -> Optional[dict]:
    """
    From Polygon 'options snapshot for underlying' JSON, pick a near-ATM CALL.
    Handles common layouts: results/options/contracts, greeks under 'greeks' or 'day'.
    """
    opts = None
    for key in ("results", "options", "contracts"):
        if key in snap_json and isinstance(snap_json[key], list):
            opts = snap_json[key]
            break
    if not opts:
        return None
    cands = []
    for o in opts:
        try:
            typ = (o.get("details", {}).get("contract_type")
                   or o.get("contract_type") or "").upper()
            if typ != "CALL":
                continue
            k = float(o.get("details", {}).get("strike_price")
                      or o.get("strike_price"))
            iv = (o.get("greeks", {}) or o.get("day", {}) or {}).get("iv") \
                 or (o.get("greeks", {}) or o.get("day", {}) or {}).get("implied_volatility")
            if iv is None:
                continue
            cands.append((abs(k - spot), o))
        except Exception:
            continue
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    return cands[0][1]

def fetch_greeks_polygon_asof(underlying: str, epoch_sec: int, spot: float,
                              r_annual: float = 0.04, q_annual: float = 0.0) -> Dict[str, float]:
    """
    Fetch Greeks for an underlying at a bucketed time.
    Returns real numbers; raises RuntimeError if cannot produce non-zero greeks.
    """
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY missing")

    bucket = _bucket_ts(epoch_sec)
    key = (underlying, bucket)
    if key in _cache:
        return _cache[key]

    url = POLYGON_GREEKS_URL.format(underlying=underlying)
    params = {"apiKey": POLYGON_API_KEY, "limit": 250}

    resp = requests.get(url, params=params, timeout=10)
    try:
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Polygon snapshot error for %s: %s", underlying, e)
        raise

    js = resp.json()
    opt = _pick_atm_contract(js, spot)
    if not opt:
        raise RuntimeError(f"No ATM call contract in snapshot for {underlying}")

    g = (opt.get("greeks") or opt.get("day") or {})
    delta = g.get("delta"); gamma = g.get("gamma"); theta = g.get("theta")
    vega  = g.get("vega");  rho   = g.get("rho")
    iv    = g.get("iv") or g.get("implied_volatility")

    # If some greeks missing, compute from Black–Scholes when IV & strike known
    if (delta is None or gamma is None or theta is None or vega is None or rho is None) and iv:
        try:
            strike = float(opt.get("details", {}).get("strike_price") or opt.get("strike_price"))
            T = 30.0 / 365.0  # proxy tenor if expiry not in payload
            bs = BSArgs(s=spot, k=strike, t=T, r=r_annual, q=q_annual, iv=float(iv), is_call=True)
            bsg = bs_greeks(bs)
            delta = delta if delta is not None else bsg.get("delta")
            gamma = gamma if gamma is not None else bsg.get("gamma")
            theta = theta if theta is not None else bsg.get("theta")
            vega  = vega  if vega  is not None else bsg.get("vega")
            rho   = rho   if rho   is not None else bsg.get("rho")
        except Exception as e:
            logger.warning("BS backfill failed for %s: %s", underlying, e)

    out = {
        "delta": float(delta) if delta is not None else float("nan"),
        "gamma": float(gamma) if gamma is not None else float("nan"),
        "theta": float(theta) if theta is not None else float("nan"),  # annualized
        "vega":  float(vega)  if vega  is not None else float("nan"),
        "rho":   float(rho)   if rho   is not None else float("nan"),
        "implied_volatility": float(iv) if iv is not None else float("nan"),
        "source": "polygon",
    }

    vals = [out[k] for k in ("delta","gamma","theta","vega","rho","implied_volatility")]
    if all((v == 0.0 or math.isnan(v)) for v in vals):
        raise RuntimeError(f"All Greeks zero/missing for {underlying}")

    _cache[key] = out
    time.sleep(0.05)  # polite pacing
    return out