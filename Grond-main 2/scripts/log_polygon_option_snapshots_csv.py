#!/usr/bin/env python3
"""
Log Polygon option snapshots to CSV (near-ATM, nearest expiries).
- Pulls per-underlying snapshots from Polygon's v3 Options Snapshot endpoint.
- Filters to contracts near-ATM by % moneyness and keeps N nearest expiries.
- Writes a timestamped CSV per run under: {OUT_DIR}/{SYMBOL}/{YYYY-MM-DD}/snapshot_{ISO}.csv

Usage (env + CLI):
  POLYGON_API_KEY=... python scripts/log_polygon_option_snapshots_csv.py \
      --out polygon_live_csv --poll 15 --atm 0.05 --exp 4

Notes:
- No historical OPRA; this logs *live snapshots* (whatever Polygon exposes off-hours will be whatever the API returns).
- Resilient to missing Greeks/IV; it writes empty fields rather than exploding.
- Tickers: tries to import config.TICKERS, else falls back to a sensible default mega-cap set.

Schema (first 24 columns):
  symbol, underlying_price, contract_symbol, contract_type, strike, expiration_date, exercise_style,
  bid, ask, mid, last_price, last_size, implied_volatility,
  delta, gamma, theta, vega, rho,
  open_interest, volume,
  updated_at, source, underlying_change_pct, moneyness_abs, distance_to_expiry_days

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

API_BASE = "https://api.polygon.io"

# ---------------------------- utils ----------------------------

def robust_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "na", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def day_str_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def http_get(url: str, params: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        return {}

# ------------------------- data sources ------------------------

def load_tickers() -> List[str]:
    """Prefer repo config.TICKERS; otherwise fallback set."""
    try:
        # If your repo already has Grond-main/config.py with TICKERS, this picks it up.
        from config import TICKERS  # type: ignore
        ts = [t.strip().upper() for t in TICKERS if t and isinstance(t, str)]
        if ts:
            return ts
    except Exception:
        pass
    # Fallback mega-caps (covers common testing)
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "TSLA"]

def fetch_underlying_price(symbol: str, api_key: str) -> Optional[float]:
    """
    Equity last trade price. We prefer /v2/last/trade/{ticker}.
    """
    url = f"{API_BASE}/v2/last/trade/{symbol}"
    data = http_get(url, {"apiKey": api_key})
    p = (
        data.get("results", {}).get("p")  # price
        if isinstance(data.get("results"), dict)
        else None
    )
    return robust_float(p)

def fetch_option_snapshots(symbol: str, api_key: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Polygon v3 snapshot for all options on an underlying.
    Endpoint variant used: /v3/snapshot/options/{underlying}
    """
    url = f"{API_BASE}/v3/snapshot/options/{symbol}"
    out: List[Dict[str, Any]] = []
    params = {"limit": min(limit, 1000), "apiKey": api_key}
    data = http_get(url, params)
    results = data.get("results", [])
    if isinstance(results, list):
        out.extend(results)
    return out

# ------------------------- filtering ---------------------------

@dataclass
class ParsedContract:
    underlying: str
    underlying_price: Optional[float]
    contract_symbol: str
    contract_type: str  # "call"/"put"
    strike: Optional[float]
    expiration_date: Optional[str]
    exercise_style: Optional[str]
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    last_price: Optional[float]
    last_size: Optional[float]
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    open_interest: Optional[float]
    volume: Optional[float]
    updated_at: Optional[str]
    source: str
    underlying_change_pct: Optional[float]
    moneyness_abs: Optional[float]
    dte_days: Optional[float]

def parse_snapshot_row(row: Dict[str, Any], und_symbol: str, und_price: Optional[float]) -> Optional[ParsedContract]:
    try:
        details = row.get("details", {}) if isinstance(row.get("details"), dict) else {}
        last_quote = row.get("last_quote", {}) if isinstance(row.get("last_quote"), dict) else {}
        day = row.get("day", {}) if isinstance(row.get("day"), dict) else {}
        greeks = row.get("greeks", {}) if isinstance(row.get("greeks"), dict) else {}

        contract_symbol = details.get("ticker") or details.get("symbol") or ""
        if not contract_symbol:
            return None

        # Contract meta
        contract_type = (details.get("contract_type") or "").lower() or ("call" if "C" in contract_symbol else "put")
        strike = robust_float(details.get("strike_price") or details.get("strike"))
        expiration_date = details.get("expiration_date") or details.get("expiration")  # YYYY-MM-DD
        exercise_style = details.get("exercise_style")

        # Quotes / prices
        bid = robust_float(last_quote.get("bid") or last_quote.get("p"))  # p sometimes bid
        ask = robust_float(last_quote.get("ask") or last_quote.get("P"))  # P sometimes ask
        mid = None
        if bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2.0

        last_trade = row.get("last_trade", {})
        last_price = robust_float(last_trade.get("price") or last_quote.get("bid") or last_quote.get("ask"))
        last_size = robust_float(last_trade.get("size"))

        # Activity
        open_interest = robust_float(day.get("open_interest"))
        volume = robust_float(day.get("volume"))

        # Greeks / IV (may be missing or null)
        iv = robust_float(row.get("implied_volatility") or greeks.get("iv"))
        delta = robust_float(greeks.get("delta"))
        gamma = robust_float(greeks.get("gamma"))
        theta = robust_float(greeks.get("theta"))
        vega = robust_float(greeks.get("vega"))
        rho = robust_float(greeks.get("rho"))

        # Updated time
        ts = row.get("updated") or row.get("updated_at")
        if ts and isinstance(ts, (int, float)):
            updated_at = datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).isoformat()
        else:
            updated_at = ts if isinstance(ts, str) else None

        # Derived
        up = robust_float(und_price)
        mny = None
        if up is not None and strike is not None and up > 0:
            mny = abs(strike - up) / up

        # Distance to expiry (approx)
        dte = None
        if expiration_date:
            try:
                dt = datetime.strptime(expiration_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dte = (dt - datetime.now(timezone.utc)).total_seconds() / 86400.0
            except Exception:
                pass

        return ParsedContract(
            underlying=und_symbol,
            underlying_price=up,
            contract_symbol=contract_symbol,
            contract_type=contract_type,
            strike=strike,
            expiration_date=expiration_date,
            exercise_style=exercise_style,
            bid=bid,
            ask=ask,
            mid=mid,
            last_price=last_price,
            last_size=last_size,
            iv=iv,
            delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
            open_interest=open_interest, volume=volume,
            updated_at=updated_at,
            source="polygon",
            underlying_change_pct=None,  # set by caller if desired
            moneyness_abs=mny,
            dte_days=dte,
        )
    except Exception:
        return None

def select_near_atm_near_expiry(rows: List[ParsedContract], atm_pct: float, max_expiries: int) -> List[ParsedContract]:
    """Keep contracts with |moneyness| <= atm_pct, across the nearest distinct expiries (max_expiries)."""
    # Filter to near-ATM
    f = [r for r in rows if r.moneyness_abs is not None and r.moneyness_abs <= atm_pct]
    if not f:
        return []
    # Group by expiry, sort by expiry ascending by DTE (closest first)
    f.sort(key=lambda r: (r.dte_days if r.dte_days is not None else 1e9, r.moneyness_abs))
    # Keep up to N distinct expiries
    out: List[ParsedContract] = []
    seen_exp: List[str] = []
    for r in f:
        exp = r.expiration_date or "UNKNOWN"
        if exp not in seen_exp:
            seen_exp.append(exp)
        if exp in seen_exp[:max_expiries]:
            out.append(r)
        if len(seen_exp) >= max_expiries and len(out) > 5000:
            break
    return out

# --------------------------- writer ----------------------------

CSV_HEADER: List[str] = [
    "symbol", "underlying_price", "contract_symbol", "contract_type", "strike", "expiration_date", "exercise_style",
    "bid", "ask", "mid", "last_price", "last_size", "implied_volatility",
    "delta", "gamma", "theta", "vega", "rho",
    "open_interest", "volume",
    "updated_at", "source", "underlying_change_pct", "moneyness_abs", "distance_to_expiry_days",
]

def row_to_list(p: ParsedContract) -> List[Any]:
    return [
        p.underlying, p.underlying_price, p.contract_symbol, p.contract_type, p.strike, p.expiration_date, p.exercise_style,
        p.bid, p.ask, p.mid, p.last_price, p.last_size, p.iv,
        p.delta, p.gamma, p.theta, p.vega, p.rho,
        p.open_interest, p.volume,
        p.updated_at, p.source, p.underlying_change_pct, p.moneyness_abs, p.dte_days,
    ]

def write_snapshot_csv(out_dir: str, symbol: str, rows: List[ParsedContract]) -> str:
    day_dir = os.path.join(out_dir, symbol, day_str_utc())
    ensure_dir(day_dir)
    path = os.path.join(day_dir, f"snapshot_{now_utc_iso()}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow(row_to_list(r))
    return path

# --------------------------- main ------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Log Polygon option snapshots to CSV.")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output root directory.")
    ap.add_argument("--poll", dest="poll_seconds", type=int, default=15, help="Polling seconds (ignored here; handled by workflow loop).")
    ap.add_argument("--atm", dest="near_atm_pct", type=float, default=0.05, help="Near-ATM band as fraction (0.05 = 5%%).")
    ap.add_argument("--exp", dest="max_expiries", type=int, default=4, help="Nearest expiries to keep.")
    args = ap.parse_args()

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY is not set.", file=sys.stderr)
        return 2

    ensure_dir(args.out_dir)

    tickers = load_tickers()
    if not tickers:
        print("ERROR: no tickers.", file=sys.stderr)
        return 3

    for sym in tickers:
        try:
            und_price = fetch_underlying_price(sym, api_key)
        except Exception as e:
            print(f"WARN: failed underlying price for {sym}: {e}", file=sys.stderr)
            und_price = None

        try:
            snaps = fetch_option_snapshots(sym, api_key, limit=1000)
        except Exception as e:
            print(f"WARN: snapshot fetch failed for {sym}: {e}", file=sys.stderr)
            snaps = []

        parsed: List[ParsedContract] = []
        for row in snaps:
            p = parse_snapshot_row(row, sym, und_price)
            if p:
                parsed.append(p)

        selected = select_near_atm_near_expiry(parsed, args.near_atm_pct, args.max_expiries)
        if not selected:
            print(f"INFO: no near-ATM rows for {sym} (und_price={und_price})")
            continue

        path = write_snapshot_csv(args.out_dir, sym, selected)
        print(f"OK: wrote {len(selected)} rows for {sym} -> {path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
