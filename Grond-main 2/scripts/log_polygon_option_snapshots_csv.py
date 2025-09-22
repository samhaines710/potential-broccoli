#!/usr/bin/env python3
"""
Log Polygon option snapshots to CSV (near-ATM, nearest expiries).

- Uses Polygon v3 Options **Underlying Snapshot** endpoint:
    GET /v3/snapshot/options/{UNDERLYING}?apiKey=...
  NOTE: This endpoint does **not** support 'limit', 'max', or pagination params.
        Passing them yields 400 "Field validation" errors.

- Filters to contracts near-ATM by % moneyness and keeps N nearest expiries.
- Writes a timestamped CSV per run under: {OUT_DIR}/{SYMBOL}/{YYYY-MM-DD}/snapshot_{ISO}.csv

Usage (env + CLI):

  POLYGON_API_KEY=... python scripts/log_polygon_option_snapshots_csv.py \
      --out polygon_live_csv --poll 15 --atm 0.05 --exp 4

The polling cadence is handled by your GitHub Actions loop; this script performs **one pass**.

Columns:
  symbol, underlying_price, contract_symbol, contract_type, strike, expiration_date, exercise_style,
  bid, ask, mid, last_price, last_size, implied_volatility, delta, gamma, theta, vega, rho,
  open_interest, volume, updated_at, source, moneyness_abs, distance_to_expiry_days
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# ----------------------- Env / Tickers -----------------------

API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
if not API_KEY:
    sys.stderr.write("FATAL: POLYGON_API_KEY missing in environment.\n")
    sys.exit(2)

BASE = "https://api.polygon.io"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "NFLX", "GOOG", "IBIT"]
try:
    try:
        from config import TICKERS as REPO_TICKERS  # type: ignore
    except Exception:
        from Grond_main_2.config import TICKERS as REPO_TICKERS  # type: ignore  # noqa: N816
    TICKERS: List[str] = [t for t in REPO_TICKERS if isinstance(t, str) and t.strip()]
    if not TICKERS:
        TICKERS = DEFAULT_TICKERS
except Exception:
    TICKERS = DEFAULT_TICKERS


# ----------------------- HTTP -----------------------

def fetch_underlying_snapshot(
    underlying: str,
    *,
    retries: int = 3,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    GET /v3/snapshot/options/{UNDERLYING}?apiKey=...

    Valid params: apiKey only (as of Polygon v3 snapshots).
    Any 'limit'/'max'/pagination params cause 400 "Field validation" errors.
    """
    url = f"{BASE}/v3/snapshot/options/{underlying}"
    params = {"apiKey": API_KEY}

    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            # Rate limiting
            if r.status_code == 429:
                backoff = min(2 * attempt, 10)
                time.sleep(backoff)
                continue

            # Hard failures (400+)
            if r.status_code >= 400:
                # Try to surface server message (helps debugging)
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text
                sys.stderr.write(f"WARN: snapshot fetch failed for {underlying}: {r.status_code} {r.reason}: {msg}\n")
                # For 4xx (except 429 handled above), don't retry
                if 400 <= r.status_code < 500:
                    return None
                # For 5xx, retry
                time.sleep(min(2 * attempt, 10))
                continue

            return r.json()

        except Exception as e:
            sys.stderr.write(f"WARN: network error for {underlying}: {e}\n")
            time.sleep(min(2 * attempt, 10))

    return None


# ----------------------- Helpers -----------------------

def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    try:
        if bid is None or ask is None:
            return None
        if bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0
    except Exception:
        return None


def _parse_iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def _days_between(d1: Optional[date], d2: Optional[date]) -> Optional[float]:
    if not d1 or not d2:
        return None
    return float((d2 - d1).days)


# ----------------------- Row model -----------------------

@dataclass
class SnapshotRow:
    symbol: str
    underlying_price: Optional[float]
    contract_symbol: str
    contract_type: str  # "call" | "put"
    strike: Optional[float]
    expiration_date: Optional[str]
    exercise_style: Optional[str]
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    last_price: Optional[float]
    last_size: Optional[float]
    implied_volatility: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    open_interest: Optional[float]
    volume: Optional[float]
    updated_at: Optional[str]
    source: str
    moneyness_abs: Optional[float]
    distance_to_expiry_days: Optional[float]

    def to_list(self) -> List[Any]:
        return [
            self.symbol,
            self.underlying_price,
            self.contract_symbol,
            self.contract_type,
            self.strike,
            self.expiration_date,
            self.exercise_style,
            self.bid,
            self.ask,
            self.mid,
            self.last_price,
            self.last_size,
            self.implied_volatility,
            self.delta,
            self.gamma,
            self.theta,
            self.vega,
            self.rho,
            self.open_interest,
            self.volume,
            self.updated_at,
            self.source,
            self.moneyness_abs,
            self.distance_to_expiry_days,
        ]


HEADER = [
    "symbol",
    "underlying_price",
    "contract_symbol",
    "contract_type",
    "strike",
    "expiration_date",
    "exercise_style",
    "bid",
    "ask",
    "mid",
    "last_price",
    "last_size",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "open_interest",
    "volume",
    "updated_at",
    "source",
    "moneyness_abs",
    "distance_to_expiry_days",
]


# ----------------------- Transform -----------------------

def flatten_snapshot_json(symbol: str, data: Dict[str, Any]) -> Tuple[Optional[float], List[SnapshotRow]]:
    """
    Convert Polygon snapshot JSON -> list[SnapshotRow].
    """
    results = data.get("results") or []
    rows: List[SnapshotRow] = []

    # Underlying price: many results include 'underlying_asset.price'. Take a robust average.
    u_prices: List[float] = []
    for r in results:
        p = _safe_get(r, "underlying_asset", "price")
        if isinstance(p, (int, float)):
            u_prices.append(float(p))
    underlying_price: Optional[float] = float(sum(u_prices) / len(u_prices)) if u_prices else None

    today = datetime.now(timezone.utc).date()

    for r in results:
        opt = r.get("option") or {}
        nbbo = r.get("nbbo") or {}
        last = r.get("last_trade") or {}
        greeks = r.get("greeks") or {}

        bid = _safe_get(nbbo, "bid", "price")
        ask = _safe_get(nbbo, "ask", "price")
        last_price = last.get("price")
        last_size = last.get("size")

        contract_symbol = opt.get("symbol") or r.get("symbol") or ""
        contract_type = (opt.get("type") or "").lower()
        strike = opt.get("strike_price") or opt.get("strike")
        expiry = opt.get("expiration_date") or opt.get("exp_date")
        exercise_style = opt.get("exercise_style")

        iv = greeks.get("implied_volatility")
        delta = greeks.get("delta"); gamma = greeks.get("gamma")
        theta = greeks.get("theta"); vega = greeks.get("vega"); rho = greeks.get("rho")

        open_interest = r.get("open_interest")
        volume = r.get("volume")
        updated_at = r.get("updated") or r.get("updated_at")

        # Deriveds
        try:
            u = underlying_price
            m_abs = abs(float(strike) / float(u) - 1.0) if (strike is not None and u is not None) else None
        except Exception:
            m_abs = None
        dte = _days_between(today, _parse_iso_to_date(expiry))

        rows.append(
            SnapshotRow(
                symbol=symbol,
                underlying_price=float(underlying_price) if underlying_price is not None else None,
                contract_symbol=contract_symbol,
                contract_type=contract_type,
                strike=float(strike) if strike is not None else None,
                expiration_date=str(expiry) if expiry else None,
                exercise_style=str(exercise_style) if exercise_style else None,
                bid=float(bid) if bid is not None else None,
                ask=float(ask) if ask is not None else None,
                mid=_mid(bid, ask),
                last_price=float(last_price) if last_price is not None else None,
                last_size=float(last_size) if last_size is not None else None,
                implied_volatility=float(iv) if iv is not None else None,
                delta=float(delta) if delta is not None else None,
                gamma=float(gamma) if gamma is not None else None,
                theta=float(theta) if theta is not None else None,
                vega=float(vega) if vega is not None else None,
                rho=float(rho) if rho is not None else None,
                open_interest=float(open_interest) if open_interest is not None else None,
                volume=float(volume) if volume is not None else None,
                updated_at=str(updated_at) if updated_at else None,
                source="polygon",
                moneyness_abs=m_abs,
                distance_to_expiry_days=dte,
            )
        )

    return underlying_price, rows


def filter_near_atm_and_top_expiries(
    rows: List[SnapshotRow],
    *,
    near_atm_pct: float,
    max_expiries: int,
) -> List[SnapshotRow]:
    """
    Keep rows within abs(moneyness) <= near_atm_pct and restrict to N nearest expiries (by |DTE|).
    """
    if not rows:
        return []
    filt = [r for r in rows if (r.moneyness_abs is not None and r.moneyness_abs <= near_atm_pct)]
    if not filt:
        return []

    # group by expiry
    by_exp: Dict[str, List[SnapshotRow]] = {}
    for r in filt:
        key = r.expiration_date or "NA"
        by_exp.setdefault(key, []).append(r)

    # nearest expiries by abs(DTE)
    def dte_key(key: str) -> float:
        ds = [rr.distance_to_expiry_days for rr in by_exp[key] if rr.distance_to_expiry_days is not None]
        return abs(min(ds) if ds else 1e9)

    top_keys = sorted(by_exp.keys(), key=dte_key)[: max(1, max_expiries)]
    out: List[SnapshotRow] = []
    for k in top_keys:
        out.extend(by_exp[k])
    return out


# ----------------------- IO -----------------------

def write_csv(out_dir: str, symbol: str, rows: List[SnapshotRow]) -> Optional[str]:
    """
    Write rows to {out_dir}/{symbol}/{YYYY-MM-DD}/snapshot_{ISO}.csv
    Returns path or None.
    """
    ts = datetime.now(timezone.utc)
    dpart = ts.strftime("%Y-%m-%d")
    isopart = ts.strftime("%Y%m%dT%H%M%SZ")
    dest_dir = os.path.join(out_dir, symbol, dpart)
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"snapshot_{isopart}.csv")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(r.to_list())
    return path


# ----------------------- One-pass -----------------------

def run_once(
    *,
    out_dir: str,
    symbols: Iterable[str],
    near_atm_pct: float,
    max_expiries: int,
) -> List[str]:
    written: List[str] = []
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue

        data = fetch_underlying_snapshot(sym)
        if not data:
            sys.stdout.write(f"WARN: no snapshot JSON for {sym}\n")
            p = write_csv(out_dir, sym, [])
            if p:
                written.append(p)
            continue

        try:
            u_px, rows = flatten_snapshot_json(sym, data)
        except Exception as e:
            sys.stdout.write(f"WARN: failed to parse snapshot for {sym}: {e}\n")
            p = write_csv(out_dir, sym, [])
            if p:
                written.append(p)
            continue

        if not rows:
            sys.stdout.write(f"INFO: zero rows in snapshot for {sym}\n")
            p = write_csv(out_dir, sym, [])
            if p:
                written.append(p)
            continue

        filtered = filter_near_atm_and_top_expiries(rows, near_atm_pct=near_atm_pct, max_expiries=max_expiries)
        if not filtered:
            sys.stdout.write(f"INFO: no near-ATM rows for {sym} (und_price={u_px})\n")
            p = write_csv(out_dir, sym, [])
            if p:
                written.append(p)
            continue

        p = write_csv(out_dir, sym, filtered)
        if p:
            written.append(p)
            sys.stdout.write(f"INFO: wrote {len(filtered)} rows for {sym} -> {p}\n")

    return written


# ----------------------- CLI -----------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Log Polygon option snapshots to CSV (near-ATM, N expiries).")
    ap.add_argument("--out", required=True, help="Output directory (created if missing).")
    ap.add_argument("--poll", type=int, default=15, help="Poll interval seconds (GH workflow uses this in a loop).")
    ap.add_argument("--atm", type=float, default=0.05, help="Near-ATM band: abs(strike/underlying-1) <= atm.")
    ap.add_argument("--exp", type=int, default=4, help="Keep N nearest expiries.")
    ap.add_argument(
        "--tickers",
        type=str,
        default=",".join(TICKERS),
        help="Comma-separated list of underlyings; defaults to repo config or a mega-cap set.",
    )
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    out_dir = ns.out
    near_atm_pct = float(ns.atm)
    max_expiries = int(ns.exp)
    tickers = [t.strip().upper() for t in ns.tickers.split(",") if t.strip()]
    os.makedirs(out_dir, exist_ok=True)

    written = run_once(
        out_dir=out_dir,
        symbols=tickers,
        near_atm_pct=near_atm_pct,
        max_expiries=max_expiries,
    )

    total_rows = 0
    for p in written:
        try:
            with open(p, "r") as f:
                n = sum(1 for _ in f) - 1  # minus header
                total_rows += max(0, n)
        except Exception:
            pass
    sys.stdout.write(f"SUMMARY: files={len(written)} total_rows={total_rows}\n")


if __name__ == "__main__":
    main()