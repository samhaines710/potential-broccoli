#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log live Polygon option snapshots to per-symbol daily CSVs (append-only),
loading the ticker universe from your repo's config.py (TICKERS) if available.

- Polls Polygon v3 snapshot endpoint for each underlying.
- Filters nearest N expiries and near-ATM contracts (|S-K|/S <= NEAR_ATM_PCT).
- Extracts NBBO (bid/ask), mid, and Polygon's Greeks/IV (delta/gamma/theta/vega/rho/iv).
- Appends rows to OUT_DIR/{SYMBOL}/{YYYY-MM-DD}.csv (stable schema).
- Skips contracts without a sane mid (no zeros or crossed markets).
- Dedupes by (ts_iso, expiry, type, strike) per batch.

Env / CLI:
  POLYGON_API_KEY  : required
  UNDERLYINGS      : fallback comma list if repo TICKERS not found and --symbols not given
  OUT_DIR          : default polygon_live_csv
  POLL_SECONDS     : default 15
  NEAR_ATM_PCT     : default 0.05
  MAX_EXPIRIES     : default 4
  LOG_ERRORS       : "1" to echo HTTP/parse errors

Schema:
  ts_iso, ingest_ts_iso, symbol_underlying, underlying_price, expiry, type, strike,
  bid, ask, mid, bid_size, ask_size, iv, delta, gamma, theta, vega, rho
"""

from __future__ import annotations
import argparse
import csv
import glob
import importlib.util
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

# Defaults (overridable via env/CLI)
DEF_OUT_DIR     = os.getenv("OUT_DIR", "polygon_live_csv")
DEF_POLL_SEC    = int(os.getenv("POLL_SECONDS", "15"))
DEF_ATM_PCT     = float(os.getenv("NEAR_ATM_PCT", "0.05"))
DEF_MAX_EXPS    = int(os.getenv("MAX_EXPIRIES", "4"))
DEF_LOG_ERRORS  = os.getenv("LOG_ERRORS", "0") == "1"
API_KEY         = os.getenv("POLYGON_API_KEY", "")

# ---------- helpers ----------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def day_str(dt: Optional[datetime] = None) -> str:
    d = dt or datetime.now(timezone.utc)
    return d.strftime("%Y-%m-%d")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def robust_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        f = float(x)
        return None if (f != f) else f  # NaN check
    except Exception:
        return None

def robust_int(x: Any) -> Optional[int]:
    try:
        if x is None: return None
        return int(x)
    except Exception:
        return None

def calc_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    try:
        if bid is None or ask is None: return None
        if bid <= 0 or ask <= 0: return None
        if ask < bid: return None
        return 0.5 * (bid + ask)
    except Exception:
        return None

# ---------- repo universe discovery ----------
def possible_repo_roots() -> List[str]:
    roots = [os.getcwd()]
    patterns = [
        "./*/config.py",
        "./*/*/config.py",
        "./potential-broccoli*/config.py",
        "./Grond*/config.py",
    ]
    for pat in patterns:
        for hit in glob.glob(pat):
            roots.append(os.path.dirname(hit))
    out, seen = [], set()
    for r in roots:
        rp = os.path.abspath(r)
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out

def import_config_tickerset(repo_root: str) -> Optional[List[str]]:
    cfg_path = os.path.join(repo_root, "config.py")
    if not os.path.isfile(cfg_path):
        return None
    spec = importlib.util.spec_from_file_location("repo_config", cfg_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        if DEF_LOG_ERRORS:
            print(f"[WARN] import config.py failed at {cfg_path}: {e}", file=sys.stderr)
        return None
    tickers = getattr(mod, "TICKERS", None)
    if not tickers:
        return None
    try:
        norm = [str(t).strip().upper() for t in tickers if str(t).strip()]
        norm = [t for t in norm if all(c.isalnum() or c in ('.','-','_') for c in t)]
        return sorted(set(norm))
    except Exception:
        return None

def load_universe(cli_symbols: Optional[str]) -> List[str]:
    for root in possible_repo_roots():
        tickers = import_config_tickerset(root)
        if tickers:
            print(f"[info] Using TICKERS from {root}/config.py: {len(tickers)} symbols")
            return tickers
    if cli_symbols:
        syms = sorted(set(s.strip().upper() for s in cli_symbols.split(",") if s.strip()))
        if syms:
            print(f"[info] Using --symbols override: {len(syms)} symbols")
            return syms
    env_syms = os.getenv("UNDERLYINGS", "")
    syms = sorted(set(s.strip().upper() for s in env_syms.split(",") if s.strip()))
    if syms:
        print(f"[info] Using UNDERLYINGS env: {len(syms)} symbols")
        return syms
    print("ERROR: No symbols found. Provide config.py TICKERS, --symbols, or UNDERLYINGS.", file=sys.stderr)
    sys.exit(1)

# ---------- polygon ----------
def polygon_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params); p["apiKey"] = API_KEY
    resp = requests.get(url, params=p, timeout=10)
    resp.raise_for_status()
    return resp.json()

def pick_underlying_price(js: Dict[str, Any]) -> Optional[float]:
    u = js.get("underlying_asset") or {}
    return robust_float(u.get("price"))

def extract_contract_rows(underlying: str, js: Dict[str, Any], near_atm_pct: float, max_expiries: int) -> List[Dict[str, Any]]:
    ingest_ts = now_utc_iso()
    raw = js.get("results") or js.get("options") or js.get("contracts") or []
    S = pick_underlying_price(js)
    norm: List[Dict[str, Any]] = []

    for o in raw:
        try:
            det  = o.get("details", {})
            typ  = (det.get("contract_type") or o.get("contract_type") or "").upper()
            if typ not in ("CALL", "PUT"):
                continue
            expiry = det.get("expiration_date") or o.get("expiration_date") or o.get("expiry")
            strike = robust_float(det.get("strike_price") or o.get("strike_price"))
            q     = o.get("quote") or {}
            bid   = robust_float(q.get("bid") if q else o.get("bid"))
            ask   = robust_float(q.get("ask") if q else o.get("ask"))
            bidz  = robust_int(q.get("bid_size") or o.get("bid_size"))
            askz  = robust_int(q.get("ask_size") or o.get("ask_size"))
            greeks = o.get("greeks") or o.get("day") or {}
            iv     = robust_float(greeks.get("iv") or greeks.get("implied_volatility"))
            delta  = robust_float(greeks.get("delta"))
            gamma  = robust_float(greeks.get("gamma"))
            theta  = robust_float(greeks.get("theta"))
            vega   = robust_float(greeks.get("vega"))
            rho    = robust_float(greeks.get("rho"))

            if expiry is None or strike is None:
                continue
            mid = calc_mid(bid, ask)
            if mid is None:
                continue

            norm.append({
                "ts_iso": ingest_ts,
                "ingest_ts_iso": ingest_ts,
                "symbol_underlying": underlying.upper(),
                "underlying_price": S,
                "expiry": str(expiry),
                "type": "C" if typ == "CALL" else "P",
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_size": bidz,
                "ask_size": askz,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
            })
        except Exception:
            if DEF_LOG_ERRORS:
                print(f"[WARN] normalize failed for one {underlying} contract", file=sys.stderr)
            continue

    if not norm:
        return []

    expiries = sorted({r["expiry"] for r in norm})
    keep_exps = set(expiries[:max_expiries])
    norm = [r for r in norm if r["expiry"] in keep_exps]

    if S is not None:
        return [r for r in norm if abs((r["strike"] - S) / S) <= near_atm_pct]
    return norm

def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    seen, deduped = set(), []
    for r in rows:
        key = (r["ts_iso"], r["expiry"], r["type"], r["strike"])
        if key not in seen:
            seen.add(key); deduped.append(r)

    write_header = not os.path.exists(path)
    ensure_dir(os.path.dirname(path))
    fieldnames = [
        "ts_iso","ingest_ts_iso","symbol_underlying","underlying_price",
        "expiry","type","strike","bid","ask","mid",
        "bid_size","ask_size","iv","delta","gamma","theta","vega","rho"
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in deduped:
            w.writerow(r)

def fetch_and_log_once(underlying: str, out_dir: str, near_atm_pct: float, max_expiries: int) -> int:
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying.upper()}"
    try:
        js = polygon_get(url, params={"limit": 1000})
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if DEF_LOG_ERRORS:
            print(f"[HTTP {code}] {underlying} snapshot failed", file=sys.stderr)
        return 0
    except Exception as e:
        if DEF_LOG_ERRORS:
            print(f"[ERR] {underlying} snapshot error: {e}", file=sys.stderr)
        return 0

    rows = extract_contract_rows(underlying, js, near_atm_pct, max_expiries)
    if not rows:
        return 0
    sym_dir = os.path.join(out_dir, underlying.upper())
    ensure_dir(sym_dir)
    path = os.path.join(sym_dir, f"{day_str()}.csv")
    write_rows_csv(path, rows)
    return len(rows)

# ---------- cli ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Log Polygon live option snapshots into per-symbol daily CSVs.")
    ap.add_argument("--symbols", default=None, help="Comma-separated underlyings (override repo TICKERS)")
    ap.add_argument("--out", default=DEF_OUT_DIR, help="Output directory")
    ap.add_argument("--poll", type=int, default=DEF_POLL_SEC, help="Polling interval seconds")
    ap.add_argument("--atm", type=float, default=DEF_ATM_PCT, help="Near-ATM band (e.g., 0.05 = 5%)")
    ap.add_argument("--exp", type=int, default=DEF_MAX_EXPS, help="Nearest expiries to keep")
    return ap.parse_args()

def main() -> None:
    if not API_KEY:
        print("ERROR: POLYGON_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    args = parse_args()
    symbols = load_universe(args.symbols)
    out_dir = args.out
    poll_s  = max(1, int(args.poll))
    print(f"[start] symbols={symbols} poll={poll_s}s atm={args.atm:.3f} exp={args.exp} out={out_dir}")
    ensure_dir(out_dir)
    try:
        while True:
            t0 = time.time()
            total = 0
            for sym in symbols:
                total += fetch_and_log_once(sym, out_dir, args.atm, args.exp)
            elapsed = time.time() - t0
            sleep_s = max(poll_s - elapsed, 1.0)
            print(f"[{now_utc_iso()}] wrote_rows={total}, cycle_s={elapsed:.2f}, sleep_s={sleep_s:.2f}")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[stop] received KeyboardInterrupt; exiting cleanly.")

if __name__ == "__main__":
    main()