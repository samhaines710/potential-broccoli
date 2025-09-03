#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration contract checks to catch stale wiring across modules.

What this does:
1) Validate utils.microstructure export surface.
2) Stub ingestion/micro builder/greeks; run prepare_training_data.extract_features_and_label.
3) Force MLClassifier to use a binary dummy model; validate Option-B probability mapping.
"""

from __future__ import annotations

import os
import sys
import types
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.append(".")

def _fail(msg: str, code: int = 1) -> None:
    print(f"[CONTRACT-FAIL] {msg}", file=sys.stderr)
    sys.exit(code)

def _ok(msg: str) -> None:
    print(f"[CONTRACT-OK] {msg}")

# 1) Microstructure export surface
REQUIRED = {
    "calc_spread",
    "midprice",
    "micro_price",
    "imbalance",
    "add_l1_features",
    "compute_l1_metrics",
    "compute_ofi",
    "compute_trade_signed_volume",
    "compute_vpin",
}
try:
    import utils.microstructure as micro
except Exception as e:
    _fail(f"Cannot import utils.microstructure: {e}")

missing = [name for name in REQUIRED if not hasattr(micro, name)]
if missing:
    _fail(f"utils.microstructure missing exports: {missing}")
_ok("utils.microstructure export surface OK")

# 2) Stub ingestion + micro feature builder + greeks; run dataset builder
try:
    import data_ingestion as di
except Exception as e:
    _fail(f"Cannot import data_ingestion: {e}")

class _StubLoader:
    def fetch_bars(self, symbol, start, end):
        ts0 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        bars = []
        price = 100.0
        for i in range(30):
            drift = (random.random() - 0.5) * 0.2
            close = max(1e-6, price + drift)
            high = max(close, price) + 0.05
            low  = min(close, price) - 0.05
            vol  = 1000 + int(200 * random.random())
            bars.append({"t": ts0 + i * 300_000, "o": price, "h": high, "l": low, "c": close, "v": vol})
            price = close
        return bars

di.HistoricalDataLoader = _StubLoader  # type: ignore[attr-defined]

try:
    from utils import micro_features as mf
except Exception:
    mf = types.ModuleType("utils.micro_features")
    sys.modules["utils.micro_features"] = mf

def _stub_build_microstructure_features(symbol, start, end, bucket="5min", include_trades=True):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="5min", inclusive="both")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=idx)
    df["ms_spread_mean"] = rng.uniform(0.005, 0.02, size=len(df))
    df["ms_depth_imbalance_mean"] = rng.uniform(-0.5, 0.5, size=len(df))
    df["ms_ofi_sum"] = rng.normal(0.0, 100.0, size=len(df))
    df["ms_signed_volume_sum"] = rng.normal(0.0, 1000.0, size=len(df))
    df["ms_vpin"] = np.clip(rng.normal(0.15, 0.05, size=len(df)), 0.0, 1.0)
    return df

mf.build_microstructure_features = _stub_build_microstructure_features  # type: ignore[attr-defined]

try:
    import utils as uroot
except Exception as e:
    _fail(f"Cannot import utils: {e}")

def _stub_fetch_option_greeks(symbol: str):
    return {
        "delta": 0.1, "gamma": 0.02, "theta": -0.05, "vega": 0.12, "rho": 0.01,
        "vanna": 0.0, "vomma": 0.0, "charm": 0.0, "veta": 0.0, "speed": 0.0,
        "zomma": 0.0, "color": 0.0, "implied_volatility": 0.35,
    }

uroot.fetch_option_greeks = _stub_fetch_option_greeks  # type: ignore[attr-defined]

try:
    import prepare_training_data as ptd
except Exception as e:
    _fail(f"Cannot import prepare_training_data: {e}")

# Reduce env horizon for speed
os.environ["HIST_DAYS"] = "2"
os.environ["LOOKBACK_BARS"] = "6"
os.environ["LOOKAHEAD_BARS"] = "2"

df = ptd.extract_features_and_label("FAKE")
if df is None or df.empty:
    _fail("prepare_training_data.extract_features_and_label produced empty output")

need_cols = {
    "symbol","breakout_prob","recent_move_pct","time_of_day","volume_ratio",
    "rsi","corr_dev","skew_ratio",
    "yield_spike_2year","yield_spike_10year","yield_spike_30year",
    "theta_day","theta_5m","movement_type",
    "ms_spread_mean","ms_depth_imbalance_mean","ms_ofi_sum",
    "ms_signed_volume_sum","ms_vpin",
}
missing_cols = [c for c in need_cols if c not in df.columns]
if missing_cols:
    _fail(f"Training dataset missing expected columns: {missing_cols}")

_ok(f"prepare_training_data emitted {len(df)} rows with required columns")

# 3) MLClassifier with binary model (Option-B mapping)
try:
    import ml_classifier as mlc
except Exception as e:
    _fail(f"Cannot import ml_classifier: {e}")

class _DummyBinaryModel:
    def __init__(self):
        self.classes_ = np.array(["CALL", "PUT"])
    def predict_proba(self, X):
        p = np.array([0.6, 0.4])
        return np.tile(p, (len(X), 1))

mlc._load_pipeline = lambda uri: _DummyBinaryModel()  # type: ignore
os.environ["MODEL_URI"] = "local://dummy"

clf = mlc.MLClassifier()
sample = df.iloc[0].to_dict()
sample.pop("movement_type", None)

out = clf.classify(sample)
if not isinstance(out, dict) or "probs" not in out or "movement_type" not in out:
    _fail("MLClassifier.classify returned invalid structure")

probs = out["probs"]
if not (len(probs) == 3 and abs(probs[0]-0.6) < 1e-6 and abs(probs[1]-0.4) < 1e-6 and abs(probs[2]-0.0) < 1e-12):
    _fail(f"Binary fallback expected [0.6,0.4,0.0], got {probs}")

if out["movement_type"] not in {"CALL","PUT","NEUTRAL"}:
    _fail(f"movement_type invalid: {out['movement_type']}")

_ok("MLClassifier Option-B fallback OK")
print("[CONTRACT-OK] All checks passed.")
