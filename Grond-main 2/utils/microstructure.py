#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compat layer for microstructure primitives expected by utils.__init__.

If your repo already implements these in utils/micro_features.py, we simply
delegate to those functions. If not, robust fallbacks are provided.
Exposes:
    - calc_spread(bid, ask)
    - midprice(bid, ask)
    - micro_price(bid, ask, bid_size, ask_size)
    - imbalance(bid_size, ask_size)
    - add_l1_features(df, bid_col="bid", ask_col="ask",
                      bid_size_col="bid_size", ask_size_col="ask_size", prefix=None)
"""

from __future__ import annotations
from typing import Union, Optional

import numpy as np
import pandas as pd

# ---- Try to reuse existing implementations if you already have micro_features.py ----
_HAS_MF = False
try:
    from .micro_features import (  # type: ignore
        calc_spread as _mf_calc_spread,
        midprice as _mf_midprice,
        micro_price as _mf_micro_price,
        imbalance as _mf_imbalance,
        add_l1_features as _mf_add_l1_features,
    )
    _HAS_MF = True
except Exception:
    _HAS_MF = False


NumberLike = Union[int, float, np.ndarray, pd.Series]


def _to_array(x: NumberLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    if isinstance(x, (int, float, np.number)):
        return np.asarray([x], dtype=float)
    return np.asarray(x)


def _return_like(arr: np.ndarray, *like_sources: NumberLike):
    for src in like_sources:
        if isinstance(src, pd.Series):
            return pd.Series(arr, index=src.index, name=getattr(src, "name", None))
    if arr.size == 1:
        return float(arr[0])
    return arr


# ----------------------------
# Delegating wrappers (prefer existing micro_features if present)
# ----------------------------

def calc_spread(bid: NumberLike, ask: NumberLike) -> NumberLike:
    if _HAS_MF:
        return _mf_calc_spread(bid, ask)
    b = _to_array(bid).astype(float)
    a = _to_array(ask).astype(float)
    sp = a - b
    sp[~np.isfinite(sp)] = np.nan
    return _return_like(sp, bid, ask)


def midprice(bid: NumberLike, ask: NumberLike) -> NumberLike:
    if _HAS_MF:
        return _mf_midprice(bid, ask)
    b = _to_array(bid).astype(float)
    a = _to_array(ask).astype(float)
    mid = 0.5 * (a + b)
    mid[~np.isfinite(mid)] = np.nan
    return _return_like(mid, bid, ask)


def micro_price(bid: NumberLike, ask: NumberLike,
                bid_size: NumberLike, ask_size: NumberLike) -> NumberLike:
    if _HAS_MF:
        return _mf_micro_price(bid, ask, bid_size, ask_size)
    b = _to_array(bid).astype(float)
    a = _to_array(ask).astype(float)
    bs = _to_array(bid_size).astype(float)
    as_ = _to_array(ask_size).astype(float)
    denom = bs + as_
    mp = np.where(np.isfinite(denom) & (denom > 0.0),
                  (a * bs + b * as_) / denom,
                  0.5 * (a + b))
    mp[~np.isfinite(mp)] = np.nan
    return _return_like(mp, bid, ask)


def imbalance(bid_size: NumberLike, ask_size: NumberLike) -> NumberLike:
    if _HAS_MF:
        return _mf_imbalance(bid_size, ask_size)
    bs = _to_array(bid_size).astype(float)
    as_ = _to_array(ask_size).astype(float)
    denom = bs + as_
    imb = np.where(np.isfinite(denom) & (denom > 0.0), (bs - as_) / denom, np.nan)
    imb[~np.isfinite(imb)] = np.nan
    return _return_like(imb, bid_size, ask_size)


def add_l1_features(df: pd.DataFrame,
                    bid_col: str = "bid",
                    ask_col: str = "ask",
                    bid_size_col: str = "bid_size",
                    ask_size_col: str = "ask_size",
                    prefix: Optional[str] = None,
                    price_scale: float = 1.0) -> pd.DataFrame:
    """
    If micro_features.add_l1_features exists, delegate to it. Otherwise add:
      spread, mid, microprice, imbalance, spread_bps, is_locked, is_crossed.
    price_scale lets you pass quotes in cents (use 0.01 to convert to dollars for bps).
    """
    if _HAS_MF:
        # If your existing implementation doesn't accept price_scale/prefix,
        # it's fine—Python will raise if signature differs. In that case,
        # fall back to our version below.
        try:
            return _mf_add_l1_features(df, bid_col, ask_col, bid_size_col, ask_size_col, prefix)
        except TypeError:
            pass

    if prefix is None:
        prefix = ""

    need = [bid_col, ask_col, bid_size_col, ask_size_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"add_l1_features: missing columns {missing}")

    out = df.copy()
    bid = out[bid_col] * price_scale
    ask = out[ask_col] * price_scale
    bsz = out[bid_size_col]
    asz = out[ask_size_col]

    out[f"{prefix}spread"] = calc_spread(bid, ask)
    out[f"{prefix}mid"] = midprice(bid, ask)
    out[f"{prefix}microprice"] = micro_price(bid, ask, bsz, asz)
    out[f"{prefix}imbalance"] = imbalance(bsz, asz)

    with np.errstate(invalid="ignore", divide="ignore"):
        out[f"{prefix}spread_bps"] = (out[f"{prefix}spread"] / out[f"{prefix}mid"]) * 1e4

    out[f"{prefix}is_locked"] = (ask == bid).astype("Int8")
    out[f"{prefix}is_crossed"] = (ask < bid).astype("Int8")
    return out


__all__ = [
    "calc_spread",
    "midprice",
    "micro_price",
    "imbalance",
    "add_l1_features",
]
