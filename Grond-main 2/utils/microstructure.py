#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Institutional microstructure utilities.

Exposes:
- calc_spread(bid, ask)
- midprice(bid, ask)
- micro_price(bid, ask, bid_size, ask_size)
- imbalance(bid_size, ask_size)
- add_l1_features(df, bid_col="bid", ask_col="ask", bid_size_col="bid_size", ask_size_col="ask_size",
                  prefix=None, price_scale=1.0)
- compute_l1_metrics(...)  <-- added for compatibility with utils.__init__ imports

Design:
- Vectorized; safe for scalars, numpy arrays, or pandas Series/DataFrames.
- Explicit NaN handling.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any

import numpy as np
import pandas as pd

# If you already have utils/micro_features.py, reuse it when possible.
_HAS_MF = False
_HAS_MF_COMPUTE = False
try:
    # type: ignore[attr-defined]
    from .micro_features import (  # noqa
        calc_spread as _mf_calc_spread,
        midprice as _mf_midprice,
        micro_price as _mf_micro_price,
        imbalance as _mf_imbalance,
        add_l1_features as _mf_add_l1_features,
    )
    _HAS_MF = True
    try:
        # optional in some trees
        from .micro_features import compute_l1_metrics as _mf_compute_l1_metrics  # type: ignore
        _HAS_MF_COMPUTE = True
    except Exception:
        _HAS_MF_COMPUTE = False
except Exception:
    _HAS_MF = False
    _HAS_MF_COMPUTE = False


NumberLike = Union[int, float, np.ndarray, pd.Series]


def _to_array(x: NumberLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(copy=False)
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    if isinstance(x, (int, float, np.number)):
        return np.asarray([x], dtype=float)
    return np.asarray(x)


def _return_like(arr: np.ndarray, *like_sources: NumberLike) -> NumberLike:
    for src in like_sources:
        if isinstance(src, pd.Series):
            return pd.Series(arr, index=src.index, name=getattr(src, "name", None))
    if arr.size == 1:
        return float(arr[0])
    return arr


# ----------------------------
# Core primitives
# ----------------------------

def calc_spread(bid: NumberLike, ask: NumberLike) -> NumberLike:
    if _HAS_MF:
        try:
            return _mf_calc_spread(bid, ask)
        except Exception:
            pass
    b = _to_array(bid).astype(float)
    a = _to_array(ask).astype(float)
    sp = a - b
    sp[~np.isfinite(sp)] = np.nan
    return _return_like(sp, bid, ask)


def midprice(bid: NumberLike, ask: NumberLike) -> NumberLike:
    if _HAS_MF:
        try:
            return _mf_midprice(bid, ask)
        except Exception:
            pass
    b = _to_array(bid).astype(float)
    a = _to_array(ask).astype(float)
    mid = 0.5 * (a + b)
    mid[~np.isfinite(mid)] = np.nan
    return _return_like(mid, bid, ask)


def micro_price(bid: NumberLike, ask: NumberLike,
                bid_size: NumberLike, ask_size: NumberLike) -> NumberLike:
    if _HAS_MF:
        try:
            return _mf_micro_price(bid, ask, bid_size, ask_size)
        except Exception:
            pass
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
        try:
            return _mf_imbalance(bid_size, ask_size)
        except Exception:
            pass
    bs = _to_array(bid_size).astype(float)
    as_ = _to_array(ask_size).astype(float)
    denom = bs + as_
    imb = np.where(np.isfinite(denom) & (denom > 0.0), (bs - as_) / denom, np.nan)
    imb[~np.isfinite(imb)] = np.nan
    return _return_like(imb, bid_size, ask_size)


# ----------------------------
# DataFrame features
# ----------------------------

def add_l1_features(df: pd.DataFrame,
                    bid_col: str = "bid",
                    ask_col: str = "ask",
                    bid_size_col: str = "bid_size",
                    ask_size_col: str = "ask_size",
                    prefix: Optional[str] = None,
                    price_scale: float = 1.0) -> pd.DataFrame:
    """
    Append: spread, mid, microprice, imbalance, spread_bps, is_locked, is_crossed.
    price_scale: multiply prices before bps calc (use 0.01 if quotes are in cents).
    """
    if _HAS_MF:
        # If your existing function doesn't support price_scale/prefix, fall back to ours.
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


# ----------------------------
# Compatibility wrapper expected by utils.__init__
# ----------------------------

def compute_l1_metrics(*args: Any,
                       df: Optional[pd.DataFrame] = None,
                       bid: Optional[NumberLike] = None,
                       ask: Optional[NumberLike] = None,
                       bid_size: Optional[NumberLike] = None,
                       ask_size: Optional[NumberLike] = None,
                       bid_col: str = "bid",
                       ask_col: str = "ask",
                       bid_size_col: str = "bid_size",
                       ask_size_col: str = "ask_size",
                       prefix: Optional[str] = None,
                       price_scale: float = 1.0,
                       as_frame: bool = True) -> Union[pd.DataFrame, Dict[str, Any], pd.Series]:
    """
    Flexible interface:

    1) DataFrame mode (recommended):
       compute_l1_metrics(df=<quotes_df>, bid_col="bid", ask_col="ask",
                          bid_size_col="bid_size", ask_size_col="ask_size",
                          prefix=None, price_scale=1.0, as_frame=True)
       -> returns DataFrame with added columns

    2) Vector/scalar mode:
       compute_l1_metrics(bid=<...>, ask=<...>, bid_size=<...>, ask_size=<...>, as_frame=False)
       -> returns dict with spread, mid, microprice, imbalance, spread_bps, is_locked, is_crossed

    Also supports legacy call pattern where the first positional arg is a DataFrame.
    """
    # Prefer an existing implementation if provided downstream
    if _HAS_MF_COMPUTE:
        try:
            return _mf_compute_l1_metrics(
                df=df if df is not None else (args[0] if (len(args) >= 1 and isinstance(args[0], pd.DataFrame)) else None),
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                bid_col=bid_col,
                ask_col=ask_col,
                bid_size_col=bid_size_col,
                ask_size_col=ask_size_col,
                prefix=prefix,
                price_scale=price_scale,
                as_frame=as_frame,
            )
        except TypeError:
            # Signature mismatch—fall back to our robust version
            pass

    # Detect legacy positional DF usage
    if df is None and len(args) >= 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]

    if df is not None:
        # DataFrame mode
        return add_l1_features(
            df=df,
            bid_col=bid_col,
            ask_col=ask_col,
            bid_size_col=bid_size_col,
            ask_size_col=ask_size_col,
            prefix=prefix,
            price_scale=price_scale,
        )

    # Vector/scalar mode
    if bid is None or ask is None or bid_size is None or ask_size is None:
        raise ValueError(
            "compute_l1_metrics requires either df=... or bid/ask/bid_size/ask_size inputs."
        )

    b = _to_array(bid).astype(float) * price_scale
    a = _to_array(ask).astype(float) * price_scale
    bs = _to_array(bid_size).astype(float)
    asz = _to_array(ask_size).astype(float)

    spread = _to_array(calc_spread(b, a)).astype(float)
    mid = _to_array(midprice(b, a)).astype(float)
    micro = _to_array(micro_price(b, a, bs, asz)).astype(float)
    imb = _to_array(imbalance(bs, asz)).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        spread_bps = (spread / mid) * 1e4

    is_locked = (a == b).astype(int)
    is_crossed = (a < b).astype(int)

    if as_frame:
        # Return as Series/DataFrame aligned to the longest input
        idx = None
        for src in (bid, ask, bid_size, ask_size):
            if isinstance(src, pd.Series):
                idx = src.index
                break
        out_df = pd.DataFrame(
            {
                "spread": spread,
                "mid": mid,
                "microprice": micro,
                "imbalance": imb,
                "spread_bps": spread_bps,
                "is_locked": is_locked,
                "is_crossed": is_crossed,
            },
            index=idx,
        )
        return out_df

    return {
        "spread": _return_like(spread, b, a),
        "mid": _return_like(mid, b, a),
        "microprice": _return_like(micro, b, a),
        "imbalance": _return_like(imb, bs, asz),
        "spread_bps": _return_like(spread_bps, b, a),
        "is_locked": _return_like(is_locked.astype(float), b, a),
        "is_crossed": _return_like(is_crossed.astype(float), b, a),
    }


__all__ = [
    "calc_spread",
    "midprice",
    "micro_price",
    "imbalance",
    "add_l1_features",
    "compute_l1_metrics",
]
