#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Institutional microstructure utilities.

Exports
-------
- calc_spread(bid, ask)
- midprice(bid, ask)
- micro_price(bid, ask, bid_size, ask_size)
- imbalance(bid_size, ask_size)
- add_l1_features(df, bid_col="bid", ask_col="ask", bid_size_col="bid_size", ask_size_col="ask_size",
                  prefix=None, price_scale=1.0)
- compute_l1_metrics(df=..., or bid/ask/bid_size/ask_size=..., as_frame=...)
- compute_ofi(df=..., method="cont"| "diff", normalize=False, window=None, as_frame=True)
- compute_trade_signed_volume(df=..., trades + (optional) quotes in same frame, or price/volume arrays,
                              method="lee_ready"|"tick", normalize=False, window=None, as_frame=True)

Design notes
------------
- Vectorized; safe for scalars/ndarrays/Series/DataFrames.
- Preserves DataFrame index when possible.
- No reliance on Pandas nullable dtypes (uses np.int8).
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# If your repo already has utils/micro_features.py, delegate to it when possible.
_HAS_MF = False
_HAS_MF_COMPUTE = False
try:
    from .micro_features import (  # type: ignore
        calc_spread as _mf_calc_spread,
        midprice as _mf_midprice,
        micro_price as _mf_micro_price,
        imbalance as _mf_imbalance,
        add_l1_features as _mf_add_l1_features,
    )
    _HAS_MF = True
    try:
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

    # robust flags (no pandas nullable int)
    out[f"{prefix}is_locked"] = (ask == bid).astype(np.int8)
    out[f"{prefix}is_crossed"] = (ask < bid).astype(np.int8)
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
    Either pass a DataFrame (preferred) or pass separate arrays/Series.
    """
    if _HAS_MF_COMPUTE:
        try:
            return _mf_compute_l1_metrics(
                df=df if df is not None else (args[0] if (len(args) >= 1 and isinstance(args[0], pd.DataFrame)) else None),
                bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
                bid_col=bid_col, ask_col=ask_col, bid_size_col=bid_size_col, ask_size_col=ask_size_col,
                prefix=prefix, price_scale=price_scale, as_frame=as_frame,
            )
        except TypeError:
            pass

    if df is None and len(args) >= 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]

    if df is not None:
        return add_l1_features(df, bid_col, ask_col, bid_size_col, ask_size_col, prefix, price_scale)

    if bid is None or ask is None or bid_size is None or ask_size is None:
        raise ValueError("compute_l1_metrics requires either df=... or bid/ask/bid_size/ask_size.")

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

    is_locked = (a == b).astype(np.int8)
    is_crossed = (a < b).astype(np.int8)

    if as_frame:
        idx = None
        for src in (bid, ask, bid_size, ask_size):
            if isinstance(src, pd.Series):
                idx = src.index
                break
        return pd.DataFrame(
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

    return {
        "spread": _return_like(spread, b, a),
        "mid": _return_like(mid, b, a),
        "microprice": _return_like(micro, b, a),
        "imbalance": _return_like(imb, bs, asz),
        "spread_bps": _return_like(spread_bps, b, a),
        "is_locked": _return_like(is_locked.astype(float), b, a),
        "is_crossed": _return_like(is_crossed.astype(float), b, a),
    }


# ----------------------------
# Order Flow Imbalance (OFI): Cont/Kukanov/Stoikov L1 approximation
# ----------------------------

def compute_ofi(*args: Any,
                df: Optional[pd.DataFrame] = None,
                bid_col: str = "bid",
                ask_col: str = "ask",
                bid_size_col: str = "bid_size",
                ask_size_col: str = "ask_size",
                method: str = "cont",
                normalize: bool = False,
                window: Optional[int] = None,
                as_frame: bool = True) -> Union[pd.Series, pd.DataFrame]:
    # legacy positional
    if df is None and len(args) >= 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]
    if df is None:
        raise ValueError("compute_ofi requires df=...")

    need = [bid_col, ask_col, bid_size_col, ask_size_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"compute_ofi: missing columns {miss}")

    pb = pd.to_numeric(df[bid_col], errors="coerce")
    pa = pd.to_numeric(df[ask_col], errors="coerce")
    qb = pd.to_numeric(df[bid_size_col], errors="coerce")
    qa = pd.to_numeric(df[ask_size_col], errors="coerce")

    dpb = pb.diff()
    dpa = pa.diff()
    qb_prev = qb.shift()
    qa_prev = qa.shift()

    if method.lower() == "cont":
        ofi_bid = (dpb.gt(0).astype(float) * qb) - (dpb.lt(0).astype(float) * qb_prev)
        ofi_ask = (dpa.lt(0).astype(float) * qa_prev) - (dpa.gt(0).astype(float) * qa)
        ofi = ofi_bid.add(ofi_ask, fill_value=0.0)
    elif method.lower() in ("diff", "simple"):
        sgn_b = np.sign(dpb.fillna(0.0))
        sgn_a = -np.sign(dpa.fillna(0.0))  # ask down = buy pressure
        ofi = sgn_b * qb.diff().fillna(0.0) + sgn_a * qa.diff().fillna(0.0)
    else:
        raise ValueError(f"compute_ofi: unknown method '{method}'")

    if normalize:
        denom = qb_prev.add(qa_prev, fill_value=0.0).replace(0.0, np.nan)
        ofi = ofi / denom

    if window is not None and window > 1:
        ofi = ofi.rolling(int(window), min_periods=1).sum()

    ofi.name = "ofi"
    if as_frame:
        return ofi.to_frame()
    return ofi


# ----------------------------
# Trade-Signed Volume (TSV): Lee–Ready with robust fallbacks
# ----------------------------

def _infer_mid(df: pd.DataFrame,
               bid_col: str, ask_col: str, mid_col: Optional[str]) -> Optional[pd.Series]:
    if mid_col and mid_col in df.columns:
        return pd.to_numeric(df[mid_col], errors="coerce")
    if bid_col in df.columns and ask_col in df.columns:
        b = pd.to_numeric(df[bid_col], errors="coerce")
        a = pd.to_numeric(df[ask_col], errors="coerce")
        return 0.5 * (a + b)
    return None


def compute_trade_signed_volume(*args: Any,
                                df: Optional[pd.DataFrame] = None,
                                price: Optional[NumberLike] = None,
                                volume: Optional[NumberLike] = None,
                                # optional quotes-in-frame columns:
                                bid_col: str = "bid",
                                ask_col: str = "ask",
                                mid_col: Optional[str] = None,
                                # trades columns (DataFrame mode)
                                price_col: str = "price",
                                volume_col: str = "size",
                                method: str = "lee_ready",
                                normalize: bool = False,
                                window: Optional[int] = None,
                                as_frame: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Trade-signed volume (TSV).

    Methods
    -------
    - 'lee_ready' (default): sign = sign(price_t - mid_{t-1}); ties -> tick test fallback.
      Requires either columns for quotes in the same DataFrame (bid/ask or mid), or it falls
      back to tick test if mid cannot be inferred.
    - 'tick': sign = sign(price_t - price_{t-1}) with first tick = 0.

    Normalization
    -------------
    If normalize=True, divides signed volume by absolute volume per tick.
    If window is provided, a rolling sum is returned.
    """
    # If an upstream implementation exists, prefer it.
    if _HAS_MF:
        try:
            from .micro_features import compute_trade_signed_volume as _mf_tsv  # type: ignore
            return _mf_tsv(
                df=df if df is not None else (args[0] if (len(args) >= 1 and isinstance(args[0], pd.DataFrame)) else None),
                price=price, volume=volume,
                bid_col=bid_col, ask_col=ask_col, mid_col=mid_col,
                price_col=price_col, volume_col=volume_col,
                method=method, normalize=normalize, window=window, as_frame=as_frame,
            )
        except Exception:
            pass

    # legacy positional DF
    if df is None and len(args) >= 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]

    # DataFrame mode
    if df is not None:
        if price_col not in df.columns or volume_col not in df.columns:
            raise KeyError(f"compute_trade_signed_volume: missing '{price_col}' or '{volume_col}' columns.")
        p = pd.to_numeric(df[price_col], errors="coerce")
        v = pd.to_numeric(df[volume_col], errors="coerce")

        method_l = method.lower()
        if method_l == "lee_ready":
            mid_prev = _infer_mid(df, bid_col, ask_col, mid_col)
            if mid_prev is not None:
                mid_prev = mid_prev.shift()  # Lee–Ready uses previous mid
                sign = np.sign(p - mid_prev)
                # tie-breaker: tick test
                tie = (sign == 0) | (~np.isfinite(sign))
                tick_sign = np.sign(p.diff().fillna(0.0))
                sign = np.where(tie, tick_sign, sign)
            else:
                # no quotes available → tick test fallback
                sign = np.sign(p.diff().fillna(0.0))
        elif method_l == "tick":
            sign = np.sign(p.diff().fillna(0.0))
        else:
            raise ValueError(f"compute_trade_signed_volume: unknown method '{method}'")

        tsv = sign * v
        if normalize:
            with np.errstate(invalid="ignore", divide="ignore"):
                tsv = tsv / v.replace(0.0, np.nan)
        if window is not None and window > 1:
            tsv = tsv.rolling(int(window), min_periods=1).sum()

        tsv.name = "ts_volume"
        if as_frame:
            return tsv.to_frame()
        return tsv

    # Vector/scalar mode
    if price is None or volume is None:
        raise ValueError("compute_trade_signed_volume requires either df=... or price/volume arrays.")

    p_arr = _to_array(price).astype(float)
    v_arr = _to_array(volume).astype(float)

    method_l = method.lower()
    if method_l == "lee_ready":
        # Without quotes, lee_ready reduces to tick fallback.
        sign = np.sign(np.diff(p_arr, prepend=p_arr[0]))
    elif method_l == "tick":
        sign = np.sign(np.diff(p_arr, prepend=p_arr[0]))
    else:
        raise ValueError(f"compute_trade_signed_volume: unknown method '{method}'")

    tsv_arr = sign * v_arr
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            tsv_arr = tsv_arr / np.where(v_arr == 0, np.nan, v_arr)
    if window is not None and window > 1:
        # simple rolling sum in numpy
        w = int(window)
        cumsum = np.cumsum(np.insert(tsv_arr, 0, 0.0))
        tsv_arr = cumsum[w:] - cumsum[:-w]
        # pad left
        pad = np.full(w - 1, np.nan)
        tsv_arr = np.concatenate([pad, tsv_arr])

    return _return_like(tsv_arr, price, volume)


__all__ = [
    "calc_spread",
    "midprice",
    "micro_price",
    "imbalance",
    "add_l1_features",
    "compute_l1_metrics",
    "compute_ofi",
    "compute_trade_signed_volume",
]
