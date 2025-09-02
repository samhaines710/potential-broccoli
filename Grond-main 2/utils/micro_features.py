"""
micro_features.py

Aggregate microstructure metrics into fixed-interval features aligned to bar time.

Workflow
--------
1) Fetch NBBO quotes (and optionally trades) using utils.polygon_nbbo.
2) Compute per-message L1 metrics and OFI.
3) Resample/aggregate into regular buckets (default 5 minutes).
4) Optionally compute VPIN from trades and join to the bucketed features.

Returns a DataFrame indexed by UTC bucket end with columns prefixed `ms_`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from utils.polygon_nbbo import fetch_nbbo_quotes, fetch_trades
from utils.microstructure import (
    compute_l1_metrics,
    compute_ofi,
    compute_trade_signed_volume,
    compute_vpin,
)


def _to_utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def build_microstructure_features(
    ticker: str,
    *,
    start: datetime,
    end: Optional[datetime] = None,
    bucket: str = "5min",
    quote_limit: int = 50_000,
    trade_limit: int = 50_000,
    include_trades: bool = True,
) -> pd.DataFrame:
    """
    Build bucketed microstructure feature set for `ticker` between start..end.

    Columns produced (ms_*):
      - ms_spread_mean
      - ms_depth_imbalance_mean
      - ms_ofi_sum
      - ms_signed_volume_sum (if trades available)
      - ms_vpin (if trades available)
    """
    end = end or datetime.now(tz=timezone.utc)

    quotes = fetch_nbbo_quotes(ticker, start_timestamp=start, limit=quote_limit)
    if quotes.empty:
        # empty but well-formed structure
        idx = pd.DatetimeIndex([], name="bucket_end")
        return pd.DataFrame(
            {
                "ms_spread_mean": pd.Series(dtype=float),
                "ms_depth_imbalance_mean": pd.Series(dtype=float),
                "ms_ofi_sum": pd.Series(dtype=float),
                "ms_signed_volume_sum": pd.Series(dtype=float),
                "ms_vpin": pd.Series(dtype=float),
            },
            index=idx,
        )

    quotes = quotes.sort_index()
    l1 = compute_l1_metrics(quotes)
    ofi = compute_ofi(l1)
    l1["ofi"] = ofi

    # Aggregate quotes to bucket
    q_buck = pd.DataFrame(
        {
            "ms_spread_mean": l1["spread"].resample(bucket).mean(),
            "ms_depth_imbalance_mean": l1["depth_imbalance"].resample(bucket).mean(),
            "ms_ofi_sum": l1["ofi"].resample(bucket).sum(),
        }
    )

    # Trades-derived features
    if include_trades:
        trades = fetch_trades(ticker, start_timestamp=start, limit=trade_limit)
        if not trades.empty:
            trades = trades.sort_index()
            signed_vol, _eff = compute_trade_signed_volume(trades, l1)
            tdf = trades.copy()
            tdf["signed_volume"] = signed_vol.reindex(trades.index)
            sv_sum = tdf["signed_volume"].resample(bucket).sum(min_count=1)
            vpin = compute_vpin(tdf[["price", "size", "signed_volume"]].dropna(), bucket_size=50_000, window=50)
            q_buck["ms_signed_volume_sum"] = sv_sum.reindex(q_buck.index)
            # Align VPIN to bucket end (asof)
            q_buck["ms_vpin"] = vpin.reindex(q_buck.index, method="pad")
        else:
            q_buck["ms_signed_volume_sum"] = np.nan
            q_buck["ms_vpin"] = np.nan
    else:
        q_buck["ms_signed_volume_sum"] = np.nan
        q_buck["ms_vpin"] = np.nan

    # Keep within [start,end]
    q_buck = q_buck.loc[(q_buck.index >= pd.to_datetime(start, utc=True)) & (q_buck.index <= pd.to_datetime(end, utc=True))]
    q_buck.index.name = "bucket_end"
    return q_buck
