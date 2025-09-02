def compute_vpin(*args: Any,
                 df: Optional[pd.DataFrame] = None,
                 price: Optional[NumberLike] = None,
                 volume: Optional[NumberLike] = None,
                 # optional quotes-in-frame columns for Lee–Ready
                 bid_col: str = "bid",
                 ask_col: str = "ask",
                 mid_col: Optional[str] = None,
                 # trades columns (DataFrame mode)
                 price_col: str = "price",
                 volume_col: str = "size",
                 ts_col: Optional[str] = None,
                 method: str = "lee_ready",      # "lee_ready" | "tick"
                 bucket_size: Optional[float] = None,
                 buckets: Optional[int] = None,  # if bucket_size not set, use total_vol / buckets
                 window: int = 50,
                 as_frame: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    VPIN time series computed on **volume buckets**.
    - Signing: 'lee_ready' uses sign(price_t - mid_{t-1}) with tick-test tie-break; falls back to tick if no quotes.
               'tick' uses sign(price_t - price_{t-1}).
    - Buckets: exact volume splitting across boundaries (no overfill). VPIN at each bucket end is
               |signed_volume_bucket| / bucket_size. The returned 'vpin' is a rolling mean over the last `window` buckets.
    Returns Series (if as_frame=False) or DataFrame with columns: ['vpin','inst_abs_imb','bucket_id','bucket_size'].
    """
    # Accept legacy positional df
    if df is None and len(args) >= 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]

    # Build DF from arrays if needed
    if df is None:
        if price is None or volume is None:
            raise ValueError("compute_vpin requires df=... or price/volume arrays.")
        df = pd.DataFrame({price_col: price, volume_col: volume})

    # Set a reliable index
    if ts_col and ts_col in df.columns:
        idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.set_index(idx)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.RangeIndex(len(df))

    # Coerce numeric
    p = pd.to_numeric(df[price_col], errors="coerce")
    v = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0).astype(float)

    # ---- Trade signing
    def _infer_mid(local_df: pd.DataFrame) -> Optional[pd.Series]:
        if mid_col and mid_col in local_df.columns:
            return pd.to_numeric(local_df[mid_col], errors="coerce")
        if bid_col in local_df.columns and ask_col in local_df.columns:
            b = pd.to_numeric(local_df[bid_col], errors="coerce")
            a = pd.to_numeric(local_df[ask_col], errors="coerce")
            return 0.5 * (a + b)
        return None

    mth = str(method).lower()
    if mth == "lee_ready":
        mid_prev = _infer_mid(df)
        if mid_prev is not None:
            mid_prev = mid_prev.shift()
            sign = np.sign(p - mid_prev)
            tie = (sign == 0) | (~np.isfinite(sign))
            tick_sign = np.sign(p.diff().fillna(0.0))
            sign = np.where(tie, tick_sign, sign)
        else:
            sign = np.sign(p.diff().fillna(0.0))
    elif mth == "tick":
        sign = np.sign(p.diff().fillna(0.0))
    else:
        raise ValueError(f"compute_vpin: unknown method '{method}'")

    sign = pd.Series(sign, index=df.index).fillna(0.0).astype(float)

    # ---- Bucket sizing
    total_vol = float(np.nansum(v.values))
    if bucket_size is None:
        if not buckets or buckets <= 0:
            buckets = 50
        bucket_size = max(total_vol / float(buckets), 1e-9)

    # ---- Volume-bucket splitting (exact)
    times = df.index
    n = len(v)
    bucket_times: list = []
    abs_imb: list = []
    bucket_ids: list = []

    current_vol = 0.0
    current_signed = 0.0
    cum_vol = 0.0
    k = 0
    threshold = bucket_size  # next boundary at this cumulative volume

    i = 0
    while i < n:
        vol_rem = float(v.iloc[i])
        sgn_i = float(sign.iloc[i])

        # Split a single trade across multiple buckets as needed
        while vol_rem > 1e-12:
            need = threshold - cum_vol
            take = vol_rem if vol_rem <= need else need

            current_vol += take
            current_signed += sgn_i * take
            cum_vol += take
            vol_rem -= take

            if cum_vol >= threshold - 1e-12:
                # close bucket
                bucket_times.append(times[i])
                abs_imb.append(abs(current_signed) / bucket_size)
                bucket_ids.append(k)

                # advance to next bucket
                k += 1
                threshold = (k + 1) * bucket_size
                current_vol = 0.0
                current_signed = 0.0

        i += 1

    if not abs_imb:
        # no buckets formed (e.g., zero volume)
        out = pd.DataFrame(columns=["vpin", "inst_abs_imb", "bucket_id", "bucket_size"])
        return out if as_frame else out["vpin"]

    inst = pd.Series(abs_imb, index=pd.Index(bucket_times, name="bucket_time"), name="inst_abs_imb")
    vpin = inst.rolling(int(max(1, window)), min_periods=1).mean()
    result = pd.DataFrame({"vpin": vpin, "inst_abs_imb": inst})
    result["bucket_id"] = bucket_ids
    result["bucket_size"] = float(bucket_size)

    return result if as_frame else result["vpin"]
