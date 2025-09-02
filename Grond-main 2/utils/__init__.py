# ─── Logging & status writes ────────────────────────────────────────────────────
from .logging_utils import (
    write_status,
    REST_CALLS,
    REST_429,
)

# ─── HTTP client & rate limiting (compat layer) ────────────────────────────────
# Prefer the namespaced client to avoid collisions with other repos.
# Falls back to local .http_client only if it exists.
try:
    from grond_http.client import (
        safe_fetch_polygon_data,
        rate_limited,
        fetch_option_greeks,
    )
except Exception:  # noqa: BLE001 - intentional broad fallback
    # If you still carry a legacy utils/http_client.py, this keeps it working.
    from .http_client import (  # type: ignore[no-redef]
        safe_fetch_polygon_data,
        rate_limited,
        fetch_option_greeks,
    )

# ─── Market‐data utilities ─────────────────────────────────────────────────────
from .market_data import (
    reformat_candles,
    fetch_premarket_early_data,
    fetch_today_5m_candles,
    fetch_latest_5m_candle,
    fetch_historical_5m_candles,
)

# ─── Core indicators & analysis ────────────────────────────────────────────────
from .analysis import (
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_signal_persistence,
    calculate_reversal_and_scope,
    calculate_risk_reward,
    calculate_time_of_day,
    calculate_volume_ratio,
    compute_skew_ratio,
    compute_corr_deviation,
    compute_rsi,
    detect_yield_spike,
)

# ─── File I/O helpers ──────────────────────────────────────────────────────────
from .file_io import (
    append_signal_log,
    load_snapshot,
    save_snapshot,
)

# ─── Greeks & Pricing helpers ──────────────────────────────────────────────────
from .greeks_helpers import (
    calculate_all_greeks,
)

# ─── Microstructure & Polygon NBBO utilities (explicitly exported) ────────────
from .microstructure import (
    compute_l1_metrics,
    compute_ofi,
    compute_trade_signed_volume,
    compute_vpin,
)

from .micro_features import (
    build_microstructure_features,
)

from .polygon_nbbo import (
    fetch_nbbo_quotes,
    fetch_trades,
)

__all__ = [
    # logging
    "write_status", "REST_CALLS", "REST_429",
    # http client
    "safe_fetch_polygon_data", "rate_limited", "fetch_option_greeks",
    # market data
    "reformat_candles", "fetch_premarket_early_data", "fetch_today_5m_candles",
    "fetch_latest_5m_candle", "fetch_historical_5m_candles",
    # analysis
    "calculate_breakout_prob", "calculate_recent_move_pct", "calculate_signal_persistence",
    "calculate_reversal_and_scope", "calculate_risk_reward", "calculate_time_of_day",
    "calculate_volume_ratio", "compute_skew_ratio", "compute_corr_deviation",
    "compute_rsi", "detect_yield_spike",
    # file I/O
    "append_signal_log", "load_snapshot", "save_snapshot",
    # greeks helpers
    "calculate_all_greeks",
    # microstructure
    "compute_l1_metrics", "compute_ofi", "compute_trade_signed_volume", "compute_vpin",
    "build_microstructure_features",
    # polygon nbbo
    "fetch_nbbo_quotes", "fetch_trades",
]
