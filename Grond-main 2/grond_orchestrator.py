#!/usr/bin/env python3
"""
Central orchestrator for the Grond trading system.

- Singleton lock (file lock).
- Movement normalization (ints/np.int64 → CALL/PUT/NEUTRAL).
- Exploration gating (NEUTRAL doesn't trade unless ALLOW_NEUTRAL_BANDIT=1).
- Safety clamp against accidental NEUTRAL execution.
- Centralized JSON logging configured once at startup.
- Always includes the 'source' feature for the ML pipeline.
"""

from __future__ import annotations

import os
import fcntl
import logging
import time
from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
from prometheus_client import Counter

from config import (
    TICKERS,
    BANDIT_EPSILON,
    ORDER_SIZE,
    SERVICE_NAME,
    tz,
)
from monitoring_ops import start_monitoring_server

from data_ingestion import (
    HistoricalDataLoader,
    RealTimeDataStreamer,
    REALTIME_CANDLES,
    REALTIME_LOCK,
)
from pricing_engines import DerivativesPricer
# === FIX: use HCBCClassifier (calibrated XGBoost bundle) instead of MLClassifier ===
from ml_classifier import HCBCClassifier
from strategy_logic import StrategyLogic
from signal_generation import BanditAllocator
from execution_layer import ManualExecutor

from utils.logging_utils import configure_logging, write_status
from utils import (
    reformat_candles,
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_time_of_day,
    calculate_volume_ratio,
    compute_rsi,
    compute_corr_deviation,
    compute_skew_ratio,
    detect_yield_spike,
    fetch_option_greeks,
    append_signal_log,
)
from utils.movement import normalize_movement_type
from utils.messaging import send_telegram

# configure logging once
configure_logging()

# ─── Prometheus metrics ────────────────────────────────────────────────────────
SIGNALS_PROCESSED = Counter(
    f"{SERVICE_NAME}_signals_total",
    "Number of signals generated",
    ["ticker", "movement_type"],
)
EXECUTIONS = Counter(
    f"{SERVICE_NAME}_executions_total",
    "Number of executions placed",
    ["ticker", "action"],
)

# ─── Exploration controls ─────────────────────────────────────────────────────
_EPSILON = float(os.getenv("BANDIT_EPSILON", str(BANDIT_EPSILON)))
_ALLOW_NEUTRAL_BANDIT = os.getenv("ALLOW_NEUTRAL_BANDIT", "0").lower() in {"1", "true", "yes"}

# ─── Singleton lock config ────────────────────────────────────────────────────
_LOCK_FILE = os.getenv("GROND_LOCK_FILE", "/tmp/grond_orchestrator.lock")
_INSTANCE_ID = os.getenv("INSTANCE_ID", f"pid-{os.getpid()}")

# ─── HCBC bundle/config (H key/value) ─────────────────────────────────────────
_MODEL_PATH = os.getenv("ML_MODEL_PATH", "Resources/xgb_hcbc.bundle.joblib")
_H_KEY = os.getenv("HCBC_H_KEY", "H")
_H_DEFAULT = int(os.getenv("HCBC_DEFAULT_H", "15"))  # horizon in minutes


class GrondOrchestrator:
    def __init__(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

        # Acquire singleton lock
        try:
            self._lock_fh = open(_LOCK_FILE, "w")
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fh.truncate(0)
            self._lock_fh.write(_INSTANCE_ID)
            self._lock_fh.flush()
            write_status(f"[{_INSTANCE_ID}] Acquired singleton lock: {_LOCK_FILE}")
        except BlockingIOError:
            write_status(f"[{_INSTANCE_ID}] Another instance holds {_LOCK_FILE}. Exiting.")
            raise SystemExit(0)

        write_status(f"[{_INSTANCE_ID}] Starting monitoring server…")
        start_monitoring_server()

        # Data & engines
        self.hist_loader = HistoricalDataLoader()
        self.rt_stream = RealTimeDataStreamer()
        self.rt_stream.start()

        self.pricer = DerivativesPricer()
        # === FIX: instantiate HCBCClassifier with bundle path ===
        self.classifier = HCBCClassifier(_MODEL_PATH)
        self.logic = StrategyLogic()
        self.bandit = BanditAllocator(
            list(self.logic.logic_branches.keys()),
            epsilon=_EPSILON,
        )
        self.executor = ManualExecutor(notify_fn=send_telegram)

        write_status(
            f"[{_INSTANCE_ID}] Orchestrator ready (epsilon={_EPSILON}, "
            f"allow_neutral_bandit={_ALLOW_NEUTRAL_BANDIT}, model='{_MODEL_PATH}', H={_H_DEFAULT})"
        )

    def _decide_movement(self, base_mv: str) -> Tuple[str, bool]:
        """Apply ε-greedy exploration with gating for NEUTRAL."""
        explored = False
        mv = base_mv
        roll = np.random.rand()

        if base_mv in ("CALL", "PUT"):
            if roll < _EPSILON:
                cand = self.bandit.select_arm()
                explored = (cand != base_mv)
                mv = cand
        elif base_mv == "NEUTRAL":
            if _ALLOW_NEUTRAL_BANDIT and roll < _EPSILON:
                cand = self.bandit.select_arm()
                mv = "CALL" if cand == "CALL" else ("PUT" if cand == "PUT" else "NEUTRAL")
                explored = (mv != base_mv)
            else:
                mv = "NEUTRAL"
        else:
            mv = "NEUTRAL"

        return mv, explored

    def run(self) -> None:
        write_status(f"[{_INSTANCE_ID}] Entering main loop.")
        while True:
            now = datetime.now(tz)
            for ticker in TICKERS:
                with REALTIME_LOCK:
                    raw = list(REALTIME_CANDLES.get(ticker, []))
                if len(raw) < 5:
                    continue

                bars = reformat_candles(raw)

                # feature computation
                breakout   = calculate_breakout_prob(bars)
                recent_pct = calculate_recent_move_pct(bars)
                vol_ratio  = calculate_volume_ratio(bars)
                rsi_val    = compute_rsi(bars)
                corr_dev   = compute_corr_deviation(ticker)
                skew       = compute_skew_ratio(ticker)
                ys2        = detect_yield_spike("2year")
                ys10       = detect_yield_spike("10year")
                ys30       = detect_yield_spike("30year")
                tod        = calculate_time_of_day(now)
                greeks     = fetch_option_greeks(ticker)

                theta_raw = float(greeks.get("theta", 0.0))
                theta_day = theta_raw
                theta_5m  = theta_day / 78.0

                # include 'source' from Greeks for ML pipeline
                source_val = str(greeks.get("source", "fallback"))

                features: Dict[str, Union[float, str, int]] = {
                    "breakout_prob":      breakout,
                    "recent_move_pct":    recent_pct,
                    "volume_ratio":       vol_ratio,
                    "rsi":                rsi_val,
                    "corr_dev":           corr_dev,
                    "skew_ratio":         skew,
                    "yield_spike_2year":  ys2,
                    "yield_spike_10year": ys10,
                    "yield_spike_30year": ys30,
                    "time_of_day":        tod,
                    "delta":  float(greeks.get("delta", 0.0)),
                    "gamma":  float(greeks.get("gamma", 0.0)),
                    "theta":  theta_raw,
                    "vega":   float(greeks.get("vega", 0.0)),
                    "rho":    float(greeks.get("rho", 0.0)),
                    "vanna":  float(greeks.get("vanna", 0.0)),
                    "vomma":  float(greeks.get("vomma", 0.0)),
                    "charm":  float(greeks.get("charm", 0.0)),
                    "veta":   float(greeks.get("veta", 0.0)),
                    "speed":  float(greeks.get("speed", 0.0)),
                    "zomma":  float(greeks.get("zomma", 0.0)),
                    "color":  float(greeks.get("color", 0.0)),
                    "implied_volatility": float(greeks.get("implied_volatility", 0.0)),
                    "theta_day": theta_day,
                    "theta_5m":  theta_5m,
                    "source": source_val,
                    # === FIX: inject HCBC horizon feature ===
                    _H_KEY: _H_DEFAULT,
                }

                cls_out = self.classifier.classify(features)
                base_mv = normalize_movement_type(cls_out.get("movement_type"))

                mv, explored = self._decide_movement(base_mv)
                if explored:
                    write_status(f"[{_INSTANCE_ID}] Exploration override: base={base_mv} → chosen={mv}")

                cls_out["movement_type"] = mv
                context = {**features, **cls_out}

                strat = self.logic.execute_strategy(mv, context)
                SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()

                # safety clamp: NEUTRAL never executes if neutral-bandit is disabled
                action = strat.get("action", "REVIEW")
                if mv == "NEUTRAL" and not _ALLOW_NEUTRAL_BANDIT:
                    if action not in ("AVOID", "REVIEW"):
                        write_status(f"[{_INSTANCE_ID}] Safety clamp: preventing execution on NEUTRAL (ticker={ticker})")
                    action = "REVIEW"

                if action not in ("AVOID", "REVIEW"):
                    side = "buy" if action.endswith("_CALL") else "sell"
                    self.executor.place_order(ticker=ticker, size=ORDER_SIZE, side=side)
                    EXECUTIONS.labels(ticker=ticker, action=action).inc()
                    append_signal_log({
                        "time": now.isoformat(),
                        "ticker": ticker,
                        "movement_type": mv,
                        "action": action,
                        **greeks,
                    })

            time.sleep(300)


if __name__ == "__main__":
    write_status(f"[{_INSTANCE_ID}] Launching Grond Orchestrator…")
    orchestrator = GrondOrchestrator()
    orchestrator.run()
