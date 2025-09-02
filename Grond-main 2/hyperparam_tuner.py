"""Hyperparameter tuning utilities for movement logic thresholds.

This script loads the current movement configuration, runs multiple backtest
trials across a defined parameter space using AdaptiveHyperparamOptimizer,
and writes the best-performing configuration back to disk.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd

from data_ingestion import HistoricalDataLoader
from backtesting_framework import run_backtest_backtrader
from signal_generation import AdaptiveHyperparamOptimizer

CONFIG_PATH = "movement_logic_config.json"
TMP_CONFIG_PATH = "movement_logic_config.tmp.json"


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load a JSON config if it exists, otherwise return an empty dict."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Write the config dictionary to a JSON file."""
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def backtest_with_config(cfg: Dict[str, Any], symbol: str = "TSLA") -> float:
    """
    Fetch 30 days of historical 5‑minute bars for `symbol`, run a backtest,
    and return the Sharpe ratio.
    """
    end = datetime.now()
    start = end - timedelta(days=30)

    # Fetch data
    loader = HistoricalDataLoader()
    raw = loader.fetch_bars(symbol, start, end)

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
            }
            for b in raw
        ]
    )
    df.index = pd.to_datetime([b["t"] for b in raw], unit="ms")

    # Save temporary config so classifier picks up thresholds
    save_config(cfg, TMP_CONFIG_PATH)

    # Run backtest
    perf = run_backtest_backtrader(df, symbol)

    # Clean up temp file
    os.remove(TMP_CONFIG_PATH)

    return perf.get("Sharpe Ratio", 0.0)


def make_trial_config(base: Dict[str, Any], params: Dict[str, float]) -> Dict[str, Any]:
    """
    Overlay ``params`` onto the ``base`` config structure. Each key in params maps
    to one threshold in the resulting JSON. Greek bands are symmetric.
    """
    cfg = base.copy()

    # --- safe_greek_bands (symmetric) ---
    for greek in ("vanna", "vomma", "charm", "veta", "speed", "zomma", "color", "rho"):
        key = f"safe_{greek}"
        val = params[key]
        cfg.setdefault("safe_greek_bands", {})[greek] = [-val, val]

    # --- tier1_greek_bands ---
    for greek in ("vanna", "vomma", "charm", "veta"):
        key = f"tier1_{greek}"
        val = params[key]
        cfg.setdefault("tier1_greek_bands", {})[greek] = [-val, val]

    # --- vol/hedge bands ---
    for greek in ("vanna", "veta"):
        vkey = f"vol_{greek}"
        hkey = f"hedge_{greek}"
        vol_val = params[vkey]
        hedge_val = params[hkey]
        cfg.setdefault("volatility_greek_bands", {})[greek] = [-vol_val, vol_val]
        cfg.setdefault("hedging_greek_bands", {})[greek] = [-hedge_val, hedge_val]

    # --- RSI & other singles ---
    cfg["rsi_overbought"] = params["rsi_overbought"]
    cfg["rsi_oversold"] = params["rsi_oversold"]
    cfg["corr_dev_threshold"] = params["corr_dev"]
    cfg["skew_extreme"] = params["skew_extreme"]
    cfg["yield_spike_threshold"] = params["yield_spike"]

    # --- micro thresholds ---
    cfg.setdefault("micro_threshold", {})
    cfg["micro_threshold"]["default"] = params["micro_default"]
    cfg["micro_threshold"]["high_vol"] = params["micro_high_vol"]
    cfg["micro_threshold"]["low_vol"] = params["micro_low_vol"]

    # --- session thresholds ---
    for sess in ("MORNING", "MIDDAY", "AFTERNOON"):
        base_s = cfg.setdefault("sessions", {}).get(sess, {})
        base_s["breakout_low"] = params[f"{sess}_b_low"]
        base_s["breakout_high"] = params[f"{sess}_b_high"]
        base_s["vol_thr"] = params[f"{sess}_vol_thr"]
        base_s["delta_thr"] = params[f"{sess}_delta_thr"]
        cfg["sessions"][sess] = base_s

    return cfg


# --- Define hyperparameter search space ---
param_space: Dict[str, Dict[str, float]] = {
    # safe greek bands
    **{
        f"safe_{g}": {"low": 0.05, "high": 1.0, "step": 0.05}
        for g in ("vanna", "vomma", "charm", "veta", "speed", "zomma", "color", "rho")
    },
    # tier1 greek bands
    **{
        f"tier1_{g}": {"low": 0.01, "high": 0.2, "step": 0.01}
        for g in ("vanna", "vomma", "charm", "veta")
    },
    # vol/hedge bands
    "vol_vanna": {"low": 0.1, "high": 2.0, "step": 0.1},
    "hedge_vanna": {"low": 0.1, "high": 1.0, "step": 0.1},
    "vol_veta": {"low": 0.1, "high": 2.0, "step": 0.1},
    "hedge_veta": {"low": 0.1, "high": 1.0, "step": 0.1},
    # RSI & others
    "rsi_overbought": {"low": 50, "high": 90, "step": 5},
    "rsi_oversold": {"low": 10, "high": 50, "step": 5},
    "corr_dev": {"low": 0.5, "high": 3.0, "step": 0.1},
    "skew_extreme": {"low": 0.1, "high": 1.0, "step": 0.1},
    "yield_spike": {"low": 0.05, "high": 0.5, "step": 0.05},
    # micro thresholds
    "micro_default": {"low": 0.1, "high": 0.5, "step": 0.05},
    "micro_high_vol": {"low": 0.2, "high": 1.0, "step": 0.1},
    "micro_low_vol": {"low": 0.05, "high": 0.3, "step": 0.05},
    # session thresholds for breakout low and high
    **{
        f"{sess}_{fld}": {
            "low": 5 if "b_low" in fld else 10,
            "high": 30 if "b_low" in fld else 60,
            "step": 5,
        }
        for sess in ("MORNING", "MIDDAY", "AFTERNOON")
        for fld in ("b_low", "b_high")
    },
    # session vol thresholds
    **{
        f"{sess}_vol_thr": {"low": 0.5, "high": 3.0, "step": 0.1}
        for sess in ("MORNING", "MIDDAY", "AFTERNOON")
    },
    # session delta thresholds
    **{
        f"{sess}_delta_thr": {"low": 0.1, "high": 1.0, "step": 0.1}
        for sess in ("MORNING", "MIDDAY", "AFTERNOON")
    },
}


def tune(n_trials: int = 50) -> None:
    """Run hyperparameter tuning for a specified number of trials."""
    base_cfg = load_config()

    def trial_fn(trial_params: Dict[str, float]) -> float:
        # Build a trial-specific config
        cfg = make_trial_config(base_cfg, trial_params)
        # Return Sharpe ratio from backtest
        return backtest_with_config(cfg)

    tuner = AdaptiveHyperparamOptimizer(
        backtest_func=trial_fn,
        param_space=param_space,
        n_trials=n_trials,
        direction="maximize",
    )
    study = tuner.optimize()
    best = study.best_params

    # Merge best back into your real config
    final_cfg = make_trial_config(base_cfg, best)
    save_config(final_cfg, CONFIG_PATH)
    print("✅ Tuning complete. Best params:", best)
    print(f"Updated thresholds written to {CONFIG_PATH}")


if __name__ == "__main__":
    tune(n_trials=100)
