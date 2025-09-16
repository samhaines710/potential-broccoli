"""
Runtime feature sanitizer to prevent all-NaN groups from reaching the sklearn pipeline.

We normalize the microstructure features to a constant (0.0) when absent/NaN so the
pipeline never sees an all-NaN input vector for that group.

This is a runtime fix: it does not retrain or modify your persisted model.
"""

from __future__ import annotations

import math
from typing import Dict, Any

# Microstructure features (based on your logs)
MS_FEATURES: tuple[str, ...] = (
    "ms_spread_mean",
    "ms_depth_imbalance_mean",
    "ms_ofi_sum",
    "ms_signed_volume_sum",
    "ms_vpin",
)

MS_DEFAULTS = {k: 0.0 for k in MS_FEATURES}


def _is_nan_like(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    return False


def sanitize_features(row: Dict[str, Any], extra_fill: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a copy of `row` with ms_* present and finite (0.0 if missing/NaN)."""
    out = dict(row)

    for k, v in MS_DEFAULTS.items():
        x = out.get(k, v)
        if _is_nan_like(x):
            x = v
        out[k] = x

    if extra_fill:
        for k, v in extra_fill.items():
            x = out.get(k, v)
            if _is_nan_like(x):
                x = v
            out[k] = x

    return out
