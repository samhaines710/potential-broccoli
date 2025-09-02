"""Movement-type normalization utilities.

This module provides helper functions for normalizing movement_type
values emitted by the ML classifier into canonical string labels.

Our models may output integers (e.g., 0, 1, 2), NumPy integer scalars
(e.g., np.int64(0)), or strings ("0", "1", "2"). Downstream strategy
logic expects human-readable labels ("CALL", "PUT", "NEUTRAL").
Normalizing in one place ensures consistency and prevents default
fallbacks due to mismatched types.
"""

from __future__ import annotations

from typing import Union

__all__ = ["normalize_movement_type"]

# Try to include NumPy integer family for robust type handling.
try:
    import numpy as _np  # type: ignore
    _NUMERIC_TYPES = (int, _np.integer)  # covers np.int64, np.int32, etc.
except Exception:  # NumPy not installed — treat like plain ints only
    _NUMERIC_TYPES = (int,)

# Central map from classifier outputs to strategy labels. Accepts both
# numeric and string forms. Unknown inputs map to "NEUTRAL".
_MOVEMENT_MAP: dict = {
    0: "CALL",
    1: "PUT",
    2: "NEUTRAL",
    "0": "CALL",
    "1": "PUT",
    "2": "NEUTRAL",
    "CALL": "CALL",
    "PUT": "PUT",
    "NEUTRAL": "NEUTRAL",
}

def _as_canonical_key(x: object) -> object:
    """Convert input to a canonical lookup key for _MOVEMENT_MAP.

    - For numeric types (int, NumPy integer scalars), coerce to Python int.
    - Otherwise, return the original if it's already a key; else uppercase string.
    """
    if isinstance(x, _NUMERIC_TYPES):
        try:
            return int(x)  # np.int64(2) -> 2
        except Exception:
            pass
    if x in _MOVEMENT_MAP:
        return x
    s = str(x).strip().upper()
    return s

def normalize_movement_type(x: Union[int, str, None, object]) -> str:
    """Return a canonical movement_type label.

    Parameters
    ----------
    x : int | str | None | object
        Raw movement_type output from the classifier (may be an integer class
        ID, a string class ID, a NumPy integer scalar, or already a canonical
        label). Unknown/None defaults to "NEUTRAL".

    Returns
    -------
    str
        One of "CALL", "PUT", or "NEUTRAL".
    """
    if x is None:
        return "NEUTRAL"
    key = _as_canonical_key(x)
    return _MOVEMENT_MAP.get(key, "NEUTRAL")
