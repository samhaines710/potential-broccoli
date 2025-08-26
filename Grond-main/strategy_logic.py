"""StrategyLogic maps movement_type signals to trading instructions.

This revised version defensively normalizes the input movement_type
before dispatching to a handler.  Unknown or malformed values no
longer silently fall through; instead they are converted via
normalize_movement_type so that numeric class IDs (0/1/2) and
string forms ("0","1","2") yield "CALL", "PUT", or
"NEUTRAL".  This eliminates warnings such as "No handler for
movement_type '2'; defaulting to REVIEW".
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict

from utils.movement import normalize_movement_type

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StrategyLogic:
    """
    Maps ML‑predicted movement_type to trade instructions.
    Supports optional per‑signal overrides via strategy_profiles.json.
    """

    def __init__(self, profiles_path: str = "strategy_profiles.json"):
        """
        :param profiles_path: Path to JSON file containing per‑movement_type overrides.
        """
        self.profiles_path = profiles_path
        self.profiles = self._load_profiles()
        self.logic_branches = {
            "CALL":    self.handle_call,
            "PUT":     self.handle_put,
            "NEUTRAL": self.handle_neutral,
        }

    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Load JSON overrides for each movement_type.
        """
        if not os.path.exists(self.profiles_path):
            logger.debug("No strategy_profiles.json found; skipping overrides")
            return {}
        try:
            with open(self.profiles_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded strategy profiles from {self.profiles_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load strategy profiles: {e}")
            return {}

    def execute_strategy(self, movement_type: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selects and runs the handler for the given movement_type,
        then merges in any JSON‑defined overrides.

        The incoming movement_type may be an integer class ID or a
        string; we normalize it to one of the canonical labels.

        :param movement_type: raw output from the classifier (int/str)
        :param context: dictionary of features + classifier output
        :return: dict with keys 'action','exit','ttl','note', plus any overrides
        """
        mt = normalize_movement_type(movement_type)
        logger.debug(f"Executing strategy for normalized movement_type={mt} (raw={movement_type!r})")
        # update context to reflect normalized type
        context = dict(context)
        context["movement_type"] = mt
        handler = self.logic_branches.get(mt, self.default_strategy)
        base_decision = handler(context)

        override = self.profiles.get(mt, {})
        if override:
            logger.info(f"Applying overrides for {mt}: {override}")
        decision = {**base_decision, **override}

        logger.debug(f"Strategy decision for {mt}: {decision}")
        return decision

    def handle_call(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handler when the model predicts a bullish (CALL) movement.
        """
        return {
            "action": "BUY_CALL",
            "exit":   "TRAIL_VOL",
            "ttl":    "SHORT",
            "note":   "ML predicted bullish movement",
        }

    def handle_put(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handler when the model predicts a bearish (PUT) movement.
        """
        return {
            "action": "BUY_PUT",
            "exit":   "TRAIL_VOL",
            "ttl":    "SHORT",
            "note":   "ML predicted bearish movement",
        }

    def handle_neutral(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handler when the model predicts no clear movement.
        """
        return {
            "action": "REVIEW",
            "exit":   "N/A",
            "ttl":    "N/A",
            "note":   "ML predicted neutral movement",
        }

    def default_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback for unknown movement_type.  Should rarely be hit thanks
        to normalization, but will handle any unforeseen labels.
        """
        mv = context.get("movement_type", "UNKNOWN")
        logger.warning(f"No handler for movement_type '{mv}'; defaulting to REVIEW")
        return {
            "action": "REVIEW",
            "exit":   "N/A",
            "ttl":    "N/A",
            "note":   f"No strategy defined for '{mv}'",
        }