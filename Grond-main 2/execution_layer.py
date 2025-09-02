"""Execution layer providing simple slippage model and manual order execution.

This module defines a SlippageModel with static methods for fixed spread,
volume impact, and volatility impact, plus a ManualExecutor class that logs
and notifies a manual trade.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict

from utils.logging_utils import write_status


logger = logging.getLogger("execution_layer")
logger.setLevel(logging.INFO)


class SlippageModel:
    """Collection of static methods to estimate slippage components."""

    @staticmethod
    def fixed(spread: float) -> float:
        """Half the bid–ask spread."""
        return spread / 2.0

    @staticmethod
    def volume_impact(
        order_size: float,
        adv: float,
        impact_coefficient: float = 0.1,
        exponent: float = 0.6,
    ) -> float:
        """
        Market impact based on participation rate.

        Parameters
        ----------
        order_size : float
            Number of shares/contracts being traded.
        adv : float
            Average daily volume for the instrument.
        impact_coefficient : float, optional
            Scaling factor for the volume impact.
        exponent : float, optional
            Exponent used in the volume impact calculation.

        Returns
        -------
        float
            Estimated impact cost.
        """
        participation = order_size / max(adv, 1e-6)
        return impact_coefficient * (participation ** exponent)

    @staticmethod
    def volatility_impact(volatility: float, vol_coeff: float = 0.5) -> float:
        """Impact proportional to volatility."""
        return vol_coeff * volatility

    @classmethod
    def total(
        cls,
        spread: float,
        order_size: float,
        adv: float,
        volatility: float,
        vol_coeff: float = 0.5,
        impact_coeff: float = 0.1,
        exponent: float = 0.6,
    ) -> float:
        """
        Sum of fixed, volume, and volatility impacts.
        """
        return (
            cls.fixed(spread)
            + cls.volume_impact(
                order_size,
                adv,
                impact_coefficient=impact_coeff,
                exponent=exponent,
            )
            + cls.volatility_impact(volatility, vol_coeff=vol_coeff)
        )


class ManualExecutor:
    """
    Executor that emits a manual signal (e.g., via Telegram) and logs it.
    """

    def __init__(self, notify_fn: Callable[[str], None]) -> None:
        """
        Initialize the executor with a notification function.

        Parameters
        ----------
        notify_fn : Callable[[str], None]
            A function to call with a markdown-formatted message.
        """
        self.notify = notify_fn

    def place_order(
        self,
        ticker: str,
        size: float,
        side: str = "buy",
    ) -> Dict[str, Any]:
        """
        Execute a manual signal by logging, notifying, and returning a report.

        Parameters
        ----------
        ticker : str
            Symbol to trade.
        size : float
            Number of contracts/shares.
        side : str, optional
            "buy" or "sell".

        Returns
        -------
        Dict[str, Any]
            A report describing the order.
        """
        report: Dict[str, Any] = {
            "ticker": ticker,
            "side": side.lower(),
            "size": size,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "note": "Manual execution required",
        }

        # Log locally
        msg = f"MANUAL EXECUTION → {side.upper()} {size} {ticker}"
        logger.info(msg)
        write_status(msg)

        # Send Telegram (or other) notification
        try:
            text = (
                f"📋 *MANUAL SIGNAL* — {side.upper()} {size} {ticker}"
                f" @ {datetime.utcnow().strftime('%H:%M')} UTC"
            )
            self.notify(text)
        except Exception as exc:
            logger.warning(f"Failed to send notification: {exc}")

        return report

    def generate_execution_report(self) -> dict[str, float]:
        """Return a summary of fills and current account value."""
        report = {
            "cash": self.cash,
            "positions": self.positions.copy(),
            "account_value": self.cash + sum(self.positions.values()),
        }

        # send Telegram or Slack message
        try:
            self.notify(f"Execution report: {report}")
        except Exception as exc:
            logger.warning(f"Failed to send notification: {exc}")

        return report
