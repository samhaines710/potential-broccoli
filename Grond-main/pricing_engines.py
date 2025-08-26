"""Pricing engines for Black‑Scholes options.

This module provides three implementations: one using QuantLib's analytic
solution, one using a JAX‑jitted function, and a pure‑Python fallback.
A dispatcher class chooses the appropriate engine based on availability.
"""

from __future__ import annotations

import math
from enum import Enum

# Attempt to import optional dependencies
try:
    import QuantLib as ql  # type: ignore
    _HAS_QUANTLIB = True
except ImportError:
    _HAS_QUANTLIB = False

try:
    import jax.numpy as jnp  # type: ignore
    from jax import jit  # type: ignore
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


class EngineType(str, Enum):
    """Enumeration of pricing engine types."""

    QUANTLIB = "QuantLib"
    JAX = "JAX"
    FALLBACK = "Fallback"


class QuantLibEngine:
    """Pricer using QuantLib's analytic Black-Scholes engine."""

    def __init__(self) -> None:
        if not _HAS_QUANTLIB:
            raise ImportError("QuantLib is not installed")
        self.calendar = ql.NullCalendar()
        self.day_count = ql.Actual365Fixed()
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        self.today = today

    def price(
        self,
        spot: float,
        strike: float,
        vol: float,
        maturity: float,
        rate: float,
        dividend: float,
        option_type: str,
    ) -> float:
        """Return the Black-Scholes price using QuantLib."""
        settlement = self.today
        rf = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement, rate, self.day_count)
        )

        dq = ql.YieldTermStructureHandle(
            ql.FlatForward(settlement, dividend, self.day_count)
        )
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(settlement, self.calendar, vol, self.day_count)
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            dq,
            rf,
            vol_ts,
        )
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type.lower() == "call" else ql.Option.Put,
            strike,
        )
        days_to_expiry = max(int(maturity * 365), 1)
        expiry = settlement + days_to_expiry
        option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return option.NPV()


if _HAS_JAX:
    @jit  # type: ignore[misc]
    def _jax_bs_price(
        spot: float,
        strike: float,
        vol: float,
        maturity: float,
        rate: float,
        dividend: float,
        is_call: int,
    ):
        sqrtT = jnp.sqrt(maturity)
        d1 = (
            jnp.log(spot / strike)
            + (rate - dividend + 0.5 * vol ** 2) * maturity
        ) / (vol * sqrtT)
        d2 = d1 - vol * sqrtT

        def _cdf(x):
            return 0.5 * (1.0 + jnp.erf(x / jnp.sqrt(2.0)))

        c1, c2 = _cdf(d1), _cdf(d2)
        df_r = jnp.exp(-rate * maturity)
        df_d = jnp.exp(-dividend * maturity)
        call = spot * df_d * c1 - strike * df_r * c2
        put = strike * df_r * (1.0 - c2) - spot * df_d * (1.0 - c1)
        return jnp.where(is_call, call, put)


class JAXEngine:
    """Pricer using a JAX-accelerated implementation."""

    def __init__(self) -> None:
        if not _HAS_JAX:
            raise ImportError("JAX is not installed")
        self._fn = _jax_bs_price

    def price(
        self,
        spot: float,
        strike: float,
        vol: float,
        maturity: float,
        rate: float,
        dividend: float,
        option_type: str,
    ) -> float:
        is_call = 1 if option_type.lower() == "call" else 0
        return float(self._fn(spot, strike, vol, maturity, rate, dividend, is_call))


class FallbackEngine:
    """Pure-Python Black-Scholes pricer."""

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def price(
        self,
        spot: float,
        strike: float,
        vol: float,
        maturity: float,
        rate: float,
        dividend: float,
        option_type: str,
    ) -> float:
        """
        Compute the Black-Scholes price. If maturity or vol is non-positive,
        return intrinsic value.
        """
        if maturity <= 0.0 or vol <= 0.0:
            return (
                max(spot - strike, 0.0)
                if option_type.lower() == "call"
                else max(strike - spot, 0.0)
            )
        sqrtT = math.sqrt(maturity)
        d1 = (
            math.log(spot / strike)
            + (rate - dividend + 0.5 * vol ** 2) * maturity
        ) / (vol * sqrtT)
        d2 = d1 - vol * sqrtT
        c1 = self._norm_cdf(d1)
        c2 = self._norm_cdf(d2)
        df_r = math.exp(-rate * maturity)
        df_d = math.exp(-dividend * maturity)
        if option_type.lower() == "call":
            return spot * df_d * c1 - strike * df_r * c2
        else:
            return strike * df_r * (1.0 - c2) - spot * df_d * (1.0 - c1)


class DerivativesPricer:
    """Dispatching pricer that selects an engine based on availability."""

    def __init__(self, engine: str = EngineType.QUANTLIB.value) -> None:
        et = EngineType(engine)
        if et == EngineType.QUANTLIB:
            self.engine = QuantLibEngine()
        elif et == EngineType.JAX:
            self.engine = JAXEngine()
        else:
            self.engine = FallbackEngine()

    def price_black_scholes(
        self,
        spot: float,
        strike: float,
        vol: float,
        maturity: float,
        rate: float,
        dividend: float,
        option_type: str,
    ) -> float:
        """Compute an option price using the selected engine."""
        return self.engine.price(
            spot,
            strike,
            vol,
            maturity,
            rate,
            dividend,
            option_type,
        )
