import math
from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


class VaRCalculator:
    """
    Value‐at‐Risk calculator supporting parametric, historical, and Monte Carlo methods.
    """

    def __init__(self, pnl: pd.Series):
        """
        :param pnl: Series of P&L values (negative for losses).
        """
        self.pnl = pnl.dropna()

    def parametric_var(self, alpha: float = 0.05) -> float:
        """
        Parametric (Gaussian) VaR at level alpha.
        """
        mu = self.pnl.mean()
        sigma = self.pnl.std(ddof=1)
        z = norm.ppf(alpha)
        # Negative because losses are negative pnl
        return float(-(mu + sigma * z))

    def historical_var(self, alpha: float = 0.05) -> float:
        """
        Historical VaR at level alpha: the alpha‐quantile of losses.
        """
        return float(-np.percentile(self.pnl.values, alpha * 100))

    def monte_carlo_var(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_value: float,
        n_sims: int = 100_000,
        alpha: float = 0.05
    ) -> float:
        """
        Monte Carlo VaR: simulate returns and compute VaR.

        :param weights: portfolio weights array.
        :param mean_returns: array of expected returns.
        :param cov_matrix: covariance matrix of returns.
        :param portfolio_value: current value of the portfolio.
        :param n_sims: number of Monte Carlo simulations.
        :param alpha: VaR confidence level.
        """
        sims = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_sims)
        port_rets = sims.dot(weights)
        port_pnls = portfolio_value * port_rets
        return float(-np.percentile(port_pnls, alpha * 100))


class CVaRCalculator(VaRCalculator):
    """
    Conditional VaR (Expected Shortfall) calculator extending VaRCalculator.
    """

    def cvar(self, alpha: float = 0.05, method: str = "historical") -> float:
        """
        Compute CVaR (ES) at level alpha.
        :param alpha: confidence level.
        :param method: "historical" or "parametric".
        """
        if method == "historical":
            var = self.historical_var(alpha)
            tail = self.pnl[self.pnl <= -var]
            if tail.empty:
                return var
            return float(-tail.mean())
        elif method == "parametric":
            mu = self.pnl.mean()
            sigma = self.pnl.std(ddof=1)
            z = norm.ppf(alpha)
            pdf = norm.pdf(z)
            # ES formula for normal distribution
            es = -(mu - sigma * pdf / alpha)
            return float(es)
        else:
            raise ValueError("method must be 'historical' or 'parametric'")


class StressTestEngine:
    """
    Applies factor return shocks to compute stressed P&L.
    """

    def __init__(self, scenario: Dict[str, float]):
        """
        :param scenario: mapping factor name → shock amount (in return units).
        """
        self.scenario = scenario

    def apply_scenario(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Add shock to each factor in the returns DataFrame.
        """
        stressed = factor_returns.copy()
        for factor, shock in self.scenario.items():
            if factor in stressed.columns:
                stressed[factor] += shock
        return stressed

    def portfolio_pnl(
        self,
        exposures: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.Series:
        """
        Compute P&L series given exposures and factor returns.
        """
        aligned = exposures.reindex_like(factor_returns).fillna(0.0)
        pnl = (aligned * factor_returns).sum(axis=1)
        return pnl


class XVAEngine:
    """
    Counterparty‐credit and funding value adjustments (CVA, DVA, FVA).
    """

    def __init__(
        self,
        exposures: pd.Series,
        hazard_rates: pd.Series,
        recovery_rate: float = 0.4,
        discount_curve: Optional[Callable[[float], float]] = None,
        discount_rate: float = 0.0
    ):
        """
        :param exposures: time‐indexed series of positive (asset) / negative (liability) exposures.
        :param hazard_rates: time‐indexed series of counterparty default hazard rates.
        :param recovery_rate: fraction recovered on default (0–1).
        :param discount_curve: optional function t→discount factor; if None, uses flat rate.
        :param discount_rate: flat discount rate if discount_curve is None.
        """
        self.exposures = exposures.sort_index()
        self.hazard    = hazard_rates.sort_index()
        self.recovery  = recovery_rate
        self.discount_curve = discount_curve
        self.discount_rate  = discount_rate

        # Build survival probability curve
        times = self.hazard.index.to_numpy()
        rates = self.hazard.values
        dt = np.diff(np.insert(times, 0.0))
        cum_hazard = np.cumsum(rates * dt)
        surv = np.exp(-cum_hazard)
        self.surv = pd.Series(surv, index=times)

    def _df(self, t: float) -> float:
        """
        Discount factor at time t.
        """
        if self.discount_curve:
            return self.discount_curve(t)
        return math.exp(-self.discount_rate * t)

    def cva(self) -> float:
        """
        Compute unilateral CVA: expected loss due to counterparty default.
        """
        times = self.exposures.index.to_numpy()
        surv  = self.surv.values
        dpd   = -np.diff(np.insert(surv, 0.0))  # default probability density
        epe   = np.maximum(self.exposures.values, 0.0)
        dfs   = np.array([self._df(t) for t in times])
        return float((1 - self.recovery) * np.sum(dfs * epe * dpd))

    def dva(self) -> float:
        """
        Compute DVA: expected gain from own default.
        """
        times = self.exposures.index.to_numpy()
        surv  = self.surv.values
        dpd   = -np.diff(np.insert(surv, 0.0))
        ene   = np.maximum(-self.exposures.values, 0.0)
        dfs   = np.array([self._df(t) for t in times])
        return float((1 - self.recovery) * np.sum(dfs * ene * dpd))

    def fva(self, funding_spread: float) -> float:
        """
        Funding Value Adjustment: cost of funding uncollateralized exposure.
        """
        times = self.exposures.index.to_numpy()
        dt    = np.diff(np.insert(times, 0.0))
        eni   = np.maximum(-self.exposures.values, 0.0)
        dfs   = np.array([self._df(t) for t in times])
        return float(funding_spread * np.sum(dfs * eni * dt))
