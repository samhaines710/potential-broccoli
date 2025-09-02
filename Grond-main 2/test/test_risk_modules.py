import numpy as np
import pandas as pd
import pytest

from risk_modules import VaRCalculator, CVaRCalculator, StressTestEngine, XVAEngine

def test_historical_var(sample_pnl):
    var_calc = VaRCalculator(sample_pnl)
    hvar = var_calc.historical_var(alpha=0.2)
    # 20th percentile of [1,-2,3,-4,5] ≈ -4 -> historical_var returns positive
    assert hvar >= 0

def test_parametric_var_zero_series():
    ser = pd.Series([0.0]*100)
    var_calc = VaRCalculator(ser)
    assert var_calc.parametric_var(0.05) == pytest.approx(0.0)

def test_cvar_methods(sample_pnl):
    ccalc = CVaRCalculator(sample_pnl)
    h = ccalc.cvar(alpha=0.2, method="historical")
    p = ccalc.cvar(alpha=0.2, method="parametric")
    assert h >= 0 and p >= 0

def test_stress_test_engine():
    df = pd.DataFrame({"f1":[0.01,0.02], "f2":[-0.01,0.00]})
    engine = StressTestEngine({"f1":0.05})
    stressed = engine.apply_scenario(df)
    assert (stressed["f1"] - df["f1"] == 0.05).all()

def test_xva_engine():
    times = [0.0, 1.0, 2.0]
    expos = pd.Series([100.0, 50.0, -20.0], index=times)
    hazard= pd.Series([0.01,0.02,0.03], index=times)
    eng = XVAEngine(expos, hazard, recovery_rate=0.4, discount_rate=0.05)
    assert eng.cva() >= 0
    assert eng.dva() >= 0
    assert eng.fva(0.01) >= 0
