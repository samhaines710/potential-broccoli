import pytest
from pricing_engines import FallbackEngine, EngineType, QuantLibEngine, JAXEngine, DerivativesPricer

def test_fallback_engine_price_call():
    eng = FallbackEngine()
    price = eng.price(spot=100, strike=100, vol=0.2, maturity=1.0, rate=0.05, dividend=0.02, option_type="call")
    # Known BS price ≈10.45
    assert pytest.approx(price, rel=1e-2) == 10.45

def test_derivatives_pricer_quantlib_or_fallback():
    # If QuantLib installed, uses it; else fallback
    pr = DerivativesPricer(engine=EngineType.QUANTLIB.value)
    price = pr.price_black_scholes(100,100,0.2,1.0,0.05,0.02,"call")
    assert isinstance(price, float)

@pytest.mark.skipif(not JAXEngine, reason="JAX not installed")
def test_jax_engine():
    eng = JAXEngine()
    price = eng.price(100,100,0.2,1.0,0.05,0.02,"put")
    assert isinstance(price, float)
