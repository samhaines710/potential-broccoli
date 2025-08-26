import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_df():
    """5m OHLCV sample DataFrame for backtests."""
    now = datetime(2021, 1, 1, 10, 0)
    times = [now + i * timedelta(minutes=5) for i in range(12)]
    data = {
        "open":   np.linspace(100, 110, 12),
        "high":   np.linspace(101, 111, 12),
        "low":    np.linspace(99, 109, 12),
        "close":  np.linspace(100.5, 110.5, 12),
        "volume": np.full(12, 1000),
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(times))
    return df

@pytest.fixture
def sample_pnl():
    """P&L series fixture for VaR tests."""
    return pd.Series([1.0, -2.0, 3.0, -4.0, 5.0])
