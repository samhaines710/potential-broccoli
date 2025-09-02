import json
from datetime import datetime, timedelta
import pytz
import pytest

from data_ingestion import HistoricalDataLoader, RealTimeDataStreamer, REALTIME_CANDLES, REALTIME_LOCK

class DummyLoader(HistoricalDataLoader):
    def _get(self, path, params):
        # return two bars, one inside range, one outside
        now = datetime(2021,1,1,10,0,tzinfo=pytz.UTC)
        inside = {"t": int(now.timestamp()*1000), "o":1, "h":2, "l":0.5, "c":1.5, "v":100}
        outside= {"t": int((now + timedelta(days=1)).timestamp()*1000), "o":1, "h":2, "l":0.5, "c":1.5, "v":100}
        return {"results":[inside, outside]}

def test_fetch_bars_filters_by_time(monkeypatch):
    loader = DummyLoader(api_key="test")
    start = datetime(2021,1,1,0,0,tzinfo=pytz.UTC)
    end   = datetime(2021,1,1,23,59,tzinfo=pytz.UTC)
    bars = loader.fetch_bars("TSLA", start, end)
    assert len(bars)==1
    assert bars[0]["o"]==1

def test_realtime_on_message(monkeypatch):
    streamer = RealTimeDataStreamer(api_key="test")
    # clear any existing bars
    REALTIME_CANDLES["TSLA"].clear()
    # simulate message payload
    msg = json.dumps({"ev":"AM","sym":"TSLA","t":1609459200000,"o":400,"h":410,"l":395,"c":405,"v":1000})
    streamer.on_message(None, msg)
    with REALTIME_LOCK:
        dq = REALTIME_CANDLES["TSLA"]
        assert len(dq)==1
        bar = dq[0]
        assert bar["open"]==400
        assert bar["close"]==405
