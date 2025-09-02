import pytest
import threading
import time

from grond_orchestrator import GrondOrchestrator

def test_orchestrator_init():
    orch = GrondOrchestrator()
    assert hasattr(orch, "run")
    assert hasattr(orch, "executor")

def test_run_one_iteration(monkeypatch):
    # monkeypatch sleep to stop after first loop
    calls = []
    def fake_sleep(sec):
        calls.append(sec)
        raise KeyboardInterrupt
    monkeypatch.setattr(time, "sleep", fake_sleep)

    orch = GrondOrchestrator()
    # run should raise KeyboardInterrupt after first .sleep()
    with pytest.raises(KeyboardInterrupt):
        orch.run()
    # ensure it attempted to sleep for 300s
    assert 300 in calls
