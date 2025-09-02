import numpy as np
from signal_generation import BanditAllocator, AdaptiveHyperparamOptimizer
import pytest

def test_bandit_allocator():
    arms = ["A","B","C"]
    b = BanditAllocator(arms, epsilon=0.5)
    choice = b.select_arm()
    assert choice in arms
    b.update(choice, reward=1.0)
    assert b.counts[choice] == 1

def test_adaptive_hyperparam_optimizer_monkey_backtest(monkeypatch):
    # dummy backtest: return sum of params
    def backtest(params):
        return sum(params.values())
    space = {"x": {"low":0, "high":1}, "y":{"choices":[1,2,3]}}
    opt = AdaptiveHyperparamOptimizer(backtest, space, n_trials=3, direction="maximize")
    study = opt.optimize()
    assert study is not None
    assert hasattr(study, "best_trial")
