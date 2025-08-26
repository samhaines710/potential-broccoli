import numpy as np
import optuna
from collections import defaultdict
from typing import Callable, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from ml_classifier import MLClassifier
from strategy_logic import StrategyLogic

class AdaptiveHyperparamOptimizer:
    def __init__(
        self,
        backtest_func: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, Any],
        n_trials: int = 50,
        direction: str = "maximize"
    ):
        self.backtest = backtest_func
        self.param_space = param_space
        self.n_trials = n_trials
        self.study = optuna.create_study(direction=direction)

    def _suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for name, opts in self.param_space.items():
            if "choices" in opts:
                params[name] = trial.suggest_categorical(name, opts["choices"])
            else:
                low, high = opts["low"], opts["high"]
                step = opts.get("step")
                if step:
                    params[name] = trial.suggest_float(name, low, high, step=step)
                else:
                    params[name] = trial.suggest_float(name, low, high)
        return params

    def optimize(self) -> optuna.Study:
        def objective(trial):
            params = self._suggest(trial)
            return self.backtest(params)
        self.study.optimize(objective, n_trials=self.n_trials)
        return self.study

class BanditAllocator:
    def __init__(self, arms: List[str], epsilon: float = 0.1):
        self.epsilon = epsilon
        self.arms = arms
        self.counts = defaultdict(int)
        self.values = defaultdict(float)

    def select_arm(self) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arms)
        avg = {a: self.values[a] / max(1, self.counts[a]) for a in self.arms}
        return max(avg, key=avg.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm] / self.counts[arm])

class TradingEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(
        self,
        feature_gen: Callable[[str], Dict[str, Any]],
        ml_classifier: MLClassifier,
        strat_logic: StrategyLogic,
        tickers: List[str],
        horizon_steps: int = 3
    ):
        super().__init__()
        self.feat       = feature_gen
        self.classifier = ml_classifier
        self.logic      = strat_logic
        self.tickers    = tickers
        self.horizon    = horizon_steps

        self.arms = list(self.logic.logic_branches.keys())
        self.action_space = spaces.Discrete(len(self.arms))

        sample = self.feat(self.tickers[0])
        obs_dim = len(sample)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_ticker = None
        self.current_step   = 0

    def reset(self):
        self.current_ticker = np.random.choice(self.tickers)
        self.current_step   = 0
        feats = self.feat(self.current_ticker)
        return np.array(list(feats.values()), dtype=np.float32)

    def step(self, action: int):
        mv    = self.arms[action]
        feats = self.feat(self.current_ticker)
        cls   = self.classifier.classify(feats)
        strat = self.logic.execute_strategy(mv, {**feats, **cls})
        reward = self._simulate_pnl(strat, self.current_ticker)

        self.current_step += 1
        done = self.current_step >= self.horizon
        next_feats = self.feat(self.current_ticker)
        obs = np.array(list(next_feats.values()), dtype=np.float32)
        return obs, reward, done, {}

    def _simulate_pnl(self, strat: Dict[str, Any], ticker: str) -> float:
        return np.random.normal(0, 1)

    def render(self, mode='human'):
        pass

class RLAgent:
    def __init__(self, env: gym.Env, **ppo_kwargs):
        self.env = env
        self.model = PPO("MlpPolicy", env, verbose=0, **ppo_kwargs)

    def train(self, timesteps: int = 100_000):
        self.model.learn(total_timesteps=timesteps)

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
