import numpy as np


class RandomPolicy:
    def __init__(self, T: int, seed=None):
        self.T = T
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.random() < 1.0 / self.T)


class ThresholdPolicy:
    def __init__(self, u_min: int):
        self.u_min = int(u_min)

    def act(self, obs) -> int:
        u = obs[1]
        return int(u >= self.u_min)


class OptimalPolicy:
    TABLE = {
        1: {1: 0, 2: 0, 3: 0, 4: 1},
        2: {1: 0, 2: 0, 3: 1, 4: 1},
        3: {1: 0, 2: 0, 3: 1, 4: 1},
        4: {1: 1, 2: 1, 3: 1, 4: 1},
    }

    def act(self, obs) -> int:
        t = int(round(float(obs[0])))
        u_obs = float(obs[1])
        u = int(round(u_obs))
        u = max(1, min(4, u))
        return self.TABLE[t][u]