import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ApartmentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T: int, K: int, seed=None, noise_std: float = 0.0):
        super().__init__()
        self.T = int(T)
        self.K = int(K)
        self.noise_std = float(noise_std)

        self.action_space = spaces.Discrete(2)

        low = np.array([1.0, -np.inf if noise_std > 0 else 1.0], dtype=np.float32)
        high = np.array([self.T, np.inf if noise_std > 0 else self.K], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._np_random = None
        self.reset(seed=seed)

    def _draw_quality(self):
        return int(self._np_random.integers(1, self.K + 1))

    def _make_obs(self):
        if self.noise_std > 0.0:
            noisy = self.U_t + self._np_random.normal(0.0, self.noise_std)
            return np.array([self.t, noisy], dtype=np.float32)
        return np.array([self.t, self.U_t], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        self.t = 1
        self.U_t = self._draw_quality()
        self.done = False
        return self._make_obs(), {"true_U": self.U_t}

    def step(self, action: int):
        assert not self.done, "step() called on terminated episode"
        assert action in (0, 1)

        terminated = False
        truncated = False
        reward = 0.0
        info = {"true_U": self.U_t}

        if action == 1:
            reward = float(self.U_t)
            terminated = True
            self.done = True
        else:
            if self.t == self.T:
                reward = 0.0
                terminated = True
                self.done = True
            else:
                self.t += 1
                self.U_t = self._draw_quality()
                info["true_U"] = self.U_t

        obs = self._make_obs()
        return obs, reward, terminated, truncated, info