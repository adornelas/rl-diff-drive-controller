import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DifferentialRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32)

        self.R = 0.02
        self.L = 0.10
        self.k = 0.5
        self.alpha = 1.0

        self.wL = 0.0
        self.wR = 0.0
        self.step_count = 0
        self.max_steps = 500

        self.v_ref = 0.0
        self.w_ref = 0.0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wL = 0.0
        self.wR = 0.0
        self.step_count = 0
        self._update_reference()
        return self._get_obs(), {}

    def _update_reference(self):
        t = self.step_count * 0.1
        self.v_ref = 0.3 * np.sin(0.2 * t)
        self.w_ref = 1.0 * np.sin(0.1 * t)

    def _get_obs(self):
        v_meas = self.R * (self.wL + self.wR) / 2
        w_meas = self.R * (self.wR - self.wL) / self.L
        return np.array([self.v_ref, v_meas, self.w_ref, w_meas], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        self._update_reference()

        pwmL, pwmR = np.clip(action, -1.0, 1.0)
        wL_d = self.k * pwmL
        wR_d = self.k * pwmR

        self.wL += self.alpha * (wL_d - self.wL)
        self.wR += self.alpha * (wR_d - self.wR)

        v_meas = self.R * (self.wL + self.wR) / 2
        w_meas = self.R * (self.wR - self.wL) / self.L

        e_v = self.v_ref - v_meas
        e_w = self.w_ref - w_meas

        reward = -20.0 * (e_v ** 2) - 10.0 * (e_w ** 2)

        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
