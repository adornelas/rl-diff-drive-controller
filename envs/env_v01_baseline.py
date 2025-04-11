import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DifferentialRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Ações: PWM esquerdo e direito normalizados entre [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Estado: [v_ref, v_meas, w_ref, w_meas]
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32)

        # Parâmetros físicos do robô
        self.R = 0.02   # raio da roda (m)
        self.L = 0.10   # distância entre rodas (m)
        self.k = 0.5    # pwm -> velocidade angular ideal (rad/s)
        self.alpha = 0.1  # fator de inércia (0 = lento, 1 = resposta instantânea)

        # Estado interno dos motores (velocidades atuais das rodas)
        self.wL = 0.0
        self.wR = 0.0

        self.step_count = 0
        self.max_steps = 500  # limite por episódio

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wL = 0.0
        self.wR = 0.0
        self.step_count = 0
        self._update_reference()
        return self._get_obs(), {}

    def _update_reference(self):
        # Referências variam com o tempo (ex: senoides)
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

        # Velocidades desejadas das rodas
        wL_d = self.k * pwmL
        wR_d = self.k * pwmR

        # Simula inércia: motores não mudam instantaneamente
        self.wL += self.alpha * (wL_d - self.wL)
        self.wR += self.alpha * (wR_d - self.wR)

        # Medições
        v_meas = self.R * (self.wL + self.wR) / 2
        w_meas = self.R * (self.wR - self.wL) / self.L

        # Erros
        e_v = self.v_ref - v_meas
        e_w = self.w_ref - w_meas

        # Recompensa: negativa proporcional ao erro quadrático
        reward = - (e_v ** 2 + e_w ** 2)

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass  # podemos adicionar visualização depois
