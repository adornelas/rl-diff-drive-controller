import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from envs.env_v01_baseline import DifferentialRobotEnv

# Cria o ambiente
env = DifferentialRobotEnv()

# Define ruído nas ações (exploração durante o treino)
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Cria o modelo TD3
model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_tensorboard/"
)

# Treinamento
model.learn(total_timesteps=100_000)

# Salva o modelo treinado
model.save("saved_models/td3_v01")
print("Modelo salvo como td3_differential_robot")
