from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from envs.env_v01_baseline import DifferentialRobotEnv

# Cria o ambiente
env = DifferentialRobotEnv()

# Aumenta o ruído para explorar mais
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# Cria o modelo TD3 com mais exploração
model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_tensorboard/"
)

# Aumenta o número de passos de treino
model.learn(total_timesteps=300_000)

# Salva o modelo treinado
model.save("saved_models/td3_v02")
print("Modelo salvo como aved_models/td3_v02")
