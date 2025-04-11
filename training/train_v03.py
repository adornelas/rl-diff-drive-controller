from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from envs.env_v03_penalized_reward import DifferentialRobotEnv

env = DifferentialRobotEnv()
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_tensorboard/v03"
)

model.learn(total_timesteps=100_000)
model.save("saved_models/td3_v03")
print("Model saved to saved_models/td3_v03.zip")
