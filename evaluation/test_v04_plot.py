import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs.env_v04_balanced_reward import DifferentialRobotEnv

env = DifferentialRobotEnv()
model = TD3.load("saved_models/td3_v04")

obs, _ = env.reset()
v_ref_list, v_meas_list = [], []
w_ref_list, w_meas_list = [], []
rewards = []
pwmA_list, pwmB_list = [], []

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = env.step(action)

    v_ref, v_meas, w_ref, w_meas = obs
    v_ref_list.append(v_ref)
    v_meas_list.append(v_meas)
    w_ref_list.append(w_ref)
    w_meas_list.append(w_meas)
    rewards.append(reward)
    pwmA_list.append(action[0])
    pwmB_list.append(action[1])

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Linear velocity
axs[0].plot(v_ref_list, label="v_ref")
axs[0].plot(v_meas_list, label="v_meas")
axs[0].set_ylabel("Linear Velocity (m/s)")
axs[0].legend()
axs[0].grid()

# Angular velocity
axs[1].plot(w_ref_list, label="w_ref")
axs[1].plot(w_meas_list, label="w_meas")
axs[1].set_ylabel("Angular Velocity (rad/s)")
axs[1].legend()
axs[1].grid()

# Reward
axs[2].plot(rewards)
axs[2].set_ylabel("Reward")
axs[2].grid()

# PWM signals
axs[3].plot(pwmA_list, label="PWM A")
axs[3].plot(pwmB_list, label="PWM B")
axs[3].set_ylabel("PWM Output")
axs[3].set_xlabel("Steps")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.savefig("docs/figures/v04_response.png")

plt.show()
