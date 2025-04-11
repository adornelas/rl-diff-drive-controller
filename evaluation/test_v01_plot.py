import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from envs.env_v01_baseline import DifferentialRobotEnv

# Load the environment and the trained model
env = DifferentialRobotEnv()
model = TD3.load("saved_models/td3_v01")

# Reset environment and initialize data logs
obs, _ = env.reset()
v_ref_list, v_meas_list = [], []
w_ref_list, w_meas_list = [], []
rewards = []

# Run the agent for 500 steps
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = env.step(action)

    v_ref, v_meas, w_ref, w_meas = obs
    v_ref_list.append(v_ref)
    v_meas_list.append(v_meas)
    w_ref_list.append(w_ref)
    w_meas_list.append(w_meas)
    rewards.append(reward)

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Linear velocity tracking
axs[0].plot(v_ref_list, label="v_ref")
axs[0].plot(v_meas_list, label="v_meas")
axs[0].set_ylabel("Linear Velocity (m/s)")
axs[0].legend()
axs[0].grid()

# Angular velocity tracking
axs[1].plot(w_ref_list, label="w_ref")
axs[1].plot(w_meas_list, label="w_meas")
axs[1].set_ylabel("Angular Velocity (rad/s)")
axs[1].legend()
axs[1].grid()

# Reward over time
axs[2].plot(rewards)
axs[2].set_ylabel("Reward")
axs[2].set_xlabel("Steps")
axs[2].grid()

plt.tight_layout()
plt.savefig("docs/figures/v01_response.png")
plt.show()
