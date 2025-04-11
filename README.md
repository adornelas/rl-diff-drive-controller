# RL Differential Drive Controller

A reinforcement learning-based low-level controller for a differential drive robot.

This project explores the use of RL agents (e.g. TD3) to map desired linear and angular velocities (`v_ref`, `w_ref`) and current feedback (`v_meas`, `w_meas`) into PWM commands for each motor (`pwmA`, `pwmB`).

The controller is trained in simulation using a simple physics model with configurable motor response (inertia), and is designed to be deployed later in embedded systems such as an ESP32-based robot.

## 🚗 Control Goal

Given:
- `v_ref` = desired linear velocity (m/s)
- `w_ref` = desired angular velocity (rad/s)
- `v_meas`, `w_meas` = current velocities (feedback)

The agent learns to produce:
- `pwmA`, `pwmB` = motor commands (normalized between -1 and 1)

## 🧪 Features

- Custom Gym-like environment (`DifferentialRobotEnv`)
- TD3 agent training with Stable-Baselines3
- Realistic motor response modeling (inertia)
- Time-varying sinusoidal reference signals
- Visualization tools to analyze performance
- Structured versioning for each trained policy
