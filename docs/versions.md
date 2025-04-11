## v0.1 - `baseline_env`
- Environment with instantaneous motor response (`alpha = 1.0`)
- Constant velocity references: `v_ref = 0.3`, `w_ref = 0.0`
- Reward: `- (e_v² + e_w²)`
- Observation: `[v_ref, v_meas, w_ref, w_meas]`
- Action: `[pwmL, pwmR]` ∈ [-1, 1]
- No inertia, no dynamics
- Agent failed to learn meaningful behavior

- Env file: `envs/env_v01_baseline.py`
- Train script: `training/train_v01.py`
- Plot: `docs/figures/v01_response.png`

## v0.2 - `dynamic_refs`
- Introduced sinusoidal reference signals:
  - `v_ref = 0.3 * sin(0.2 * t)`
  - `w_ref = 1.0 * sin(0.1 * t)`
- Maintains instant motor response (`alpha = 1.0`)
- Reward function unchanged: `- (e_v² + e_w²)`
- Observation: `[v_ref, v_meas, w_ref, w_meas]`
- Action: `[pwmL, pwmR]` ∈ [-1, 1]
- Purpose: test agent's ability to track time-varying references

- Env file: `envs/env_v02_dynamic_refs.py`
- Train script: `training/train_v02.py`
- Plot: `docs/figures/v02_response.png`

## v0.3 - `penalized_reward`
- Increased reward penalty to emphasize error minimization:
  - Reward: `-20 * (e_v² + e_w²)`
- Maintains sinusoidal references and instant motor response
- Helps the agent distinguish smaller improvements in tracking
- Action space remains normalized in `[-1.0, 1.0]`

- Env file: `envs/env_v03_penalized_reward.py`
- Train script: `training/train_v03.py`
- Plot: `docs/figures/v03_response.png`
