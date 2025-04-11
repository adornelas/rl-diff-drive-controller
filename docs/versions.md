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

## v0.4 - `balanced_reward`
- Reward function adjusted to prioritize linear velocity tracking:
  - `reward = -20 * e_v² - 10 * e_w²`
- Purpose: encourage agent to also learn to follow `v_ref`, not just `w_ref`
- Maintains sinusoidal references and instant response
- Useful for evaluating controller balance

- Env file: `envs/env_v04_balanced_reward.py`
- Train script: `training/train_v04.py`
- Plot: `docs/figures/v04_response.png`

## v0.5 - `slow_vref`
- Introduced slower `v_ref` signal: `0.3 * sin(0.1 * t)`
- Maintains same `w_ref` and motor model (`alpha = 1.0`)
- Reward: `-40 * e_v² - 10 * e_w²` to emphasize linear tracking
- Action noise increased to improve exploration (`σ = 0.3`)
- Trained for 300k steps to improve policy convergence

- Env file: `envs/env_v05_slow_vref.py`
- Train script: `training/train_v05.py`
- Plot: `docs/figures/v05_response.png`

## v0.6 - `no_wref`
- Reference angular velocity (`w_ref`) set to zero
- Focused training on linear velocity tracking only
- Reward: `-40 * e_v² - 10 * e_w²`
- Slower sine wave used for `v_ref`: `0.3 * sin(0.1 * t)`
- Same motor model (instantaneous response), trained with more noise and 300k steps

- Env file: `envs/env_v06_no_wref.py`
- Train script: `training/train_v06.py`
- Plot: `docs/figures/v06_response.png`
