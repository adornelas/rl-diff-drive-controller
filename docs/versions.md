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
