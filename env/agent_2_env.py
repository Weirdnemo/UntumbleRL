import numpy as np
from gymnasium import spaces

import mujoco
from env.base_env import BaseDetumbleEnv, UnTumbleRewardPolicyV3


class Agent2Env(BaseDetumbleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.phase_flag = 0.0
        self.PHASE_TRANSITION_THRESH = 0.1
        self.reward_policy = UnTumbleRewardPolicyV3(action_dim=16)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.phase_flag = 0.0
        obs_with_phase = np.append(obs, self.phase_flag)
        self.reward_policy.reset(obs, np.zeros(16))
        return obs_with_phase, info

    def step(self, action):
        rcs_raw = action[0:12]
        rw_cmd = action[12:16]
        rcs_cmd = np.where(rcs_raw > 0.0, 1.0, 0.0)

        omega = self.data.sensordata[self._gyro_id : self._gyro_id + 3]

        if np.linalg.norm(omega) < self.PHASE_TRANSITION_THRESH:
            self.phase_flag = 1.0
            rw_vels = np.array([self.data.qvel[dof_idx] for dof_idx in self._rw_dofadr])
            if not np.any(np.abs(rw_vels) > 0.9 * self.RW_MAX_SPEED):
                rcs_cmd.fill(0)
        else:
            self.phase_flag = 0.0
            rw_cmd.fill(0.0)

        self.data.ctrl[0:12] = rcs_cmd
        self.data.ctrl[12:16] = rw_cmd

        for _ in range(self.PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        applied_action = np.concatenate([rcs_cmd, rw_cmd])
        reward, done, breakdown = self.reward_policy.step(
            obs, applied_action, self._step_count
        )

        truncated = self._step_count >= self.MAX_STEPS
        obs_with_phase = np.append(obs, self.phase_flag)
        info = {"residual_omega": float(np.linalg.norm(obs[0:3])), **breakdown}

        return obs_with_phase, reward, done, truncated, info
