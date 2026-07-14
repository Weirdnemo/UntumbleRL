import numpy as np
from envio.base_env import BaseDetumbleEnv, PWPFModulator, UnTumbleRewardPolicyV3
from gymnasium import spaces

import mujoco


class Agent4Env(BaseDetumbleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=np.array([0.0] * 12 + [-1.0] * 4),
            high=np.array([1.0] * 12 + [1.0] * 4),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.pwpf = PWPFModulator()
        self.reward_policy = UnTumbleRewardPolicyV3(action_dim=16)

    def reset(self, seed=None, options=None):
        self.pwpf.reset()
        obs, info = super().reset(seed, options)
        self.reward_policy.reset(obs, np.zeros(16))
        return obs, info

    def step(self, action):
        rcs_duty = np.clip(action[0:12], 0.0, 1.0)
        rw_cmd = np.clip(action[12:16], -1.0, 1.0)
        dt = self.model.opt.timestep

        for _ in range(self.PHYSICS_STEPS):
            actual_firing = self.pwpf.step(rcs_duty, dt)
            self.data.ctrl[0:12] = actual_firing
            self.data.ctrl[12:16] = rw_cmd
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()

        combined_action = np.concatenate([rcs_duty, rw_cmd])
        reward, done, breakdown = self.reward_policy.step(
            obs, combined_action, self._step_count
        )
        truncated = self._step_count >= self.MAX_STEPS

        info = {"residual_omega": float(np.linalg.norm(obs[0:3])), **breakdown}
        return obs, reward, done, truncated, info
