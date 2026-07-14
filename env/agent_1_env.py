import numpy as np
from gymnasium import spaces

import mujoco
from env.base_env import BaseDetumbleEnv, UnTumbleRewardPolicyV3


class Agent1Env(BaseDetumbleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.MultiDiscrete([2] * 12)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.reward_policy = UnTumbleRewardPolicyV3(action_dim=12)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.reward_policy.reset(obs, np.zeros(12))
        return obs, info

    def step(self, action):
        self.data.ctrl[0:12] = action
        self.data.ctrl[12:16] = 0.0

        for _ in range(self.PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()

        reward, done, breakdown = self.reward_policy.step(obs, action, self._step_count)
        truncated = self._step_count >= self.MAX_STEPS

        info = {"residual_omega": float(np.linalg.norm(obs[0:3])), **breakdown}
        return obs, reward, done, truncated, info
