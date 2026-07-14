import numpy as np
from envio.base_env import BaseDetumbleEnv, HybridRewardPolicy, PWPFModulator
from gymnasium import spaces

import mujoco


class Agent3Env(BaseDetumbleEnv):
    """
    Agent 3: Residual Reinforcement Learning (Hybrid).
    A Classical PD controller generates a baseline detumble thrust.
    The RL Agent outputs small residual corrections [-0.1, 0.1] to damp flex vibrations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # RESIDUAL ACTION SPACE: AI can tweak duty cycles by +/- 10%
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(12,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        self.pwpf = PWPFModulator()
        self.reward_policy = HybridRewardPolicy(action_dim=12)
        self.internal_step_counter = 0

        # Classical Controller Gain (Gentle Brake)
        self.K_D = 0.3

    def _allocate_thrusters(self, ideal_torque):
        """
        Reverse-engineered from scene.xml.
        Maps 3D torque (Roll, Pitch, Yaw) to the 12 cross-coupled RCS valves.
        Actuator array: [px1, px2, nx1, nx2, py1, py2, ny1, ny2, pz1, pz2, nz1, nz2]
        """
        duty = np.zeros(12, dtype=np.float32)

        # Roll (+X / -X)
        if ideal_torque[0] > 0:
            duty[5] += ideal_torque[0]  # py2
            duty[7] += ideal_torque[0]  # ny2
            duty[8] += ideal_torque[0]  # pz1
            duty[10] += ideal_torque[0]  # nz1
        else:
            abs_t = abs(ideal_torque[0])
            duty[4] += abs_t  # py1
            duty[6] += abs_t  # ny1
            duty[9] += abs_t  # pz2
            duty[11] += abs_t  # nz2

        # Pitch (+Y / -Y)
        if ideal_torque[1] > 0:
            duty[0] += ideal_torque[1]  # px1
            duty[2] += ideal_torque[1]  # nx1
            duty[9] += ideal_torque[1]  # pz2
            duty[10] += ideal_torque[1]  # nz1
        else:
            abs_t = abs(ideal_torque[1])
            duty[1] += abs_t  # px2
            duty[3] += abs_t  # nx2
            duty[8] += abs_t  # pz1
            duty[11] += abs_t  # nz2

        # Yaw (+Z / -Z)
        if ideal_torque[2] > 0:
            duty[1] += ideal_torque[2]  # px2
            duty[2] += ideal_torque[2]  # nx1
            duty[4] += ideal_torque[2]  # py1
            duty[7] += ideal_torque[2]  # ny2
        else:
            abs_t = abs(ideal_torque[2])
            duty[0] += abs_t  # px1
            duty[3] += abs_t  # nx2
            duty[5] += abs_t  # py2
            duty[6] += abs_t  # ny1

        # Normalize to preserve thrust vector direction if over-saturated
        max_duty = np.max(duty)
        if max_duty > 1.0:
            duty /= max_duty

        return duty

    def reset(self, seed=None, options=None):
        self.pwpf.reset()
        self.total_timesteps = self.internal_step_counter
        obs, info = super().reset(seed=seed, options=options)
        self.reward_policy.reset(obs, np.zeros(12))
        return obs, info

    def step(self, action):
        # 1. CLASSICAL BRAIN: PD Controller calculates ideal torque (POSITIVE SIGN = BRAKE)
        omega = self.data.sensordata[self._gyro_id : self._gyro_id + 3]
        ideal_torque = self.K_D * omega

        # Map torque to baseline duty cycle [0, 1]
        baseline_duty = self._allocate_thrusters(ideal_torque)

        # 2. RL BRAIN: The Neural Network provides the residual correction
        rl_residual = np.zeros(12)

        # 3. MERGE: Combine and clip back to valid [0, 1] bounds for PWPF
        final_duty_cycle = np.clip(baseline_duty + rl_residual, 0.0, 1.0)

        dt = self.model.opt.timestep

        # Physical Step
        for _ in range(self.PHYSICS_STEPS):
            actual_firing = self.pwpf.step(final_duty_cycle, dt)
            self.data.ctrl[0:12] = actual_firing
            self.data.ctrl[12:16] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        self.internal_step_counter += 1

        obs = self._get_obs()

        # Evaluate based on the final combined duty cycle
        reward, done, breakdown = self.reward_policy.step(
            obs, final_duty_cycle, self._step_count, self.internal_step_counter
        )

        truncated = self._step_count >= self.MAX_STEPS

        info = {
            "metrics/residual_omega": float(np.linalg.norm(obs[0:3])),
            "metrics/peak_flex_angle": breakdown.get("peak_flex_angle", 0.0),
            "metrics/flex_breach_event": float(
                breakdown.get("flex_limit_breach", False)
            ),
            "metrics/stable_steps": breakdown.get("stable_count", 0.0),
            "reward_dist/r_flex_base": breakdown.get("r_flex_base", 0.0),
            "reward_dist/r_delta_action": breakdown.get("r_delta_action", 0.0),
            "reward_dist/r_terminal": breakdown.get("r_terminal", 0.0),
        }

        return obs, reward, done, truncated, info
