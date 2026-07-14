import gymnasium as gym
import numpy as np
from gymnasium import spaces

import mujoco


class PWPFModulator:
    def __init__(self, tau_m=0.1, U_on=0.55, U_off=0.35, K_m=1.0):
        self.tau_m, self.U_on, self.U_off, self.K_m = tau_m, U_on, U_off, K_m
        self.x_f = np.zeros(12, dtype=np.float32)
        self.state = np.zeros(12, dtype=np.float32)

    def step(self, u_cmd, dt):
        self.x_f += dt * (self.K_m * u_cmd - self.x_f) / self.tau_m
        self.state[self.x_f > self.U_on] = 1.0
        self.state[self.x_f < self.U_off] = 0.0
        return self.state.copy()

    def reset(self):
        self.x_f.fill(0.0)
        self.state.fill(0.0)


class HybridRewardPolicy:
    """
    Streamlined reward policy for Residual RL.
    The PD controller handles the detumble; the RL agent handles structural safety.
    """

    OMEGA_THRESH = 0.10
    FLEX_SUCCESS = 0.08
    FLEX_LIMIT = 0.15
    FLEX_WARN = 0.07

    K_FLEX_BASE = 10.0
    K_FLEX_SOFT = 120.0
    K_DELTA_ACT = 2.0

    C_SUCCESS = 5000.0
    C_FAIL = -600.0
    N_STABLE = 3
    MAX_STEPS = 1000

    def __init__(self, action_dim=12):
        self.action_dim = action_dim
        self.reset(np.zeros(13), np.zeros(action_dim))

    def reset(self, obs, initial_action):
        self._prev_flex_pos = obs[3:6].copy()
        self._prev_action = np.array(initial_action, dtype=np.float32)
        self._stable_count = 0

    def step(self, obs, action, step, total_timesteps):
        omega = obs[0:3]
        flex_pos = obs[3:6]

        # 1. Structural Safety Penalties (The Agent's primary job)
        r_flex_base = -self.K_FLEX_BASE * float(np.dot(flex_pos, flex_pos))
        flex_excess = np.maximum(0.0, np.abs(flex_pos) - self.FLEX_WARN)
        r_flex_soft = -self.K_FLEX_SOFT * float(np.sum(flex_excess**2))

        # 2. Smoothness Penalty
        delta_act = action - self._prev_action
        r_delta_action = -self.K_DELTA_ACT * float(np.dot(delta_act, delta_act))

        r_terminal = 0.0
        done = False

        # Hard Failure (Snapping the panels)
        flex_breach = bool(np.any(np.abs(flex_pos) >= self.FLEX_LIMIT))
        if flex_breach:
            r_terminal = self.C_FAIL
            # Give the agent a 500k step shield to explore before death is enabled
            if total_timesteps > 500_000:
                done = True

        # Composite Success Gate
        omega_quiet = float(np.max(np.abs(omega))) < self.OMEGA_THRESH
        flex_quiet = float(np.max(np.abs(flex_pos))) < self.FLEX_SUCCESS

        if omega_quiet and flex_quiet:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if not done:
            if self._stable_count >= self.N_STABLE:
                r_terminal = self.C_SUCCESS
                done = True
            elif step >= self.MAX_STEPS - 1:
                done = True

        self._prev_flex_pos = flex_pos.copy()
        self._prev_action = action.copy()

        total = r_flex_base + r_flex_soft + r_delta_action + r_terminal

        breakdown = {
            "r_flex_base": r_flex_base,
            "r_flex_soft": r_flex_soft,
            "r_delta_action": r_delta_action,
            "r_terminal": r_terminal,
            "peak_flex_angle": float(np.max(np.abs(flex_pos))),
            "stable_count": float(self._stable_count),
            "flex_limit_breach": flex_breach,
        }
        return total, done, breakdown


class BaseDetumbleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}
    MAX_STEPS, PHYSICS_STEPS = 1000, 10

    def __init__(self, xml_path="assets/xml/scene.xml", render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        self._debris_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "debris_core"
        )

        flex_names = ["flex_joint_py", "flex_joint_ny", "flex_joint_px"]
        self._flex_qposadr = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in flex_names
        ]
        self._flex_dofadr = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in flex_names
        ]

        rw_names = ["rw_joint_1", "rw_joint_2", "rw_joint_3", "rw_joint_4"]
        self._rw_dofadr = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in rw_names
        ]

        self.RW_MAX_SPEED, self._step_count, self.total_timesteps = 300.0, 0, 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # --- HOLLOW SPHERE INITIALIZATION ---
        progress = np.clip(self.total_timesteps / 5_000_000, 0, 1)

        # Min spin is strictly > OMEGA_THRESH (0.10) to prevent instant-wins
        min_spin = 0.12
        max_spin = 0.12 + progress * 0.38

        spin_magnitude = self.np_random.uniform(min_spin, max_spin)
        direction = self.np_random.normal(size=3)
        direction /= np.linalg.norm(direction)

        initial_omega = spin_magnitude * direction

        chaser_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "chaser_free")
        ]
        debris_dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "debris_free")
        ]

        # --- MUJOCO WELD KINEMATICS FIX ---
        # The Chaser is welded 4 meters below the Debris on the Z-axis.
        r_chaser_to_debris = np.array([0.0, 0.0, -4.0])
        v_chaser = np.cross(initial_omega, r_chaser_to_debris)

        # Debris spins in place
        self.data.qvel[debris_dof + 3 : debris_dof + 6] = initial_omega
        self.data.qvel[debris_dof : debris_dof + 3] = np.zeros(3)

        # Chaser spins AND translates to satisfy the weld constraint
        self.data.qvel[chaser_dof + 3 : chaser_dof + 6] = initial_omega
        self.data.qvel[chaser_dof : chaser_dof + 3] = v_chaser

        for qpos_idx in self._flex_qposadr:
            self.data.qpos[qpos_idx] = self.np_random.uniform(-0.02, 0.02)
        for qvel_idx in self._flex_dofadr:
            self.data.qvel[qvel_idx] = self.np_random.uniform(-0.01, 0.01)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        omega = self.data.sensordata[
            self._gyro_id : self._gyro_id + 3
        ] + self.np_random.normal(0, 0.0001, size=3)
        flex_pos = np.array(
            [self.data.qpos[idx] for idx in self._flex_qposadr]
        ) + self.np_random.normal(0, 0.0005, size=3)
        flex_vel = np.array([self.data.qvel[idx] for idx in self._flex_dofadr])
        rw_vels = np.array([self.data.qvel[idx] for idx in self._rw_dofadr])
        return np.concatenate(
            [omega, flex_pos, flex_vel, rw_vels / self.RW_MAX_SPEED], dtype=np.float32
        )
