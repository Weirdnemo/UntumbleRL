"""
rigid_untumble_env.py  —  Project UnTumble: Rigid Body Phase
=============================================================
Supports all four agent modes. Updated for symmetric continuous bounds on Agent 3
with a non-finite arithmetic check and a precision Linear-Log reward landscape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import mujoco.viewer

    import mujoco

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False
    print("[WARNING] mujoco package not found - env will raise on reset().")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N_THRUSTERS: int = 20

K_FUEL: float = 0.005  # fuel-consumption linear penalty weight
SUCCESS_BONUS: float = 250.0  # Boosted bonus on success termination

OMEGA_THRESHOLD: float = 0.01  # rad/s  -- detumble success criterion
MAX_STEPS: int = 4000  # ~20 s at dt=0.005 s
INITIAL_OMEGA_RANGE: float = 0.5  # rad/s  -- uniform random initial tumble

PD_KD: float = 8.0  # derivative gain for Agent 4 baseline

# -----------------------------------------------------------------------------
# Thruster index groups
# -----------------------------------------------------------------------------
IDX_ROLL_PLUS = [1, 6]
IDX_ROLL_MINUS = [0, 7]
IDX_PITCH_PLUS = [2, 9]
IDX_PITCH_MINUS = [3, 8]
IDX_YAW_PLUS = [12, 13, 14, 15]
IDX_YAW_MINUS = [16, 17, 18, 19]
IDX_LINEAR_PZ = [4, 10]
IDX_LINEAR_NZ = [5, 11]

AXIS_MAP: Dict[str, Dict[int, list]] = {
    "roll": {
        1: [(i, 1.0) for i in IDX_ROLL_PLUS],
        2: [(i, 1.0) for i in IDX_ROLL_MINUS],
    },
    "pitch": {
        1: [(i, 1.0) for i in IDX_PITCH_PLUS],
        2: [(i, 1.0) for i in IDX_PITCH_MINUS],
    },
    "yaw": {
        1: [(i, 1.0) for i in IDX_YAW_PLUS],
        2: [(i, 1.0) for i in IDX_YAW_MINUS],
    },
}


class RigidUnTumbleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        agent_mode: int = 3,
        xml_path: str | Path = "scene_rigid.xml",
        render_mode: Optional[str] = None,
        randomise_omega: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert agent_mode in (1, 2, 3, 4), "agent_mode must be 1, 2, 3, or 4"
        self.agent_mode = agent_mode
        self.xml_path = str(xml_path)
        self.render_mode = render_mode
        self.randomise_omega = randomise_omega

        self._step_count = 0
        self._last_ctrl = np.zeros(N_THRUSTERS, dtype=np.float32)
        self._viewer = None
        self._renderer = None

        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(13,), dtype=np.float32
        )

        if agent_mode == 1:
            self.action_space = spaces.MultiBinary(N_THRUSTERS)
        elif agent_mode == 2:
            self.action_space = spaces.MultiDiscrete([3, 3, 3])
        elif agent_mode == 3:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(N_THRUSTERS,), dtype=np.float32
            )
        elif agent_mode == 4:
            self.action_space = spaces.Box(
                low=-0.2, high=0.2, shape=(N_THRUSTERS,), dtype=np.float32
            )

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        if _MUJOCO_AVAILABLE:
            self._load_model()

    def _load_model(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        assert self.model.nu == N_THRUSTERS

        def _sensor_adr(name: str) -> int:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sid >= 0
            return int(self.model.sensor_adr[sid])

        self._adr_gyro = _sensor_adr("imu_gyro")
        self._adr_quat = _sensor_adr("imu_quat")
        self._adr_angvel = _sensor_adr("sat_angvel")

        jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "satellite_free"
        )
        dof_adr = int(self.model.jnt_dofadr[jnt_id])
        qpos_adr = int(self.model.jnt_qposadr[jnt_id])

        self._sl_lin_vel = slice(dof_adr, dof_adr + 3)
        self._sl_ang_vel = slice(dof_adr + 3, dof_adr + 6)
        self._sl_pos = slice(qpos_adr, qpos_adr + 3)
        self._sl_quat = slice(qpos_adr + 3, qpos_adr + 7)

    def _get_omega(self) -> np.ndarray:
        return self.data.sensordata[self._adr_gyro : self._adr_gyro + 3].copy()

    def _get_quat(self) -> np.ndarray:
        return self.data.sensordata[self._adr_quat : self._adr_quat + 4].copy()

    def _get_angvel_world(self) -> np.ndarray:
        return self.data.sensordata[self._adr_angvel : self._adr_angvel + 3].copy()

    def _decode_action(self, action) -> np.ndarray:
        ctrl = np.zeros(N_THRUSTERS, dtype=np.float32)

        if self.agent_mode == 1:
            ctrl[:] = np.asarray(action, dtype=np.float32)

        elif self.agent_mode == 2:
            channels = ["roll", "pitch", "yaw"]
            for ch_idx, ch_name in enumerate(channels):
                cmd = int(action[ch_idx])
                if cmd == 0:
                    continue
                for thr_idx, throttle in AXIS_MAP[ch_name][cmd]:
                    ctrl[thr_idx] = min(1.0, ctrl[thr_idx] + throttle)

        elif self.agent_mode == 3:
            normalized_action = (action + 1.0) / 2.0
            ctrl[:] = np.clip(normalized_action, 0.0, 1.0).astype(np.float32)

        elif self.agent_mode == 4:
            omega = self._get_omega()
            baseline = self._pd_baseline(omega)
            residual = np.clip(action, -0.2, 0.2).astype(np.float32)
            ctrl[:] = np.clip(baseline + residual, 0.0, 1.0)

        return ctrl

    def _pd_baseline(self, omega: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(N_THRUSTERS, dtype=np.float32)
        wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])

        def _throttle(w: float, scale: float = 1.0) -> float:
            return float(np.clip(abs(w) * PD_KD * scale / 100.0, 0.0, 1.0))

        pitch_group = IDX_PITCH_MINUS if wx > 0 else IDX_PITCH_PLUS
        for i in pitch_group:
            ctrl[i] = _throttle(wx)

        roll_group = IDX_ROLL_MINUS if wy > 0 else IDX_ROLL_PLUS
        for i in roll_group:
            ctrl[i] = _throttle(wy)

        yaw_group = IDX_YAW_MINUS if wz > 0 else IDX_YAW_PLUS
        yaw_t = _throttle(wz, scale=0.2)
        for i in yaw_group:
            ctrl[i] = max(ctrl[i], yaw_t)

        return ctrl

    def _build_obs(self) -> np.ndarray:
        quat = self._get_quat()
        omega = self._get_omega()
        av_world = self._get_angvel_world()

        ctrl_x = float(np.mean(self._last_ctrl[[0, 1, 6, 7]]))
        ctrl_y = float(np.mean(self._last_ctrl[[2, 3, 8, 9]]))
        ctrl_z = float(
            np.mean(self._last_ctrl[[4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        )

        obs = np.concatenate([quat, omega, av_world, [ctrl_x, ctrl_y, ctrl_z]]).astype(
            np.float32
        )

        obs = np.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)
        return np.clip(obs, -50.0, 50.0)

    def _compute_reward(
        self, omega: np.ndarray, ctrl: np.ndarray, terminated: bool
    ) -> float:
        omega_norm = float(np.linalg.norm(omega))

        # Precision Multi-Stage Reward Strategy
        if omega_norm > 0.15:
            # Quadratic scaling locks onto rough counter-damping fields
            omega_pen = 2.0 * (omega_norm**2)
        else:
            # High-slope linear reward provides sharp gradient tracking below 0.15 rad/s
            omega_pen = 0.3 * omega_norm

        fuel_pen = float(np.sum(ctrl))
        bonus = SUCCESS_BONUS if terminated else 0.0

        return -(omega_pen + K_FUEL * fuel_pen) + bonus

    def _is_success(self, omega: np.ndarray) -> bool:
        return float(np.linalg.norm(omega)) < OMEGA_THRESHOLD

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        q = self.np_random.standard_normal(4)
        q /= np.linalg.norm(q)
        self.data.qpos[self._sl_quat] = q

        if self.randomise_omega:
            omega_init = self.np_random.uniform(
                -INITIAL_OMEGA_RANGE, INITIAL_OMEGA_RANGE, size=3
            )
        else:
            omega_init = np.array([0.3, -0.2, 0.1])

        self.data.qvel[self._sl_ang_vel] = omega_init
        self.data.qvel[self._sl_lin_vel] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_ctrl = np.zeros(N_THRUSTERS, dtype=np.float32)

        obs = self._build_obs()
        info = {
            "omega_norm": float(np.linalg.norm(self._get_omega())),
            "omega": self._get_omega().tolist(),
        }
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        ctrl = self._decode_action(action)
        self._last_ctrl = ctrl.copy()

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        omega = self._get_omega()
        obs = self._build_obs()
        success = self._is_success(omega)
        terminated = success
        truncated = self._step_count >= MAX_STEPS
        reward = self._compute_reward(omega, ctrl, terminated=terminated)

        info = {
            "omega_norm": float(np.linalg.norm(omega)),
            "omega": omega.tolist(),
            "fuel_used": float(np.sum(ctrl)),
            "success": success,
            "step": self._step_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
