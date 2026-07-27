"""
train.py  —  Project UnTumble: Rigid Body Phase
================================================
Stable-Baselines 3 PPO training script optimized for breaking out of local
minima via wider clip fractions and relaxed exploration variance frequencies.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    sync_envs_normalization,
)

sys.path.insert(0, str(Path(__file__).parent))
from rigid_untumble_env import RigidUnTumbleEnv


def get_hyperparams(agent_mode: int) -> Dict[str, Any]:
    if agent_mode == 1:
        return dict(
            ppo_policy="MlpPolicy",
            ppo_learning_rate=2e-4,
            ppo_n_steps=4096,
            ppo_batch_size=512,
            ppo_n_epochs=10,
            ppo_gamma=0.99,
            ppo_gae_lambda=0.95,
            ppo_clip_range=0.2,
            ppo_ent_coef=0.02,
            ppo_vf_coef=0.5,
            ppo_max_grad_norm=0.5,
            ppo_policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn_name="tanh",
            ),
            train_total_timesteps=5_000_000,
            train_n_envs=8,
            train_use_vecnorm=False,
            train_eval_freq=50_000,
            train_eval_episodes=20,
            train_save_freq=100_000,
        )

    elif agent_mode == 2:
        return dict(
            ppo_policy="MlpPolicy",
            ppo_learning_rate=3e-4,
            ppo_n_steps=2048,
            ppo_batch_size=256,
            ppo_n_epochs=10,
            ppo_gamma=0.99,
            ppo_gae_lambda=0.95,
            ppo_clip_range=0.2,
            ppo_ent_coef=0.01,
            ppo_vf_coef=0.5,
            ppo_max_grad_norm=0.5,
            ppo_policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn_name="tanh",
            ),
            train_total_timesteps=3_000_000,
            train_n_envs=8,
            train_use_vecnorm=False,
            train_eval_freq=30_000,
            train_eval_episodes=20,
            train_save_freq=60_000,
        )

    elif agent_mode == 3:
        return dict(
            ppo_policy="MlpPolicy",
            ppo_learning_rate=1.5e-4,
            ppo_n_steps=2048,
            ppo_batch_size=256,
            ppo_n_epochs=10,
            ppo_gamma=0.99,
            ppo_gae_lambda=0.95,
            ppo_clip_range=0.25,
            ppo_ent_coef=0.0,
            ppo_vf_coef=0.5,
            ppo_max_grad_norm=0.5,
            ppo_policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn_name="tanh",
                squash_output=True,
                use_sde=True,
                sde_sample_freq=16,
            ),
            train_total_timesteps=8_000_000,
            train_n_envs=8,
            train_use_vecnorm=True,
            train_eval_freq=40_000,
            train_eval_episodes=20,
            train_save_freq=120_000,
        )

    elif agent_mode == 4:
        return dict(
            ppo_policy="MlpPolicy",
            ppo_learning_rate=1e-4,
            ppo_n_steps=8192,
            ppo_batch_size=512,
            ppo_n_epochs=5,
            ppo_gamma=0.995,
            ppo_gae_lambda=0.97,
            ppo_clip_range=0.1,
            ppo_ent_coef=0.001,
            ppo_vf_coef=0.5,
            ppo_max_grad_norm=0.3,
            ppo_policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn_name="tanh",
            ),
            train_total_timesteps=6_000_000,
            train_n_envs=8,
            train_use_vecnorm=True,
            train_eval_freq=80_000,
            train_eval_episodes=20,
            train_save_freq=160_000,
        )
    else:
        raise ValueError(f"Unknown agent_mode {agent_mode}")


class TrainMetricsCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._csv_file = None
        self._csv_writer = None
        self._episode_buffer: list = []

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._csv_file = open(self.log_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "timestep",
                "mean_ep_reward",
                "mean_ep_len",
                "mean_omega_norm_final",
                "success_rate",
                "mean_fuel",
            ],
        )
        self._csv_writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_buffer.append(
                    {
                        "reward": info["episode"]["r"],
                        "length": info["episode"]["l"],
                        "omega_norm": info.get("omega_norm", float("nan")),
                        "success": float(info.get("success", False)),
                        "fuel": info.get("fuel_used", float("nan")),
                    }
                )
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_buffer:
            return

        rewards = [e["reward"] for e in self._episode_buffer]
        lengths = [e["length"] for e in self._episode_buffer]
        omegas = [
            e["omega_norm"]
            for e in self._episode_buffer
            if not np.isnan(e["omega_norm"])
        ]
        successes = [e["success"] for e in self._episode_buffer]
        fuels = [e["fuel"] for e in self._episode_buffer if not np.isnan(e["fuel"])]

        mean_reward = float(np.mean(rewards))
        mean_len = float(np.mean(lengths))
        mean_omega = float(np.mean(omegas)) if omegas else float("nan")
        sr = float(np.mean(successes))
        mean_fuel = float(np.mean(fuels)) if fuels else float("nan")

        row = {
            "timestep": self.num_timesteps,
            "mean_ep_reward": mean_reward,
            "mean_ep_len": mean_len,
            "mean_omega_norm_final": mean_omega,
            "success_rate": sr,
            "mean_fuel": mean_fuel,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        print(f"\n⚡ [ROLLOUT TELEMETRY | TIMESTEP {self.num_timesteps:,}]")
        print(
            f"  ├─ Completed Episodes inside rollout block: {len(self._episode_buffer)}"
        )
        print(f"  ├─ Success Rate                           : {sr * 100:.2f}%")
        print(f"  ├─ Mean Episode Reward                    : {mean_reward:.2f}")
        print(f"  ├─ Mean Episode Steps                     : {mean_len:.1f}")
        print(f"  ├─ Mean Final Omega                       : {mean_omega:.5f} rad/s")
        print(f"  └─ Mean Fuel Usage Scalar                 : {mean_fuel:.2f}\n")

        for k, v in row.items():
            if k != "timestep" and not np.isnan(v):
                self.logger.record(f"untumble/{k}", v)

        self._episode_buffer.clear()

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()


def _make_env_fn(agent_mode: int, xml_path: str, seed: int, rank: int):
    def _init():
        set_random_seed(seed + rank)
        env = RigidUnTumbleEnv(
            agent_mode=agent_mode,
            xml_path=xml_path,
            render_mode=None,
            randomise_omega=True,
            seed=seed + rank,
        )
        return Monitor(env)

    return _init


def build_vec_env(
    agent_mode: int, xml_path: str, n_envs: int, seed: int, use_vecnorm: bool
):
    fns = [_make_env_fn(agent_mode, xml_path, seed, i) for i in range(n_envs)]
    if n_envs == 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        venv = DummyVecEnv(fns)
    else:
        venv = SubprocVecEnv(fns)

    if use_vecnorm:
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    return venv


def _resolve_policy_kwargs(raw: dict) -> dict:
    import torch.nn as nn

    resolved = dict(raw)
    name = resolved.pop("activation_fn_name", "tanh")
    resolved["activation_fn"] = {"tanh": nn.Tanh, "relu": nn.ReLU}[name]
    return resolved


def train(
    agent_mode: int, xml_path: str, seed: int, run_dir: str, resume: str | None = None
) -> PPO:
    cfg = get_hyperparams(agent_mode)
    os.makedirs(run_dir, exist_ok=True)
    tb_path = os.path.join(run_dir, "tb")
    csv_path = os.path.join(run_dir, "train_log.csv")

    if torch.cuda.is_available():
        device = "cuda"
        print(
            f"--- Running Parallel Subprocess Workers mapped to CUDA Device: {torch.cuda.get_device_name(0)} ---"
        )
    else:
        device = "cpu"

    print(f"\n{'=' * 60}")
    print(
        f"  Project UnTumble — Agent {agent_mode} Step 0.6 Breakthrough Build Launched"
    )
    print(f"  run_dir  : {run_dir}")
    print(f"  n_envs   : {cfg['train_n_envs']}")
    print(f"  device   : {device}")
    print(f"{'=' * 60}\n")

    train_env = build_vec_env(
        agent_mode=agent_mode,
        xml_path=xml_path,
        n_envs=cfg["train_n_envs"],
        seed=seed,
        use_vecnorm=cfg["train_use_vecnorm"],
    )

    eval_env = build_vec_env(
        agent_mode=agent_mode,
        xml_path=xml_path,
        n_envs=1,
        seed=seed + 9999,
        use_vecnorm=cfg["train_use_vecnorm"],
    )

    raw_policy_kwargs = dict(cfg["ppo_policy_kwargs"])
    use_sde = raw_policy_kwargs.pop("use_sde", False)
    sde_sample_freq = raw_policy_kwargs.pop("sde_sample_freq", -1)

    policy_kwargs = _resolve_policy_kwargs(raw_policy_kwargs)

    ppo_kwargs = dict(
        policy=cfg["ppo_policy"],
        env=train_env,
        learning_rate=cfg["ppo_learning_rate"],
        n_steps=cfg["ppo_n_steps"],
        batch_size=cfg["ppo_batch_size"],
        n_epochs=cfg["ppo_n_epochs"],
        gamma=cfg["ppo_gamma"],
        gae_lambda=cfg["ppo_gae_lambda"],
        clip_range=cfg["ppo_clip_range"],
        ent_coef=cfg["ppo_ent_coef"],
        vf_coef=cfg["ppo_vf_coef"],
        max_grad_norm=cfg["ppo_max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_path,
        verbose=1,
        seed=seed,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        device=device,
    )

    if resume:
        print(f"  Resuming from: {resume}")
        model = PPO.load(
            resume,
            env=train_env,
            **{
                k: v
                for k, v in ppo_kwargs.items()
                if k not in ("policy", "env", "use_sde", "sde_sample_freq")
            },
        )
    else:
        model = PPO(**ppo_kwargs)

    class VecNormalizeSaveCallback(EvalCallback):
        def __init__(self, vecnorm_env, vecnorm_path: str, **kwargs):
            super().__init__(**kwargs)
            self._vecnorm_env = vecnorm_env
            self._vecnorm_path = vecnorm_path

        def _on_step(self) -> bool:
            if self._vecnorm_env is not None:
                sync_envs_normalization(self.training_env, self.eval_env)

            result = super()._on_step()
            if self.last_mean_reward == self.best_mean_reward:
                if self._vecnorm_env is not None:
                    self._vecnorm_env.save(self._vecnorm_path)
            return result

    vecnorm_save_path = os.path.join(run_dir, "vecnorm.pkl")

    eval_cb = VecNormalizeSaveCallback(
        vecnorm_env=train_env if cfg["train_use_vecnorm"] else None,
        vecnorm_path=vecnorm_save_path,
        eval_env=eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(cfg["train_eval_freq"] // cfg["train_n_envs"], 1),
        n_eval_episodes=cfg["train_eval_episodes"],
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(cfg["train_save_freq"] // cfg["train_n_envs"], 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=f"agent{agent_mode}",
    )

    metrics_cb = TrainMetricsCallback(log_path=csv_path)
    callbacks = CallbackList([eval_cb, ckpt_cb, metrics_cb])

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["train_total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=resume is None,
        tb_log_name=f"agent{agent_mode}",
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed / 3600:.2f} h")

    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    if cfg["train_use_vecnorm"]:
        train_env.save(os.path.join(run_dir, "vecnorm_final.pkl"))
    train_env.close()
    eval_env.close()
    return model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--xml", type=str, default="scene_rigid.xml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def _next_run_dir(agent_mode: int) -> str:
    base = Path("runs")
    k = 0
    while (base / f"agent{agent_mode}_run{k}").exists():
        k += 1
    return str(base / f"agent{agent_mode}_run{k}")


if __name__ == "__main__":
    args = _parse_args()
    run_dir = args.run_dir or _next_run_dir(args.agent)
    train(
        agent_mode=args.agent,
        xml_path=args.xml,
        seed=args.seed,
        run_dir=run_dir,
        resume=args.resume,
    )
