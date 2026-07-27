import argparse
import os
import sys
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

# from env.agent_1_env import Agent1Env
# from env.agent_2_env import Agent2Env
from env.agent_3_env import Agent3Env

# from env.agent_4_env import Agent4Env
from env.callbacks import WelfordMetricsCallback


def make_env(env_class, rank, seed=0):
    def _init():
        env = env_class()
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def linear_schedule(
    initial_value: float, final_value: float
) -> Callable[[float], float]:
    """Generates a callable for SB3 to linearly decay learning rate."""

    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and goes to 0.0
        return progress_remaining * (initial_value - final_value) + final_value

    return func


# --- PROJECT UNTUMBLE HYPERPARAMETER GRIDS ---
PPO_CONFIGS = {
    "Agent1": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "gae_lambda": 0.95,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "Agent2": {
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.015,
        "gae_lambda": 0.95,
        "policy_kwargs": dict(net_arch=[512, 256, 128]),
    },
    "Agent3": {  # PWPF Only - V2 Exploit Hardened
        "learning_rate": linear_schedule(3e-4, 1e-5),
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "Agent4": {  # PWPF + RW Synergy - V2 Exploit Hardened
        "learning_rate": linear_schedule(3e-4, 5e-5),
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "policy_kwargs": dict(net_arch=[256, 256, 128]),
    },
}

ENV_MAP = {
    # 1: ("Agent1", Agent1Env),
    # 2: ("Agent2", Agent2Env),
    3: ("Agent3", Agent3Env),
    # 4: ("Agent4", Agent4Env),
}


def train_agent(
    agent_num, num_envs=8, total_timesteps=5_000_000, seeds=[10, 20, 30, 40, 50]
):
    agent_name, env_class = ENV_MAP[agent_num]
    hyperparams = PPO_CONFIGS[agent_name]

    print("=" * 60)
    print(f"INITIALIZING PROJECT UNTUMBLE TRAINING: {agent_name}")
    print(f"Hyperparameters: {hyperparams}")
    print("=" * 60)

    for seed in seeds:
        print(f"\n--- Starting Seed: {seed} ---")
        run_name = f"{agent_name}_seed_{seed}"
        log_dir = f"./tensorboard_logs/{agent_name}/"
        model_dir = f"./models/{agent_name}/seed_{seed}/"
        os.makedirs(model_dir, exist_ok=True)

        envs = SubprocVecEnv([make_env(env_class, i, seed) for i in range(num_envs)])
        envs = VecMonitor(envs)
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(500_000 // num_envs, 1),
            save_path=model_dir,
            name_prefix=run_name,
        )
        metrics_callback = WelfordMetricsCallback()

        model = PPO(
            "MlpPolicy",
            envs,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            **hyperparams,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, metrics_callback],
            tb_log_name=run_name,
            reset_num_timesteps=False,
        )

        model.save(f"{model_dir}/final_model")
        envs.save(f"{model_dir}/vec_normalize.pkl")
        torch.save(model.policy.state_dict(), f"{model_dir}/policy_weights.pth")
        envs.close()
        print(f"Finished Seed {seed}.")
        print(f" -> Brain (SB3): {model_dir}/final_model.zip")
        print(f" -> Eyes (VecNorm): {model_dir}/vec_normalize.pkl")
        print(f" -> Raw Weights: {model_dir}/policy_weights.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Project UnTumble Agents")
    parser.add_argument("--agent", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5_000_000)
    args = parser.parse_args()

    train_agent(args.agent, num_envs=args.envs, total_timesteps=args.steps)
