"""
eval.py  —  Project UnTumble: Quick Evaluation
===============================================
Rolls out a saved PPO model and prints per-episode stats + a summary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from rigid_untumble_env import OMEGA_THRESHOLD, RigidUnTumbleEnv


def _bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(frac * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _omega_grade(omega_f: float) -> str:
    if omega_f < OMEGA_THRESHOLD:
        return "✓ DETUMBLED"
    elif omega_f < 0.05:
        return "~ almost"
    elif omega_f < 0.15:
        return "○ partial"
    else:
        return "✗ failed"


def run_eval(
    agent_mode: int,
    model_path: str,
    xml_path: str,
    n_episodes: int,
    render: bool,
    vecnorm_path: str | None,
    seed: int,
) -> None:

    render_mode = "human" if render else None

    raw_env = RigidUnTumbleEnv(
        agent_mode=agent_mode,
        xml_path=xml_path,
        render_mode=render_mode,
        randomise_omega=True,
        seed=seed,
    )

    if vecnorm_path:
        venv = DummyVecEnv([lambda: Monitor(raw_env)])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        use_vec = True
    else:
        use_vec = False

    model = PPO.load(model_path)

    print()
    print("=" * 65)
    print(f"  Project UnTumble — Evaluation")
    print(f"  Agent   : {agent_mode}  |  Action space: {raw_env.action_space}")
    print(f"  Model   : {model_path}")
    print("=" * 65)
    print(
        f"  {'ep':>3}  {'steps':>5}  {'ω₀ (r/s)':>9}  {'ωf (r/s)':>9}  "
        f"{'fuel':>7}  {'reward':>9}  result"
    )
    print(
        f"  {'─' * 3}  {'─' * 5}  {'─' * 9}  {'─' * 9}  {'─' * 7}  {'─' * 9}  {'─' * 11}"
    )

    records = []

    for ep in range(n_episodes):
        if use_vec:
            obs = venv.reset()
        else:
            obs, info = raw_env.reset()

        omega_0 = float(np.linalg.norm(raw_env._get_omega()))

        ep_reward = 0.0
        ep_fuel = 0.0
        ep_steps = 0
        omega_f = omega_0
        success = False

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if use_vec:
                obs, rew, done_arr, info_arr = venv.step(action)
                rew = float(rew[0])
                done = bool(done_arr[0])
                info = info_arr[0]
            else:
                obs, rew, terminated, truncated, info = raw_env.step(action)
                done = terminated or truncated

            ep_reward += rew
            ep_fuel += float(info.get("fuel_used", 0.0))
            ep_steps += 1
            omega_f = float(info.get("omega_norm", omega_f))
            success = bool(info.get("success", False))

        records.append(
            {
                "omega_0": omega_0,
                "omega_f": omega_f,
                "steps": ep_steps,
                "fuel": ep_fuel,
                "reward": ep_reward,
                "success": success,
            }
        )

        grade = _omega_grade(omega_f)
        print(
            f"  {ep + 1:>3}  {ep_steps:>5}  {omega_0:>9.4f}  {omega_f:>9.4f}  "
            f"{ep_fuel:>7.1f}  {ep_reward:>9.2f}  {grade}"
        )

    n = len(records)
    successes = [r for r in records if r["success"]]
    sr = len(successes) / n
    rewards = [r["reward"] for r in records]
    steps_list = [r["steps"] for r in records]
    fuels = [r["fuel"] for r in records]
    omegas_f = [r["omega_f"] for r in records]

    print("\n" + "=" * 65 + "\n  SUMMARY\n" + "=" * 65)
    print(f"  Success rate   : {sr * 100:>5.1f}%  {_bar(sr, 0, 1)}")
    print(f"  Mean Reward    : {np.mean(rewards):>9.2f}")
    print(f"  Mean Steps     : {np.mean(steps_list):>7.1f}")
    print(f"  Mean Final ω   : {np.mean(omegas_f):>8.5f} rad/s")
    print(f"  Mean Fuel Vol  : {np.mean(fuels):>8.1f}")

    if use_vec:
        venv.close()
    else:
        raw_env.close()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--agent", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--xml", default="scene_rigid.xml")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run_eval(
        agent_mode=args.agent,
        model_path=args.model,
        xml_path=args.xml,
        n_episodes=args.episodes,
        render=args.render,
        vecnorm_path=args.vecnorm,
        seed=args.seed,
    )
