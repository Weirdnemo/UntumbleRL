import os

import gymnasium as gym
import numpy as np

# Import your agent's specific environment
from envio.agent_1_env import Agent1Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate_agent(agent_name="Agent2", seed="10", num_episodes=1000):
    print(f"==================================================")
    print(f"🚀 INITIATING PHASE 4 EVALUATION: {agent_name} (Seed {seed}) 🚀")
    print(f"==================================================")

    model_dir = f"./models/{agent_name}/seed_{seed}/"
    model_path = os.path.join(model_dir, "final_model.zip")
    stats_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(f"❌ Error: Could not find model or normalization files in {model_dir}")
        return


    env = DummyVecEnv([lambda: Agent1Env()])

    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    print(f"Loaded successfully. Running {num_episodes} deterministic episodes...\n")
    eval_metrics = {
        "residual_omega": [],
        "detumble_time": [],
        "total_dv_used": [],
        "peak_flex_angle": [],
        "rms_flex_vel": [],
        "max_flex_energy": [],
        "rw_sat_events": [],
    }

    for ep in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = dones[0]

            if done:
                info = infos[0]
                eval_metrics["residual_omega"].append(info.get("residual_omega", 0.0))
                eval_metrics["detumble_time"].append(info.get("detumble_time", 0.0))
                eval_metrics["total_dv_used"].append(info.get("total_dv_used", 0.0))
                eval_metrics["peak_flex_angle"].append(info.get("peak_flex_angle", 0.0))
                eval_metrics["rms_flex_vel"].append(info.get("rms_flex_vel", 0.0))
                eval_metrics["max_flex_energy"].append(info.get("max_flex_energy", 0.0))
                eval_metrics["rw_sat_events"].append(info.get("rw_sat_events", 0))

        if (ep + 1) % 100 == 0:
            print(f"Completed {ep + 1}/{num_episodes} episodes...")

    # 4. Calculate and Print Final Statistics for the Paper
    print("\n==================================================")
    print(f"📊 FINAL PHASE 4 RESULTS: {agent_name} 📊")
    print("==================================================")

    for key, values in eval_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key:<20}: {mean_val:>8.4f} ± {std_val:>6.4f}")

    print("==================================================")


if __name__ == "__main__":
    evaluate_agent(agent_name="Agent1", seed="10", num_episodes=1000)
