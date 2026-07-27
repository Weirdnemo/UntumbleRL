import time

from stable_baselines3 import PPO

from rigid_untumble_env import RigidUnTumbleEnv


def evaluate_model():
    print("Initializing MuJoCo Viewer...")

    # 1. Initialize the environment in 'human' render mode
    # Ensure xml_path points correctly to your scene_rigid.xml
    env = RigidUnTumbleEnv(xml_path="scene_rigid.xml", render_mode="human")

    # 2. Load the trained model
    # The EvalCallback from our train.py saved the absolute best performing
    # deterministic weights here. We want to use these over the final timestep weights.
    model_path = "ppo_rigid_untumble_final.zip"

    try:
        model = PPO.load(model_path, env=env)
        print(f"Successfully loaded: {model_path}")
    except FileNotFoundError:
        backup_path = "ppo_rigid_untumble_final"
        print(f"Could not find best_model.zip. Falling back to: {backup_path}")
        model = PPO.load(backup_path, env=env)

    # 3. Run the evaluation loop
    episodes = 500
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Starting Episode {ep + 1} ---")

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Render the frame to the MuJoCo window
            env.render()

            time.sleep(0.02)

        print(f"Episode {ep + 1} finished.")
        print(f"Steps to stabilize: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Angular Velocity Norm: {info['omega_norm']:.4f} rad/s")
        print(f"Final Linear Velocity Norm: {info['v_norm']:.4f} m/s")
        print(f"Mission Success: {info['is_success']}")

        time.sleep(1.5)

    env.close()
    print("\nEvaluation complete. Shutting down viewer.")


if __name__ == "__main__":
    evaluate_model()
