# UntumbleRL

Reinforcement learning for **spacecraft detumbling** — training a control policy to stabilize a tumbling satellite's angular velocity, in a scenario aligned with JAXA's CRD2 (Commercial Removal of Debris Demonstration) program, where an approaching servicer must handle an uncooperative, tumbling target.

Detumbling is a foundational problem for on-orbit servicing and debris removal: before any capture or docking maneuver, the target's rotation has to be brought under control. This project trains and compares RL agents across different actuator assumptions to identify which control strategies are actually effective for this regime.

## Problem setup

- **Simulator**: MuJoCo, using a custom rigid-body scene (`scene_rigid.xml`) representing the tumbling satellite
- **Environment**: `rigid_untumble_env.py` — a Gym-style environment exposing angular velocity/attitude state and actuator commands, with reward shaped around reducing tumble rate to near-zero within an episode

## Ablation study

The core of this project is a 4-agent ablation comparing actuator strategies, isolating two independent design choices:

|  | RCS-only | RW-augmented |
|---|---|---|
| **Bang-bang control** | Agent A | Agent B |
| **PWPF modulation** | Agent C | Agent D |

- **Bang-bang** — simple on/off thruster firing, the simplest actuation policy
- **PWPF (Pulse-Width Pulse-Frequency) modulation** — pulse modulation that approximates continuous thrust from on/off actuators, standard in real spacecraft attitude control
- **RCS-only vs. RW-augmented** — reaction-control-system thrusters alone, vs. thrusters combined with reaction wheels for finer control authority

This isolates *how much* control smoothing (PWPF vs. bang-bang) and *actuator augmentation* (adding reaction wheels) each contribute to detumbling performance, rather than training one agent and reporting a single number.

## Reward design

Reward shaping went through multiple iterations to close exploits the agent found in earlier versions — cases where it satisfied the immediate reward signal without actually stabilizing the tumble. The final reward function jointly penalizes residual angular velocity, control effort, and time-to-stabilize, closing off the degenerate shortcuts found during earlier training runs.

## Training and evaluation

- `train.py` — trains an agent (PPO) for a given actuator configuration
- `eval.py` — evaluates a trained policy, logging performance against the baseline
- `logs/rigid_baseline/` — baseline training logs
- A trained model checkpoint (`ppo_rigid_untumble_final.zip`) is included for evaluation without retraining
- `flow.md` documents the project's development flow and notes

## Status

Active research project exploring detumbling control for uncooperative-target servicing scenarios. Techniques from this line of work have informed real onboard control applications. Code is in local-research form — no packaged release yet; run directly via `train.py` / `eval.py` with MuJoCo installed.

## Tech

Python, MuJoCo, Stable-Baselines3 (PPO), custom Gym-style environment
