

# Everything You Need — Project UnTumble Research Paper

---

## 🚀 Executive Project Summary

**Project UnTumble** is an Active Debris Removal (ADR) Deep Reinforcement Learning (DRL) study. It investigates continuous vs. discrete control techniques for stabilizing a flexible, non-cooperative target spacecraft post-capture. The 4-part ablation study uses Proximal Policy Optimization (PPO) to prove that a hybrid architecture—combining Pseudo-Random Pulse-Width Pulse-Frequency (PWPF) thruster modulation with continuous Reaction Wheels (RWs)—synergistically breaks the Minimum Impulse Bit (MIB) performance floor while minimizing structural excitation.

**Current Repository Layout**

```bash
.
├── agents
│   └── train.py
├── assets
│   └── xml
│       └── scene.xml
├── env
│   ├── agent_1_env.py
│   ├── agent_2_env.py
│   ├── agent_3_env.py
│   ├── agent_4_env.py
│   ├── base_env.py
│   └── callbacks.py
```

---

## PHASE 1 — Foundation & Experimental Design

### 1.1 Problem Formalization

- ~~**Mathematical Problem Statement:**~~ *(Done: Defined explicitly before coding.)*
- ~~**System State Vector:**~~ *(Done: Defined in 17 dimensions: q, ω, flex_pos, flex_vel, rw_vels.)*
- ~~**Control Objective:**~~ *(Done: Minimize effort and flex energy while driving ω to zero.)*
- ~~**Assumptions:**~~ *(Done: Locked to rigid post-dock weld, zero-disturbance environment, lumped-parameter flex.)*
- ~~**Numerical Success Criteria:**~~ *(Done: Residual ω < 0.015 rad/s held for 50 steps [5 seconds]. Time limit = 1000 steps [100s]. Structural failure boundary = 0.15 rad deflection.)*

### 1.2 Hypothesis Statement

- ~~**H1 (Sub-Impulse):** PWPF breaks the Bang-Bang MIB floor.~~ *(Done)*
- ~~**H2 (Structural Isolation):** RWs absorb momentum without impulsive flex excitation.~~ *(Done)*
- ~~**H3 (Heterogeneous Synergy):** Agent 4 (PWPF + RW) achieves lowest combined flex/fuel cost.~~ *(Done)*
- ~~**H4 (Autonomous Saturation):** PPO manages momentum dumping without explicit state machines.~~ *(Done)*

### 1.3 Metrics Definition (define ALL before training)

*(Done: All explicitly defined mathematically and tracked via Welford's online algorithm in callbacks.)*

| Metric                        | What it measures                                      |
|-------------------------------|-------------------------------------------------------|
| **Residual ω (rad/s)**        | Final angular rate at episode end                     |
| **Detumble time (s)**         | Time to cross ω threshold                             |
| **Total ΔV used (m/s)**       | RCS fuel cost                                         |
| **Peak flex angle (rad)**     | Max structural deflection                             |
| **RMS flex velocity (rad/s)** | Sustained structural vibration energy                 |
| **Flex excitation energy (J)**| `0.5 * k * θ² + 0.5 * I * θ̇²` — key non-rigid metric  |
| **RW saturation events**      | How many times wheels hit speed limit                 |

---

## PHASE 2 — Simulation Environment

### 2.1 MuJoCo Scene Validation

- ~~**Combined COM Position:**~~ *(Done: True Global COM calculated at `[0.0111, 0.0000, 3.5575]`. The 1.1cm X-axis shift is intentional due to asymmetric boom load, strengthening the RL robustness claim.)*
- ~~**Inertia Tensor:**~~ *(Done: Total system mass accurately calculated at 3760.38 kg, including 320.38 kg of dynamically calculated flexible mass.)*
- ~~**RCS Thruster Torque:**~~ *(Done: Actuator geometries verified. High cross-coupling confirmed due to asymmetric lever arms from the Z=3.5575m COM.)*
- ~~**RW Pyramid Geometry:**~~ *(Done: Torque Allocation Matrix is mathematically Rank 3, granting full 3-axis authority.)*
- ~~**Debris-side IMU:**~~ *(Done: Duplicate PAF removed, debris gyro added.)*
- ~~**nstack Deprecation:**~~ *(Done: Warning fixed.)*

### 2.2 Environment Wrapper (per agent, but shared base)

- ~~**Domain Randomization `reset()`:**~~ *(Done: Initial ω uniform [-0.5, 0.5] rad/s. Flex stiffness ±30% [range 5.6 to 10.4]. Inertia/Mass ±15%.)*
- ~~**Actuator Logic `step()`:**~~ *(Done: Re-architected into `base_env.py` and four modular agent wrappers to ensure physics consistency across discrete/continuous spaces.)*
- ~~**Observation Space:**~~ *(Done: Normalized using VecNormalize wrapper.)*
- ~~**Action Space:**~~ *(Done: Defined via gym spaces per agent.)*
- ~~**Termination Conditions:**~~ *(Done: Success threshold, timeout, and flex failure logic active.)*

### 2.3 Action Spaces Per Agent

*(Done: Mapped precisely in respective env files.)*

| Agent       | Space                                      | Dims |
|-------------|--------------------------------------------|------|
| **Agent 1** | MultiDiscrete [2]*12                       | 12   |
| **Agent 2** | Tuple(MultiDiscrete [2]*12, Box[-1,1]^4)   | 16   |
| **Agent 3** | Box[0,1]^12                                | 12   |
| **Agent 4** | Box[0,1]^12 + Box[-1,1]^4                  | 16   |

### 2.4 Reward Function Design

*(Done: Hyperparameters locked in `base_env.py`.)*

- ~~**Coarse phase reward:**~~ *(Done: `-10.0 * ‖ω‖`)*
- ~~**Flex penalty:**~~ *(Done: `-100.0 * flex_excitation_energy` - High weight because Joules are numerically small)*
- ~~**Fuel penalty:**~~ *(Done: `-0.01`)*
- ~~**Time penalty:**~~ *(Done: `-0.01` per step)*
- ~~**Saturation penalty:**~~ *(Done: `-5.0` for Agents 2 and 4)*
- ~~**Success bonus:**~~ *(Done: `+100.0`)*

### 2.5 PWPF Modulator (Agent 3 & 4)

- ~~**Dynamic Block Implementation:**~~ *(Done: `PWPFModulator` class instantiated. Runs at high-frequency 100Hz MuJoCo physics loop inside the standard 10Hz RL step loop.)*

---

## PHASE 3 — Training

### 3.1 PPO Hyperparameter Tuning

- ~~**Network Architectures:**~~ *(Done: Agents 1/3 use `[256, 256]`. Agents 2/4 use deeper `[512, 256, 128]` networks due to heterogeneous actuator complexity.)*
- ~~**Entropy Coefficient:**~~ *(Done: Agent 2 uses higher entropy `0.015` to explore state machine boundaries. Others use `0.005` to `0.01`.)*

### 3.2 Training Protocol

- ~~**Seed Requirement:**~~ *(Done: Script configured for seeds 10, 20, 30, 40, 50.)*
- ~~**TensorBoard Logging:**~~ *(Done: Extended with `WelfordMetricsCallback` to track exact ablation metrics in the console and TensorBoard.)*
- ~~**Normalization Save Protocol:**~~ *(Done: Explicitly saving `VecNormalize.pkl` alongside `.zip` brain and raw `.pth` PyTorch weights to ensure evaluation scaling matches training scaling.)*

### 3.3 Computational Requirements

- ~~**Parallelization:**~~ *(Done: `SubprocVecEnv` configured, defaulting to 8 parallel cores.)*
- **Estimate Training Time:** Estimate time per agent and document for reproducibility.

---

## PHASE 4 — Evaluation

### 4.1 Evaluation Protocol

- **1000 Random Episodes:** Evaluate each trained agent (not training seeds).
- **Test Distribution:** Use same domain randomization range as training.
- **Out-of-Distribution Test:** Push ω to edges (e.g. 0.8 rad/s initial rate).
- **Metrics Calculation:** Compute mean ± std for all 7 metrics across 1000 episodes per agent.

### 4.2 Statistical Analysis

- **Data Reporting:** Report mean, std, median, 5th/95th percentile for every metric.
- **Significance Tests:** Run Welch's t-test or Mann-Whitney U between agent pairs.
- **P-values:** Report p-values to prove statistical significance.
- **Learning Curves:** Plot with shaded std across 5 seeds.

### 4.3 Ablation Verification

Build the 2×2 result table:

Every cell: residual ω, flex energy, fuel cost. (This is the main result.):

|               | No Reaction Wheel     | With Reaction Wheel    |
|---------------|-----------------------|------------------------|
| **Bang-Bang** | Agent 1               | Agent 2                |
|   **PWPF**    | Agent 3               | Agent 4                |

### 4.4 Visualization

- **Figure 1:** System diagram (chaser+debris stack, labeled with actuators, sensors, COM).
- **Figure 2:** State machine diagram (Agent 2 phase transitions).
- **Figure 3:** PWPF block diagram (filter + Schmitt trigger).
- **Figure 4:** Learning curves (all 4 agents overlaid, mean ± std).
- **Figure 5:** Angular rate convergence (time-series ω for best episode per agent).
- **Figure 6:** Flex excitation comparison (bar chart, flex energy per agent ± std).
- **Figure 7:** 2×2 ablation table (main result, all metrics).
- **Figure 8:** Phase space trajectory (ωx, ωy, ωz 3D trajectory for Agent 1 vs Agent 4).
- **Figure 9:** Torque profile (thruster firing pattern vs PWPF output).
- **Figure 10:** Reward curve components (stacked plot showing how each reward term evolves).

---

## PHASE 5 — Paper Writing

### 5.1 Paper Structure

**Abstract** (150–200 words)  
**1. Introduction** (Kessler Syndrome, non-rigid dynamics, gap in existing work, contribution).  
**2. Related Work** (ADR missions, RL attitude control, classical GNC, flexible dynamics).  
**3. System Model** (Rigid body, flexible appendage, RCS bang-bang, PWPF modulator, RW pyramid, MuJoCo parameters).  
**4. GNC Architecture & RL Formulation** (MDP, observation/action spaces, reward equation, PPO multi-categorical handling).  
**5. Experimental Setup** (Scene parameters including 3760kg baseline, domain randomization, hyperparameters).  
**6. Results** (Learning curves, 2×2 table, angular rate convergence, flex excitation analysis, fuel efficiency, generalization).  
**7. Discussion** (MIB floor physics, why PWPF breaks the floor, sim-to-real gap limitations).  
**8. Conclusion** (Restate contributions, quantitative findings, hardware-in-loop future work).  
**References** (25–35 sources).  
**Appendix** (Full hyperparameter/PWPF tables, additional curves).

### 5.2 LaTeX Requirements

- **Template:** Use AIAA or AAS template.
- **Formatting:** All equations numbered. All figures vector (PDF/SVG).
- **Self-Contained Captions:** Reader shouldn't need body text to understand a figure.
- **Acronyms:** Define all on first use.

---

## PHASE 6 — Submission

### 6.1 Target Venues (in order)

1. **AAS/AIAA Astrodynamics Specialist Conference** (Best fit).
2. **AIAA SciTech Forum** (January event).
3. **ICRA** (If emphasizing robotics/manipulation).
4. **Acta Astronautica** (Journal track).

### 6.2 Pre-submission Checklist

- **Reproducibility:** Release code on GitHub with README, conda env file, trained model weights.
- **Statistical Backing:** All claims backed by stat tests with p-values.
- **References:** Every figure referenced in text, no orphaned references.
- **Final Checks:** Page limit check, plagiarism checker, external reader review.

---

## PHASE 7 — Parallel Track

- **Literature Review:** 30+ papers mapped to the research gap.
- **Nomenclature Table:** Every symbol defined from day 1.
- **Lab Notebook:** Date-stamped notes on design decisions (e.g., specific reward weights).
- **Version Control:** Every experiment tagged in git.
- **Early Figures:** Rough versions drafted before writing prose.

---

## Rough Timeline

| Week     | Milestone                                                      |
|----------|----------------------------------------------------------------|
| **1–2**  | Env base class, domain randomization, all metrics instrumented |
| **3**    | Agent 2 env + state machine, verify in sim                     |
| **4–5**  | Train Agent 2, 5 seeds                                         |
| **6**    | PWPF implementation + Agent 3 env                              |
| **7–8**  | Train Agent 3, 5 seeds                                         |
| **9**    | Agent 4 env + training                                         |
| **10**   | Full evaluation, 1000 episodes per agent                       |
| **11**   | All figures generated                                          |
| **12–14**| Paper writing                                                  |
| **15**   | Internal review + revision                                     |
| **16**   | Submit                                                         |
