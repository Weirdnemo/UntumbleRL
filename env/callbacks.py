from stable_baselines3.common.callbacks import BaseCallback


class WelfordMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WelfordMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                # Log all metrics/ keys
                for key in [
                    "residual_omega",
                    "peak_flex_angle",
                    "flex_breach_event",
                    "stable_steps",
                ]:
                    full_key = f"metrics/{key}"
                    if full_key in info:
                        self.logger.record(full_key, info[full_key])

                # Log all reward components directly
                for key in [
                    "r_omega_diff",
                    "r_omega_near",
                    "r_flex_base",
                    "r_flex_soft",
                    "r_delta_flex",
                    "r_flex_mech",
                    "r_delta_action",
                    "r_rw_sat",
                    "r_fuel",
                    "r_time",
                    "r_terminal",
                    "total_reward",
                ]:
                    if key in info:
                        self.logger.record(f"reward_dist/{key}", info[key])

                # Log curriculum param
                if "k_delta_flex_current" in info:
                    self.logger.record(
                        "curriculum/k_delta_flex", info["k_delta_flex_current"]
                    )

        return True
