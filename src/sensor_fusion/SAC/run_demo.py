import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import torch
from ..types import Modality
from .env import MASK_ORDER, NavEnvConfig, SensorFusionNavEnv
from .masking import MaskingConfig, MaskingMode, ModalityMaskingWrapper
from .networks import NetworkConfig, ObsShape
from .sac_agent import SACAgent, SACConfig

_MOD_TO_OBSKEY = {
    Modality.IMU: "imu",
    Modality.LIDAR: "lidar",
    Modality.RGBD: "rgbd",
}


class DeterministicMaskWrapper:
    
    def __init__(self, env, drop: set[Modality]) -> None:
        self.env = env
        self.drop =set(drop)

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        obs, info =self.env.reset(seed=seed)
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated,truncated, info = self.env.step(action)
        return self._mask(obs), reward,terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _mask(self, obs: dict) -> dict:
        new_obs = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in obs.items()}
        new_mask = new_obs["mask"].copy()
        for i, m in enumerate(MASK_ORDER):
            if m in self.drop:
                key = _MOD_TO_OBSKEY[m]
                new_obs[key] = np.zeros_like(new_obs[key])
                new_mask[i] = 0.0
        new_obs["mask"] = new_mask
        return new_obs


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable                  


def _default_scenarios(mask_seed: int = 0) -> list[Scenario]:
   
    return [
        Scenario(
            "clean", "no masking (baseline)",
            lambda e: e,
        ),
        Scenario(
            "no_imu", "IMU permanently dropped",
            lambda e: DeterministicMaskWrapper(e, {Modality.IMU}),
        ),
        Scenario(
            "no_lidar", "LiDAR permanently dropped",
            lambda e: DeterministicMaskWrapper(e, {Modality.LIDAR}),
        ),
        Scenario(
            "no_rgbd", "RGB-D permanently dropped",
            lambda e: DeterministicMaskWrapper(e, {Modality.RGBD}),
        ),
        Scenario(
            "train_dist", "training-distribution block masking",
            lambda e: ModalityMaskingWrapper(
                e, MaskingConfig(mode=MaskingMode.BLOCK, seed=mask_seed)),
        ),
        Scenario(
            "all_dropped", "all sensors permanently dropped",
            lambda e: DeterministicMaskWrapper(
                e, {Modality.IMU, Modality.LIDAR, Modality.RGBD}),
        ),
    ]


@dataclass
class DemoConfig:
    checkpoint_path: str = "sac.pt"
    n_episodes_per_scenario: int = 10
    deterministic_policy: bool =True
    #plot_path: [str] = "sac_demo.png"
    seed: int = 0


@dataclass
class EpisodeResult:
    total_return: float
    length: int
    reason: str
    final_distance: float


def evaluate_with_details(
    env, agent: SACAgent, n_episodes: int, deterministic: bool = True,
) -> list[EpisodeResult]:
   
    results: list[EpisodeResult] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden = agent.init_actor_hidden()
        total_reward =0.0
        steps =0
        info: dict = {}
        while True:
            action, hidden = agent.act(obs, hidden, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
            obs = next_obs
        results.append(EpisodeResult(
            total_return=total_reward,
            length=steps,
            reason=info.get("reason", "unknown"),
            final_distance=float(info.get("goal_distance", float("nan"))),
        ))
    return results



def _aggregate(results: list[EpisodeResult]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {"n": 0, "success": 0.0, "collision": 0.0, "timeout": 0.0,
                "return_mean": 0.0, "return_std": 0.0, "length_mean": 0.0}
    success = sum(1 for r in results if r.reason == "goal_reached") / n
    collision = sum(1 for r in results if r.reason == "collision") / n
    timeout = sum(1 for r in results if r.reason == "timeout") / n
    rets = np.array([r.total_return for r in results])
    lens = np.array([r.length for r in results])
    return {
        "n":          n,
        "success": success,
        "collision": collision,
        "timeout":timeout,
        "return_mean": float(rets.mean()),
        "return_std":float(rets.std()),
        "length_mean": float(lens.mean()),
    }


def _print_table(per_scenario: dict[str, dict[str, float]]) -> None:
    header = (f"{'scenario':>14}  {'N':>3}  {'success%':>8}  "
              f"{'coll%':>5}  {'time%':>5}  {'ret_mean':>9}  "
              f"{'ret_std':>8}  {'len_mean':>8}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, stats in per_scenario.items():
        print(f"{name:>14}  {stats['n']:>3}  "
              f"{stats['success']*100:>7.1f}%  "
              f"{stats['collision']*100:>4.1f}%  "
              f"{stats['timeout']*100:>4.1f}%  "
              f"{stats['return_mean']:>+9.2f}  "
              f"{stats['return_std']:>8.2f}  "
              f"{stats['length_mean']:>8.1f}")
    print("=" * len(header))


"""

def _try_plot(per_scenario: dict[str, dict[str, float]], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot.")
        return

    names = list(per_scenario.keys())
    x = np.arange(len(names))

    success   = np.array([per_scenario[n]["success"]   for n in names]) * 100
    collision = np.array([per_scenario[n]["collision"] for n in names]) * 100
    timeout   = np.array([per_scenario[n]["timeout"]   for n in names]) * 100

    ret_mean = np.array([per_scenario[n]["return_mean"] for n in names])
    ret_std  = np.array([per_scenario[n]["return_std"]  for n in names])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # ---- Panel 1: stacked outcome bars ----
    ax = axes[0]
    ax.bar(x, success, label="goal reached",
           color="tab:green", edgecolor="black")
    ax.bar(x, collision, bottom=success, label="collision",
           color="tab:red", edgecolor="black")
    ax.bar(x, timeout, bottom=success + collision, label="timeout",
           color="lightgray", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("episodes (%)")
    ax.set_title("Termination outcome by scenario")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # ---- Panel 2: mean return with std error bars ----
    ax = axes[1]
    ax.bar(x, ret_mean, yerr=ret_std,
           color="tab:blue", edgecolor="black",
           ecolor="black", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("episode return")
    ax.set_title("Mean episode return by scenario (error bars: std)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("SAC sensor-fusion: generalization across dropout scenarios",
                 fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved to {path}")
"""

def run_demo(
    cfg: Optional[DemoConfig] = None,
    nav_cfg: Optional[NavEnvConfig] = None,
    net_cfg: Optional[NetworkConfig] = None,
    sac_cfg: Optional[SACConfig] = None,
) -> dict[str, dict[str, float]]:
    
    cfg = cfg or DemoConfig()
    nav_cfg = nav_cfg or NavEnvConfig(seed=cfg.seed)
    net_cfg = net_cfg or NetworkConfig()
    sac_cfg = sac_cfg or SACConfig()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    base_env = SensorFusionNavEnv(nav_cfg)
    spec = base_env.observation_spec
    obs_shape = ObsShape(
        imu=spec["imu"][0],
        lidar=spec["lidar"][0],
        rgbd=spec["rgbd"][0],
        goal_rel=spec["goal_rel"][0],
        mask=spec["mask"][0],
    )
    action_dim = base_env.action_spec["shape"][0]

    agent = SACAgent(obs_shape, action_dim, net_cfg, sac_cfg)

    print(f"{'='*60}")
    print(f"SAC eval demo")
    print(f"{'='*60}")
    print(f"Checkpoint:{cfg.checkpoint_path}")
    print(f"Episodes/scen.: {cfg.n_episodes_per_scenario}")
    print(f"Deterministic:{cfg.deterministic_policy}")

    if os.path.exists(cfg.checkpoint_path):
        agent.load(cfg.checkpoint_path)
        print(f"Loaded checkpoint from {cfg.checkpoint_path}\n")
    else:
        print(f"\n no checkpoint at '{cfg.checkpoint_path}'.")

    scenarios = _default_scenarios(mask_seed=cfg.seed)
    per_scenario: dict[str, dict[str, float]] = {}
    for sc in scenarios:
        print(f"--- {sc.name}: {sc.description} ---")
        env = sc.build(base_env)
        results = evaluate_with_details(
            env, agent, cfg.n_episodes_per_scenario,
            deterministic=cfg.deterministic_policy,
        )
        per_scenario[sc.name] = _aggregate(results)
        s = per_scenario[sc.name]
        print(f"   success={s['success']*100:.1f}% collision={s['collision']*100:.1f}%  "
              f"timeout={s['timeout']*100:.1f}%  return={s['return_mean']:+.2f}")

    _print_table(per_scenario)

    if cfg.plot_path is not None:
        _try_plot(per_scenario, cfg.plot_path)

    base_env.close()
    return per_scenario


if __name__ == "__main__":
    run_demo()
