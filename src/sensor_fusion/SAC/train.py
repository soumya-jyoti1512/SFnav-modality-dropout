import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from .env import NavEnvConfig, SensorFusionNavEnv
from .masking import MaskingConfig, ModalityMaskingWrapper
from .networks import NetworkConfig, ObsShape
from .replay_buffer import Episode, SequenceReplayBuffer
from .sac_agent import SACAgent, SACConfig

@dataclass
class TrainConfig:

    n_episodes: int = 150
    warmup_episodes: int= 10

    batch_size: int = 32
    seq_len: int = 15
    buffer_capacity: int= 100      
    updates_per_episode: int = 50
    min_buffer_episodes_for_updates: int =10

    eval_every_n_episodes: int = 25
    n_eval_episodes: int = 3

    log_every_n_episodes: int =5

    save_path: Optional[str] = None        
    #plot_path: Optional[str] = ""

    seed: int = 0

def collect_episode(
    env,
    agent: SACAgent,
    random_action: bool,
    action_dim: int,
    rng: np.random.Generator,
) -> tuple[Episode, float, str]:
   
    obs, _ = env.reset()
    hidden = agent.init_actor_hidden()

    obs_list: list[dict] = [{k: v.copy() for k, v in obs.items()}]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    total_reward =0.0
    reason = "unknown"

    while True:
        if random_action:
            action = rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        else:
            action, hidden = agent.act(obs, hidden, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(bool(terminated))  
        obs_list.append({k: v.copy() for k, v in next_obs.items()})
        total_reward +=float(reward)

        if terminated or truncated:
            reason = info.get("reason", "unknown")
            break
        obs = next_obs

    obs_stacked = {
        k: np.stack([o[k] for o in obs_list], axis=0)
        for k in obs_list[0]
    }
    ep = Episode(
        obs=obs_stacked,
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
    )
    return ep, total_reward, reason


def evaluate(env, agent: SACAgent, n_episodes: int) -> dict[str, float]:
    
    returns: list[float] = []
    lengths: list[int] = []
    goal_count = collision_count = timeout_count = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden = agent.init_actor_hidden()
        total_reward= 0.0
        steps = 0
        while True:
            action, hidden = agent.act(obs, hidden, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                reason = info.get("reason", "unknown")
                if reason == "goal_reached":
                    goal_count += 1
                elif reason =="collision":
                    collision_count += 1
                elif reason == "timeout":
                    timeout_count += 1
                break
            obs = next_obs
        returns.append(total_reward)
        lengths.append(steps)

    n = max(n_episodes, 1)
    return {
        "eval/return_mean":   float(np.mean(returns)),
        "eval/return_std":    float(np.std(returns)),
        "eval/length_mean":   float(np.mean(lengths)),
        "eval/goal_rate":     goal_count / n,
        "eval/collision_rate": collision_count / n,
        "eval/timeout_rate":  timeout_count / n,
    }


def train(
    train_cfg: Optional[TrainConfig] = None,
    nav_cfg: Optional[NavEnvConfig] = None,
    mask_cfg: Optional[MaskingConfig]= None,
    net_cfg: Optional[NetworkConfig] = None,
    sac_cfg: Optional[SACConfig] = None,
) -> dict:
    
    train_cfg = train_cfg or TrainConfig()
    nav_cfg = nav_cfg or NavEnvConfig(seed=train_cfg.seed)
    mask_cfg = mask_cfg or MaskingConfig(seed=train_cfg.seed)
    net_cfg = net_cfg or NetworkConfig()
    sac_cfg = sac_cfg or SACConfig()

    rng = np.random.default_rng(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    env_inner = SensorFusionNavEnv(nav_cfg)
    env = ModalityMaskingWrapper(env_inner, mask_cfg)

    spec = env.observation_spec
    obs_shape = ObsShape(
        imu=spec["imu"][0],
        lidar=spec["lidar"][0],
        rgbd=spec["rgbd"][0],
        goal_rel=spec["goal_rel"][0],
        mask=spec["mask"][0],
    )
    action_dim = env.action_spec["shape"][0]

    agent = SACAgent(obs_shape, action_dim, net_cfg, sac_cfg)
    buffer = SequenceReplayBuffer(
        capacity=train_cfg.buffer_capacity,
        seq_len=train_cfg.seq_len,
        seed=train_cfg.seed,
    )

    history: dict[str, list] = {
        "episode": [],
        "return":[],
        "length": [],
        "reason": [],
        "alpha":[],
        "loss/actor": [],
        "loss/critic1": [],
        "loss/alpha": [],
        "eval_episode": [],
        "eval_return":[],
        "eval_goal_rate": [],
    }

    print(f"\n{'=' * 60}")
    print(f"SAC sensor-fusion training")
    print(f"{'=' * 60}")
    print(f"Device: {agent.device}")
    print(f"Episodes: {train_cfg.n_episodes}")
    print(f"Warmup episodes: {train_cfg.warmup_episodes}")
    print(f"Buffer capacity: {train_cfg.buffer_capacity} episodes")
    print(f"Updates/episode: {train_cfg.updates_per_episode}")
    print(f"Batch / seq_len: {train_cfg.batch_size} / {train_cfg.seq_len}")
    print(f"Target entropy:{agent.target_entropy}")

    t_start = time.time()

    print(f"\n--- Warmup: {train_cfg.warmup_episodes} random episodes ")
    for i in range(train_cfg.warmup_episodes):
        ep, total_reward, reason = collect_episode(
            env, agent, random_action=True,
            action_dim=action_dim, rng=rng,
        )
        buffer.push_episode(ep)
        if (i + 1) % 5 == 0 or i + 1 == train_cfg.warmup_episodes:
            print(f"  warmup {i+1:>3}/{train_cfg.warmup_episodes}: "
                  f"return={total_reward:>+7.2f}, len={ep.length:>3}, "
                  f"reason={reason}")

    print(f"\n--- Training: {train_cfg.n_episodes} episodes ")
    print(f"{'ep':>4}  {'return':>8}  {'len':>4}  {'reason':>13}  "
          f"{'alpha':>6}  {'L_actor':>9}  {'L_critic':>9}  "
          f"{'eval_ret':>9}  {'goal%':>5}")

    last_eval_ret = float('nan')
    last_eval_goal = float('nan')

    for ep_idx in range(train_cfg.n_episodes):
        ep, total_reward, reason = collect_episode(
            env, agent, random_action=False,
            action_dim=action_dim, rng=rng,
        )
        buffer.push_episode(ep)

        ep_metrics: list[dict] = []
        if len(buffer) >= train_cfg.min_buffer_episodes_for_updates:
            for _ in range(train_cfg.updates_per_episode):
                batch = buffer.sample(train_cfg.batch_size)
                ep_metrics.append(agent.update(batch))

        if ep_metrics:
            avg = {k: float(np.mean([m[k] for m in ep_metrics]))
                   for k in ep_metrics[0]}
        else:
            avg = {"alpha": float('nan'),
                   "loss/actor": float('nan'),
                   "loss/critic1": float('nan'),
                   "loss/alpha": float('nan')}

        history["episode"].append(ep_idx)
        history["return"].append(total_reward)
        history["length"].append(ep.length)
        history["reason"].append(reason)
        for k in ["alpha", "loss/actor", "loss/critic1", "loss/alpha"]:
            history[k].append(avg.get(k, float('nan')))

        if (ep_idx + 1) % train_cfg.eval_every_n_episodes == 0:
            m = evaluate(env, agent, train_cfg.n_eval_episodes)
            history["eval_episode"].append(ep_idx)
            history["eval_return"].append(m["eval/return_mean"])
            history["eval_goal_rate"].append(m["eval/goal_rate"])
            last_eval_ret = m["eval/return_mean"]
            last_eval_goal = m["eval/goal_rate"] * 100.0

        if (ep_idx + 1) % train_cfg.log_every_n_episodes == 0 or ep_idx == 0:
            print(f"{ep_idx+1:>4}  {total_reward:>+8.2f}  {ep.length:>4}  "
                  f"{reason:>13}  {avg['alpha']:>6.3f}  "
                  f"{avg['loss/actor']:>+9.3f}  {avg['loss/critic1']:>+9.3f}  "
                  f"{last_eval_ret:>+9.2f}  {last_eval_goal:>5.1f}")

    elapsed = time.time() - t_start
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print("\n--- Final evaluation (10 episodes, deterministic) ---")
    final = evaluate(env, agent, n_episodes=10)
    for k, v in final.items():
        print(f"  {k:>22}: {v:.4f}")

    if train_cfg.save_path is not None:
        agent.save(train_cfg.save_path)
        print(f"\nCheckpoint saved to {train_cfg.save_path}")

    env.close()

    if train_cfg.plot_path is not None:
        _try_plot(history, train_cfg.plot_path)

    return history
"""
def _try_plot(history: dict, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    
        return

    eps = np.array(history["episode"])
    rets = np.array(history["return"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # (1) Episode return + smoothed + eval
    ax = axes[0, 0]
    ax.plot(eps, rets, alpha=0.3, label="per-episode")
    if len(rets) >= 10:
        window = max(5, len(rets) // 20)
        smoothed = np.convolve(rets, np.ones(window) / window, mode='valid')
        ax.plot(eps[window - 1:], smoothed,
                label=f"rolling mean (w={window})", linewidth=2)
    if history["eval_episode"]:
        ax.plot(history["eval_episode"], history["eval_return"],
                "o-", label="eval (deterministic)", linewidth=2)
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("Episode return")
    ax.legend()
    ax.grid(True)

    # (2) Alpha
    ax = axes[0, 1]
    ax.plot(eps, history["alpha"])
    ax.set_xlabel("episode")
    ax.set_ylabel("alpha")
    ax.set_title("Entropy temperature over training")
    ax.grid(True)

    # (3) Losses
    ax = axes[1, 0]
    ax.plot(eps, history["loss/actor"], label="actor", alpha=0.8)
    ax.plot(eps, history["loss/critic1"], label="critic1", alpha=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("loss")
    ax.set_title("Training losses (per-episode avg)")
    ax.legend()
    ax.grid(True)

    # (4) Goal-reached rate (eval)
    ax = axes[1, 1]
    if history["eval_episode"]:
        goal_pct = [g * 100.0 for g in history["eval_goal_rate"]]
        ax.plot(history["eval_episode"], goal_pct, "o-", linewidth=2)
        ax.set_ylim(-5, 105)
    ax.set_xlabel("episode")
    ax.set_ylabel("goal-reached rate (%)")
    ax.set_title("Eval goal-reached rate")
    ax.grid(True)

    fig.suptitle("SAC sensor-fusion training")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Training plot saved to {path}")
"""

if __name__ == "__main__":
    
    QUICK = False

    if QUICK:
        train(train_cfg=TrainConfig(
            n_episodes=10,
            warmup_episodes=3,
            updates_per_episode=10,
            eval_every_n_episodes=5,
            n_eval_episodes=2,
            log_every_n_episodes=1,
            plot_path=None,
        ))
    else:
        train()
