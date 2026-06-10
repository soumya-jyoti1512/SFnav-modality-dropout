import copy
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch
from torch.optim import Adam
from .networks import (
    Actor,
    Critic,
    NetworkConfig,
    ObsShape,
    dict_obs_to_torch,
    init_hidden,
)


@dataclass
class SACConfig:

    gamma: float = 0.99                  
    tau: float =0.005                   
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_log_alpha: float= 0.0       
    target_entropy: Optional[float] = None  
    grad_clip_norm: float = 1.0
    device: str = "cpu"


class SACAgent:

    def __init__(
        self,
        obs_shape: ObsShape,
        action_dim: int,
        net_cfg: NetworkConfig,
        sac_cfg: SACConfig,
    ) -> None:
        self.action_dim = action_dim
        self.net_cfg =net_cfg
        self.cfg =sac_cfg
        self.device = torch.device(sac_cfg.device)

        self.target_entropy = float(
            sac_cfg.target_entropy if sac_cfg.target_entropy is not None
            else -float(action_dim)
        )

        self.actor = Actor(obs_shape, action_dim, net_cfg).to(self.device)
        self.critic1 = Critic(obs_shape, action_dim, net_cfg).to(self.device)
        self.critic2 = Critic(obs_shape, action_dim, net_cfg).to(self.device)

        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        for p in self.critic1_target.parameters():
            p.requires_grad =False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(
            sac_cfg.initial_log_alpha,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        self.actor_optim = Adam(self.actor.parameters(), lr=sac_cfg.actor_lr)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=sac_cfg.critic_lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=sac_cfg.critic_lr)
        self.alpha_optim = Adam([self.log_alpha], lr=sac_cfg.alpha_lr)


    @torch.no_grad()
    def act(
        self,
        obs: dict[str, np.ndarray],
        hidden: tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        
        obs_t = dict_obs_to_torch(obs, device=self.device, add_batch_seq=True)
        if deterministic:
            action, new_hidden = self.actor.act_deterministic(obs_t, hidden)
        else:
            action, _, new_hidden = self.actor.sample(obs_t, hidden)
        action_np = action.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        return action_np, new_hidden

    def init_actor_hidden(self) -> tuple[torch.Tensor, torch.Tensor]:
        return init_hidden(1, self.net_cfg, device=self.device)


    def update(self, batch: dict[str, Any]) -> dict[str, float]:
       
        obs = {k: torch.from_numpy(v).float().to(self.device)
               for k, v in batch["obs"].items()}
        next_obs = {k: torch.from_numpy(v).float().to(self.device)
                    for k, v in batch["next_obs"].items()}
        actions = torch.from_numpy(batch["actions"]).float().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)
        valid = torch.from_numpy(batch["valid"]).float().to(self.device)

        valid_sum = valid.sum().clamp(min=1.0)
        alpha = self.log_alpha.exp().detach()  

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_t, _ =self.critic1_target(next_obs, next_action)
            q2_t, _ =self.critic2_target(next_obs, next_action)
            min_q_target = torch.min(q1_t, q2_t) - alpha * next_log_prob
            target = rewards + self.cfg.gamma * (1.0 - dones) * min_q_target

        q1, _ = self.critic1(obs, actions)
        q2, _ = self.critic2(obs, actions)

        critic1_loss = (((q1 - target) ** 2) * valid).sum()/ valid_sum
        critic2_loss = (((q2 - target) ** 2) * valid).sum() / valid_sum

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(), self.cfg.grad_clip_norm)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(), self.cfg.grad_clip_norm)
        self.critic2_optim.step()

        new_action, new_log_prob, _ = self.actor.sample(obs)
        q1_new, _ = self.critic1(obs, new_action)
        q2_new, _ = self.critic2(obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = ((alpha * new_log_prob - min_q_new) * valid).sum() / valid_sum

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_optim.step()

        alpha_loss = -((self.log_alpha
                        * (new_log_prob.detach() +self.target_entropy))
                       * valid).sum() / valid_sum
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        self._polyak(self.critic1_target, self.critic1)
        self._polyak(self.critic2_target, self.critic2)

        with torch.no_grad():
            metrics = {
                "loss/critic1": float(critic1_loss.item()),
                "loss/critic2": float(critic2_loss.item()),
                "loss/actor":float(actor_loss.item()),
                "loss/alpha": float(alpha_loss.item()),
                "alpha": float(alpha.item()),
                "q_mean": float(((q1 + q2) * 0.5 * valid).sum().item()
                                / valid_sum.item()),
                "target_mean":float((target * valid).sum().item()
                                     / valid_sum.item()),
                "log_prob_mean": float((new_log_prob * valid).sum().item()
                                       / valid_sum.item()),
            }
        return metrics

    def _polyak(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau)
                tp.data.add_(self.cfg.tau * sp.data)


    def save(self, path: str) -> None:
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha":      self.log_alpha.detach().cpu(),
            "actor_optim":    self.actor_optim.state_dict(),
            "critic1_optim":  self.critic1_optim.state_dict(),
            "critic2_optim":  self.critic2_optim.state_dict(),
            "alpha_optim":    self.alpha_optim.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.log_alpha.data = ckpt["log_alpha"].to(self.device)
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic1_optim.load_state_dict(ckpt["critic1_optim"])
        self.critic2_optim.load_state_dict(ckpt["critic2_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
