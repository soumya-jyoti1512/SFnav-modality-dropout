import math
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

_LOG_2 = math.log(2.0)

@dataclass
class ObsShape:

    imu: int = 6
    lidar: int =36
    rgbd: int = 8
    goal_rel: int =3
    mask: int = 3


@dataclass
class NetworkConfig:

    imu_embed_dim: int = 32
    lidar_embed_dim: int =64
    rgbd_embed_dim: int = 32
    goal_embed_dim: int =16

    encoder_out_dim: int = 128

    lstm_hidden_dim: int= 128
    lstm_layers: int =1

    head_hidden_dim: int = 128

    log_std_min: float =-5.0
    log_std_max: float = 2.0

class ObservationEncoder(nn.Module):

    def __init__(self, shape: ObsShape, cfg: NetworkConfig) -> None:
        super().__init__()
        self.imu_enc = nn.Sequential(
            nn.Linear(shape.imu, cfg.imu_embed_dim), nn.ReLU())
        self.lidar_enc = nn.Sequential(
            nn.Linear(shape.lidar, cfg.lidar_embed_dim), nn.ReLU())
        self.rgbd_enc = nn.Sequential(
            nn.Linear(shape.rgbd, cfg.rgbd_embed_dim), nn.ReLU())
        self.goal_enc = nn.Sequential(
            nn.Linear(shape.goal_rel, cfg.goal_embed_dim), nn.ReLU())

        concat_dim = (
            cfg.imu_embed_dim
            + cfg.lidar_embed_dim
            + cfg.rgbd_embed_dim
            + cfg.goal_embed_dim
            + shape.mask  
        )
        self.fuse = nn.Sequential(
            nn.Linear(concat_dim, cfg.encoder_out_dim), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
       
        mask = obs["mask"]  
        imu_feat = self.imu_enc(obs["imu"]) * mask[..., 0:1]
        lidar_feat =self.lidar_enc(obs["lidar"]) * mask[..., 1:2]
        rgbd_feat = self.rgbd_enc(obs["rgbd"]) * mask[..., 2:3]
        goal_feat =self.goal_enc(obs["goal_rel"])
        cat = torch.cat(
            [imu_feat, lidar_feat, rgbd_feat, goal_feat, mask], dim=-1)
        return self.fuse(cat)


class Actor(nn.Module):

    def __init__(
        self, shape: ObsShape, action_dim: int, cfg: NetworkConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        self.encoder = ObservationEncoder(shape, cfg)
        self.lstm =nn.LSTM(
            input_size=cfg.encoder_out_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.mean_head =nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, cfg.head_hidden_dim), nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, action_dim),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, cfg.head_hidden_dim), nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, action_dim),
        )

    def _forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        feats = self.encoder(obs)                       
        lstm_out, new_hidden = self.lstm(feats, hidden) 
        mean = self.mean_head(lstm_out)                
        log_std = self.log_std_head(lstm_out).clamp(
            self.cfg.log_std_min, self.cfg.log_std_max)
        return mean, log_std, new_hidden

    def sample(
        self,
        obs: dict[str, torch.Tensor],
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        mean, log_std, new_hidden = self._forward(obs, hidden)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()                            
        action = torch.tanh(x)
       
        log_prob = normal.log_prob(x) - 2.0 * (
            _LOG_2 - x - F.softplus(-2.0 * x))
        log_prob = log_prob.sum(dim=-1)                 
        return action, log_prob, new_hidden

    @torch.no_grad()
    def act_deterministic(
        self,
        obs: dict[str, torch.Tensor],
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean, _, new_hidden = self._forward(obs, hidden)
        return torch.tanh(mean), new_hidden

class Critic(nn.Module):

    def __init__(
        self, shape: ObsShape, action_dim: int, cfg: NetworkConfig,
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(shape, cfg)
        self.lstm = nn.LSTM(
            input_size=cfg.encoder_out_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim + action_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
      
        feats = self.encoder(obs)
        lstm_out, new_hidden = self.lstm(feats, hidden)
        q_input = torch.cat([lstm_out, action], dim=-1)
        q = self.head(q_input).squeeze(-1)               
        return q, new_hidden


def dict_obs_to_torch(
    obs: dict[str, np.ndarray],
    device: str | torch.device = "cpu",
    add_batch_seq: bool = True,
) -> dict[str, torch.Tensor]:
   
    out: dict[str, torch.Tensor] ={}
    for k, v in obs.items():
        t = torch.from_numpy(np.asarray(v)).float().to(device)
        if add_batch_seq:
            t = t.unsqueeze(0).unsqueeze(0)
        out[k] = t
    return out


def init_hidden(
    batch_size: int,
    cfg: NetworkConfig,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
   
    h = torch.zeros(cfg.lstm_layers, batch_size, cfg.lstm_hidden_dim,
                    device=device)
    c = torch.zeros(cfg.lstm_layers, batch_size, cfg.lstm_hidden_dim,
                    device=device)
    return (h, c)
