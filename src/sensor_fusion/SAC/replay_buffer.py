from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class Episode:

    obs: dict[str, np.ndarray]    
    actions: np.ndarray             
    rewards: np.ndarray            
    dones: np.ndarray               

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class SequenceReplayBuffer:

    def __init__(self, capacity: int, seq_len: int, seed: int = 0) -> None:
        
        self.capacity = capacity
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)
        self._episodes: list[Episode] = []


    def push_episode(self, ep: Episode) -> None:
       
        if ep.length == 0:
            return
        if len(self._episodes) >=self.capacity:
            self._episodes.pop(0)
        self._episodes.append(ep)

  
    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def n_steps(self) -> int:
    
        return sum(ep.length for ep in self._episodes)

    def sample(self, batch_size: int) -> dict[str, Any]:

        B, T = batch_size, self.seq_len

        ep_idxs = self._rng.integers(0, len(self._episodes), size=B)

        template = self._episodes[0]
        obs_keys = list(template.obs.keys())
        obs_shapes = {k: template.obs[k].shape[1:] for k in obs_keys}
        action_shape = template.actions.shape[1:]

        out_obs = {
            k: np.zeros((B, T, *obs_shapes[k]), dtype=template.obs[k].dtype)
            for k in obs_keys
        }
        out_next_obs = {
            k: np.zeros((B, T, *obs_shapes[k]), dtype=template.obs[k].dtype)
            for k in obs_keys
        }
        out_actions = np.zeros((B, T, *action_shape),dtype=template.actions.dtype)
        out_rewards = np.zeros((B, T), dtype=np.float32)
        out_dones = np.zeros((B, T), dtype=np.float32)
        out_valid = np.zeros((B, T), dtype=np.float32)

        for b, ep_idx in enumerate(ep_idxs):
            ep = self._episodes[int(ep_idx)]
            ep_len = ep.length

            if ep_len >= T:
                start = int(self._rng.integers(0, ep_len - T + 1))
                length = T
            else:
                start = 0
                length = ep_len

            for k in obs_keys:
                out_obs[k][b, :length] = ep.obs[k][start:start + length]
                out_next_obs[k][b, :length] = ep.obs[k][start + 1:start + length + 1]
            out_actions[b, :length] = ep.actions[start:start + length]
            out_rewards[b, :length] = ep.rewards[start:start + length]
            out_dones[b, :length] = ep.dones[start:start + length].astype(np.float32)
            out_valid[b, :length] = 1.0

        return {
            "obs": out_obs,
            "next_obs": out_next_obs,
            "actions": out_actions,
            "rewards": out_rewards,
            "dones": out_dones,
            "valid": out_valid,
        }
