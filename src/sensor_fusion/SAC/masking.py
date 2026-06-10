from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np
from ..types import Modality
from .env import MASK_ORDER, SensorFusionNavEnv

_OBS_KEY: dict[Modality, str] = {
    Modality.IMU: "imu",
    Modality.LIDAR: "lidar",
    Modality.RGBD: "rgbd",
}


class MaskingMode(str, Enum):
    INDEPENDENT = "independent"
    BLOCK = "block"

@dataclass
class MaskingConfig:

    mode: MaskingMode = MaskingMode.BLOCK

    per_step_prob: dict[Modality, float] = field(default_factory=lambda: {
        Modality.IMU: 0.0,    
        Modality.LIDAR: 0.1,
        Modality.RGBD: 0.1,
    })

    block_enter_prob: dict[Modality, float] = field(default_factory=lambda: {
        Modality.IMU: 0.001, 
        Modality.LIDAR: 0.01,
        Modality.RGBD: 0.01,
    })
    block_min_steps: int = 5
    block_max_steps: int =30

    seed: int =0


class _BlockState:

    __slots__ = ("remaining",)

    def __init__(self) -> None:
        self.remaining: int = 0

    def in_block(self) -> bool:
        return self.remaining > 0

    def maybe_enter(self, rng: np.random.Generator, p_enter: float,
                    dur_min: int, dur_max: int) -> None:
        if self.remaining == 0 and rng.random() < p_enter:
            self.remaining = int(rng.integers(dur_min, dur_max + 1))

    def decrement(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1

    def reset(self) -> None:
        self.remaining = 0


class ModalityMaskingWrapper:

    def __init__(
        self,
        env: SensorFusionNavEnv,
        cfg: Optional[MaskingConfig] = None,
    ) -> None:
        self.env = env
        self.cfg = cfg or MaskingConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._blocks: dict[Modality, _BlockState] = {
            m: _BlockState() for m in MASK_ORDER
        }

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng =np.random.default_rng(seed)
        for b in self._blocks.values():
            b.reset()
        obs, info= self.env.reset(seed=seed)
        return self._apply_mask(obs), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_mask(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _apply_mask(self, obs: dict) -> dict:
       
        masked: dict[Modality, bool] = {m: False for m in MASK_ORDER}

        if self.cfg.mode is MaskingMode.INDEPENDENT:
            for m in MASK_ORDER:
                if self._rng.random() < self.cfg.per_step_prob[m]:
                    masked[m] = True
        else:  
            for m in MASK_ORDER:
                bs = self._blocks[m]
                bs.maybe_enter(
                    self._rng,
                    self.cfg.block_enter_prob[m],
                    self.cfg.block_min_steps,
                    self.cfg.block_max_steps,
                )
                masked[m] = bs.in_block()
                bs.decrement()

        new_obs: dict[str, np.ndarray] = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in obs.items()
        }
        new_mask = new_obs["mask"].copy()
        for i, m in enumerate(MASK_ORDER):
            if masked[m]:
                key = _OBS_KEY[m]
                new_obs[key] = np.zeros_like(new_obs[key])
                new_mask[i] = 0.0
        new_obs["mask"] = new_mask
        return new_obs
