import abc
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np
from .types import IMUReading, LidarScan, Modality, RGBDFrame

class CorruptionType(str, Enum):

    DROP = "drop"
    FROZEN ="frozen"
    NOISY = "noisy"

class DropoutSchedule(abc.ABC):

    @abc.abstractmethod
    def active(self, t: float) -> bool: ...


@dataclass
class AlwaysOn(DropoutSchedule):

    def active(self, t: float) -> bool:  
        return True


@dataclass
class TimeWindow(DropoutSchedule):

    start: float
    end: float

    def active(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass
class Bernoulli(DropoutSchedule):

    p: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def active(self, t: float) -> bool:
        return bool(self.rng.random() < self.p)


@dataclass
class Markov(DropoutSchedule):

    fail_rate: float      
    recover_rate: float    
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    _unhealthy: bool = False
    _last_t: float = 0.0

    def active(self, t: float) -> bool:
        dt = max(0.0, t - self._last_t)
        self._last_t = t
        if self._unhealthy:
            if self.rng.random() < self.recover_rate * dt:
                self._unhealthy = False
        else:
            if self.rng.random() < self.fail_rate * dt:
                self._unhealthy = True
        return self._unhealthy

_HEAVY_NOISE = {
    Modality.IMU:   dict(accel_std=0.25, gyro_std=0.025),  
    Modality.LIDAR: dict(range_std=0.10),                 
    Modality.RGBD:  dict(depth_std=0.15),                 
}


@dataclass
class DropoutRule:

    modality: Modality
    corruption: CorruptionType
    schedule: DropoutSchedule
    noise_scale: float = 1.0


class DropoutSimulator:

    def __init__(
        self,
        rules: list[DropoutRule],
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rules = rules
        self.rng = rng or np.random.default_rng()
        self._last_good: dict[Modality, Any] = {}

    def apply(
        self,
        readings: dict[Modality, Any],
        sim_time: float,
    ) -> dict[Modality, Any]:
        out: dict[Modality, Any] = {}
        for modality, reading in readings.items():
            rule = self._active_rule(modality, sim_time)
            if rule is None:
                out[modality] = reading
                self._last_good[modality] = reading
                continue

            if rule.corruption is CorruptionType.DROP:
                continue

            if rule.corruption is CorruptionType.FROZEN:
                last = self._last_good.get(modality)
                if last is None:
                    continue
                out[modality] = last
                continue

            if rule.corruption is CorruptionType.NOISY:
                noisy = self._add_heavy_noise(modality, reading, rule.noise_scale)
                out[modality] = noisy
                self._last_good[modality] = noisy
                continue

        return out

    def _active_rule(self, modality: Modality, t: float) -> Optional[DropoutRule]:
        for r in self.rules:
            if r.modality ==modality and r.schedule.active(t):
                return r
        return None

    def _add_heavy_noise(
        self,
        modality: Modality,
        reading: Any,
        scale: float,
    ) -> Any:
        noise = _HEAVY_NOISE[modality]
        if modality is Modality.IMU:
            r: IMUReading = reading
            return IMUReading(
                timestamp=r.timestamp,
                accel=r.accel + self.rng.normal(
                    0.0, scale * noise["accel_std"], size=3),
                gyro=r.gyro + self.rng.normal(
                    0.0, scale * noise["gyro_std"], size=3),
            )
        if modality is Modality.LIDAR:
            r_l: LidarScan = reading
            noisy_ranges = r_l.ranges + self.rng.normal(
                0.0, scale * noise["range_std"], size=r_l.ranges.shape)
            noisy_ranges = np.clip(noisy_ranges, 0.0, r_l.max_range)
            return LidarScan(
                timestamp=r_l.timestamp,
                ranges=noisy_ranges,
                angles=r_l.angles,
                max_range=r_l.max_range,
            )
        if modality is Modality.RGBD:
            r_d: RGBDFrame = reading
            depth_noise = self.rng.normal(
                0.0, scale * noise["depth_std"], size=r_d.depth.shape
            ).astype(np.float32)
            return RGBDFrame(
                timestamp=r_d.timestamp,
                rgb=r_d.rgb,
                depth=r_d.depth + depth_noise,
                fx=r_d.fx, fy=r_d.fy, cx=r_d.cx, cy=r_d.cy,
            )


@dataclass
class HealthState:

    healthy: bool = True
    reason: str = ""


@dataclass
class HealthConfig:

    expected_rates: dict[Modality, float] = field(default_factory=lambda: {
        Modality.IMU: 200.0,
        Modality.LIDAR: 15.0,
        Modality.RGBD: 10.0,
    })
    timeout_multiplier: float = 3.0
    stuck_window: int=5


class StreamHealthMonitor:

    def __init__(self, cfg: Optional[HealthConfig] = None) -> None:
        self.cfg = cfg or HealthConfig()
        self._last_seen: dict[Modality, Optional[float]] = {
            m: None for m in self.cfg.expected_rates
        }
        self._recent_hashes: dict[Modality, deque] = {
            m: deque(maxlen=self.cfg.stuck_window)
            for m in self.cfg.expected_rates
        }
        self._state: dict[Modality, HealthState] = {
            m: HealthState() for m in self.cfg.expected_rates
        }

    def update(
        self,
        readings: dict[Modality, Any],
        sim_time: float,
    ) -> None:
        for modality, reading in readings.items():
            if modality not in self._last_seen:
                continue 
            self._last_seen[modality] = sim_time
            self._recent_hashes[modality].append(_reading_hash(modality, reading))

        for modality, rate in self.cfg.expected_rates.items():
            timeout = self.cfg.timeout_multiplier/ rate
            last_t = self._last_seen[modality]

            if last_t is None:

                if sim_time > timeout:
                    self._state[modality] = HealthState(
                        False, f"no samples  within {timeout:.3f}s of start")
                else:
                    self._state[modality] = HealthState(True, "")
                continue

            if sim_time - last_t > timeout:
                self._state[modality] = HealthState(
                    False, f"no sample for {sim_time - last_t:.3f}s")
                continue

            buf = self._recent_hashes[modality]
            if len(buf) == buf.maxlen and len(set(buf)) == 1:
                self._state[modality] = HealthState(
                    False, f"stream stuck  {len(buf)} identical samples")
                continue

            self._state[modality] = HealthState(True, "")

    def is_healthy(self, modality: Modality) -> bool:
        return self._state[modality].healthy

    def reason(self, modality: Modality) -> str:
        return self._state[modality].reason

    def all_states(self) -> dict[Modality, HealthState]:
        return dict(self._state)


def _reading_hash(modality: Modality, reading: Any) -> bytes:
   
    if modality is Modality.IMU:
        r: IMUReading = reading
        return r.accel.tobytes() + r.gyro.tobytes()
    if modality is Modality.LIDAR:
        r_l: LidarScan = reading
        return r_l.ranges.tobytes()
    if modality is Modality.RGBD:
        r_d: RGBDFrame = reading
        return r_d.depth.tobytes()
    raise ValueError(f"Unknown modality: {modality!r}")
