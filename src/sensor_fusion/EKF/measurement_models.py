import abc
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ..types import State
from .motion_model import quat_inv, quat_log, quat_mul

@dataclass
class Lidar2DPoseMeas:

    timestamp: float
    x: float
    y: float
    yaw: float


@dataclass
class RGBDPoseMeas:

    timestamp: float
    position: np.ndarray         
    orientation: np.ndarray      

class MeasurementModel(abc.ABC):

    @abc.abstractmethod
    def innovation(self, state: State, measurement) -> np.ndarray:

    @abc.abstractmethod
    def jacobian(self, state: State) -> np.ndarray:

    @abc.abstractmethod
    def noise(self) -> np.ndarray:

def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    siny_cosp =2.0 * (w * z + x * y)
    cosy_cosp= 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Lidar2DPoseModel(MeasurementModel):

    def __init__(
        self,
        xy_noise_std: float = 0.05,    
        yaw_noise_std: float = 0.01,  
    ) -> None:
        self._R = np.diag([
            xy_noise_std ** 2,
            xy_noise_std ** 2,
            yaw_noise_std ** 2,
        ])

    def innovation(self, state: State, measurement: Lidar2DPoseMeas) -> np.ndarray:
        yaw_pred = _yaw_from_quat(state.orientation)
        return np.array([
            measurement.x - state.position[0],
            measurement.y - state.position[1],
            _wrap_to_pi(measurement.yaw - yaw_pred),
        ])

    def jacobian(self, state: State) -> np.ndarray:
       
        H = np.zeros((3, 15))
        H[0, 0] = 1.0   
        H[1, 1] = 1.0   
        H[2, 8] = 1.0   
        return H

    def noise(self) -> np.ndarray:
        return self._R.copy()


class RGBDPoseModel(MeasurementModel):

    def __init__(
        self,
        pos_noise_std: float = 0.10,   
        rot_noise_std: float = 0.05,   
    ) -> None:
        self._R = np.diag(
            [pos_noise_std ** 2] * 3
            + [rot_noise_std ** 2] * 3
        )

    def innovation(self, state: State, measurement: RGBDPoseMeas) -> np.ndarray:
        dp = measurement.position - state.position
        q_err = quat_mul(quat_inv(state.orientation), measurement.orientation)
        if q_err[0] < 0:
            q_err = -q_err
        dtheta = quat_log(q_err)
        return np.concatenate([dp, dtheta])

    def jacobian(self, state: State) -> np.ndarray:
        H = np.zeros((6, 15))
        H[0:3, 0:3] =np.eye(3)  
        H[3:6, 6:9] =np.eye(3)   
        return H

    def noise(self) -> np.ndarray:
        return self._R.copy()
