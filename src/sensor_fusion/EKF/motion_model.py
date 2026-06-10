from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ..types import IMUReading, State

def skew(v: np.ndarray) -> np.ndarray:

    x, y, z = v
    return np.array([
        [0.0, -z,   y  ],
        [z,    0.0, -x ],
        [-y,   x,   0.0],
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:

    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),2*(x*y - z*w), 2*(x*z + y*w)    ],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z),2*(y*z - x*w)    ],
        [2*(x*z - y*w),2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_exp(omega: np.ndarray) -> np.ndarray:
   
    angle = float(np.linalg.norm(omega))
    if angle < 1e-8:
        q = np.array([1.0, 0.5 * omega[0], 0.5 * omega[1], 0.5 * omega[2]])
        return q / np.linalg.norm(q)
    axis = omega / angle
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s])


def quat_log(q: np.ndarray) -> np.ndarray:
  
    w = float(q[0])
    v = np.asarray(q[1:4], dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return 2.0 * v
    angle = 2.0 * np.arctan2(n, w)
    return angle * v/n


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


@dataclass
class MotionModelConfig:

    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    accel_noise_psd: float =0.05 ** 2    
    gyro_noise_psd: float= 0.005 ** 2    
    accel_bias_rw_psd: float = 1e-4 ** 2  
    gyro_bias_rw_psd: float = 1e-5 ** 2    


class MotionModel:

    def __init__(self, cfg: Optional[MotionModelConfig] = None) -> None:
        self.cfg = cfg or MotionModelConfig()

    def predict(self, state: State, imu: IMUReading) -> State:
        dt = imu.timestamp - state.timestamp
        if dt <= 0.0:
            return state

        new_pos, new_vel, new_quat = self._integrate_nominal(state, imu, dt)
        new_cov = self._propagate_covariance(state, imu, dt)

        return State(
            timestamp=imu.timestamp,
            position=new_pos,
            velocity=new_vel,
            orientation=new_quat,
            accel_bias=state.accel_bias.copy(),
            gyro_bias=state.gyro_bias.copy(),
            covariance=new_cov,
        )

    def _integrate_nominal(
        self,
        state: State,
        imu: IMUReading,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R_wb = quat_to_rot(state.orientation)
        a_hat = imu.accel - state.accel_bias
        omega_hat = imu.gyro -state.gyro_bias

        a_world = R_wb @ a_hat + self.cfg.gravity

        new_pos = state.position + state.velocity * dt + 0.5 * a_world * dt * dt
        new_vel = state.velocity + a_world * dt

        dq = quat_exp(omega_hat * dt)
        new_quat = quat_normalize(quat_mul(state.orientation, dq))
        return new_pos, new_vel, new_quat

    def _propagate_covariance(
        self,
        state: State,
        imu: IMUReading,
        dt: float,
    ) -> np.ndarray:
        R_wb = quat_to_rot(state.orientation)
        a_hat = imu.accel - state.accel_bias
        omega_hat = imu.gyro - state.gyro_bias

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)                      
        F[3:6, 6:9] = -R_wb @ skew(a_hat)             
        F[3:6, 9:12] = -R_wb                          
        F[6:9, 6:9] = -skew(omega_hat)                
        F[6:9, 12:15] = -np.eye(3)                    

        Phi = np.eye(15) + F * dt

        Q = np.zeros((15, 15))
        Q[3:6, 3:6]= self.cfg.accel_noise_psd * dt * np.eye(3)
        Q[6:9, 6:9]  = self.cfg.gyro_noise_psd * dt * np.eye(3)
        Q[9:12, 9:12] = self.cfg.accel_bias_rw_psd * dt * np.eye(3)
        Q[12:15, 12:15] = self.cfg.gyro_bias_rw_psd * dt * np.eye(3)

        P_new = Phi @ state.covariance @ Phi.T + Q
        return 0.5 * (P_new + P_new.T)
