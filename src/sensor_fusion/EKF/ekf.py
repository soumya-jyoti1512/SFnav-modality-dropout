from typing import Any
import numpy as np
from ..types import IMUReading, Modality, State
from .measurement_models import MeasurementModel
from .motion_model import (
    MotionModel,
    quat_exp,
    quat_mul,
    quat_normalize,
    quat_to_rot,
)


class EKF:

    def __init__(
        self,
        motion_model: MotionModel,
        measurement_models: dict[Modality, MeasurementModel],
        initial_state: State,
    ) -> None:
        self.motion_model = motion_model
        self.measurement_models = dict(measurement_models)
        self._state = initial_state

    @property
    def state(self) -> State:
        return self._state

    def predict(self, imu: IMUReading) -> None:
        self._state = self.motion_model.predict(self._state, imu)

    def update(self, modality: Modality, measurement: Any) -> None:

        if measurement.timestamp < self._state.timestamp:
            return 
        if measurement.timestamp >self._state.timestamp:
            self._dead_reckon_to(measurement.timestamp)

        model = self.measurement_models[modality]
        self._kf_update(model,measurement)

    def tick(self, timestamp: float) -> None:
       
        if timestamp > self._state.timestamp:
            self._dead_reckon_to(timestamp)


    def _kf_update(self, model: MeasurementModel, measurement: Any) -> None:
    
        H = model.jacobian(self._state)
        R = model.noise()
        y = model.innovation(self._state, measurement)

        P = self._state.covariance
        S = H @ P @ H.T + R
       
        K =np.linalg.solve(S.T, (P @ H.T).T).T  

        dx = K @ y
        self._state = self._apply_error_state(self._state,dx)

        I_KH = np.eye(15) - K @ H
        new_P = I_KH @ P @ I_KH.T + K @ R @ K.T
        
        new_P = 0.5 * (new_P + new_P.T)
        self._state.covariance = new_P

    def _dead_reckon_to(self, target_t: float) -> None:
        
        if target_t <= self._state.timestamp:
            return
        R_bw = quat_to_rot(self._state.orientation).T
        rest_accel = R_bw @ (-self.motion_model.cfg.gravity) + self._state.accel_bias
        rest_gyro = self._state.gyro_bias.copy()
        fake_imu = IMUReading(
            timestamp=target_t, accel=rest_accel, gyro=rest_gyro)
        self._state = self.motion_model.predict(self._state, fake_imu)

    @staticmethod
    def _apply_error_state(state: State, dx: np.ndarray) -> State:
        
        dp = dx[0:3]
        dv = dx[3:6]
        dtheta = dx[6:9]
        db_a = dx[9:12]
        db_g = dx[12:15]
        return State(
            timestamp=state.timestamp,
            position=state.position + dp,
            velocity=state.velocity + dv,
            orientation=quat_normalize(
                quat_mul(state.orientation, quat_exp(dtheta))),
            accel_bias=state.accel_bias + db_a,
            gyro_bias=state.gyro_bias + db_g,
            covariance=state.covariance, 
        )
