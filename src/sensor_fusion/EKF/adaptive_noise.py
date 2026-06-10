from typing import Any
import numpy as np
from ..dropout import StreamHealthMonitor
from ..types import Modality, State
from .measurement_models import MeasurementModel


class AdaptiveNoiseModel(MeasurementModel):

    def __init__(
        self,
        base: MeasurementModel,
        modality: Modality,
        monitor: StreamHealthMonitor,
        inflation_factor: float = 1.0e6,
    ) -> None:
        self.base= base
        self.modality = modality
        self.monitor =monitor
        self.inflation_factor = inflation_factor

    def innovation(self, state: State, measurement: Any) -> np.ndarray:
        return self.base.innovation(state, measurement)

    def jacobian(self, state: State) -> np.ndarray:
        return self.base.jacobian(state)

    def noise(self) -> np.ndarray:
        R = self.base.noise()
        if not self.monitor.is_healthy(self.modality):
            return R * self.inflation_factor
        return R
