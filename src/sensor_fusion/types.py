from dataclasses import dataclass
from enum import Enum
import numpy as np



class Modality(str, Enum):
    
    IMU = "imu"
    LIDAR = "lidar"
    RGBD = "rgbd"


@dataclass
class IMUReading:
    
    timestamp: float
    accel: np.ndarray 
    gyro: np.ndarray   


@dataclass
class LidarScan:

    timestamp: float
    ranges: np.ndarray    
    angles: np.ndarray   
    max_range: float


@dataclass
class RGBDFrame:

    timestamp: float
    rgb: np.ndarray      
    depth: np.ndarray     
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class State:

    timestamp: float
    position: np.ndarray        
    velocity: np.ndarray       
    orientation: np.ndarray    
    accel_bias: np.ndarray     
    gyro_bias: np.ndarray      
    covariance: np.ndarray     

    @classmethod
    def zero(cls, timestamp: float = 0.0) -> "State":
        return cls(
            timestamp=timestamp,
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            covariance=np.eye(15) * 1e-3,
        )


@dataclass
class GroundTruth:

    timestamp: float
    position: np.ndarray        
    orientation: np.ndarray     
    linear_velocity: np.ndarray 
    angular_velocity: np.ndarray 
