import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pybullet as p
import pybullet_data
from .types import GroundTruth, IMUReading, LidarScan, Modality, RGBDFrame


@dataclass
class WorldConfig:

    timestep: float = 1.0 /240.0          

    imu_rate: float = 200.0
    lidar_rate: float =15.0
    camera_rate: float = 10.0

    accel_noise_std: float = 0.05         
    gyro_noise_std: float =0.005          
    accel_bias_random_walk: float = 1e-4  
    gyro_bias_random_walk: float= 1e-5   

    lidar_n_beams: int = 360
    lidar_fov: float = 2.0 * math.pi      
    lidar_max_range: float = 30.0
    lidar_min_range: float = 0.1
    lidar_noise_std: float = 0.02          
    lidar_height: float = 0.4              

    camera_width: int = 320
    camera_height: int= 240
    camera_fov_deg: float =60.0          
    camera_near: float = 0.05
    camera_far: float= 20.0
    camera_height_offset: float = 0.6      
    camera_forward_offset: float = 0.2     

    wheel_separation: float =0.555       
    wheel_radius: float =0.165            
    wheel_force: float =100.0            

    use_gui: bool = False
    seed: int =0


def _quat_xyzw_to_wxyz(q: tuple[float, float, float, float]) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> tuple[float, float, float, float]:
    return (float(q[1]), float(q[2]), float(q[3]), float(q[0]))


class PyBulletWorld:
   
    def __init__(self, cfg: Optional[WorldConfig] = None) -> None:
        self.cfg = cfg or WorldConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        self._client: Optional[int] = None
        self._robot_id: Optional[int] =None
        self._plane_id: Optional[int] = None
        self._left_wheel_joints: list[int] = []
        self._right_wheel_joints: list[int] = []

        self._sim_time: float = 0.0
        self._last_lin_vel_world: np.ndarray =np.zeros(3)
        self._accel_bias: np.ndarray = np.zeros(3)
        self._gyro_bias: np.ndarray= np.zeros(3)
        self._first_imu_sample: bool = True

        self._next_imu_t: float = 0.0
        self._next_lidar_t: float= 0.0
        self._next_camera_t: float = 0.0

        self._connect()

    def _connect(self) -> None:
        mode = p.GUI if self.cfg.use_gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

   
    def reset(
        self,
        start_position: Optional[np.ndarray] = None,
        start_orientation_wxyz: Optional[np.ndarray] = None,
    ) -> None:
       
        assert self._client is not None
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client)
        p.setTimeStep(self.cfg.timestep, physicsClientId=self._client)

        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        if start_position is None:
            position = [0.0, 0.0, 0.1]
        else:
            position = list(np.asarray(start_position, dtype=float))

        if start_orientation_wxyz is None:
            orientation_xyzw = [0.0, 0.0, 0.0, 1.0]
        else:
            orientation_xyzw = list(_quat_wxyz_to_xyzw(
                np.asarray(start_orientation_wxyz, dtype=float)))

        self._robot_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=position,
            baseOrientation=orientation_xyzw,
            physicsClientId=self._client,
        )

        self._left_wheel_joints = []
        self._right_wheel_joints = []
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client)
        for j in range(n_joints):
            info = p.getJointInfo(self._robot_id, j,
                                  physicsClientId=self._client)
            name = info[1].decode("utf-8").lower()
            if "wheel" not in name:
                continue
            if "left" in name:
                self._left_wheel_joints.append(j)
            elif "right" in name:
                self._right_wheel_joints.append(j)
        if not self._left_wheel_joints or not self._right_wheel_joints:

        self._sim_time = 0.0
        self._last_lin_vel_world =np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._gyro_bias= np.zeros(3)
        self._first_imu_sample = True
        self._next_imu_t =0.0
        self._next_lidar_t = 0.0
        self._next_camera_t = 0.0

    def set_velocity(self, linear: float, angular: float) -> None:
        
        v_left = (linear - angular * self.cfg.wheel_separation / 2.0) / self.cfg.wheel_radius
        v_right = (linear + angular * self.cfg.wheel_separation / 2.0) / self.cfg.wheel_radius

        for j in self._left_wheel_joints:
            p.setJointMotorControl2(
                self._robot_id, j, p.VELOCITY_CONTROL,
                targetVelocity=v_left, force=self.cfg.wheel_force,
                physicsClientId=self._client,
            )
        for j in self._right_wheel_joints:
            p.setJointMotorControl2(
                self._robot_id, j, p.VELOCITY_CONTROL,
                targetVelocity=v_right, force=self.cfg.wheel_force,
                physicsClientId=self._client,
            )

    def step(self) -> None:
        p.stepSimulation(physicsClientId=self._client)
        self._sim_time += self.cfg.timestep

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def get_ground_truth(self) -> GroundTruth:
        pos, quat_xyzw = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._client)

        R_wb = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
        ang_vel_body = R_wb.T @ np.asarray(ang_vel_world, dtype=float)

        return GroundTruth(
            timestamp=self._sim_time,
            position=np.asarray(pos, dtype=float),
            orientation=_quat_xyzw_to_wxyz(quat_xyzw),
            linear_velocity=np.asarray(lin_vel_world, dtype=float),
            angular_velocity=ang_vel_body,
        )

    def _sample_imu(self) -> IMUReading:
  
        pos, quat_xyzw = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._client)
        lin_vel_world = np.asarray(lin_vel_world, dtype=float)
        ang_vel_world = np.asarray(ang_vel_world, dtype=float)
        R_wb = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)

        if self._first_imu_sample:
            lin_acc_world = np.zeros(3)
            self._first_imu_sample = False
        else:
            lin_acc_world = (lin_vel_world - self._last_lin_vel_world) / max(
                self.cfg.timestep, 1e-9)
        self._last_lin_vel_world = lin_vel_world.copy()

        gravity_world = np.array([0.0, 0.0, 9.81])
        accel_body_true = R_wb.T @ (lin_acc_world + gravity_world)
        gyro_body_true = R_wb.T @ ang_vel_world

        self._accel_bias = self._accel_bias + self._rng.normal(
            0.0, self.cfg.accel_bias_random_walk, size=3)
        self._gyro_bias = self._gyro_bias + self._rng.normal(
            0.0, self.cfg.gyro_bias_random_walk, size=3)
        accel = (accel_body_true + self._accel_bias
                 + self._rng.normal(0.0, self.cfg.accel_noise_std, size=3))
        gyro = (gyro_body_true + self._gyro_bias
                + self._rng.normal(0.0, self.cfg.gyro_noise_std, size=3))

        return IMUReading(timestamp=self._sim_time,accel=accel, gyro=gyro)

   
    def _sample_lidar(self) -> LidarScan:
      
        pos, quat_xyzw = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client)
        R_wb = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
        base_pos = np.asarray(pos, dtype=float)
        sensor_pos = base_pos + R_wb @ np.array([0.0, 0.0, self.cfg.lidar_height])

        angles = np.linspace(
            -self.cfg.lidar_fov / 2.0,
            self.cfg.lidar_fov / 2.0,
            self.cfg.lidar_n_beams,
            endpoint=False,
        )
        
        dir_body = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        dir_world = dir_body @ R_wb.T  

        from_pts = np.broadcast_to(sensor_pos, dir_world.shape)
        to_pts = from_pts + dir_world * self.cfg.lidar_max_range
        results = p.rayTestBatch(
            from_pts.tolist(), to_pts.tolist(),
            physicsClientId=self._client,
        )

        ranges = np.full(self.cfg.lidar_n_beams, self.cfg.lidar_max_range,
                         dtype=float)
        for i, r in enumerate(results):
            hit_object_id, _, hit_fraction, _, _ = r
            if hit_object_id >= 0:
                ranges[i] = hit_fraction * self.cfg.lidar_max_range

        ranges = ranges + self._rng.normal(
            0.0, self.cfg.lidar_noise_std, size=ranges.shape)
        ranges = np.clip(ranges, self.cfg.lidar_min_range, self.cfg.lidar_max_range)

        return LidarScan(
            timestamp=self._sim_time,
            ranges=ranges,
            angles=angles,
            max_range=self.cfg.lidar_max_range,
        )

    def _sample_rgbd(self) -> RGBDFrame:
      
        pos, quat_xyzw = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client)
        R_wb = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
        base_pos = np.asarray(pos, dtype=float)

        cam_offset_body = np.array(
            [self.cfg.camera_forward_offset, 0.0, self.cfg.camera_height_offset])
        cam_pos_world = base_pos + R_wb @ cam_offset_body
        forward_world = R_wb @ np.array([1.0, 0.0, 0.0])
        up_world = R_wb @ np.array([0.0, 0.0, 1.0])
        target_world = cam_pos_world + forward_world

        view = p.computeViewMatrix(
            cam_pos_world.tolist(), target_world.tolist(), up_world.tolist())
        aspect = self.cfg.camera_width / self.cfg.camera_height
        proj = p.computeProjectionMatrixFOV(
            self.cfg.camera_fov_deg, aspect,
            self.cfg.camera_near, self.cfg.camera_far)

        _, _, rgba, depth_buf, _ = p.getCameraImage(
            self.cfg.camera_width, self.cfg.camera_height,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(
            self.cfg.camera_height, self.cfg.camera_width, 4)[..., :3].copy()
        depth_buf = np.asarray(depth_buf, dtype=np.float32).reshape(
            self.cfg.camera_height, self.cfg.camera_width)

       
        n, f = self.cfg.camera_near, self.cfg.camera_far
        depth = (f * n) / (f - (f - n) * depth_buf)
        depth[depth_buf >= 1.0 - 1e-6] = np.nan  

        fy = (self.cfg.camera_height / 2.0) / math.tan(
            math.radians(self.cfg.camera_fov_deg) / 2.0)
        fx = fy
        cx = self.cfg.camera_width / 2.0
        cy = self.cfg.camera_height / 2.0

        return RGBDFrame(
            timestamp=self._sim_time, rgb=rgb, depth=depth,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )

    def read_sensors(self) -> dict[Modality, object]:
        
        out: dict[Modality, object] = {}
        if self._sim_time >= self._next_imu_t:
            out[Modality.IMU] = self._sample_imu()
            self._next_imu_t += 1.0 / self.cfg.imu_rate
        if self._sim_time >= self._next_lidar_t:
            out[Modality.LIDAR] = self._sample_lidar()
            self._next_lidar_t += 1.0 / self.cfg.lidar_rate
        if self._sim_time >= self._next_camera_t:
            out[Modality.RGBD] = self._sample_rgbd()
            self._next_camera_t += 1.0 / self.cfg.camera_rate
        return out

    def close(self) -> None:
        if self._client is not None:
            try:
                p.disconnect(self._client)
            except p.error:
                pass
            self._client = None
