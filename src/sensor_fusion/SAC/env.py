import math
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pybullet as p
from ..pybullet_world import PyBulletWorld, WorldConfig
from ..types import LidarScan, Modality, RGBDFrame


MASK_ORDER: tuple[Modality, ...] = (Modality.IMU, Modality.LIDAR, Modality.RGBD)

@dataclass
class NavEnvConfig:

    linear_max: float =0.5           
    angular_max: float= 1.0          
    agent_rate: float = 10.0          

    max_episode_steps: int = 300       
    goal_threshold: float = 0.3       

    world_half_size: float =4.0       
    goal_min_distance: float= 1.5     
    goal_max_distance: float = 3.5     

    n_obstacles: int = 5
    obstacle_radius: float = 0.25      
    obstacle_height: float =1.0       
    obstacle_min_clearance: float = 0.8  

    lidar_n_sectors: int= 36
    rgbd_n_sectors: int = 8

    accel_scale: float = 20.0         
    gyro_scale: float =5.0
    range_scale: float= 10.0          
    goal_scale: float = 5.0

    goal_reward: float = 10.0
    collision_penalty: float = -10.0
    step_cost: float= -0.005
    smoothness_weight: float = 0.005

    world: WorldConfig = field(default_factory=WorldConfig)

    seed: int = 0


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
  
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z),2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),1 - 2*(x*x + y*y)],
    ])


class SensorFusionNavEnv:

    def __init__(self, cfg: Optional[NavEnvConfig] = None) -> None:
        self.cfg = cfg or NavEnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._world = PyBulletWorld(self.cfg.world)

        self._obstacle_ids: list[int] = []
        self._goal_xy: np.ndarray= np.zeros(2)
        self._steps: int = 0
        self._prev_dist: float = 0.0

        self._sensor_cache: dict[Modality, Any] = {}

    def close(self) -> None:
        self._world.close()


    def reset(self, seed: Optional[int] =None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng= np.random.default_rng(seed)

        self._world.reset()
        self._goal_xy = self._sample_goal()
        self._spawn_obstacles()

        self._steps = 0
        self._sensor_cache = {}
        self._world.set_velocity(0.0, 0.0)
        self._world.step()
        self._collect_sensors()

        gt = self._world.get_ground_truth()
        self._prev_dist = float(np.linalg.norm(self._goal_xy - gt.position[:2]))

        return self._make_obs(), {"goal_xy": self._goal_xy.copy()}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        linear = float(action[0]) * self.cfg.linear_max
        angular = float(action[1]) * self.cfg.angular_max

        self._world.set_velocity(linear, angular)

        n_substeps = max(1, int(round(
            (1.0 / self.cfg.agent_rate) / self._world.cfg.timestep)))

        collision = False
        for _ in range(n_substeps):
            self._world.step()
            self._collect_sensors()
            if self._check_collision():
                collision = True
                break

        self._steps +=1

        obs = self._make_obs()

        gt = self._world.get_ground_truth()
        curr_dist = float(np.linalg.norm(self._goal_xy - gt.position[:2]))

        reward = self.cfg.step_cost
        reward += (self._prev_dist - curr_dist)
        reward -= self.cfg.smoothness_weight * abs(float(action[1]))
        self._prev_dist = curr_dist

        terminated = False
        truncated = False
        info: dict[str, Any] = {"goal_distance": curr_dist}

        if curr_dist < self.cfg.goal_threshold:
            reward += self.cfg.goal_reward
            terminated = True
            info["reason"] = "goal_reached"
        elif collision:
            reward += self.cfg.collision_penalty
            terminated = True
            info["reason"] = "collision"
        elif self._steps >= self.cfg.max_episode_steps:
            truncated = True
            info["reason"] = "timeout"

        return obs, float(reward), terminated, truncated, info


    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        return {
            "imu": (6,),
            "lidar": (self.cfg.lidar_n_sectors,),
            "rgbd": (self.cfg.rgbd_n_sectors,),
            "goal_rel": (3,),
            "mask": (3,),
        }

    @property
    def action_spec(self) -> dict[str, Any]:
        return {"shape": (2,), "low": -1.0, "high": 1.0, "dtype": np.float32}

    @property
    def goal(self) -> np.ndarray:
        return self._goal_xy.copy()


    def _sample_goal(self) -> np.ndarray:
  
        margin = 0.5
        for _ in range(50):
            angle = self._rng.uniform(-math.pi, math.pi)
            r = self._rng.uniform(
                self.cfg.goal_min_distance, self.cfg.goal_max_distance)
            xy = r * np.array([math.cos(angle), math.sin(angle)])
            if np.all(np.abs(xy) < self.cfg.world_half_size - margin):
                return xy
        return np.array([self.cfg.goal_min_distance, 0.0])

    def _spawn_obstacles(self) -> None:
       
        client = self._world._client          
        if client is None:
            return
        self._obstacle_ids = []
        for _ in range(self.cfg.n_obstacles):
            xy = self._sample_obstacle_xy()
            if xy is None:
                continue
            col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self.cfg.obstacle_radius,
                height=self.cfg.obstacle_height,
                physicsClientId=client,
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=self.cfg.obstacle_radius,
                length=self.cfg.obstacle_height,
                rgbaColor=[0.6, 0.3, 0.3, 1.0],
                physicsClientId=client,
            )
            body = p.createMultiBody(
                baseMass=0.0,                 
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(xy[0]), float(xy[1]),
                              self.cfg.obstacle_height / 2.0],
                physicsClientId=client,
            )
            self._obstacle_ids.append(body)

    def _sample_obstacle_xy(self) -> Optional[np.ndarray]:
       
        for _ in range(50):
            xy = self._rng.uniform(
                -self.cfg.world_half_size + 0.5,
                self.cfg.world_half_size - 0.5,
                size=2,
            )
            if (np.linalg.norm(xy) > self.cfg.obstacle_min_clearance
                and np.linalg.norm(xy - self._goal_xy)
                > self.cfg.obstacle_min_clearance):
                return xy
        return None


    def _collect_sensors(self) -> None:
        
        for modality, reading in self._world.read_sensors().items():
            self._sensor_cache[modality] = reading

    def _check_collision(self) -> bool:
       
        client =self._world._client          
        robot_id= self._world._robot_id      
        plane_id =self._world._plane_id      
        if client is None or robot_id is None:
            return False
        contacts = p.getContactPoints(bodyA=robot_id, physicsClientId=client)
        for c in contacts:
            body_b = c[2]
            if body_b != plane_id and body_b != robot_id:
                return True
        return False


    def _make_obs(self) -> dict[str, np.ndarray]:
       
        imu_reading = self._sensor_cache.get(Modality.IMU)
        if imu_reading is None:
            imu_vec = np.zeros(6, dtype=np.float32)
        else:
            imu_vec = np.concatenate([
                imu_reading.accel / self.cfg.accel_scale,
                imu_reading.gyro / self.cfg.gyro_scale,
            ]).astype(np.float32)

        lidar_reading = self._sensor_cache.get(Modality.LIDAR)
        if lidar_reading is None:
            lidar_vec = np.ones(self.cfg.lidar_n_sectors, dtype=np.float32)
        else:
            lidar_vec = self._downsample_lidar(lidar_reading)

        rgbd_reading = self._sensor_cache.get(Modality.RGBD)
        if rgbd_reading is None:
            rgbd_vec = np.ones(self.cfg.rgbd_n_sectors, dtype=np.float32)
        else:
            rgbd_vec =self._summarize_rgbd(rgbd_reading)

        gt = self._world.get_ground_truth()
        goal_world_rel = np.array(
            [self._goal_xy[0] - gt.position[0],
             self._goal_xy[1] - gt.position[1],
             0.0])
        R_wb = _quat_to_rot(gt.orientation)
        goal_body = R_wb.T @ goal_world_rel
        dist = float(np.linalg.norm(goal_body[:2]))
        goal_vec = np.array([
            goal_body[0] / self.cfg.goal_scale,
            goal_body[1] / self.cfg.goal_scale,
            dist / self.cfg.goal_scale,
        ], dtype=np.float32)

        mask = np.ones(3, dtype=np.float32)

        return {
            "imu": imu_vec,
            "lidar": lidar_vec,
            "rgbd": rgbd_vec,
            "goal_rel": goal_vec,
            "mask": mask,
        }

    def _downsample_lidar(self, scan: LidarScan) -> np.ndarray:
      
        n_sectors = self.cfg.lidar_n_sectors
        beams_per_sector = max(1, len(scan.ranges) // n_sectors)
        usable = beams_per_sector * n_sectors
        ranges = scan.ranges[:usable].reshape(n_sectors, beams_per_sector)
        sector_min = ranges.min(axis=1)
        return np.clip(
            sector_min / self.cfg.range_scale, 0.0, 1.0).astype(np.float32)

    def _summarize_rgbd(self, frame: RGBDFrame) -> np.ndarray:
        
        n_sectors = self.cfg.rgbd_n_sectors
        h, w = frame.depth.shape
        cols_per_sector = max(1, w // n_sectors)
        sector_mins = np.empty(n_sectors, dtype=np.float32)
        for i in range(n_sectors):
            col = frame.depth[:, i * cols_per_sector:(i + 1) * cols_per_sector]
            finite = col[np.isfinite(col)]
            sector_mins[i] = (float(finite.min()) if finite.size > 0
                              else self.cfg.range_scale)
        return np.clip(
            sector_mins / self.cfg.range_scale, 0.0, 1.0).astype(np.float32)
