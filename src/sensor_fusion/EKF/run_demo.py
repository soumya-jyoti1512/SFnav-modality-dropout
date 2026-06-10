import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ..dropout import (
    CorruptionType,
    DropoutRule,
    DropoutSimulator,
    HealthConfig,
    StreamHealthMonitor,
    TimeWindow,
)
from ..pybullet_world import PyBulletWorld, WorldConfig
from ..types import IMUReading, LidarScan, Modality, RGBDFrame, State
from .adaptive_noise import AdaptiveNoiseModel
from .ekf import EKF
from .measurement_models import (
    Lidar2DPoseMeas,
    Lidar2DPoseModel,
    RGBDPoseMeas,
    RGBDPoseModel,
    _yaw_from_quat,
)
from .motion_model import MotionModel, quat_exp, quat_mul

def synth_lidar_pose(
    world: PyBulletWorld,
    timestamp: float,
    xy_noise_std: float,
    yaw_noise_std: float,
    rng: np.random.Generator,
) -> Lidar2DPoseMeas:
    
    gt = world.get_ground_truth()
    return Lidar2DPoseMeas(
        timestamp=timestamp,
        x=gt.position[0] + rng.normal(0.0, xy_noise_std),
        y=gt.position[1] + rng.normal(0.0, xy_noise_std),
        yaw=_yaw_from_quat(gt.orientation) + rng.normal(0.0, yaw_noise_std),
    )


def synth_rgbd_pose(
    world: PyBulletWorld,
    timestamp: float,
    pos_noise_std: float,
    rot_noise_std: float,
    rng: np.random.Generator,
) -> RGBDPoseMeas:

    gt = world.get_ground_truth()
    pos = gt.position + rng.normal(0.0,pos_noise_std, size=3)
    q_perturb = quat_exp(rng.normal(0.0, rot_noise_std,size=3))
    orientation = quat_mul(gt.orientation, q_perturb)
    return RGBDPoseMeas(timestamp=timestamp, position=pos,
                        orientation=orientation)



@dataclass
class DemoLog:

    t: list[float] = field(default_factory=list)
    gt_pos: list[np.ndarray] = field(default_factory=list)
    est_pos: list[np.ndarray] = field(default_factory=list)
    pos_err: list[float] = field(default_factory=list)
    cov_trace: list[float] = field(default_factory=list)
    health_imu: list[bool] = field(default_factory=list)
    health_lidar: list[bool] = field(default_factory=list)
    health_rgbd: list[bool] = field(default_factory=list)

    def append(self, t, gt_pos, est_pos, cov_trace, health):
        self.t.append(t)
        self.gt_pos.append(gt_pos.copy())
        self.est_pos.append(est_pos.copy())
        self.pos_err.append(float(np.linalg.norm(gt_pos - est_pos)))
        self.cov_trace.append(float(cov_trace))
        self.health_imu.append(health[Modality.IMU])
        self.health_lidar.append(health[Modality.LIDAR])
        self.health_rgbd.append(health[Modality.RGBD])


def run_demo(
    horizon_seconds: float= 10.0,
    linear_speed: float = 0.3,
    angular_speed: float =0.2,
    seed: int = 0,
    plot_path: Optional[str] = "ekf_demo.png",
) -> DemoLog:
    
    rng = np.random.default_rng(seed)

    world = PyBulletWorld(WorldConfig(seed=seed))
    world.reset()

    dropout = DropoutSimulator(
        rules=[
            DropoutRule(
                modality=Modality.LIDAR,
                corruption=CorruptionType.DROP,
                schedule=TimeWindow(start=3.0, end=6.0),
            ),
        ],
        rng=rng,
    )
    monitor = StreamHealthMonitor(HealthConfig())

    motion = MotionModel()
    
    lidar_xy_std, lidar_yaw_std = 0.03, 0.005
    rgbd_pos_std, rgbd_rot_std = 0.08, 0.04

    lidar_base = Lidar2DPoseModel(xy_noise_std=lidar_xy_std,
                                  yaw_noise_std=lidar_yaw_std)
    rgbd_base = RGBDPoseModel(pos_noise_std=rgbd_pos_std,
                              rot_noise_std=rgbd_rot_std)

    measurement_models = {
        Modality.LIDAR: AdaptiveNoiseModel(lidar_base, Modality.LIDAR, monitor),
        Modality.RGBD: AdaptiveNoiseModel(rgbd_base, Modality.RGBD, monitor),
    }

    
    gt0 = world.get_ground_truth()
    init_state = State.zero(timestamp=0.0)
    init_state.position = gt0.position.copy()
    init_state.orientation = gt0.orientation.copy()
    ekf = EKF(motion, measurement_models, init_state)

    log = DemoLog()
    dt = world.cfg.timestep
    n_steps = int(horizon_seconds / dt)

    print(f"\nRunning the EKF demo: {horizon_seconds:.1f}s, "
          f"LiDAR DROP from t=3.0s to t=6.0s.\n")
    print(f"{'t [s]':>6}  {'pos err [m]':>11}  {'trace(P)':>10}  "
          f"{'imu':>9}  {'lidar':>9}  {'rgbd':>9}")

    next_status = 0.0
    for k in range(n_steps):
        world.set_velocity(linear_speed, angular_speed)
        world.step()

        readings = world.read_sensors()
        readings = dropout.apply(readings, world.sim_time)
        monitor.update(readings,world.sim_time)

        imu_reading = readings.get(Modality.IMU)
        if imu_reading is not None and monitor.is_healthy(Modality.IMU):
            ekf.predict(imu_reading)

        if Modality.LIDAR in readings:
            pose = synth_lidar_pose(
                world, world.sim_time, lidar_xy_std, lidar_yaw_std, rng)
            ekf.update(Modality.LIDAR, pose)

        if Modality.RGBD in readings:
            pose = synth_rgbd_pose(
                world, world.sim_time, rgbd_pos_std, rgbd_rot_std, rng)
            ekf.update(Modality.RGBD, pose)

        ekf.tick(world.sim_time)

        gt = world.get_ground_truth()
        health = {m: monitor.is_healthy(m) for m in Modality}
        log.append(
            t=world.sim_time, gt_pos=gt.position, est_pos=ekf.state.position,
            cov_trace=np.trace(ekf.state.covariance), health=health,
        )

        if world.sim_time >= next_status:
            print(
                f"{world.sim_time:6.2f} "
                f"{log.pos_err[-1]:11.4f} "
                f"{log.cov_trace[-1]:10.3e} "
                f"{_hstr(health[Modality.IMU]):>9}  "
                f"{_hstr(health[Modality.LIDAR]):>9}  "
                f"{_hstr(health[Modality.RGBD]):>9}"
            )
            next_status += 0.5

    world.close()

    _print_summary(log)
    if plot_path is not None:
        _try_plot(log, plot_path)

    return log


def _hstr(b: bool) -> str:
    return "ok" if b else "DROPPED"


def _print_summary(log: DemoLog) -> None:
    pos_err = np.array(log.pos_err)
    t = np.array(log.t)

    pre = (t < 3.0)
    out = (t >= 3.0) & (t < 6.0)
    post = (t >= 6.0)

    def _stats(name, mask):
        if mask.any():
            sub = pos_err[mask]
            print(f"  {name:>11}: mean = {sub.mean():.4f} m, "
                  f"max = {sub.max():.4f} m")
        else:
            print(f"  {name:>11}: (no samples)")

    print("\nPosition error summary (Euclidean, meters):")
    _stats("pre outage",  pre)
    _stats("during DROP", out)
    _stats("post-outage", post)


def _try_plot(log: DemoLog, path: str) -> None:
    
    try:
        import matplotlib.pyplot as plt
   
        return

    t = np.array(log.t)
    gt = np.array(log.gt_pos)
    est = np.array(log.est_pos)
    err = np.array(log.pos_err)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

    ax = axes[0]
    ax.plot(gt[:, 0], gt[:, 1], label="ground truth", linewidth=2)
    ax.plot(est[:, 0], est[:, 1], label="EKF estimate", linestyle="--")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(t, err, label="‖p_gt - p_est‖")
    ax.axvspan(3.0, 6.0, color="tab:red", alpha=0.15, label="LiDAR DROP")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error vs time")
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    h_imu   = np.array(log.health_imu,   dtype=float)
    h_lidar = np.array(log.health_lidar, dtype=float)
    h_rgbd  = np.array(log.health_rgbd,  dtype=float)
    ax.step(t, h_imu   + 2.2, where="post", label="IMU")
    ax.step(t, h_lidar + 1.1, where="post", label="LiDAR")
    ax.step(t, h_rgbd  + 0.0, where="post", label="RGB-D")
    ax.set_yticks([])
    ax.set_xlabel("t [s]")
    ax.set_title("Monitor health (1 = healthy, 0 = unhealthy)")
    ax.legend(loc="center right")
    ax.grid(True, axis="x")

    fig.suptitle("EKF + adaptive R-inflation under LiDAR dropout")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    run_demo()
