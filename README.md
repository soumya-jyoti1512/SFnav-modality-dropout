# Robust Sensor Fusion for Navigation Under Modality Dropout (0ngoing)

> A simulation-based navigation framework designed to remain functional under sensor dropout and partial observability - implemented and benchmarked across a classical EKF pipeline and a learning-based SAC pipeline.

---

## Overview

This project implements and benchmarks two parallel navigation pipelines against identical sensor dropout scenarios:

- **Classical Pipeline** - Extended Kalman Filter (EKF) sensor fusion with Cartographer SLAM and A\* / DWA planning
- **Learning-Based Pipeline** - Soft Actor-Critic (SAC) adaptive sensor fusion with a hybrid SLAM and reinforcement learning navigation approach

Both pipelines share the same sensor input layer and are evaluated under the same dropout conditions to enable direct, fair comparison.

---

## Problem Statement

Real-world robot navigation is fragile. Sensors fail, degrade, or produce corrupted data during deployment. Most navigation stacks assume complete, consistent sensor availability, an assumption that fails in the field.

This project targets two specific failure modes:

- **Sensor Dropout** - a sensor modality goes partially or completely offline (camera occlusion, LiDAR noise, IMU drift)
- **Partial Observability** - even with all sensors active, the robot has only an egocentric, incomplete view of its dynamic environment

Robustly handling these two problems is a prerequisite for deploying autonomous robots in real-world settings such as warehouses, field robotics, and search and rescue operations.

---

## Sensor Suite

| Sensor         | Modality          | Status    |
|----------------|-------------------|-----------|
| RGB Camera     | Visual            | Active    |
| 2D LiDAR       | Geometric         | Active    |
| RGB-D Camera   | Visual + Depth    | Planned   |
| 3D LiDAR       | Geometric         | Planned   |
| IMU            | Inertial          | Planned   |

---

## System Architecture

<img width="3276" height="2826" alt="Image" src="https://github.com/user-attachments/assets/b90ce81e-cc0e-4fbc-91ca-13f7b844c272" />

---

## Methodology

### Classical Pipeline

The EKF fuses RGB-D camera, 3D LiDAR, and IMU observations using a standard two-stage predict-update cycle. IMU measurements drive the prediction step, exteroceptive sensors provide corrections in the update step.

**Dropout handling:** Each sensor carries an adaptive measurement noise covariance matrix **R** tuned to its normal operating profile. When a dropout is detected via stream health monitoring, the corresponding **R** is dynamically inflated to a large value reducing that sensor's Kalman gain to near zero and effectively removing it from the update step without interrupting the filter. The EKF continues running on IMU dead reckoning until the modality recovers and is seamlessly reintegrated.

### Classical Navigation

Navigation uses **Cartographer** as the SLAM backend, a graph-optimization-based system supporting 3D LiDAR, IMU, and RGB-D camera inputs with loop closure for globally consistent maps. Path planning uses **A\*** on the Cartographer-generated occupancy grid for global planning, with the **Dynamic Window Approach (DWA)** as the local planner for real-time obstacle avoidance.

---

### Learning-Based Pipeline

Sensor fusion is formulated as a **Partially Observable Markov Decision Process (POMDP)**. The SAC policy is trained with two complementary mechanisms:

**Stochastic modality masking** - randomly removes sensor inputs during training to prevent over-reliance on any single modality
**LSTM belief state** - an LSTM layer in both actor and critic maintains an observation history across timesteps, allowing the policy to reason through sensor dropouts mid-episode

Together these address both dropout and partial observability as a unified problem.

> **Planned extension:** Cross-modal attention-based weighting will replace implicit LSTM fusion, enabling the policy to explicitly score and select modalities based on current signal quality.

### Hybrid Navigation

Global navigation uses Cartographer for mapping and localization, with A\* providing a waypoint sequence. The SAC policy acts as the **local planner**, trained to navigate between waypoints while avoiding dynamic obstacles, a task where classical planners such as DWA underperform.

**Frontier-based exploration** enables autonomous goal selection: the system identifies boundaries between mapped and unmapped space and navigates toward them without requiring human-specified goals, making the pipeline suitable for fully autonomous operation in unknown environments.

> **Planned extension:** Hierarchical RL navigation where both global and local planning are handled by RL policies at different time scales, with Cartographer used only for localization.

---

## Multi-Agent Extension *(Learning Pipeline Only)*

The learning-based pipeline will be extended to decentralized multi-agent navigation for heterogeneous robotic systems:

- **Phase 1 - MAPPO baseline:** Multi-Agent Proximal Policy Optimization for homogeneous agent teams under centralized training, decentralized execution (CTDE)
- **Phase 2 - HetGPPO:** Graph Neural Network-based approach supporting heterogeneous agent teams where robots carry different sensor suites and learn specialized behaviors through differentiable inter-agent communication

> The classical pipeline is not extended to multi-agent. Classical SLAM was designed for single-robot operation and does not scale naturally to coordinated multi-robot deployment.

---

## Evaluation Scenarios

Both pipelines are evaluated across four conditions:

| Condition | Description |
|-----------|-------------|
| Full sensor availability | Baseline - all modalities active |
| Single modality dropout | e.g., camera failure only |
| Multi-modality dropout | e.g., camera and LiDAR simultaneously |
| Degraded signal quality | Noise injection, partial occlusion |

Environments progress from static and structured to dynamic and unstructured as the project matures.

---

## Preliminary Results

> Results will be updated as experiments progress. Current work is focused on the sensor fusion module: **EKF baseline complete**, SAC-based fusion in progress.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python, C++ |
| Middleware | ROS 2 Jazzy |
| Deep Learning | PyTorch |
| RL Framework | Stable-Baselines3 / custom SAC implementation |
| SLAM | Google Cartographer |
| Simulation | Gazebo, Isaac Sim |
