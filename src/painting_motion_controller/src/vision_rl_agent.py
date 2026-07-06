#!/usr/bin/env python3
"""
Vision-side RL agent.

Runs the SAME PPO+TD3 ensemble architecture as rl_agent_node.py, but the
observation comes from the real-camera defect_matrix published by
Defect_detection_with_Coverage_Map.py on /spray/defect_matrix, instead of
the simulated thickness_matrix.

defect_matrix cell semantics (from build_defect_matrix() in the CV script):
    -1.0        -> cell outside the detected part ROI (ignore)
     0.0         -> cell fully covered (>= CELL_COVERED_THRESH)
    (0.0, 1.0]  -> under-coverage severity, 1.0 = completely unpainted

This is a SEPARATE node/agent from RLAgentNode (simulation). It does not
touch cartesian_trajectory_controller.cpp or the sim RL's checkpoints.
It publishes on its own /spray/vision_* topics so you can decide how (or
whether) to gate/blend its corrective path against the sim RL's output --
that arbitration is a robot-execution decision left to you.
"""

import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node
import rclpy.qos

from std_msgs.msg import Float32, Float32MultiArray, String, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

# Reuse the exact same networks / action decoder / geometry helper as the
# simulation agent, so both loops share one implementation.
from rl_agent_node import (
    PPOAgent, TD3Agent, decode_action, _orientation_facing_normal,
    OBS_DIM, ACT_DIM, STANDOFF_MIN, STANDOFF_MAX,
)

# ── Vision-loop constants ──────────────────────────────────────────────
VISION_GRID_ROWS = 20          # must match GRID_ROWS in the CV script
VISION_GRID_COLS = 20          # must match GRID_COLS in the CV script

VISION_DECISION_INTERVAL = 3.0  # seconds between vision RL decisions

DBSCAN_EPS         = 2
DBSCAN_MIN_SAMPLES = 3

PATCH_HALF_WIDTH  = 0.15
PATCH_HALF_HEIGHT = 0.15
PATCH_STEP        = 0.02

DEFAULT_SURFACE_NX = 1.0
DEFAULT_SURFACE_NY = 0.0
DEFAULT_SURFACE_NZ = 0.0

CHECKPOINT_DIR = "/home/user/car_spraying_ws/rl_checkpoints/vision"


class VisionRLAgentNode(Node):

    def __init__(self):
        super().__init__('vision_rl_agent')

        # ── Physical patch bounds (same panel patch the sim RL operates on) ──
        self.declare_parameter('y_min', -0.20)
        self.declare_parameter('y_max',  0.00)
        self.declare_parameter('z_min',  0.45)
        self.declare_parameter('z_max',  0.70)
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value

        # ── State ──────────────────────────────────────────────────────
        self.defect_matrix = np.full(
            (VISION_GRID_ROWS, VISION_GRID_COLS), -1.0, dtype=np.float32)
        self.lock = threading.Lock()
        self._pass1_done = False

        self._current_ee_pose = None
        self._tracking_lock = threading.Lock()

        # ── Independent PPO+TD3 ensemble (own weights, own checkpoints) ──
        self.ppo = PPOAgent()
        self.td3 = TD3Agent()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._try_load_checkpoints()

        self._prev_obs = None
        self._prev_action = None
        self._episode_step = 0
        self._total_reward = 0.0

        # ── QoS ────────────────────────────────────────────────────────
        _tl_qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/spray/defect_matrix',
            self._defect_matrix_cb, rclpy.qos.qos_profile_sensor_data)

        self.create_subscription(
            Bool, '/spray/pass1_done',
            self._pass1_done_cb, _tl_qos)

        self.create_subscription(
            PoseStamped, '/spray/tracking_pose',
            self._tracking_pose_cb, rclpy.qos.qos_profile_sensor_data)

        # ── Publishers (own namespace -- does not collide with sim RL) ──
        self.action_pub = self.create_publisher(Float32MultiArray, '/spray/vision_rl_action', 10)
        self.path_pub    = self.create_publisher(Path,             '/spray/vision_rl_path',   10)
        self.reward_pub  = self.create_publisher(Float32,          '/spray/vision_reward',    10)
        self.status_pub  = self.create_publisher(String,           '/spray/vision_rl_status', 10)

        self.create_timer(VISION_DECISION_INTERVAL, self._decision_step)

        self.get_logger().info(
            f'Vision RL Agent (PPO+TD3) started | '
            f'grid={VISION_GRID_ROWS}x{VISION_GRID_COLS} | sklearn={_HAVE_SKLEARN} | '
            f'Waiting for /spray/defect_matrix + /spray/pass1_done')

    # ─────────────────────────────────────────────────────────
    #  CHECKPOINTING
    # ─────────────────────────────────────────────────────────

    def _try_load_checkpoints(self):
        ppo_path = os.path.join(CHECKPOINT_DIR, 'ppo.npz')
        td3_path = os.path.join(CHECKPOINT_DIR, 'td3.npz')
        if os.path.exists(ppo_path):
            try:
                self.ppo.load(ppo_path)
                self.get_logger().info(f'Vision PPO checkpoint loaded from {ppo_path}')
            except Exception as e:
                self.get_logger().warning(f'Vision PPO load failed: {e}')
        if os.path.exists(td3_path):
            try:
                self.td3.load(td3_path)
                self.get_logger().info(f'Vision TD3 checkpoint loaded from {td3_path}')
            except Exception as e:
                self.get_logger().warning(f'Vision TD3 load failed: {e}')

    def _save_checkpoints(self):
        try:
            self.ppo.save(os.path.join(CHECKPOINT_DIR, 'ppo.npz'))
            self.td3.save(os.path.join(CHECKPOINT_DIR, 'td3.npz'))
        except Exception as e:
            self.get_logger().warning(f'Vision checkpoint save failed: {e}')

    # ─────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────

    def _tracking_pose_cb(self, msg: PoseStamped):
        with self._tracking_lock:
            self._current_ee_pose = msg.pose

    def _pass1_done_cb(self, msg: Bool):
        if msg.data and not self._pass1_done:
            self._pass1_done = True
            self.get_logger().info('PASS 1 complete — vision RL agent now active.')

    def _defect_matrix_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        try:
            reshaped = data.reshape((VISION_GRID_ROWS, VISION_GRID_COLS))
            with self.lock:
                self.defect_matrix = reshaped
        except Exception as e:
            self.get_logger().error(f'defect_matrix reshape failed: {e}')

    # ─────────────────────────────────────────────────────────
    #  OBSERVATION BUILDER  (mirrors _build_obs in rl_agent_node.py,
    #  but reads coverage SEVERITY instead of paint THICKNESS, and masks
    #  out cells outside the detected part ROI, i.e. matrix == -1)
    # ─────────────────────────────────────────────────────────

    def _build_obs(self, matrix: np.ndarray) -> np.ndarray:
        valid = matrix >= 0.0
        n_valid = float(np.sum(valid))
        n_valid = max(n_valid, 1.0)

        unpainted = float(np.sum(valid & (matrix >= 0.999))) / n_valid
        weak      = float(np.sum(valid & (matrix > 0.0) & (matrix < 0.999))) / n_valid
        good      = float(np.sum(valid & (matrix <= 0.0))) / n_valid
        over      = 0.0  # the CV coverage matrix has no over-spray concept

        # Fill invalid (-1) cells with the mean of valid cells before taking
        # the gradient, so the ROI boundary itself doesn't register as a
        # fake defect edge.
        fill_val = float(np.mean(matrix[valid])) if np.any(valid) else 0.0
        clean = np.where(valid, matrix, fill_val)
        gy, gx = np.gradient(clean)
        grad_mag = np.sqrt(gx**2 + gy**2)
        uneven = float(np.sum(valid & (grad_mag > 0.4))) / n_valid

        mean_s = float(np.mean(clean[valid])) if np.any(valid) else 0.0
        std_s  = float(np.std(clean[valid])) if np.any(valid) else 0.0
        grms   = float(np.sqrt(np.mean(grad_mag[valid]**2))) if np.any(valid) else 0.0

        obs = np.array([unpainted, weak, good, over, uneven,
                        np.clip(mean_s, 0, 1),
                        np.clip(std_s, 0, 1),
                        np.clip(grms, 0, 1)], dtype=np.float32)
        return obs

    # ─────────────────────────────────────────────────────────
    #  REWARD  (same shape as sim reward, no over-spray term)
    # ─────────────────────────────────────────────────────────

    def _compute_reward(self, obs: np.ndarray) -> float:
        unpainted, weak, good, over, uneven = obs[0], obs[1], obs[2], obs[3], obs[4]
        reward = good * 200.0 - unpainted * 80.0 - uneven * 40.0 - weak * 20.0
        return float(reward)

    # ─────────────────────────────────────────────────────────
    #  DEFECT CLUSTERING  (severity > 0 cells = anything not fully covered)
    # ─────────────────────────────────────────────────────────

    def _defect_cells(self, matrix: np.ndarray) -> list:
        valid = matrix >= 0.0
        defect_mask = valid & (matrix > 0.0)
        return np.argwhere(defect_mask).tolist()

    def _cluster_defect_cells(self, cells: list) -> list:
        if not cells:
            return []
        arr = np.array(cells, dtype=np.int32)
        if not _HAVE_SKLEARN or len(arr) < DBSCAN_MIN_SAMPLES:
            return [arr]
        labels = DBSCAN(eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MIN_SAMPLES).fit_predict(arr)
        regions = [arr[labels == l] for l in np.unique(labels) if l >= 0]
        regions.sort(key=len, reverse=True)
        return regions

    # ─────────────────────────────────────────────────────────
    #  PATH GENERATION  (same boustrophedon patch geometry as sim RL)
    # ─────────────────────────────────────────────────────────

    def _surface_normal_from_ee(self):
        with self._tracking_lock:
            ee_pose = self._current_ee_pose
        if ee_pose is None:
            return DEFAULT_SURFACE_NX, DEFAULT_SURFACE_NY, DEFAULT_SURFACE_NZ
        ox, oy, oz, ow = (ee_pose.orientation.x, ee_pose.orientation.y,
                          ee_pose.orientation.z, ee_pose.orientation.w)
        nx = 2.0 * (ox * oz + ow * oy)
        ny = 2.0 * (oy * oz - ow * ox)
        nz = 1.0 - 2.0 * (ox * ox + oy * oy)
        mag = (nx*nx + ny*ny + nz*nz) ** 0.5
        if mag > 1e-6:
            return nx / mag, ny / mag, nz / mag
        return DEFAULT_SURFACE_NX, DEFAULT_SURFACE_NY, DEFAULT_SURFACE_NZ

    def _patch_poses(self, target_y, target_z, standoff, nx, ny, nz, stamp) -> list:
        with self._tracking_lock:
            ee_pose = self._current_ee_pose
        surface_x = (ee_pose.position.x + standoff * nx) if ee_pose is not None \
                    else (standoff * nx)
        qx, qy, qz, qw = _orientation_facing_normal(nx, ny, nz)

        y_start = np.clip(target_y - PATCH_HALF_WIDTH,  self.y_min, self.y_max)
        y_end   = np.clip(target_y + PATCH_HALF_WIDTH,  self.y_min, self.y_max)
        z_start = np.clip(target_z - PATCH_HALF_HEIGHT, self.z_min, self.z_max)
        z_end   = np.clip(target_z + PATCH_HALF_HEIGHT, self.z_min, self.z_max)

        poses = []
        z_vals = np.arange(z_start, z_end + 1e-6, PATCH_STEP)
        left_to_right = True
        for z in z_vals:
            y_vals = np.arange(y_start, y_end + 1e-6, PATCH_STEP)
            if not left_to_right:
                y_vals = y_vals[::-1]
            left_to_right = not left_to_right
            for y in y_vals:
                nozzle_x = surface_x - standoff * nx
                nozzle_y = float(y)  - standoff * ny
                nozzle_z = float(z)  - standoff * nz
                ps = PoseStamped()
                ps.header.frame_id = 'world'
                ps.header.stamp = stamp
                ps.pose.position.x = nozzle_x
                ps.pose.position.y = nozzle_y
                ps.pose.position.z = nozzle_z
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw
                poses.append(ps)
        return poses

    def _generate_paint_path(self, target_y, target_z, standoff, clusters=None) -> Path:
        path = Path()
        path.header.frame_id = 'world'
        path.header.stamp = self.get_clock().now().to_msg()
        stamp = path.header.stamp
        nx, ny, nz = self._surface_normal_from_ee()

        if clusters:
            for cluster_arr in clusters:
                centroid = cluster_arr.mean(axis=0)
                res_y = (self.y_max - self.y_min) / VISION_GRID_COLS
                res_z = (self.z_max - self.z_min) / VISION_GRID_ROWS
                cy = self.y_min + centroid[1] * res_y
                cz = self.z_min + centroid[0] * res_z
                path.poses.extend(
                    self._patch_poses(cy, cz, standoff, nx, ny, nz, stamp))
        else:
            path.poses.extend(
                self._patch_poses(target_y, target_z, standoff, nx, ny, nz, stamp))
        return path

    # ─────────────────────────────────────────────────────────
    #  MAIN DECISION STEP
    # ─────────────────────────────────────────────────────────

    def _decision_step(self):
        if not self._pass1_done:
            return  # same PASS-1 silence convention as the sim agent

        self._episode_step += 1

        with self.lock:
            matrix = self.defect_matrix.copy()

        obs = self._build_obs(matrix)
        reward = self._compute_reward(obs)
        self._total_reward += reward

        # ── TD3: store transition from last step ──────────────────────
        if self._prev_obs is not None and self._prev_action is not None:
            self.td3.buffer.add(self._prev_obs, self._prev_action,
                                reward, obs, done=False)
            self.td3.update()

        # ── PPO: store transition + rollout update ─────────────────────
        if self._prev_obs is not None and self._prev_action is not None:
            _, prev_lp, prev_val = self.ppo.select_action(self._prev_obs)
            self.ppo.store(self._prev_obs, self._prev_action,
                           reward, prev_val, prev_lp, done=False)
            if self.ppo.ready():
                self.ppo.update(obs, last_done=False)

        # ── Action selection ────────────────────────────────────────────
        ppo_raw, _, _ = self.ppo.select_action(obs)
        params = decode_action(ppo_raw, self.y_min, self.y_max, self.z_min, self.z_max)
        standoff = float(np.clip(params['standoff'], STANDOFF_MIN, STANDOFF_MAX))
        flow     = params['flow']
        target_y = float(np.clip(params['target_y'], self.y_min, self.y_max))
        target_z = float(np.clip(params['target_z'], self.z_min, self.z_max))

        cells = self._defect_cells(matrix)
        clusters = self._cluster_defect_cells(cells)

        if clusters:
            paint_path = self._generate_paint_path(target_y, target_z, standoff,
                                                    clusters=clusters)
            self.path_pub.publish(paint_path)
            self.get_logger().info(
                f'[vision {self._episode_step}] {len(clusters)} defect cluster(s) | '
                f'standoff={standoff:.3f}m flow={flow:.2f} | reward={reward:.2f}')
        else:
            self.path_pub.publish(Path())
            self.get_logger().info(
                f'[vision {self._episode_step}] no defects | reward={reward:.2f}')

        act_msg = Float32MultiArray()
        act_msg.data = [standoff, flow]
        self.action_pub.publish(act_msg)

        r_msg = Float32(); r_msg.data = reward
        self.reward_pub.publish(r_msg)

        s_msg = String()
        s_msg.data = (f'[Vision step {self._episode_step}] '
                     f'good={obs[2]*100:.1f}% unpainted={obs[0]*100:.1f}% '
                     f'weak={obs[1]*100:.1f}% uneven={obs[4]*100:.1f}% | '
                     f'reward={reward:.2f} totalR={self._total_reward:.1f}')
        self.status_pub.publish(s_msg)
        self.get_logger().info(s_msg.data)

        if self._episode_step % 20 == 0:
            self._save_checkpoints()

        self._prev_obs = obs
        self._prev_action = ppo_raw

    def destroy_node(self):
        self.get_logger().info(
            f'Vision RL shutting down | steps={self._episode_step} | '
            f'total_reward={self._total_reward:.2f}')
        self._save_checkpoints()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionRLAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
