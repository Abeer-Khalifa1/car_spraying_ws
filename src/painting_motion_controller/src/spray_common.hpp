#pragma once

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <Eigen/Dense>

#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <string>

// =========================================================
//  Surface waypoint — carries position AND surface normal
// =========================================================
struct SurfaceWaypoint // : corresponds to one row in the CSV
{
  double x, y, z;         // surface position
  double nx, ny, nz;      // outward surface normal (unit vector)
};

// Build a perpendicular spray pose from a surface waypoint.
geometry_msgs::msg::Pose pose_from_surface(
  const SurfaceWaypoint & wp,
  double standoff);

// Load surface waypoints (position + normal) from the trajectory CSV.
// Returns an empty vector and logs an error if the file can't be read
// or has no valid rows.
std::vector<SurfaceWaypoint> load_surface_waypoints(
  const std::string & csv_path,
  rclcpp::Logger logger);

// Force the spray state OFF/ON regardless of the spray_enabled gate.
void set_spray_force(
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub,
  rclcpp::Logger logger,
  bool active);

// Set spray state, respecting the spray_enabled gate from /spray/enable.
void set_spray(
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub,
  rclcpp::Logger logger,
  bool active,
  const std::atomic<bool> & spray_enabled);

// Singularity check using Jacobian SVD over a whole planned trajectory.
bool trajectory_is_singular(
  moveit::planning_interface::MoveGroupInterface & move_group,
  const moveit_msgs::msg::RobotTrajectory & traj,
  rclcpp::Logger logger,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub);

// Plan + execute a Cartesian segment, with singularity/IK-failure
// avoidance retries (orientation perturbation, relaxed first waypoint)
// constant end-effector-speed retiming.
bool execute_cartesian_segment(
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::vector<geometry_msgs::msg::Pose> & segment,
  double jump_threshold,
  double fraction_min,
  double ee_speed,
  rclcpp::Logger logger,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub);

// Build correction waypoints perpendicular to a surface normal
// (boustrophedon patch used for both legacy Point corrections and,
std::vector<geometry_msgs::msg::Pose> build_correction_patch(
  const geometry_msgs::msg::Point & target,
  const Eigen::Vector3d & surface_normal,
  double standoff,
  double stroke,
  double step);

// Execute a nav_msgs::msg::Path-derived corrective pass directly via
bool execute_rl_path(
  moveit::planning_interface::MoveGroupInterface & move_group,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr spray_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr planning_failed_pub,
  const std::vector<geometry_msgs::msg::PoseStamped> & path_poses,
  const std::atomic<bool> & spray_enabled,
  double jump_threshold,
  double fraction_min,
  double ee_speed,
  rclcpp::Logger logger);

// Drain the legacy Point-based correction queue (older RL interface).
// Vision executor passes an empty queue that nothing ever populates.
int drain_rl_corrections(
  moveit::planning_interface::MoveGroupInterface & move_group,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr spray_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr planning_failed_pub,
  std::queue<geometry_msgs::msg::Point> & rl_target_queue,
  std::mutex & rl_queue_mutex,
  std::atomic<float> & rl_standoff,
  std::atomic<float> & rl_flow,
  const std::atomic<bool> & spray_enabled,
  rclcpp::Logger logger,
  double jump_threshold,
  double fraction_min,
  double ee_speed,
  int max_corrections,
  const std::vector<SurfaceWaypoint> & surface_waypoints,
  geometry_msgs::msg::Point * last_target = nullptr);
