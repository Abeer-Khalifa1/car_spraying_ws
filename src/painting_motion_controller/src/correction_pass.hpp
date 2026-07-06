#pragma once
// =====================================================================
//  correction_pass.hpp
//
//  The PASS 2 correction loop, extracted out of
//  cartesian_trajectory_controller.cpp so it can be reused, byte-for-byte
//  identical in behavior, by BOTH:
//    - the main controller (sim RL, /spray/rl_path)
//    - vision_pass_executor.cpp (vision RL, /spray/vision_rl_path)
//
//  Nothing about the safety logic (singularity checks, spray gating,
//  idle timeout, correction caps) differs between the two callers —
//  only which topic/queues are wired in and what gets printed in logs.
// =====================================================================

#include "spray_common.hpp"
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <std_msgs/msg/float64.hpp>

#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <string>

struct CorrectionPassConfig
{
  int max_correction_passes = 200;
  int detect_timeout_ms     = 5000;   // how long to wait for the agent to show signs of life
  int idle_timeout_ms       = 8000;   // exit after this long with no work
  double jump_threshold     = 0.0;
  double fraction_min       = 0.5;
  double desired_ee_speed   = 0.05;
  int max_inline_corrections = 5;     // cap for legacy Point-based drains per call
  std::string path_topic_label = "rl_path";  // just for log readability
};

struct CorrectionPassResult
{
  int correction_count        = 0;   // real Path executions only
  int rl_path_executions      = 0;
  int legacy_correction_count = 0;
  bool agent_detected         = false;
};

// Runs the full PASS 2 loop: probes for agent activity, then repeatedly
// drains whichever of (Path, legacy Point queue) has work, executing
// corrective motion via execute_rl_path / drain_rl_corrections, until
// either max_correction_passes is hit or the agent goes idle for
// idle_timeout_ms.
//
// path_poses / path_mutex / path_available: populated by the caller's own
// subscription to whichever topic it cares about (/spray/rl_path or
// /spray/vision_rl_path) — this function only reads them.
//
// rl_target_queue / rl_queue_mutex / rl_standoff / rl_flow: legacy
// Point-based correction support. Pass genuinely empty/unused ones if the
// caller's agent never publishes on that interface (e.g. the vision
// agent doesn't) — they'll simply never have work and cost nothing.
CorrectionPassResult run_correction_pass(
  moveit::planning_interface::MoveGroupInterface & move_group,
  rclcpp::Node::SharedPtr node,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr spray_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr planning_failed_pub,
  std::vector<geometry_msgs::msg::PoseStamped> & path_poses,
  std::mutex & path_mutex,
  std::atomic<bool> & path_available,
  std::atomic<bool> & agent_detected,
  std::queue<geometry_msgs::msg::Point> & rl_target_queue,
  std::mutex & rl_queue_mutex,
  std::atomic<float> & rl_standoff,
  std::atomic<float> & rl_flow,
  const std::atomic<bool> & spray_enabled,
  const std::vector<SurfaceWaypoint> & surface_waypoints,
  const CorrectionPassConfig & cfg);
