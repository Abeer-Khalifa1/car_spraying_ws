#include "correction_pass.hpp"
#include <chrono>
#include <thread>

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
  const CorrectionPassConfig & cfg)
{
  CorrectionPassResult result;
  auto logger = node->get_logger();

  // ── Probe: wait for the agent to show signs of life ──────────────────
  RCLCPP_INFO(logger,
    "PASS 2 (%s): probing for agent (timeout %d ms)...",
    cfg.path_topic_label.c_str(), cfg.detect_timeout_ms);
  {
    int waited = 0;
    while (!agent_detected.load() && waited < cfg.detect_timeout_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      waited += 250;
    }
  }

  if (!agent_detected.load()) {
    RCLCPP_INFO(logger,
      "No agent detected within %d ms on %s -- PASS 2 skipped.",
      cfg.detect_timeout_ms, cfg.path_topic_label.c_str());
    result.agent_detected = false;
    return result;
  }
  result.agent_detected = true;

  RCLCPP_INFO(logger,
    "=== PASS 2 (%s): correction loop (max %d iterations) ===",
    cfg.path_topic_label.c_str(), cfg.max_correction_passes);

  auto idle_start = std::chrono::steady_clock::now();

  while (result.correction_count < cfg.max_correction_passes)
  {
    bool did_work = false;

    // Priority 1: Path-based correction (nav_msgs::Path -> PoseStamped vector)
    if (path_available.load())
    {
      std::vector<geometry_msgs::msg::PoseStamped> path_snapshot;
      {
        std::lock_guard<std::mutex> lock(path_mutex);
        path_snapshot = path_poses;
        path_available.store(false);
      }

      if (!path_snapshot.empty()) {
        RCLCPP_INFO(logger,
          "PASS 2 (%s) [%d]: executing path (%zu waypoints).",
          cfg.path_topic_label.c_str(), result.correction_count, path_snapshot.size());

        bool ok = execute_rl_path(
          move_group,
          spray_pub, singularity_pub, manipulability_pub,
          planning_failed_pub,
          path_snapshot, spray_enabled,
          cfg.jump_threshold, cfg.fraction_min, cfg.desired_ee_speed,
          logger);

        result.rl_path_executions++;
        result.correction_count++;
        did_work = true;
        idle_start = std::chrono::steady_clock::now();

        if (ok)
          RCLCPP_INFO(logger,
            "  %s execution %d complete.", cfg.path_topic_label.c_str(),
            result.rl_path_executions);
        else
          RCLCPP_WARN(logger,
            "  %s execution %d failed.", cfg.path_topic_label.c_str(),
            result.rl_path_executions);

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        continue;
      }
    }

    // Priority 2: legacy Point-based corrections
    {
      bool legacy_has_work = false;
      {
        std::lock_guard<std::mutex> lock(rl_queue_mutex);
        legacy_has_work = !rl_target_queue.empty();
      }

      if (legacy_has_work) {
        int drained = drain_rl_corrections(
          move_group, spray_pub, singularity_pub, manipulability_pub,
          planning_failed_pub,
          rl_target_queue, rl_queue_mutex,
          rl_standoff, rl_flow, spray_enabled,
          logger,
          cfg.jump_threshold, cfg.fraction_min, cfg.desired_ee_speed,
          cfg.max_inline_corrections, surface_waypoints);

        result.legacy_correction_count += drained;
        if (drained > 0) {
          did_work = true;
          idle_start = std::chrono::steady_clock::now();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        continue;
      }
    }

    // Nothing to do -- check idle timeout
    if (!did_work) {
      auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - idle_start).count();

      if (idle_ms >= cfg.idle_timeout_ms) {
        RCLCPP_INFO(logger,
          "PASS 2 (%s): agent idle for %lld ms -- assuming done, exiting.",
          cfg.path_topic_label.c_str(), (long long)idle_ms);
        break;
      }

      RCLCPP_INFO(logger,
        "PASS 2 (%s): waiting for agent... (%lld / %d ms idle)",
        cfg.path_topic_label.c_str(), (long long)idle_ms, cfg.idle_timeout_ms);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

  if (result.correction_count >= cfg.max_correction_passes)
    RCLCPP_WARN(node->get_logger(),
      "Reached MAX_CORRECTION_PASSES (%d real %s executions). Stopping PASS 2.",
      cfg.max_correction_passes, cfg.path_topic_label.c_str());
  else
    RCLCPP_INFO(node->get_logger(),
      "PASS 2 (%s) complete -- %d real executions, %d legacy drains.",
      cfg.path_topic_label.c_str(), result.correction_count, result.legacy_correction_count);

  return result;
}
