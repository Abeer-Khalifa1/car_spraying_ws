#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>

#include "spray_common.hpp"
#include "correction_pass.hpp"

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "vision_pass_executor_node",
    rclcpp::NodeOptions()
      .automatically_declare_parameters_from_overrides(true)
      .append_parameter_override("use_sim_time", true)
  );

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner_thread([&executor]() { executor.spin(); });

  // Cartesian execution parameters — kept identical to the main
  // controller's defaults so corrective motion behaves the same way.
  const double jump_threshold    = 0.0;
  const double fraction_min      = 0.5;
  const double desired_ee_speed  = 0.05;   // m/s
  const int    MAX_INLINE_CORRECTIONS = 5;

  // ── /spray/pass1_done — wait for PASS 1 to finish ─────────────────────
  std::atomic<bool> pass1_done{false};
  auto pass1_done_qos = rclcpp::QoS(1).transient_local().reliable();
  auto pass1_done_sub = node->create_subscription<std_msgs::msg::Bool>(
    "/spray/pass1_done", pass1_done_qos,
    [&](const std_msgs::msg::Bool::SharedPtr msg) {
      if (msg->data && !pass1_done.load()) {
        pass1_done.store(true);
        RCLCPP_INFO(node->get_logger(), "PASS 1 complete — vision executor may now act.");
      }
    }
  );

  // ── /spray/vision_rl_path — vision RL's corrective Path ───────────────
  std::vector<geometry_msgs::msg::PoseStamped> vision_path_poses;
  std::mutex vision_path_mutex;
  std::atomic<bool> vision_path_available{false};
  std::atomic<bool> vision_agent_detected{false};

  auto vision_path_sub = node->create_subscription<nav_msgs::msg::Path>(
    "/spray/vision_rl_path", rclcpp::QoS(10),
    [&](const nav_msgs::msg::Path::SharedPtr msg) {
      {
        std::lock_guard<std::mutex> lock(vision_path_mutex);
        vision_path_poses = msg->poses;
        vision_path_available.store(!msg->poses.empty());
      }
      if (!vision_agent_detected.load()) {
        vision_agent_detected.store(true);
        RCLCPP_INFO(node->get_logger(), "Vision RL agent detected via /spray/vision_rl_path.");
      }
      RCLCPP_INFO(node->get_logger(),
        "/spray/vision_rl_path received: %zu waypoints", msg->poses.size());
    }
  );

  // Also treat /spray/vision_rl_action as a heartbeat, same pattern as
  // the sim controller treats /spray/rl_action / /spray/enable — some
  // decision steps produce no defect clusters and publish an empty Path,
  // so the action topic is a more reliable "agent is alive" signal.
  auto vision_action_sub = node->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/spray/vision_rl_action", rclcpp::QoS(10),
    [&](const std_msgs::msg::Float32MultiArray::SharedPtr) {
      if (!vision_agent_detected.load()) {
        vision_agent_detected.store(true);
        RCLCPP_INFO(node->get_logger(), "Vision RL agent detected via /spray/vision_rl_action.");
      }
    }
  );

  // ── Legacy Point-based correction queue: unused for the vision agent ──
  // vision_rl_agent_node.py only ever publishes Path + action messages,
  // never legacy /spray/rl_target Points, so this queue simply never has
  // work. It exists only because run_correction_pass()'s signature is
  // shared with the sim controller.
  std::queue<geometry_msgs::msg::Point> unused_legacy_queue;
  std::mutex unused_legacy_mutex;
  std::atomic<float> unused_standoff{0.20f};
  std::atomic<float> unused_flow{0.50f};

  // ── spray_enabled: vision agent has no /spray/enable-style gate today.
  // Always true here — spray is only ever forced off by run_correction_pass
  // itself between strokes. If you later want the vision agent to be able
  // to disable spray (mirroring /spray/enable for the sim agent), add a
  // /spray/vision_enable subscriber here and wire it into this atomic.
  std::atomic<bool> spray_enabled{true};

  // ── Publishers (same topics the main controller publishes to) ─────────
  auto spray_pub = node->create_publisher<std_msgs::msg::Bool>(
    "/spray/active", rclcpp::QoS(1).transient_local());
  auto singularity_pub =
    node->create_publisher<std_msgs::msg::Bool>("/singularity_warning", 10);
  auto manipulability_pub =
    node->create_publisher<std_msgs::msg::Float64>("/manipulability", 10);
  auto planning_failed_pub =
    node->create_publisher<std_msgs::msg::Bool>("/spray/planning_failed", 10);

  set_spray_force(spray_pub, node->get_logger(), false);

  // ── MoveIt setup — identical to the main controller's configuration ──
  moveit::planning_interface::MoveGroupInterface move_group(node, "car_spraying_arm");
  move_group.setPoseReferenceFrame("world");
  move_group.setEndEffectorLink("link_6");
  move_group.setPlanningTime(1.0);
  move_group.setNumPlanningAttempts(5);
  move_group.setGoalPositionTolerance(0.01);      // 1 cm
  move_group.setGoalOrientationTolerance(0.05);   // ~2.9 degrees
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  // ── Surface waypoints — loaded only for legacy-correction normal lookup.
  // Harmless if unused; kept for signature compatibility and in case you
  // later add legacy Point corrections to the vision agent too.
  std::string file_path = node->has_parameter("csv_path")
    ? node->get_parameter("csv_path").as_string()
    : std::string{};
  if (file_path.empty()) {
    file_path = "/home/user/car_spraying_ws/src/square_trajectory/peya.csv";
    RCLCPP_WARN(node->get_logger(),
      "csv_path parameter not set — falling back to default: %s", file_path.c_str());
  }
  std::vector<SurfaceWaypoint> surface_waypoints =
    load_surface_waypoints(file_path, node->get_logger());
  // Not fatal if empty here — the vision agent doesn't use legacy
  // corrections in practice, so an empty lookup table just means the
  // (never-exercised) fallback normal of +X would be used.

  // ── Wait for PASS 1 to finish (with 120s timeout) ────────────────────
  RCLCPP_INFO(node->get_logger(), "vision_pass_executor: waiting for /spray/pass1_done...");
  auto start_wait = std::chrono::steady_clock::now();
  const auto PASS1_TIMEOUT = std::chrono::seconds(120);
  while (rclcpp::ok() && !pass1_done.load()) {
    auto elapsed = std::chrono::steady_clock::now() - start_wait;
    if (elapsed > PASS1_TIMEOUT) {
      RCLCPP_ERROR(node->get_logger(),
        "TIMEOUT: /spray/pass1_done not received after %.1f seconds. "
        "Check if PASS 1 (cartesian_trajectory_controller) completed successfully.",
        std::chrono::duration<double>(elapsed).count());
      executor.cancel(); spinner_thread.join(); rclcpp::shutdown(); return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  if (!rclcpp::ok()) {
    executor.cancel(); spinner_thread.join(); rclcpp::shutdown(); return 0;
  }

  // ── Run the corrective pass — same function, vision-side wiring ──────
  CorrectionPassConfig cfg;
  cfg.max_correction_passes  = 200;
  cfg.detect_timeout_ms      = 5000;
  cfg.idle_timeout_ms        = 8000;
  cfg.jump_threshold         = jump_threshold;
  cfg.fraction_min           = fraction_min;
  cfg.desired_ee_speed       = desired_ee_speed;
  cfg.max_inline_corrections = MAX_INLINE_CORRECTIONS;
  cfg.path_topic_label       = "vision /spray/vision_rl_path";

  CorrectionPassResult result = run_correction_pass(
    move_group, node,
    spray_pub, singularity_pub, manipulability_pub, planning_failed_pub,
    vision_path_poses, vision_path_mutex, vision_path_available,
    vision_agent_detected,
    unused_legacy_queue, unused_legacy_mutex, unused_standoff, unused_flow,
    spray_enabled, surface_waypoints, cfg);

  set_spray_force(spray_pub, node->get_logger(), false);

  RCLCPP_INFO(node->get_logger(),
    "=== VISION PASS COMPLETE === executions: %d  legacy drains: %d",
    result.rl_path_executions, result.legacy_correction_count);

  executor.cancel();
  spinner_thread.join();
  rclcpp::shutdown();
  return 0;
}
