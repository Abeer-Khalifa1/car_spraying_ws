#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <builtin_interfaces/msg/duration.hpp>

#include <vector>
#include <queue>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <limits>
#include <algorithm>  // for std::clamp
#include <cmath>

#include "spray_common.hpp"
#include "correction_pass.hpp"

using namespace std::chrono_literals;

// ======
//  Main
// ======

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "cartesian_trajectory_controller",
    rclcpp::NodeOptions()
      .automatically_declare_parameters_from_overrides(true)
      .append_parameter_override("use_sim_time", true)
  );

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner_thread([&executor]() { executor.spin(); });

  // Cartesian execution parameters 
  const size_t segment_size          = 30;
  const double jump_threshold        = 0.0;
  const double fraction_min          = 0.5;
  const double desired_ee_speed      = 0.05;   // m/s
  const int    MAX_INLINE_CORRECTIONS = 5;

  // Standoff clamping limits (metres)
  const double STANDOFF_MIN     = 0.15;
  const double STANDOFF_MAX     = 0.25;
  const double DEFAULT_STANDOFF = 0.20;

  //  RL-agent detection
  std::atomic<bool> rl_agent_detected{false};

  std::atomic<bool> spray_enabled{true};   // default ON until RL takes control

  // Guards spray_enabled against RL messages that arrive during PASS 1.
  // The RL node publishes /spray/enable=false at startup; honouring that
  // during PASS 1 would suppress spray for the entire CSV trajectory.
  // Only after PASS 1 is complete do we let the RL agent gate spray state.
  std::atomic<bool> pass1_complete{false};

  auto spray_enable_sub = node->create_subscription<std_msgs::msg::Bool>(
    "/spray/enable", rclcpp::QoS(10),
    [&](const std_msgs::msg::Bool::SharedPtr msg) {
      // Any message on /spray/enable means the RL agent is alive
      if (!rl_agent_detected.load()) {
        rl_agent_detected.store(true);
        RCLCPP_INFO(node->get_logger(), "RL agent detected via /spray/enable.");
      }
      // Only honour the gate after PASS 1 — ignore early false published at RL startup
      if (pass1_complete.load()) {
        spray_enabled.store(msg->data);
      }
      RCLCPP_INFO(node->get_logger(),
        "/spray/enable received: %s%s", msg->data ? "ON" : "OFF",
        pass1_complete.load() ? "" : " (ignored — PASS 1 in progress)");
    }
  );


  std::vector<geometry_msgs::msg::PoseStamped> rl_path_poses;
  std::mutex rl_path_mutex;
  std::atomic<bool> rl_path_available{false};

  auto rl_path_sub = node->create_subscription<nav_msgs::msg::Path>(
    "/spray/rl_path", rclcpp::QoS(10),
    [&](const nav_msgs::msg::Path::SharedPtr msg) {
      {
        std::lock_guard<std::mutex> lock(rl_path_mutex);
        rl_path_poses = msg->poses;
        rl_path_available.store(!msg->poses.empty());
      }
      // rl_path arriving = definitive proof agent is alive
      if (!rl_agent_detected.load()) {
        rl_agent_detected.store(true);
        RCLCPP_INFO(node->get_logger(), "RL agent detected via /spray/rl_path.");
      }
      RCLCPP_INFO(node->get_logger(),
        "/spray/rl_path received: %zu waypoints", msg->poses.size());
    }
  );

  // ===================================================
  //  RL target queue  (legacy Point-based corrections)
  // ===================================================

  std::queue<geometry_msgs::msg::Point> rl_target_queue;
  std::mutex rl_queue_mutex;

  auto rl_target_sub = node->create_subscription<geometry_msgs::msg::Point>(
    "/spray/rl_target", rclcpp::QoS(20),
    [&](const geometry_msgs::msg::Point::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(rl_queue_mutex);
      rl_target_queue.push(*msg);
      RCLCPP_INFO(node->get_logger(),
        "RL target queued: x=%.3f y=%.3f z=%.3f  (queue depth: %zu)",
        msg->x, msg->y, msg->z, rl_target_queue.size());
    }
  );

  // ===============================
  //  RL spray parameter subscriber
  // ===============================

  std::atomic<float> rl_standoff{static_cast<float>(DEFAULT_STANDOFF)};
  std::atomic<float> rl_flow{0.50f};

  auto rl_action_sub = node->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/spray/rl_action", rclcpp::QoS(10),
    [&](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
      if (msg->data.size() >= 2) {
        float clamped = std::clamp(msg->data[0],
                                   static_cast<float>(STANDOFF_MIN),
                                   static_cast<float>(STANDOFF_MAX));
        rl_standoff.store(clamped);
        rl_flow.store(msg->data[1]);
        RCLCPP_INFO(node->get_logger(),
          "RL spray params updated: standoff=%.3f m (clamped from %.3f)  flow=%.3f",
          clamped, msg->data[0], msg->data[1]);
      }
    }
  );

  // =============
  //  Publishers
  // =============

  auto spray_pub = node->create_publisher<std_msgs::msg::Bool>(
    "/spray/active", rclcpp::QoS(1).transient_local());
  auto singularity_pub =
    node->create_publisher<std_msgs::msg::Bool>("/singularity_warning", 10);
  auto manipulability_pub =
    node->create_publisher<std_msgs::msg::Float64>("/manipulability", 10);

  // /spray/planning_failed — fires whenever MoveIt could not plan/execute a
  // requested motion (unreachable target, IK failure, execution rejected)
  // for a reason OTHER than the singularity check above. This is the signal
  // rl_agent_node.py needs to learn "don't ask for that pose again" instead
  // of just watching PASS 2 quietly time out. Non-latched: the RL node
  // treats it as a sticky "seen since last decision step" event.
  auto planning_failed_pub =
    node->create_publisher<std_msgs::msg::Bool>("/spray/planning_failed", 10);

  // /spray/pass1_done  — latched Bool that rl_agent_node waits for
  auto pass1_done_pub = node->create_publisher<std_msgs::msg::Bool>(
    "/spray/pass1_done",
    rclcpp::QoS(1).transient_local().reliable());

  // /spray/tracking_pose — current EE pose, published at ~10 Hz during motion
  auto tracking_pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/spray/tracking_pose", rclcpp::QoS(10));

  // tracking_active controls the background pose-publisher thread.
  std::atomic<bool> tracking_active{true};

  set_spray_force(spray_pub, node->get_logger(), false);

  // ==============
  //  MoveIt setup
  // ==============

  moveit::planning_interface::MoveGroupInterface move_group(node, "car_spraying_arm");
  move_group.setPoseReferenceFrame("world");
  move_group.setEndEffectorLink("link_6");
  move_group.setPlanningTime(1.0);
  move_group.setNumPlanningAttempts(5);

  // Spraying doesn't need millimeter/sub-degree exactness on the nozzle
  // pose — MoveIt's defaults (~0.0001 m / ~0.001 rad) are tight enough
  // that many poses with a perfectly good *nearby* IK solution get
  // rejected outright ("Unable to sample any valid states for goal
  // tree"), which looks like "unreachable" but is really "no exact
  // solution, and we never asked it to consider a close one."
  move_group.setGoalPositionTolerance(0.01);      // 1 cm
  move_group.setGoalOrientationTolerance(0.05);   // ~2.9 degrees
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  // Start background EE-pose publisher (tracking) 
  std::thread tracking_thread([&]() {
    while (tracking_active.load()) {
      try {
        geometry_msgs::msg::PoseStamped ps = move_group.getCurrentPose();
        ps.header.stamp = node->get_clock()->now();
        tracking_pose_pub->publish(ps);
      } catch (...) {}
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // =========
  // Load CSV
  // =========

  // csv_path is injected by filter_and_forward at launch time (peya_validated.csv).
  // Falls back to the original peya.csv when run without the validator.
  // Override at runtime: --ros-args -p csv_path:=/path/to/peya_validated.csv
  std::string file_path = node->has_parameter("csv_path")
    ? node->get_parameter("csv_path").as_string()
    : std::string{};

  if (file_path.empty()) {
    file_path = "/home/user/car_spraying_ws/src/square_trajectory/peya.csv";
    RCLCPP_WARN(node->get_logger(),
      "csv_path parameter not set — falling back to default: %s", file_path.c_str());
  }

  RCLCPP_INFO(node->get_logger(), "Loading CSV: %s", file_path.c_str());

  std::vector<SurfaceWaypoint> surface_waypoints =
    load_surface_waypoints(file_path, node->get_logger());

  if (surface_waypoints.empty()) {
    // load_surface_waypoints() already logged the specific reason
    // (file not opened, or no valid rows).
    executor.cancel(); spinner_thread.join(); rclcpp::shutdown(); return 1;
  }

  RCLCPP_WARN(node->get_logger(),
    "CSV columns 3-6 (quaternion) are IGNORED — orientation is recomputed "
    "from the surface normal (cols 7-9) by pose_from_surface(). "
    "The quaternion columns in the original peya.csv were inconsistent with "
    "the normals. Use the fixed peya.csv for a consistent file, but robot "
    "behaviour is unaffected either way.");

  // ===========================
  // Build 6-DOF pose waypoints
  // ===========================

  auto build_waypoints = [&](double standoff) {
    std::vector<geometry_msgs::msg::Pose> poses;
    poses.reserve(surface_waypoints.size());
    for (const auto & wp : surface_waypoints)
      poses.push_back(pose_from_surface(wp, standoff));
    return poses;
  };

  double current_standoff = DEFAULT_STANDOFF;
  std::vector<geometry_msgs::msg::Pose> waypoints = build_waypoints(current_standoff);

  RCLCPP_INFO(node->get_logger(),
    "Built %zu 6-DOF waypoints at standoff=%.3f m.",
    waypoints.size(), current_standoff);

  for (size_t idx : {(size_t)0, waypoints.size()/2, waypoints.size()-1}) {
    const auto & p = waypoints[idx];
    RCLCPP_INFO(node->get_logger(),
      "  Nozzle[%3zu]: pos=(%.3f, %.3f, %.3f)  quat=(%.3f, %.3f, %.3f, %.3f)",
      idx,
      p.position.x, p.position.y, p.position.z,
      p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
  }

  // ==========================
  // Move to start (spray OFF)
  // ==========================

  RCLCPP_INFO(node->get_logger(), "Moving to start waypoint — spray OFF.");
  move_group.setPoseTarget(waypoints[0]);

  if (move_group.move() != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(node->get_logger(), "Failed to reach starting waypoint.");
    executor.cancel(); spinner_thread.join(); rclcpp::shutdown(); return 1;
  }

  // ========================================
  // PASS 1 — Continuous Cartesian execution
  // ========================================

  RCLCPP_INFO(node->get_logger(), "=== PASS 1: CSV trajectory ===");
  set_spray(spray_pub, node->get_logger(), true, spray_enabled);

  int skipped_segments  = 0;
  int executed_segments = 0;

  bool pass1_ok = execute_cartesian_segment(
    move_group, waypoints, jump_threshold, fraction_min,
    desired_ee_speed, node->get_logger(),
    singularity_pub, manipulability_pub);

  {
    auto final_wp  = waypoints.back();
    auto cur_pose  = move_group.getCurrentPose().pose;
    RCLCPP_INFO(node->get_logger(),
      "Requested final waypoint: x=%.3f y=%.3f z=%.3f",
      final_wp.position.x, final_wp.position.y, final_wp.position.z);
    RCLCPP_INFO(node->get_logger(),
      "Current pose:             x=%.3f y=%.3f z=%.3f",
      cur_pose.position.x, cur_pose.position.y, cur_pose.position.z);
  }

  if (pass1_ok) {
    executed_segments = 1;
    RCLCPP_INFO(node->get_logger(),
      "PASS 1 complete — continuous CSV path executed.");
    // Publish pass1_done for the success (single-shot) path
    {
      set_spray_force(spray_pub, node->get_logger(), false);
      pass1_complete.store(true);
      std_msgs::msg::Bool p1_msg; p1_msg.data = true;
      pass1_done_pub->publish(p1_msg);
      RCLCPP_INFO(node->get_logger(),
        "Published /spray/pass1_done=true -- RL agent may now act.");
      std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }
  } else {
    skipped_segments = 1;
    RCLCPP_WARN(node->get_logger(),
      "Continuous CSV path failed — falling back to segmented execution.");

    size_t i = segment_size * 2;

    auto find_nearest_waypoint_index =
      [&](const geometry_msgs::msg::Point & target) {
        double best_dist = std::numeric_limits<double>::infinity();
        size_t best_idx  = i;
        for (size_t k = i; k < waypoints.size(); ++k) {
          double dx = waypoints[k].position.x - target.x;
          double dy = waypoints[k].position.y - target.y;
          double dz = waypoints[k].position.z - target.z;
          double d2 = dx*dx + dy*dy + dz*dz;
          if (d2 < best_dist) { best_dist = d2; best_idx = k; }
        }
        return (best_idx / segment_size) * segment_size;
      };

    while (i < waypoints.size()) {

      {
        double new_standoff = static_cast<double>(rl_standoff.load());
        if (std::abs(new_standoff - current_standoff) > 1e-4) {
          current_standoff = new_standoff;
          waypoints = build_waypoints(current_standoff);
          RCLCPP_INFO(node->get_logger(),
            "Standoff updated to %.3f m — waypoints rebuilt.", current_standoff);
        }
      }

      geometry_msgs::msg::Point last_target;
      int pre_corrections = drain_rl_corrections(
        move_group, spray_pub, singularity_pub, manipulability_pub,
        planning_failed_pub,
        rl_target_queue, rl_queue_mutex,
        rl_standoff, rl_flow, spray_enabled,
        node->get_logger(),
        jump_threshold, fraction_min, desired_ee_speed,
        MAX_INLINE_CORRECTIONS, surface_waypoints, &last_target);

      if (pre_corrections > 0) {
        size_t resume_idx = find_nearest_waypoint_index(last_target);
        if (resume_idx > i) {
          RCLCPP_INFO(node->get_logger(),
            "Resuming CSV at waypoint %zu after adaptive correction.", resume_idx);
          i = resume_idx;
        }
      }

      std::vector<geometry_msgs::msg::Pose> segment;
      for (size_t j = i; j < i + segment_size && j < waypoints.size(); ++j)
        segment.push_back(waypoints[j]);

      if (segment.empty()) { i += segment_size; continue; }

      set_spray(spray_pub, node->get_logger(), true, spray_enabled);

      bool ok = execute_cartesian_segment(
        move_group, segment, jump_threshold, fraction_min,
        desired_ee_speed, node->get_logger(),
        singularity_pub, manipulability_pub);

      if (ok) {
        executed_segments++;
        auto cur_pose = move_group.getCurrentPose().pose;
        RCLCPP_INFO(node->get_logger(),
          "  ✓ Segment %zu — current pose: x=%.3f y=%.3f z=%.3f",
          i, cur_pose.position.x, cur_pose.position.y, cur_pose.position.z);
      } else {
        skipped_segments++;
        size_t next = i + segment_size;
        if (next < waypoints.size()) {
          set_spray_force(spray_pub, node->get_logger(), false);
          auto cur_pose = move_group.getCurrentPose().pose;
          std::vector<geometry_msgs::msg::Pose> hop = { cur_pose, waypoints[next] };
          moveit_msgs::msg::RobotTrajectory hop_traj;
          double hop_frac = move_group.computeCartesianPath(
            hop, 0.01, hop_traj, true);
          if (hop_frac >= fraction_min) {
            move_group.execute(hop_traj);
            RCLCPP_INFO(node->get_logger(),
              "  Cartesian hop to segment %zu (frac=%.2f).", next, hop_frac);
          } else {
            move_group.setPoseTarget(waypoints[next]);
            move_group.move();
            RCLCPP_WARN(node->get_logger(),
              "  Joint-space fallback hop to segment %zu.", next);
          }
        }
      }

      drain_rl_corrections(
        move_group, spray_pub, singularity_pub, manipulability_pub,
        planning_failed_pub,
        rl_target_queue, rl_queue_mutex,
        rl_standoff, rl_flow, spray_enabled,
        node->get_logger(),
        jump_threshold, fraction_min, desired_ee_speed,
        MAX_INLINE_CORRECTIONS, surface_waypoints);

      i += segment_size;
    }

    RCLCPP_INFO(node->get_logger(),
      "PASS 1 segmented fallback complete — executed: %d  skipped: %d",
      executed_segments, skipped_segments);
  }

  // Signal rl_agent_node that PASS 1 is complete
  {
    set_spray_force(spray_pub, node->get_logger(), false);
    pass1_complete.store(true);
    std_msgs::msg::Bool p1_msg;
    p1_msg.data = true;
    pass1_done_pub->publish(p1_msg);
    RCLCPP_INFO(node->get_logger(),
      "Published /spray/pass1_done=true -- RL agent may now act.");
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
  }

  // ======================================================
  // PASS 2 — RL correction loop  (skipped if no RL agent)
  // ======================================================

  // MAX_CORRECTION_PASSES now counts ONLY real /spray/rl_path executions.
  // Previously it also counted legacy /spray/rl_target Point drains, which
  // silently ate the whole budget — field log showed "RL path executions:
  // 11  total corrections: 51", i.e. ~40 of 50 slots were burned by the
  // legacy queue before real work got a fair share. Raised from 50 -> 200
  // now that the count is honest.
  CorrectionPassConfig pass2_cfg;
  pass2_cfg.max_correction_passes  = 200;
  pass2_cfg.detect_timeout_ms      = 5000;   // 5 s to detect the RL agent
  pass2_cfg.idle_timeout_ms        = 8000;   // exit after 8 s of no work
  pass2_cfg.jump_threshold         = jump_threshold;
  pass2_cfg.fraction_min           = fraction_min;
  pass2_cfg.desired_ee_speed       = desired_ee_speed;
  pass2_cfg.max_inline_corrections = MAX_INLINE_CORRECTIONS;
  pass2_cfg.path_topic_label       = "sim /spray/rl_path";

  CorrectionPassResult pass2_result = run_correction_pass(
    move_group, node,
    spray_pub, singularity_pub, manipulability_pub, planning_failed_pub,
    rl_path_poses, rl_path_mutex, rl_path_available,
    rl_agent_detected,
    rl_target_queue, rl_queue_mutex, rl_standoff, rl_flow,
    spray_enabled, surface_waypoints, pass2_cfg);

  // ================
  // After Execution
  // ================

  set_spray_force(spray_pub, node->get_logger(), false);

  RCLCPP_INFO(node->get_logger(),
    "=== COMPLETE ===  "
    "CSV segments executed: %d  skipped: %d  |  "
    "RL path executions: %d  total corrections: %d",
    executed_segments, skipped_segments,
    pass2_result.rl_path_executions, pass2_result.correction_count);

  tracking_active.store(false);
  if (tracking_thread.joinable()) tracking_thread.join();

  executor.cancel();
  spinner_thread.join();
  rclcpp::shutdown();
  return 0;
}