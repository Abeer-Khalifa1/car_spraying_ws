#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit/robot_state/robot_state.hpp>
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

using namespace std::chrono_literals;

// =========================================================
//  Surface waypoint — carries position AND surface normal
// =========================================================

struct SurfaceWaypoint // : corresponds to one row in the CSV
{
  double x, y, z;         // surface position
  double nx, ny, nz;      // outward surface normal (unit vector)
};

// =========================================================
//  Build a perpendicular spray pose from a surface waypoint.
// =========================================================

static geometry_msgs::msg::Pose pose_from_surface(
  const SurfaceWaypoint & wp,
  double standoff)
{
  geometry_msgs::msg::Pose pose; // position of the nozzle tip (standoff distance from surface)
  // wp: surface point, standoff along normal 
  // standoff: distance from surface to nozzle tip
  pose.position.x = wp.x - standoff * wp.nx; 
  pose.position.y = wp.y - standoff * wp.ny; 
  pose.position.z = wp.z - standoff * wp.nz; 

  Eigen::Vector3d tool_z(wp.nx, wp.ny, wp.nz); 
  tool_z.normalize();

  Eigen::Vector3d ref = (std::abs(tool_z.dot(Eigen::Vector3d::UnitZ())) < 0.9)
                        ? Eigen::Vector3d::UnitZ()
                        : Eigen::Vector3d::UnitX();

  Eigen::Vector3d tool_y = tool_z.cross(ref);  tool_y.normalize();
  Eigen::Vector3d tool_x = tool_y.cross(tool_z); tool_x.normalize();

  Eigen::Matrix3d R;
  R.col(0) = tool_x;
  R.col(1) = tool_y;
  R.col(2) = tool_z;

  Eigen::Quaterniond q(R);
  q.normalize();

  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();

  return pose; // Position of the nozzle tip, oriented perpendicularly to the surface
}

// ================================================================================
// Force the spray state OFF regardless of the spray_enabled gate (Safety standard)
// ================================================================================
static void set_spray_force(
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub,
  rclcpp::Logger logger,
  bool active)
{
  std_msgs::msg::Bool msg;
  msg.data = active;
  pub->publish(msg);
  RCLCPP_INFO(logger, "Spray %s (forced)", active ? "STARTED" : "STOPPED");
}

// =====================================================================
// Set spray state using RL, respecting the spray_enabled gate from /spray/enable
// =====================================================================
static void set_spray(
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub,
  rclcpp::Logger logger,
  bool active,
  const std::atomic<bool> & spray_enabled)
{
  // If the RL agent has disabled spray, honour it — never turn ON against it.
  // Turning OFF is always allowed.
  if (active && !spray_enabled.load()) {
    RCLCPP_DEBUG(logger, "set_spray(true) suppressed — /spray/enable is false");
    return;
  }
  std_msgs::msg::Bool msg;
  msg.data = active;
  pub->publish(msg);
  RCLCPP_INFO(logger, "Spray %s", active ? "STARTED" : "STOPPED");
}

// ==========================================
//  Singularity check using Jacobian and SVD
// ==========================================
static bool trajectory_is_singular(
    moveit::planning_interface::MoveGroupInterface & move_group,
    const moveit_msgs::msg::RobotTrajectory & traj,
    rclcpp::Logger logger,
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub)
{
    const double MANIP_THRESHOLD = 0.01;

    auto robot_model = move_group.getRobotModel();
    auto joint_model_group =
        robot_model->getJointModelGroup(move_group.getName());

    moveit::core::RobotState state(robot_model);
    double worst_sigma = std::numeric_limits<double>::infinity();

    for (const auto & point : traj.joint_trajectory.points)
    {
        state.setJointGroupPositions(joint_model_group, point.positions);
        state.update();

        Eigen::MatrixXd jacobian;
        state.getJacobian(
          joint_model_group,
          state.getLinkModel(joint_model_group->getLinkModelNames().back()),
          Eigen::Vector3d::Zero(),
          jacobian);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double sigma = svd.singularValues().minCoeff();
        worst_sigma = std::min(worst_sigma, sigma);

        if (worst_sigma < MANIP_THRESHOLD)
        {
            std_msgs::msg::Float64 manip_msg;
            manip_msg.data = worst_sigma;
            manipulability_pub->publish(manip_msg);

            std_msgs::msg::Bool warn_msg;
            warn_msg.data = true;
            singularity_pub->publish(warn_msg);

            RCLCPP_WARN(logger,
              "Singularity detected. worst_sigma = %.6f", worst_sigma);
            return true;
        }
    }

    if (worst_sigma == std::numeric_limits<double>::infinity())
      worst_sigma = 0.0;

    std_msgs::msg::Float64 manip_msg;
    manip_msg.data = worst_sigma;
    manipulability_pub->publish(manip_msg);

    std_msgs::msg::Bool warn_msg;
    warn_msg.data = false;
    singularity_pub->publish(warn_msg);

    return false;
}

// =====================================================================
// Planning, Execution of trajectory and dealing with found singularity 
// =====================================================================
static bool execute_cartesian_segment(
  moveit::planning_interface::MoveGroupInterface & move_group,
  const std::vector<geometry_msgs::msg::Pose> & segment,
  double jump_threshold,
  double fraction_min,
  double ee_speed,
  rclcpp::Logger logger,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub)
{
  if (segment.empty()) return false;

  moveit_msgs::msg::RobotTrajectory traj;
  double fraction = move_group.computeCartesianPath(
    segment, 0.01, traj, true);

  RCLCPP_INFO(logger, "Cartesian fraction = %.2f", fraction);

  if (fraction >= fraction_min) {
    if (trajectory_is_singular(move_group, traj, logger,
                               singularity_pub, manipulability_pub))
    {
      RCLCPP_WARN(logger, "Trajectory appears singular — attempting avoidance.");

      std::vector<double> deltas = { -0.10, -0.05, 0.05, 0.10 };
      bool found_alternative = false;
      moveit_msgs::msg::RobotTrajectory alt_traj;
      double alt_fraction = 0.0;

      for (double d : deltas) {
        std::vector<geometry_msgs::msg::Pose> perturbed;
        perturbed.reserve(segment.size());
        for (const auto & p : segment) {
          tf2::Quaternion q_orig(p.orientation.x, p.orientation.y,
                                 p.orientation.z, p.orientation.w);
          tf2::Quaternion rot;
          rot.setRPY(0.0, d, 0.0);
          tf2::Quaternion q_new = rot * q_orig;
          geometry_msgs::msg::Pose np = p;
          np.orientation.x = q_new.x(); np.orientation.y = q_new.y();
          np.orientation.z = q_new.z(); np.orientation.w = q_new.w();
          perturbed.push_back(np);
        }

        moveit_msgs::msg::RobotTrajectory try_traj;
        double try_fraction = move_group.computeCartesianPath(
          perturbed, 0.01, try_traj, true);

        RCLCPP_INFO(logger,
          "Tried perturbation %.3f — fraction=%.2f", d, try_fraction);

        if (try_fraction >= fraction_min) {
          if (!trajectory_is_singular(move_group, try_traj, logger,
                                      singularity_pub, manipulability_pub)) {
            found_alternative = true;
            alt_traj = try_traj;
            alt_fraction = try_fraction;
            break;
          }
        }
      }

      if (found_alternative) {
        RCLCPP_INFO(logger,
          "Found alternative non-singular path (fraction %.2f).", alt_fraction);
        traj = alt_traj;
      } else {
        RCLCPP_WARN(logger, "Trajectory rejected due to singularity.");
        return false;
      }
    }


    // Constant end-effector speed retiming 
    if (ee_speed > 0.0 && !traj.joint_trajectory.points.empty()) {
      auto robot_model = move_group.getRobotModel();
      auto joint_model_group =
          robot_model->getJointModelGroup(move_group.getName());

      moveit::core::RobotState state(robot_model);
      std::string ee_link = joint_model_group->getLinkModelNames().back();

      std::vector<Eigen::Vector3d> ee_positions;
      ee_positions.reserve(traj.joint_trajectory.points.size());

      for (const auto & point : traj.joint_trajectory.points) {
        state.setJointGroupPositions(joint_model_group, point.positions);
        state.update();
        Eigen::Isometry3d tf = state.getGlobalLinkTransform(ee_link);
        ee_positions.emplace_back(tf.translation());
      }

      std::vector<double> arc_len(ee_positions.size(), 0.0);
      for (size_t k = 1; k < ee_positions.size(); ++k)
        arc_len[k] = arc_len[k-1] + (ee_positions[k] - ee_positions[k-1]).norm();

      double total_len = arc_len.back();
      if (total_len >= 1e-6) {
        for (size_t k = 0; k < traj.joint_trajectory.points.size(); ++k) {
          double t = arc_len[k] / ee_speed;
          builtin_interfaces::msg::Duration d;
          d.sec     = static_cast<int32_t>(std::floor(t));
          d.nanosec = static_cast<uint32_t>((t - d.sec) * 1e9);
          traj.joint_trajectory.points[k].time_from_start = d;
        }

        size_t joint_count = traj.joint_trajectory.joint_names.size();

        for (size_t k = 0; k + 1 < traj.joint_trajectory.points.size(); ++k) {
          double dt = (arc_len[k+1] - arc_len[k]) / ee_speed;
          if (dt <= 1e-9) continue;
          const auto & q0 = traj.joint_trajectory.points[k].positions;
          const auto & q1 = traj.joint_trajectory.points[k+1].positions;
          traj.joint_trajectory.points[k].velocities.assign(joint_count, 0.0);
          for (size_t j = 0; j < joint_count; ++j)
            traj.joint_trajectory.points[k].velocities[j] = (q1[j] - q0[j]) / dt;
        }
        traj.joint_trajectory.points.back().velocities.assign(joint_count, 0.0);

        const auto & bounds = joint_model_group->getActiveJointModelsBounds();
        for (auto & pt : traj.joint_trajectory.points) {
          if (pt.velocities.empty()) continue;
          for (size_t j = 0; j < joint_count && j < bounds.size(); ++j) {
            if (!bounds[j] || bounds[j]->empty()) continue;
            double v_max = (*bounds[j])[0].max_velocity_;
            if (v_max <= 0.0) continue;
            pt.velocities[j] = std::clamp(pt.velocities[j], -v_max, v_max);
          }
        }

        for (auto & pt : traj.joint_trajectory.points)
          pt.accelerations.assign(joint_count, 0.0);

        for (size_t k = 1; k + 1 < traj.joint_trajectory.points.size(); ++k) {
          double dt = (arc_len[k+1] - arc_len[k-1]) / (2.0 * ee_speed);
          if (dt <= 1e-9) continue;
          for (size_t j = 0; j < joint_count; ++j) {
            double v_prev = traj.joint_trajectory.points[k-1].velocities.size() > j
                              ? traj.joint_trajectory.points[k-1].velocities[j] : 0.0;
            double v_next = traj.joint_trajectory.points[k+1].velocities.size() > j
                              ? traj.joint_trajectory.points[k+1].velocities[j] : 0.0;
            traj.joint_trajectory.points[k].accelerations[j] = (v_next - v_prev) / dt;
          }
        }
      }
    }

    auto result = move_group.execute(traj);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) return true;

    RCLCPP_WARN(logger, "Execution failed (error %d)", result.val);
    return false;
  }

  RCLCPP_WARN(logger,
    "Fraction %.2f below %.2f — singularity, skipping segment.",
    fraction, fraction_min);
  return false;
}
// ==============================================================
//  Build correction waypoints perpendicular to a surface normal 
// ==============================================================
static std::vector<geometry_msgs::msg::Pose> build_correction_patch(
  const geometry_msgs::msg::Point & target,
  const Eigen::Vector3d & surface_normal,
  double standoff,
  double stroke,
  double step)
{
  Eigen::Vector3d n = surface_normal.normalized();
  Eigen::Vector3d ref = (std::abs(n.dot(Eigen::Vector3d::UnitZ())) < 0.9)
                        ? Eigen::Vector3d::UnitZ()
                        : Eigen::Vector3d::UnitX();

  Eigen::Vector3d axis1 = n.cross(ref).normalized();
  Eigen::Vector3d axis2 = n.cross(axis1).normalized();

  Eigen::Vector3d tool_z = n;
  Eigen::Vector3d tool_y = tool_z.cross(ref); tool_y.normalize();
  Eigen::Vector3d tool_x = tool_y.cross(tool_z); tool_x.normalize();
  Eigen::Matrix3d R;
  R.col(0) = tool_x; R.col(1) = tool_y; R.col(2) = tool_z;
  Eigen::Quaterniond q(R); q.normalize();

  Eigen::Vector3d center(target.x, target.y, target.z);
  Eigen::Vector3d nozzle_center = center - standoff * n;

  std::vector<geometry_msgs::msg::Pose> waypoints;
  bool forward = true;

  for (double s2 = -stroke; s2 <= stroke + 1e-6; s2 += step) {
    double a1_start = forward ? -stroke :  stroke;
    double a1_end   = forward ?  stroke : -stroke;
    double da       = forward ?   step  :   -step;

    for (double s1 = a1_start;
         forward ? (s1 <= a1_end + 1e-6) : (s1 >= a1_end - 1e-6);
         s1 += da)
    {
      Eigen::Vector3d pos = nozzle_center + s1 * axis1 + s2 * axis2;

      geometry_msgs::msg::Pose p;
      p.position.x = pos.x();
      p.position.y = pos.y();
      p.position.z = pos.z();
      p.orientation.x = q.x();
      p.orientation.y = q.y();
      p.orientation.z = q.z();
      p.orientation.w = q.w();
      waypoints.push_back(p);
    }
    forward = !forward;
  }
  return waypoints;
}

// =============================================================
//  Execute a nav_msgs::msg::Path directly via Cartesian motion
// =============================================================

static bool execute_rl_path(
  moveit::planning_interface::MoveGroupInterface & move_group,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr spray_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub,
  const std::vector<geometry_msgs::msg::PoseStamped> & path_poses,
  const std::atomic<bool> & spray_enabled,
  double jump_threshold,
  double fraction_min,
  double ee_speed,
  rclcpp::Logger logger)
{
  if (path_poses.empty()) {
    RCLCPP_INFO(logger, "execute_rl_path: empty path — nothing to do.");
    return true;
  }

  // ── Approach the first waypoint with spray OFF ─────────────────────────
  set_spray(spray_pub, logger, false, spray_enabled);

  auto current_pose = move_group.getCurrentPose().pose;

  std::vector<geometry_msgs::msg::Pose> approach_seg = {
    current_pose, path_poses.front().pose
  };
  moveit_msgs::msg::RobotTrajectory approach_traj;
  double approach_frac = move_group.computeCartesianPath(
    approach_seg, 0.01, approach_traj, true);

  moveit::core::MoveItErrorCode approach_result;
  if (approach_frac >= fraction_min) {
    approach_result = move_group.execute(approach_traj);
  } else {
    move_group.setPoseTarget(path_poses.front().pose);
    approach_result = move_group.move();
  }

  if (approach_result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_WARN(logger,
      "execute_rl_path: could not reach first waypoint — aborting path.");
    return false;
  }

  // ── Extract plain Pose vector ──────────────────────────────────────────
  std::vector<geometry_msgs::msg::Pose> poses;
  poses.reserve(path_poses.size());
  for (const auto & ps : path_poses)
    poses.push_back(ps.pose);

  // ── Turn spray ON (if permitted) and execute ───────────────────────────
  set_spray(spray_pub, logger, true, spray_enabled);

  bool ok = execute_cartesian_segment(
    move_group, poses,
    jump_threshold, fraction_min, ee_speed,
    logger, singularity_pub, manipulability_pub);

  // ── Spray OFF when done ────────────────────────────────────────────────
  // Use force-OFF so it always fires regardless of gate state.
  set_spray_force(spray_pub, logger, false);

  if (ok)
    RCLCPP_INFO(logger,
      "execute_rl_path: ✓ %zu waypoints executed.", path_poses.size());
  else
    RCLCPP_WARN(logger,
      "execute_rl_path: ✗ path execution failed (singularity / reachability).");

  return ok;
}

// ===========================================================
//  RL processes the queue of legacy Point-based corrections
// ===========================================================

static int drain_rl_corrections(
  moveit::planning_interface::MoveGroupInterface & move_group,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr spray_pub,
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr singularity_pub,
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manipulability_pub,
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
  const std::vector<SurfaceWaypoint> & surface_waypoints,   // ← for normal lookup
  geometry_msgs::msg::Point * last_target = nullptr)
{
  const double CORRECTION_STROKE = 0.06;
  const double CORRECTION_STEP   = 0.01;
  // DEFAULT_NORMAL removed — we look up the nearest waypoint normal instead.

  int correction_count = 0;

  while (correction_count < max_corrections) {
    geometry_msgs::msg::Point target;
    {
      std::lock_guard<std::mutex> lock(rl_queue_mutex);
      if (rl_target_queue.empty()) break;
      target = rl_target_queue.front();
      rl_target_queue.pop();
    }

    correction_count++;
    if (last_target) *last_target = target;

    float standoff = rl_standoff.load();
    float flow     = rl_flow.load();

    RCLCPP_INFO(logger,
      "Adaptive correction %d/%d — target x=%.3f y=%.3f z=%.3f "
      "standoff=%.3f m flow=%.3f",
      correction_count, max_corrections,
      target.x, target.y, target.z, standoff, flow);

    // Look up the surface normal from the nearest waypoint to the RL target.
    // This ensures the correction patch is perpendicular to the actual curved
    // panel surface rather than always assuming a flat +X wall.
    Eigen::Vector3d surface_normal(1.0, 0.0, 0.0);  // fallback
    {
      double best_dist2 = std::numeric_limits<double>::infinity();
      for (const auto & wp : surface_waypoints) {
        double dx = wp.x - target.x;
        double dy = wp.y - target.y;
        double dz = wp.z - target.z;
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_dist2) {
          best_dist2  = d2;
          surface_normal = Eigen::Vector3d(wp.nx, wp.ny, wp.nz);
        }
      }
    }

    auto correction_waypoints = build_correction_patch(
      target, surface_normal, standoff,
      CORRECTION_STROKE, CORRECTION_STEP);

    set_spray(spray_pub, logger, false, spray_enabled);

    auto current_pose = move_group.getCurrentPose().pose;
    std::vector<geometry_msgs::msg::Pose> approach_seg = {
      current_pose, correction_waypoints.front()
    };
    moveit_msgs::msg::RobotTrajectory approach_traj;
    double approach_frac = move_group.computeCartesianPath(
      approach_seg, 0.01, approach_traj, true);

    moveit::core::MoveItErrorCode approach_result;
    if (approach_frac >= fraction_min) {
      approach_result = move_group.execute(approach_traj);
    } else {
      move_group.setPoseTarget(correction_waypoints.front());
      approach_result = move_group.move();
    }

    if (approach_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(logger,
        "Could not reach adaptive correction approach point — skipping.");
      set_spray(spray_pub, logger, true, spray_enabled);
      continue;
    }

    set_spray(spray_pub, logger, true, spray_enabled);

    bool ok = execute_cartesian_segment(
      move_group, correction_waypoints, jump_threshold, fraction_min,
      ee_speed, logger, singularity_pub, manipulability_pub);

    if (ok)
      RCLCPP_INFO(logger, "  ✓ Adaptive correction stroke executed.");
    else
      RCLCPP_WARN(logger,
        "  ✗ Adaptive correction stroke failed (singularity / reachability).");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  return correction_count;
}

// ======
//  Main
// ======

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "square_xz_node",
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
  move_group.setPlanningTime(5.0);
  move_group.setNumPlanningAttempts(5);
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

  std::string file_path =
    "/home/user/car_spraying_ws/src/square_trajectory/peya.csv"; // Trajectory excel path
  std::ifstream file(file_path);

  if (!file.is_open()) {
    RCLCPP_ERROR(node->get_logger(),
      "Cannot open CSV file: %s", file_path.c_str());
    executor.cancel(); spinner_thread.join(); rclcpp::shutdown(); return 1;
  }

  std::vector<SurfaceWaypoint> surface_waypoints;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ','))
      row.push_back(std::stod(value));

    if (row.size() < 10) {
      RCLCPP_WARN(node->get_logger(),
        "Skipping short CSV row (%zu cols < 10)", row.size());
      continue;
    }

    SurfaceWaypoint wp;
    wp.x  = row[0];
    wp.y  = row[1];
    wp.z  = row[2];
    wp.nx = row[7];
    wp.ny = row[8];
    wp.nz = row[9];

    double mag = std::sqrt(wp.nx*wp.nx + wp.ny*wp.ny + wp.nz*wp.nz);
    if (mag < 1e-6) {
      RCLCPP_WARN(node->get_logger(),
        "Near-zero normal at waypoint %zu — defaulting to +X",
        surface_waypoints.size());
      wp.nx = 1.0; wp.ny = 0.0; wp.nz = 0.0;
    } else {
      wp.nx /= mag; wp.ny /= mag; wp.nz /= mag;
    }

    surface_waypoints.push_back(wp);
  }
  file.close();

  if (surface_waypoints.empty()) {
    RCLCPP_ERROR(node->get_logger(), "No valid waypoints loaded from CSV.");
    executor.cancel(); spinner_thread.join(); rclcpp::shutdown(); return 1;
  }

  RCLCPP_INFO(node->get_logger(),
    "Loaded %zu surface waypoints from CSV.", surface_waypoints.size());

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

  const int  MAX_CORRECTION_PASSES = 50;
  const int  RL_DETECT_TIMEOUT_MS  = 5000;   // 5 s to detect the RL agent
  const int  RL_IDLE_TIMEOUT_MS    = 8000;   // exit after 8 s of no work

  int correction_count   = 0;
  int rl_path_executions = 0;

  // Probe: wait for RL agent heartbeat on /spray/enable
  RCLCPP_INFO(node->get_logger(),
    "PASS 2: probing for RL agent (timeout %d ms)...", RL_DETECT_TIMEOUT_MS);
  {
    int waited = 0;
    while (!rl_agent_detected.load() && waited < RL_DETECT_TIMEOUT_MS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      waited += 250;
    }
  }

  if (!rl_agent_detected.load()) {
    // No RL agent: exit normally without PASS 2
    RCLCPP_INFO(node->get_logger(),
      "No RL agent detected within %d ms -- PASS 2 skipped, exiting normally.",
      RL_DETECT_TIMEOUT_MS);

  } else {
    // RL agent alive: run correction loop
    RCLCPP_INFO(node->get_logger(),
      "=== PASS 2: RL correction loop (max %d iterations) ===",
      MAX_CORRECTION_PASSES);

    auto idle_start = std::chrono::steady_clock::now();

    while (correction_count < MAX_CORRECTION_PASSES)
    {
      bool did_work = false;

      // Priority 1: /spray/rl_path
      if (rl_path_available.load())
      {
        std::vector<geometry_msgs::msg::PoseStamped> path_snapshot;
        {
          std::lock_guard<std::mutex> lock(rl_path_mutex);
          path_snapshot = rl_path_poses;
          rl_path_available.store(false);
        }

        if (!path_snapshot.empty()) {
          RCLCPP_INFO(node->get_logger(),
            "PASS 2 [%d]: executing /spray/rl_path (%zu waypoints).",
            correction_count, path_snapshot.size());

          bool ok = execute_rl_path(
            move_group,
            spray_pub, singularity_pub, manipulability_pub,
            path_snapshot, spray_enabled,
            jump_threshold, fraction_min, desired_ee_speed,
            node->get_logger());

          rl_path_executions++;
          correction_count++;
          did_work = true;
          idle_start = std::chrono::steady_clock::now();   // reset idle timer

          if (ok)
            RCLCPP_INFO(node->get_logger(),
              "  rl_path execution %d complete.", rl_path_executions);
          else
            RCLCPP_WARN(node->get_logger(),
              "  rl_path execution %d failed.", rl_path_executions);

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
            rl_target_queue, rl_queue_mutex,
            rl_standoff, rl_flow, spray_enabled,
            node->get_logger(),
            jump_threshold, fraction_min, desired_ee_speed,
            MAX_INLINE_CORRECTIONS, surface_waypoints);

          correction_count += drained;
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

        if (idle_ms >= RL_IDLE_TIMEOUT_MS) {
          RCLCPP_INFO(node->get_logger(),
            "PASS 2: RL agent idle for %lld ms -- assuming done, exiting.",
            (long long)idle_ms);
          break;
        }

        RCLCPP_INFO(node->get_logger(),
          "PASS 2: waiting for RL agent... (%lld / %d ms idle)",
          (long long)idle_ms, RL_IDLE_TIMEOUT_MS);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }

    if (correction_count >= MAX_CORRECTION_PASSES)
      RCLCPP_WARN(node->get_logger(),
        "Reached MAX_CORRECTION_PASSES (%d). Stopping.", MAX_CORRECTION_PASSES);
    else
      RCLCPP_INFO(node->get_logger(),
        "PASS 2 complete -- %d iterations (%d rl_path).",
        correction_count, rl_path_executions);

  }  // end if (rl_agent_detected)

  // ================
  // After Execution
  // ================

  set_spray_force(spray_pub, node->get_logger(), false);

  RCLCPP_INFO(node->get_logger(),
    "=== COMPLETE ===  "
    "CSV segments executed: %d  skipped: %d  |  "
    "RL path executions: %d  total corrections: %d",
    executed_segments, skipped_segments,
    rl_path_executions, correction_count);

  tracking_active.store(false);
  if (tracking_thread.joinable()) tracking_thread.join();

  executor.cancel();
  spinner_thread.join();
  rclcpp::shutdown();
  return 0;
}