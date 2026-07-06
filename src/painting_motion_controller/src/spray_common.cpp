#include "spray_common.hpp"

#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <builtin_interfaces/msg/duration.hpp>

#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cmath>

using namespace std::chrono_literals;

// =========================================================
//  Build a perpendicular spray pose from a surface waypoint.
// =========================================================
geometry_msgs::msg::Pose pose_from_surface(
  const SurfaceWaypoint & wp,
  double standoff)
{
  geometry_msgs::msg::Pose pose;
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

  return pose;
}

// =========================================================
//  Load surface waypoints (position + normal) from CSV.
//  Extracted verbatim from main()'s CSV-loading block so both
//  executables can load the same file identically.
// =========================================================
std::vector<SurfaceWaypoint> load_surface_waypoints(
  const std::string & csv_path,
  rclcpp::Logger logger)
{
  std::vector<SurfaceWaypoint> surface_waypoints;

  std::ifstream file(csv_path);
  if (!file.is_open()) {
    RCLCPP_ERROR(logger, "Cannot open CSV file: %s", csv_path.c_str());
    return surface_waypoints;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ','))
      row.push_back(std::stod(value));

    if (row.size() < 10) {
      RCLCPP_WARN(logger, "Skipping short CSV row (%zu cols < 10)", row.size());
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
      RCLCPP_WARN(logger,
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
    RCLCPP_ERROR(logger, "No valid waypoints loaded from CSV: %s", csv_path.c_str());
  } else {
    RCLCPP_INFO(logger, "Loaded %zu surface waypoints from CSV.", surface_waypoints.size());
  }

  return surface_waypoints;
}

// ================================================================================
// Force the spray state OFF/ON regardless of the spray_enabled gate (safety path)
// ================================================================================
void set_spray_force(
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
void set_spray(
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub,
  rclcpp::Logger logger,
  bool active,
  const std::atomic<bool> & spray_enabled)
{
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
bool trajectory_is_singular(
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
// Planning, execution of a trajectory segment, with singularity retries
// =====================================================================
bool execute_cartesian_segment(
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

  move_group.setStartStateToCurrentState();

  moveit_msgs::msg::RobotTrajectory traj;
  double fraction = move_group.computeCartesianPath(
    segment, 0.01, jump_threshold, traj, true);

  RCLCPP_INFO(logger, "Cartesian fraction = %.2f", fraction);

  bool path_ok = true;
  if (fraction >= fraction_min) {
    if (trajectory_is_singular(move_group, traj, logger,
                               singularity_pub, manipulability_pub))
    {
      RCLCPP_WARN(logger, "Trajectory appears singular — attempting avoidance.");
      path_ok = false;
    }
  } else {
    RCLCPP_WARN(logger,
      "Fraction %.2f below %.2f — path incomplete, attempting singularity/IK avoidance.",
      fraction, fraction_min);
    path_ok = false;
  }

  if (!path_ok) {
      std::vector<double> deltas = { -0.25, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.25 };
      bool found_alternative = false;
      moveit_msgs::msg::RobotTrajectory alt_traj;
      double alt_fraction = 0.0;
      const double retry_jump_threshold = std::max(jump_threshold, 0.10);
      const double eef_step = 0.01;

      for (double d : deltas) {
        move_group.setStartStateToCurrentState();

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
          perturbed, eef_step, retry_jump_threshold, try_traj, true);

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

      if (!found_alternative && !segment.empty()) {
        move_group.setStartStateToCurrentState();
        std::vector<geometry_msgs::msg::Pose> relaxed = segment;
        auto current_pose = move_group.getCurrentPose().pose;
        relaxed.front().orientation = current_pose.orientation;

        moveit_msgs::msg::RobotTrajectory try_traj;
        double try_fraction = move_group.computeCartesianPath(
          relaxed, eef_step, retry_jump_threshold, try_traj, true);

        RCLCPP_INFO(logger,
          "Tried relaxed first waypoint orientation — fraction=%.2f", try_fraction);

        if (try_fraction >= fraction_min) {
          if (!trajectory_is_singular(move_group, try_traj, logger,
                                      singularity_pub, manipulability_pub)) {
            found_alternative = true;
            alt_traj = try_traj;
            alt_fraction = try_fraction;
          }
        }
      }

      if (found_alternative) {
        RCLCPP_INFO(logger,
          "Found alternative non-singular path (fraction %.2f).", alt_fraction);
        traj = alt_traj;
      } else {
        RCLCPP_WARN(logger, "Trajectory rejected due to singularity / IK failure.");
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

// ==============================================================
//  Build correction waypoints perpendicular to a surface normal
// ==============================================================
std::vector<geometry_msgs::msg::Pose> build_correction_patch(
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
  rclcpp::Logger logger)
{
  if (path_poses.empty()) {
    RCLCPP_INFO(logger, "execute_rl_path: empty path — nothing to do.");
    return true;
  }

  // ── Approach the first waypoint with spray OFF ─────────────────────────
  set_spray(spray_pub, logger, false, spray_enabled);

  // ---------------------------------------------------------
  // 1. FREE-SPACE APPROACH MOVE (Do not use Cartesian for this)
  // ---------------------------------------------------------
  RCLCPP_INFO(logger, "Planning free-space approach to patch start...");

  move_group.clearPoseTargets();
  move_group.setStartStateToCurrentState();
  move_group.setPoseTarget(path_poses.front().pose);

  moveit::planning_interface::MoveGroupInterface::Plan approach_plan;
  auto plan_result = move_group.plan(approach_plan);

  if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_INFO(logger, "Approach plan found. Executing...");
    move_group.execute(approach_plan);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  } else {
    RCLCPP_WARN(logger, "Failed to plan free-space approach. Target is unreachable.");
    std_msgs::msg::Bool pf_msg; pf_msg.data = true;
    planning_failed_pub->publish(pf_msg);
    return false;
  }

  // ---------------------------------------------------------
  // 2. CARTESIAN GRID EXECUTION
  // ---------------------------------------------------------
  std::vector<geometry_msgs::msg::Pose> poses;
  poses.reserve(path_poses.size());
  for (const auto & ps : path_poses)
    poses.push_back(ps.pose);

  set_spray(spray_pub, logger, true, spray_enabled);

  bool ok = execute_cartesian_segment(
    move_group, poses,
    jump_threshold, fraction_min, ee_speed,
    logger, singularity_pub, manipulability_pub);

  set_spray_force(spray_pub, logger, false);

  std_msgs::msg::Bool pf_msg; pf_msg.data = !ok;
  planning_failed_pub->publish(pf_msg);

  if (ok)
    RCLCPP_INFO(logger,
      "execute_rl_path: ✓ %zu waypoints executed.", path_poses.size());
  else
    RCLCPP_WARN(logger,
      "execute_rl_path: ✗ path execution failed (singularity / reachability).");

  return ok;
}

// ===========================================================
//  Drain the queue of legacy Point-based corrections
// ===========================================================
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
  geometry_msgs::msg::Point * last_target)
{
  const double CORRECTION_STROKE = 0.06;
  const double CORRECTION_STEP   = 0.01;

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
      std_msgs::msg::Bool pf_msg; pf_msg.data = true;
      planning_failed_pub->publish(pf_msg);
      set_spray(spray_pub, logger, true, spray_enabled);
      continue;
    }

    set_spray(spray_pub, logger, true, spray_enabled);

    bool ok = execute_cartesian_segment(
      move_group, correction_waypoints, jump_threshold, fraction_min,
      ee_speed, logger, singularity_pub, manipulability_pub);

    std_msgs::msg::Bool pf_msg; pf_msg.data = !ok;
    planning_failed_pub->publish(pf_msg);

    if (ok)
      RCLCPP_INFO(logger, "  ✓ Adaptive correction stroke executed.");
    else
      RCLCPP_WARN(logger,
        "  ✗ Adaptive correction stroke failed (singularity / reachability).");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  return correction_count;
}
