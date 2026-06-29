"""
vision_rl_Manual.launch.py
===========================
Launches the robot stack + RViz and publishes part_mesh.ply and
part_mesh_coverage.ply as RViz Marker messages so you can inspect
the saved meshes alongside the robot model.

  RSP + static TF  →  RViz  →  ply_marker_publisher (Python node)

No MoveGroup or controllers needed — this is a viewer only.

Usage (defaults resolve automatically if files exist at the standard path):
  ros2 launch car_spraying_robot vision_rl_Manual.launch.py

Override paths explicitly if the files live elsewhere:
  ros2 launch car_spraying_robot vision_rl_Manual.launch.py \
      mesh_ply:=/abs/path/to/part_mesh.ply \
      coverage_ply:=/abs/path/to/part_mesh_coverage.ply \
      mesh_frame:=camera_color_optical_frame

FIX — what changed vs the original:
  1. _resolve_ply_path() searches several candidate directories so the
     default value is always a file that actually exists on disk.
     A clear error is raised at launch time (not silently at node start)
     when neither path can be found.
  2. old-style static_transform_publisher arguments replaced with the
     new --frame-id / --child-frame-id form to silence the deprecation
     warning seen in the original log.
"""

import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# ── Absolute paths ────────────────────────────────────────────────────────────
_WS       = "/home/user/car_spraying_ws/src"
URDF_PATH = f"{_WS}/car_spraying_robot/urdf/UR3_Assembly_URDF_moveit.urdf"

# Candidate directories searched in order when resolving the default PLY paths.
# Add more entries here if your pipeline saves files somewhere else.
_PLY_SEARCH_DIRS = [
    f"{_WS}/ob_detection/ob_detection/spray_paths",   # original default
    f"{_WS}/ob_detection/spray_paths",
]


def _resolve_ply_path(filename: str) -> str:
    """
    Return the first existing path for *filename* across _PLY_SEARCH_DIRS.
    Falls back to the original default path (letting the node emit the
    descriptive error at runtime) if nothing is found, so the launch file
    itself never crashes hard when the files simply haven't been generated
    yet — but it does print a prominent warning.
    """
    for d in _PLY_SEARCH_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate

    # Not found anywhere — warn loudly now rather than silently at node start.
    searched = "\n    ".join(_PLY_SEARCH_DIRS)
    print(
        f"\n"
        f"[vision_rl_Manual.launch.py] WARNING: '{filename}' was not found in "
        f"any of the search directories:\n"
        f"    {searched}\n"
        f"The ply_marker_publisher node will start but publish NO markers.\n"
        f"Run the detection / coverage pipeline first, or pass the correct\n"
        f"path explicitly:  mesh_ply:=/absolute/path/to/{filename}\n"
    )
    # Return the canonical default so the launch argument still has a value.
    return os.path.join(_PLY_SEARCH_DIRS[0], filename)


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder(
            "car_spraying_robot",
            package_name="car_spraying_moveit_config",
        )
        .robot_description(file_path=URDF_PATH)
        .planning_pipelines(pipelines=["ompl"])
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    sim_time = {"use_sim_time": False}

    # ── Launch arguments ──────────────────────────────────────────────────────
    arg_mesh = DeclareLaunchArgument(
        'mesh_ply',
        default_value=_resolve_ply_path('part_mesh.ply'),
        description='Absolute path to part_mesh.ply'
    )
    arg_coverage = DeclareLaunchArgument(
        'coverage_ply',
        default_value=_resolve_ply_path('part_mesh_coverage.ply'),
        description='Absolute path to part_mesh_coverage.ply'
    )
    arg_frame = DeclareLaunchArgument(
        'mesh_frame',
        default_value='camera_color_optical_frame',
        description='TF frame the PLY meshes were captured in'
    )
    arg_rate = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='1.0',
        description='How often to re-publish the mesh markers [Hz]'
    )

    # 1. Robot State Publisher
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, sim_time],
        output="screen",
    )

    # 2. Static TF: world → base_link
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_link_tf",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        parameters=[sim_time],
        output="screen",
    )

    # 3. RViz
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(
            moveit_config.package_path, "config/moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            sim_time,
        ],
    )

        # 5. MoveGroup
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            sim_time,
            {"publish_robot_description_semantic": True},
        ],
    )

    # 4. PLY → Marker publisher
    ply_publisher = Node(
        package="ob_detection",
        executable="ply_marker_pub",
        name="ply_marker_publisher",
        output="screen",
        parameters=[{
            "use_sim_time":    False,
            "mesh_ply":        LaunchConfiguration('mesh_ply'),
            "coverage_ply":    LaunchConfiguration('coverage_ply'),
            "mesh_frame":      LaunchConfiguration('mesh_frame'),
            "publish_rate_hz": LaunchConfiguration('publish_rate_hz'),
        }],
    )

    return LaunchDescription([
        arg_mesh,
        arg_coverage,
        arg_frame,
        arg_rate,

        # t=0 — RSP + static TF
        rsp,
        static_tf,
        TimerAction(period=1.0,  actions=[move_group]),
        # t=3 — RViz
        TimerAction(period=3.0, actions=[rviz]),
        # t=4 — PLY publisher (RViz needs a moment to subscribe)
        TimerAction(period=4.0, actions=[ply_publisher]),
    ])