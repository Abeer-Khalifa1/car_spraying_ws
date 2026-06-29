"""
full_system.launch.py
=====================
ONE launch file for the entire car-spraying pipeline:

  Step 0  filter_and_forward   — validates peya.csv, strips out-of-reach
                                  points (≤ 10 %), writes peya_validated.csv.
                                  If > 10 % are unreachable → launch aborts here.

  Step 1  Gazebo               — physics sim
  Step 2  ROS-GZ Bridge        — gz ↔ ROS 2 topics
  Step 3  RSP + static TF      — /robot_description, world→base_link
  Step 4  MoveGroup            — motion planning
  Step 5  Controllers          — joint_state_broadcaster, joint_trajectory_controller
  Step 6  RViz2                — visualisation (MoveIt config)
  Step 7  PLY marker publisher — part_mesh.ply + coverage mesh in RViz
  Step 8  square_xz_node       — Cartesian trajectory executor (reads validated CSV)
  Step 9  spray_sim_node       — paint cone simulation
  Step 10 coverage_map_node    — 3-D surface coverage map
  Step 11 coverage_quality_node— quality metrics overlay
  Step 12 rl_agent_node        — PPO/TD3 defect-correction agent (PASS 2)

Usage
-----
    ros2 launch car_spraying_robot full_system.launch.py

    # Custom CSV / threshold:
    ros2 launch car_spraying_robot full_system.launch.py \
        csv_path:=/home/user/car_spraying_ws/src/square_trajectory/peya.csv \
        threshold:=0.10

    # Skip RL agent:
    ros2 launch car_spraying_robot full_system.launch.py enable_rl:=false

    # Skip PLY viewer:
    ros2 launch car_spraying_robot full_system.launch.py enable_ply:=false
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    RegisterEventHandler,
    EmitEvent,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# ── Workspace-root paths ──────────────────────────────────────────────────────
_WS = "/home/user/car_spraying_ws/src"

SDF_PATH      = f"{_WS}/car_spraying_robot/models/Spraying_Arm_moveit.sdf"
URDF_PATH     = f"{_WS}/car_spraying_robot/urdf/UR3_Assembly_URDF_moveit.urdf"
BRIDGE_CONFIG = f"{_WS}/car_spraying_robot/config/bridge_moveit.yaml"

_DEFAULT_CSV     = f"{_WS}/square_trajectory/peya.csv"
_DEFAULT_VAL_CSV = f"{_WS}/square_trajectory/peya_validated.csv"

_PLY_SEARCH_DIRS = [
    f"{_WS}/ob_detection/ob_detection/spray_paths",
    f"{_WS}/ob_detection/spray_paths",
]


def _find_ply(filename: str) -> str:
    for d in _PLY_SEARCH_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    print(
        f"[full_system.launch] WARNING: '{filename}' not found — "
        f"ply_marker_publisher will start but publish no markers."
    )
    return os.path.join(_PLY_SEARCH_DIRS[0], filename)


# ─────────────────────────────────────────────────────────────────────────────

def generate_launch_description() -> LaunchDescription:

    # ── MoveIt config ─────────────────────────────────────────────────────────
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

    sim_time = {"use_sim_time": True}

    # ═══════════════════════════════════════════════════════════════════════════
    # LAUNCH ARGUMENTS
    # ═══════════════════════════════════════════════════════════════════════════

    arg_csv = DeclareLaunchArgument(
        'csv_path', default_value=_DEFAULT_CSV,
        description='Input trajectory CSV (peya.csv)')

    arg_val_csv = DeclareLaunchArgument(
        'validated_csv_path', default_value=_DEFAULT_VAL_CSV,
        description='Filtered output CSV written by filter_and_forward')

    arg_threshold = DeclareLaunchArgument(
        'threshold', default_value='0.10',
        description='Max fraction of unsafe waypoints before aborting (0.0–1.0)')

    arg_gz_world = DeclareLaunchArgument(
        'gz_world_name', default_value='world_demo',
        description='Gazebo world name — must match <world name=...> in SDF')

    arg_cone_len = DeclareLaunchArgument(
        'cone_length', default_value='0.20',
        description='Spray cone length [m]')

    arg_half_angle = DeclareLaunchArgument(
        'cone_half_angle_deg', default_value='15.0',
        description='Spray cone half-angle [degrees]')

    arg_sigma = DeclareLaunchArgument(
        'sigma', default_value='0.03',
        description='Gaussian sigma for paint density [m]')

    arg_spray_active = DeclareLaunchArgument(
        'spray_active', default_value='true',
        description='Start with spray on (true/false)')

    arg_enable_rl = DeclareLaunchArgument(
        'enable_rl', default_value='true',
        description='Launch the RL agent node for PASS 2 defect correction')

    arg_enable_ply = DeclareLaunchArgument(
        'enable_ply', default_value='true',
        description='Launch the PLY mesh marker publisher for RViz visualisation')

    arg_mesh_ply = DeclareLaunchArgument(
        'mesh_ply', default_value=_find_ply('part_mesh.ply'),
        description='Absolute path to part_mesh.ply')

    arg_coverage_ply = DeclareLaunchArgument(
        'coverage_ply', default_value=_find_ply('part_mesh_coverage.ply'),
        description='Absolute path to part_mesh_coverage.ply')

    arg_mesh_frame = DeclareLaunchArgument(
        'mesh_frame', default_value='camera_color_optical_frame',
        description='TF frame the PLY meshes were captured in')

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 0 — filter_and_forward  (BLOCKING GATE)
    # Runs first, exits immediately.
    # Exit code 0 → validation passed, rest of launch continues normally.
    # Exit code 1 → too many unsafe points, handler emits Shutdown.
    # ═══════════════════════════════════════════════════════════════════════════

    filter_node = Node(
        package='trajectory_validator',
        executable='filter_and_forward',
        name='filter_and_forward',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'csv_path':    LaunchConfiguration('csv_path'),
            'output_path': LaunchConfiguration('validated_csv_path'),
            'threshold':   LaunchConfiguration('threshold'),
        }],
    )

    # ── KEY FIX: only emit Shutdown when returncode != 0 ─────────────────────
    # OnProcessExit fires for every exit (including clean exit 0).
    # We guard with a lambda that checks event.returncode before acting.
    def _on_filter_exit(event, context):
        if event.returncode != 0:
            # Trajectory was rejected — kill the whole launch
            print(
                '\n'
                '╔══════════════════════════════════════════════════════════╗\n'
                '║  filter_and_forward: trajectory REJECTED                ║\n'
                '║  Too many waypoints outside the workspace.              ║\n'
                '║  Fix peya.csv or raise the threshold.                   ║\n'
                '║  Aborting — Gazebo will NOT be started.                 ║\n'
                '╚══════════════════════════════════════════════════════════╝'
            )
            return [EmitEvent(event=Shutdown(
                reason='Trajectory validation failed — path out of reach'))]
        # returncode == 0: validation passed, do nothing — launch continues
        return []

    shutdown_on_filter_fail = RegisterEventHandler(
        OnProcessExit(
            target_action=filter_node,
            on_exit=_on_filter_exit,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Gazebo
    # ═══════════════════════════════════════════════════════════════════════════

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", SDF_PATH],
        output="screen",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2 — ROS-GZ Bridge   (t=3 s)
    # ═══════════════════════════════════════════════════════════════════════════

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[{"config_file": BRIDGE_CONFIG}, sim_time],
        output="screen",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Robot State Publisher + static TF   (t=4 s)
    # ═══════════════════════════════════════════════════════════════════════════

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, sim_time],
        output="screen",
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_link_tf",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        parameters=[sim_time],
        output="screen",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — MoveGroup   (t=5 s)
    # ═══════════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Controllers
    # ═══════════════════════════════════════════════════════════════════════════

    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        parameters=[sim_time],
        output="screen",
    )

    jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
        parameters=[sim_time],
        output="screen",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6 — RViz2
    # ═══════════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 7 — PLY mesh publisher   (optional)
    # ═══════════════════════════════════════════════════════════════════════════

    ply_publisher = Node(
        package="ob_detection",
        executable="ply_marker_pub",
        name="ply_marker_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration('enable_ply')),
        parameters=[{
            "use_sim_time":    True,
            "mesh_ply":        LaunchConfiguration('mesh_ply'),
            "coverage_ply":    LaunchConfiguration('coverage_ply'),
            "mesh_frame":      LaunchConfiguration('mesh_frame'),
            "publish_rate_hz": 1.0,
        }],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 8 — square_xz_node  (reads validated CSV via csv_path param)
    # ═══════════════════════════════════════════════════════════════════════════

    trajectory_node = Node(
        package="square_trajectory",
        executable="square_xz",
        name="square_xz_node",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            sim_time,
            {"csv_path": LaunchConfiguration('validated_csv_path')},
        ],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 9 — Spray sim
    # ═══════════════════════════════════════════════════════════════════════════

    spray_sim = Node(
        package='car_spraying_spray_sim',
        executable='spray_sim_node',
        name='spray_sim_node',
        output='screen',
        parameters=[{
            'use_sim_time':        True,
            'end_effector_frame':  'link_6',
            'world_frame':         'world',
            'gz_world_name':       LaunchConfiguration('gz_world_name'),
            'cone_length':         LaunchConfiguration('cone_length'),
            'cone_half_angle_deg': LaunchConfiguration('cone_half_angle_deg'),
            'sigma':               LaunchConfiguration('sigma'),
            'spray_active':        LaunchConfiguration('spray_active'),
            'num_sample_rings':    8,
            'num_angular_pts':     36,
            'paint_point_spacing': 0.0015,
            'max_paint_points':    300000,
            'max_gz_spheres':      8000,
            'gz_sphere_radius':    0.010,
            'publish_rate_hz':     10.0,
            'gz_spawn_every_n':    5,
        }],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Coverage map
    # ═══════════════════════════════════════════════════════════════════════════

    coverage_map_node = Node(
        package='car_spraying_spray_sim',
        executable='coverage_map_node',
        name='coverage_map_generator',
        output='screen',
        parameters=[{
            'use_sim_time':   True,
            'resolution':     0.02,
            'trajectory_csv': LaunchConfiguration('validated_csv_path'),
        }],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 11 — Coverage quality
    # ═══════════════════════════════════════════════════════════════════════════

    coverage_quality_node = Node(
        package='car_spraying_spray_sim',
        executable='coverage_quality_node',
        name='coverage_quality_node',
        output='screen',
        parameters=[{
            'use_sim_time':   True,
            'trajectory_csv': LaunchConfiguration('validated_csv_path'),
        }],
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 12 — RL agent   (optional)
    # ═══════════════════════════════════════════════════════════════════════════

    # rl_agent_node = Node(
    #     package='square_trajectory',
    #     executable='rl_agent_node.py',
    #     name='rl_agent_node',
    #     output='screen',
    #     condition=IfCondition(LaunchConfiguration('enable_rl')),
    #     parameters=[{'use_sim_time': True}],
    # )

    # ═══════════════════════════════════════════════════════════════════════════
    # ASSEMBLE
    # ═══════════════════════════════════════════════════════════════════════════

    return LaunchDescription([

        # Arguments
        arg_csv, arg_val_csv, arg_threshold,
        arg_gz_world, arg_cone_len, arg_half_angle, arg_sigma, arg_spray_active,
        arg_enable_rl, arg_enable_ply, arg_mesh_ply, arg_coverage_ply, arg_mesh_frame,

        # STEP 0 — validate CSV first (gate)
        filter_node,
        shutdown_on_filter_fail,

        # STEP 1 — Gazebo (small delay so filter prints its report cleanly)
        TimerAction(period=1.0,  actions=[gazebo]),

        # STEP 2 — Bridge
        TimerAction(period=4.0,  actions=[bridge]),

        # STEP 3 — RSP + TF
        TimerAction(period=5.0,  actions=[rsp, static_tf]),

        # STEP 4 — MoveGroup
        TimerAction(period=6.0,  actions=[move_group]),

        # STEP 5 — Controllers
        TimerAction(period=7.0,  actions=[jsb_spawner]),
        TimerAction(period=8.0,  actions=[jtc_spawner]),

        # STEP 6 — RViz
        TimerAction(period=9.0,  actions=[rviz]),

        # STEP 7 — PLY mesh viewer
        TimerAction(period=10.0, actions=[ply_publisher]),

        # STEP 8 — Trajectory executor
        TimerAction(period=16.0, actions=[trajectory_node]),

        # STEPS 9–12 — Spray + Coverage + RL
        TimerAction(period=18.0, actions=[
            spray_sim,
            coverage_map_node,
            coverage_quality_node,
            # rl_agent_node,
        ]),
    ])