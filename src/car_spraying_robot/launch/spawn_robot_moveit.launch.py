"""
simulation.launch.py
=====================
Single entry-point that starts the complete car-spraying simulation:

  Gazebo  →  ROS-GZ Bridge  →  RSP + static TF  →  MoveGroup
  →  Controllers  →  RViz  →  Trajectory node  →  Spray sim node  →  Coverage Map Node
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# ── Absolute paths (edit these if your workspace root changes) ───────────────
_WS = "/home/user/car_spraying_ws/src"
SDF_PATH      = f"{_WS}/car_spraying_robot/models/Spraying_Arm_moveit.sdf"
URDF_PATH     = f"{_WS}/car_spraying_robot/urdf/UR3_Assembly_URDF_moveit.urdf"
BRIDGE_CONFIG = f"{_WS}/car_spraying_robot/config/bridge_moveit.yaml"
CSV_PATH      = f"{_WS}/square_trajectory/peya.csv"


def generate_launch_description():

    # ── MoveIt config (built once, reused by multiple nodes) ─────────────────
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

    # ── Launch arguments ──────────────────────────────────────────────────────
    arg_gz_world = DeclareLaunchArgument(
        'gz_world_name', default_value='world_demo',
        description='Gazebo world name — must match <world name=...> in SDF'
    )
    arg_cone_len = DeclareLaunchArgument(
        'cone_length', default_value='0.20',
        description='Spray cone length [m]'
    )
    arg_half_angle = DeclareLaunchArgument(
        'cone_half_angle_deg', default_value='15.0',
        description='Spray cone half-angle [degrees]'
    )
    arg_sigma = DeclareLaunchArgument(
        'sigma', default_value='0.03',
        description='Gaussian sigma for paint density [m]'
    )
    arg_spray_active = DeclareLaunchArgument(
        'spray_active', default_value='true',
        description='Start with spray on (true/false)'
    )

    # ── Node definitions ──────────────────────────────────────────────────────

    # 1. Gazebo
    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", SDF_PATH],
        output="screen",
    )

    # 2. ROS-GZ Bridge
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[{"config_file": BRIDGE_CONFIG}, sim_time],
        output="screen",
    )

    # 3. Robot State Publisher
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, sim_time],
        output="screen",
    )

    # 4. Static TF: world → base_link
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_base_link_tf",
        arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        parameters=[sim_time],
        output="screen",
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

    # 6a. joint_state_broadcaster
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        parameters=[sim_time],
        output="screen",
    )

    # 6b. joint_trajectory_controller
    jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
        parameters=[sim_time],
        output="screen",
    )

    # 7. RViz
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

    # 8. Trajectory node
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
        ],
    )

    # 9. Spray sim node
    spray_sim = Node(
        package='car_spraying_spray_sim',
        executable='spray_sim_node', # Ensure extension matches workspace script names
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

    # 10. Coverage Map Generator Node
    coverage_map_node = Node(
        package='car_spraying_spray_sim',
        executable='coverage_map_node',
        name='coverage_map_generator',
        output='screen',
        parameters=[{
            'use_sim_time':   True,
            'resolution':     0.02,
            'trajectory_csv': CSV_PATH,
        }]
    )

    coverage_quality_node = Node(
        package='car_spraying_spray_sim',
        executable='coverage_quality_node',
        name='coverage_quality_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'trajectory_csv': CSV_PATH,

        }]
    )


    # rl_agent_node = Node(
    #     package='square_trajectory',
    #     executable='rl_agent_node',
    #     name='rl_agent_node',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True,
    #     }]
    # )

    # ── Sequenced launch (staggered TimerActions) ─────────────────────────────
    return LaunchDescription([
        # Arguments
        arg_gz_world,
        arg_cone_len,
        arg_half_angle,
        arg_sigma,
        arg_spray_active,

        # t=0  — Gazebo first
        gazebo,

        # t=3  — Bridge (Gazebo needs ~3 s to open transport sockets)
        TimerAction(period=3.0,  actions=[bridge]),

        # t=4  — RSP + static TF (need clock from bridge)
        TimerAction(period=4.0,  actions=[rsp, static_tf]),

        # t=5  — MoveGroup (needs /robot_description from RSP)
        TimerAction(period=5.0,  actions=[move_group]),

        # t=6  — joint_state_broadcaster
        TimerAction(period=6.0,  actions=[jsb_spawner]),

        # t=7  — joint_trajectory_controller (after JSB)
        TimerAction(period=7.0,  actions=[jtc_spawner]),

        # t=8  — RViz
        TimerAction(period=8.0,  actions=[rviz]),

        # t=15 — Trajectory (controllers fully ready + MoveGroup ready)
        TimerAction(period=15.0, actions=[trajectory_node]),

        # t=17 — Spray sim & Coverage Map (Launches together when TF world→link_6 is live)
        TimerAction(period=17.0, actions=[spray_sim, coverage_map_node, coverage_quality_node]),
    ])