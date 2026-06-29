"""
validate.launch.py
==================
Launch the trajectory_validator node for car_spraying_robot.

Usage
-----
    # Minimal — loads CSV immediately
    ros2 launch trajectory_validator validate.launch.py \
        csv_path:=/abs/path/to/trajectory.csv

    # With RViz2
    ros2 launch trajectory_validator validate.launch.py \
        csv_path:=/abs/path/to/trajectory.csv \
        rviz:=true

    # Custom frame / rate
    ros2 launch trajectory_validator validate.launch.py \
        csv_path:=/abs/path/to/trajectory.csv \
        frame_id:=world  rate_hz:=2.0  clamp:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:

    # ── arguments ─────────────────────────────────────────────────────────
    csv_arg = DeclareLaunchArgument(
        'csv_path', default_value='',
        description='Absolute path to trajectory CSV file')

    frame_arg = DeclareLaunchArgument(
        'frame_id', default_value='base_link',
        description='TF frame for RViz2 markers')

    rate_arg = DeclareLaunchArgument(
        'rate_hz', default_value='1.0',
        description='Marker republish rate in Hz')

    clamp_arg = DeclareLaunchArgument(
        'clamp', default_value='false',
        description='Log clamped position alongside each violation')

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Launch RViz2 with preconfigured display')

    # ── validator node ────────────────────────────────────────────────────
    validator_node = Node(
        package    = 'trajectory_validator',
        executable = 'trajectory_validator_node',
        name       = 'trajectory_validator',
        output     = 'screen',
        emulate_tty= True,
        parameters = [{
            'csv_path' : LaunchConfiguration('csv_path'),
            'frame_id' : LaunchConfiguration('frame_id'),
            'rate_hz'  : LaunchConfiguration('rate_hz'),
            'clamp'    : LaunchConfiguration('clamp'),
        }],
    )

    # ── optional RViz2 ─────────────────────────────────────────────────────
    rviz_cfg = PathJoinSubstitution([
        FindPackageShare('trajectory_validator'),
        'config', 'rviz2.rviz',
    ])

    rviz_node = Node(
        package   = 'rviz2',
        executable= 'rviz2',
        name      = 'rviz2',
        arguments = ['-d', rviz_cfg],
        condition = IfCondition(LaunchConfiguration('rviz')),
        output    = 'screen',
    )

    return LaunchDescription([
        csv_arg,
        frame_arg,
        rate_arg,
        clamp_arg,
        rviz_arg,
        validator_node,
        rviz_node,
    ])
