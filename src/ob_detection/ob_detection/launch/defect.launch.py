from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ob_detection',
            executable='defect',
            name='defect_inspection',
            output='screen',
            arguments=['--ros', '--camera_topic', '/color_image/compressed'],
        )
    ])
