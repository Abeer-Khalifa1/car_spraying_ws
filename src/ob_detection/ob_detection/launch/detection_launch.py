from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ob_detection',
            executable='detection_node',
            name='ob_detection_node',
            output='screen',
            parameters=[{
                'model_path': 'car_parts_best.pt',
                'input_topic': '/color_image/compressed',
                'confidence_threshold': 0.5
            }]
        )
    ])
