from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    sdf_path = "/home/user/car_spraying_ws/src/car_spraying_robot/models/Spraying_Arm.sdf"
    urdf_path = "/home/user/car_spraying_ws/src/car_spraying_robot/urdf/UR3_Assembly_URDF.urdf"
    bridge_config = "/home/user/car_spraying_ws/src/car_spraying_robot/config/bridge.yaml"
    # arm_sim_path = "/home/user/car_spraying_ws/src/car_spraying_robot/launch/ik_square.py"
    arm_sim_path = "/home/user/car_spraying_ws/src/car_spraying_robot/car_spraying_robot/peya_subscriber.py"
    return LaunchDescription([
        # Gazebo Simulation

        ExecuteProcess(
            cmd=["gz", "sim", "-r", sdf_path], # Added '-r' to auto-run/unpause simulation
            output="screen"
        ),
        # # 🔄 ROS-Gazebo Bridge for each joint position topic
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        #          '/joint_0/position_cmd@std_msgs/msg/Float64@ignition.msgs.Double'],
        #     output='screen'
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        #          '/joint_1/position_cmd@std_msgs/msg/Float64@ignition.msgs.Double'],
        #     output='screen'
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        #          '/joint_2/position_cmd@std_msgs/msg/Float64@ignition.msgs.Double'],
        #     output='screen'
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        #          '/joint_3/position_cmd@std_msgs/msg/Float64@ignition.msgs.Double'],
        #     output='screen'
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        #          '/joint_4/position_cmd@std_msgs/msg/Float64@ignition.msgs.Double'],
        #     output='screen'
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        #          '/joint_5/position_cmd@std_msgs/msg/Float64@ignition.msgs.Double'],
        #     output='screen'
        # ),
        
        # ROS-Gazebo Bridge (Updated)
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            parameters=[{"config_file": bridge_config}],
            output="screen"
        ),

        # Robot State Publisher
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": open(urdf_path).read(),
                "use_sim_time": True
            }],
            output="screen"
        ),

         # RViz2
         Node(
             package="rviz2",
             executable="rviz2",
             name="rviz2",
             parameters=[{"use_sim_time": True}]
         ),

        # Instead of ExecuteProcess, use this:
        Node(
            package="car_spraying_robot",
            executable="peya_subscriber", # Ensure this is marked executable in your setup.py
            output="screen",
            parameters=[{"use_sim_time": True}]
        ),
        
        # # Add this node to your return LaunchDescription([])
        # Node(
        #     package="joint_state_publisher",
        #     executable="joint_state_publisher",
        #     name="joint_state_publisher",
        #     parameters=[{"use_sim_time": True}]
        # ),

        #  # Run script directly with python3
        # ExecuteProcess(
        #     cmd=['python3',arm_sim_path],
        #     output='screen'
        # )

    ])
