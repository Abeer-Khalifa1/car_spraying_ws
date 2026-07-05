from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # ── Launch arguments ──────────────────────────────────────────────────────
    spray_active_arg = DeclareLaunchArgument(
        'spray_active', default_value='true',
        description='Enable or disable spray cone visualisation'
    )
    cone_length_arg = DeclareLaunchArgument(
        'cone_length', default_value='0.20',
        description='Length of the spray cone [m]'
    )
    cone_half_angle_arg = DeclareLaunchArgument(
        'cone_half_angle_deg', default_value='15.0',
        description='Half-angle of the spray cone [degrees]'
    )
    sigma_arg = DeclareLaunchArgument(
        'sigma', default_value='0.03',
        description='Gaussian sigma for paint density [m]'
    )
    gz_world_arg = DeclareLaunchArgument(
        'gz_world_name', default_value='world_demo',
        description='Gazebo world name (must match <world name=...> in SDF)'
    )

    # ── Spray sim node ────────────────────────────────────────────────────────
    spray_sim_node = Node(
        package='car_spraying_spray_sim',
        executable='spray_sim_node',
        name='spray_sim_node',
        output='screen',
        parameters=[{
            'use_sim_time':        True,
            'end_effector_frame':  'link_6',
            'world_frame':         'world',
            'gz_world_name':       LaunchConfiguration('gz_world_name'),
            'spray_active':        LaunchConfiguration('spray_active'),
            'cone_length':         LaunchConfiguration('cone_length'),
            'cone_half_angle_deg': LaunchConfiguration('cone_half_angle_deg'),
            'sigma':               LaunchConfiguration('sigma'),
            'num_sample_rings':    8,
            'num_angular_pts':     36,
            'paint_point_spacing': 0.008,
            'max_paint_points':    30000,
            'max_gz_spheres':      5000,
            'gz_sphere_radius':    0.008,
            'publish_rate_hz':     10.0,  
            'gz_spawn_every_n':    3,       
        }],
    )

    return LaunchDescription([
        spray_active_arg,
        cone_length_arg,
        cone_half_angle_arg,
        sigma_arg,
        gz_world_arg,
        spray_sim_node,
    ])