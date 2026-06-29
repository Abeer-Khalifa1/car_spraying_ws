from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_validator'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required for ament resource index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml') + glob('config/*.csv')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    zip_safe=True,
    maintainer='You',
    maintainer_email='you@example.com',
    description='Trajectory CSV validator for car_spraying_robot (ROS 2 Jazzy)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ROS 2 node
            'trajectory_validator_node = '
            'trajectory_validator.trajectory_validator_node:main',
            # Standalone CLI tools (also ros2 run-able)
            'validate_trajectory     = '
            'trajectory_validator.validate_trajectory:main',
            'visualize_workspace     = '
            'trajectory_validator.visualize_workspace:main',
            # Filter + forward: validate peya.csv, strip unsafe points, hand off to square_xz
            'filter_and_forward      = '
            'trajectory_validator.filter_and_forward:main',
        ],
    },
)
