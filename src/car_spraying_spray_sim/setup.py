from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'car_spraying_spray_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Gaussian spray-cone simulation node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'spray_sim_node = car_spraying_spray_sim.spray_sim_node:main',
            'coverage_map_node = car_spraying_spray_sim.coverage_map_node:main',
            'coverage_quality_node = car_spraying_spray_sim.coverage_quality_node:main', 
        ],
    },
)
