from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'ob_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'defect = ob_detection.Defect_detection_with_Coverage_Map_connected:main',
            'defect_detection_connected = ob_detection.Defect_detection_with_Coverage_Map_connected:main',
            'detect = ob_detection.detection_node_fixed:main',
            'segmentation_spray = ob_detection.detection_node_fixed:main',
            'ply_marker_pub = ob_detection.ply_marker_pub:main',
            'vision_stream_window = ob_detection.vision_stream_window:main',
        ],
    },
)
