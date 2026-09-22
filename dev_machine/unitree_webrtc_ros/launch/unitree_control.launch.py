#!/usr/bin/env python3
"""
Launch file for Unitree WebRTC control node.
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Unitree control node."""

    # Defaults come from config/unitree_params.yaml, so editing that file
    # actually takes effect without also having to pass launch arguments.
    config_path = os.path.join(
        get_package_share_directory('unitree_webrtc_ros'),
        'config',
        'unitree_params.yaml'
    )
    with open(config_path, 'r') as f:
        default_params = yaml.safe_load(f)['unitree_control']['ros__parameters']

    # Declare launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value=default_params['robot_ip'],
        description='IP address of the Unitree Go2 robot'
    )

    connection_method_arg = DeclareLaunchArgument(
        'connection_method',
        default_value=default_params['connection_method'],
        description='Connection method: LocalAP, LocalSTA, or Remote'
    )

    control_mode_arg = DeclareLaunchArgument(
        'control_mode',
        default_value=default_params['control_mode'],
        description='Control mode: sport_cmd or wireless_controller'
    )

    # Create node
    unitree_control_node = Node(
        package='unitree_webrtc_ros',
        executable='unitree_control',
        name='unitree_control',
        output='screen',
        parameters=[{
            'robot_ip': LaunchConfiguration('robot_ip'),
            'connection_method': LaunchConfiguration('connection_method'),
            'control_mode': LaunchConfiguration('control_mode'),
        }],
        remappings=[
            # Uncomment if you need to remap topics
            # ('cmd_vel', '/robot/cmd_vel'),
        ]
    )

    return LaunchDescription([
        robot_ip_arg,
        connection_method_arg,
        control_mode_arg,
        unitree_control_node,
    ])
