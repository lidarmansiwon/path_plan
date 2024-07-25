# path_plan/launch/path_plan_launch.py
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value='/home/sw/ros2_ws/src/path_plan/config/param.yaml',
            description='Path to the parameter file'
        ),
    
        Node(
            package='path_plan',
            executable='path_plan_node',
            name='path_plan_node',
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/sw/ros2_ws/src/path_plan/rviz/rviz.rviz']
        ),
    ])
