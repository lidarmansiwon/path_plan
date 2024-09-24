import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo,IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    include_pointcloud_to_grid = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('pointcloud_to_grid'), '/launch/pcd_map_generator.launch.py']),
        launch_arguments={'pcd_file': LaunchConfiguration('pcd_file')}.items(),
    )

    return LaunchDescription([

        include_pointcloud_to_grid,
        
        DeclareLaunchArgument(
            'param_file',
            default_value='/home/macroorin1/pass_ws/src/path_plan/config/param.yaml',
            description='Path to the parameter file'
        ),
    
        Node(
            package='path_plan',
            executable='global_path_planner',
            name='global_path_planner',
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/macroorin1/pass_ws/src/path_plan/rviz/rviz.rviz']
        ),
    ])
