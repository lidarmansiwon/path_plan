import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def select_pcd_file():
    folder_path = '/home/macroorin1/slamMAP'
    pcd_files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')]

    if not pcd_files:
        print("\033[31mNo .pcd files found in the folder.\033[0m")
        return None

    print("\033[32mSelect a pcd file:\033[0m")
    for idx, file_name in enumerate(pcd_files):
        print(f"\033[33m{idx + 1}: {file_name}\033[0m")

    selected_idx = int(input("\033[32mEnter the number of the file: \033[0m")) - 1

    if 0 <= selected_idx < len(pcd_files):
        return os.path.join(folder_path, pcd_files[selected_idx])
    else:
        print("\033[31mInvalid selection.\033[0m")
        return None

def generate_launch_description():

    pcd_file = select_pcd_file()

    if not pcd_file:
        raise RuntimeError("No valid .pcd file selected.")

    include_pointcloud_to_grid = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('pointcloud_to_grid'), '/launch/pcd_map_generator.launch.py']),
        launch_arguments={'pcd_file': pcd_file}.items(),
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
            executable='global_path_planner3',
            name='global_path_planner3',
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
