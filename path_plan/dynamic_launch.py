#!/usr/bin/env python3

import sys
import subprocess
import os

def launch_node(node_name):
    launch_command = f"ros2 launch path_plan {node_name}"
    print(f"Executing command: {launch_command}")

    subprocess.run(launch_command, shell=True)

def main():
    print("\033[33mSelect a node to launch:\033[0m")
    print("\033[32m1: path_plan_node\033[0m")
    print("\033[32m2: path_plan_vectorField\033[0m")

    choice = input("\033[33mEnter your choice\033[0m \033[31m(1 or 2):\033[0m ").strip()

    '''1 선택 시 기본 psi_d추종 코드 실행'''
    '''2 선택 시 벡터필드 기반 psi_d추종 코드 실행'''
    if choice == '1':
        launch_name = 'path_plan.launch.py'
    elif choice == '2':
        launch_name = 'path_plan_vectorField.launch.py'
    else:
        print("\033[31mInvalid choice. Please enter 1 or 2.\033[0m")
        sys.exit(1)

    # Launch the selected node
    launch_node(launch_name)

if __name__ == '__main__':
    main()