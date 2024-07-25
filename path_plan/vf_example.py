#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy

'''
[ 해야할 것 ]
1. 벡터필드 구성 방법 학습 (O)
2. 벡터필드 구성
3. 벡터필드에서의 선박 psi 추종
4. 벡터필드에서의 선박 psi 추종 불가능 상태 분류
5. 벡터필드에서의 선박 psi 추종 불가능 상태 flag
6. 벡터필드에서의 선박 psi 추종 불가능할 경우 어떻게 작동할지에 대한 고민
'''
class CurveFitting(Node):
    def __init__(self):
        super().__init__('path_plan_node')
        package_directory = get_package_share_directory('path_plan')
        path_file = os.path.join(package_directory, 'path', 'path.txt')
        self.path = self.read_waypoints_from_file(path_file)
        
        '''전역 경로 --> 매끄러운 곡선 변환'''
        self.interpolated_path = self.interpolate_path(self.path)

        self.publisher_desired_path = self.create_publisher(Path, 'desired_path', 10)
        self.publisher_origin_path = self.create_publisher(Path, '/origin/path', 10)

        self.timer = self.create_timer(0.01, self.process)

    def process(self):
        '''매끄러운 곡선으로 변환한 전역 경로'''
        self.publish_path(self.interpolated_path, self.publisher_desired_path)
        '''기존의 전역 경로'''
        self.publish_path(self.path, self.publisher_origin_path)

    ## txt파일로부터 전역 경로 생성
    def read_waypoints_from_file(self, file_path):
        waypoints = []
        scale = 1.5
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                x = x * scale
                y = y * scale
                waypoints.append([-y, x])
        return np.array(waypoints)

    def interpolate_path(self, path, num_points=3000):
        '''
        [ num_points에 따른 장/단점 ]
        장점 : 더 부드러운 곡선, 높은 해상도
        단점 : 계산 cost 증가, 네트워크 부하
        '''
        x = path[:, 0]
        y = path[:, 1]
        
        t = np.linspace(0, 1, len(path))
        fx = interp1d(t, x, kind='cubic')
        fy = interp1d(t, y, kind='cubic')
        
        # Generate new points for the interpolated path
        t_new = np.linspace(0, 1, num_points)
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        interpolated_path = np.vstack((x_new, y_new)).T
        return interpolated_path

    def publish_path(self, path, publisher):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CurveFitting()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
