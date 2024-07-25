#!/usr/bin/env python3
import rclpy, os, math, tf2_ros
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from ament_index_python.packages import get_package_share_directory
from mk3_msgs.msg import GuidanceType, NavigationType
from visualization_msgs.msg import Marker, MarkerArray

class Pose:
    x              : float = 0.0
    y              : float = 0.0
    linear_velocity: float = 0.0
    psi            : float = 0.0

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('path_plan_node')
        self.declare_parameter('path_data', 'path.txt')
        self.declare_parameter('lookahead_distance', 0.2)
        self.declare_parameter('scale', 1.5)
        self.declare_parameter('desired_u', 0.0)

        path_data               = self.get_parameter('path_data').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.scale              = self.get_parameter('scale').value
        self.desired_u          = self.get_parameter('desired_u').value

        package_share_directory = get_package_share_directory('path_plan')
        file_path               = os.path.join(package_share_directory, 'path', path_data)

        '''Read and interpolate waypoints'''
        self.path               = self.read_waypoints_from_file(file_path)
        self.interpolated_path  = self.interpolate_path(self.path) ## 

        self.navi_subscription  = self.create_subscription(
            NavigationType,
            '/pass/navigation',
            self.navigation_callback,
            10
        )
        self.boat_subscription  = self.create_subscription(
            Odometry,
            '/boat_odometry',
            self.boat_callback,
            10
        )
        self.current_state: Pose = Pose()

        self.data_check          = 0
        self.current_index       = 0
        self.start_check         = False
        self.navigation_data     = None
        self.boat_data           = None
        self.pop                 = np.zeros(2)
        self.current_position    = np.array([0.0, 0.0])

        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.publisher_            = self.create_publisher(Path, 'planned_path', 10)
        self.pop_publisher         = self.create_publisher(Marker, 'pop',10)
        self.psi_d_publisher       = self.create_publisher(Marker, 'psi_d', 10)
        self.desiredData_publisher = self.create_publisher(GuidanceType, '/pass/guidance', 10)

        self.timer = self.create_timer(0.01, self.process)

    def process(self):
        if (self.navigation_data is None):
            if self.start_check == False:
                print("\033[33m" + " --> " + "\033[0m" + "\033[31m" + "'navigation_data'" + "\033[0m" + "\033[33m" + " doesn't arrived yet" + "\033[0m")
                self.start_check = True
                self.data_check += 1
            if self.data_check % 1000 == 0:
                print("\033[33m" + " --> " + "\033[0m" + "\033[31m" + "'navigation_data'" + "\033[0m" + "\033[33m" + " doesn't arrived yet" + "\033[0m")
                self.data_check += 1
            else:
                self.data_check += 1
            return

        self.publish_path()
        self.update_boat_state()
        self.publish_desiredData()
        
        self.pop = self.find_point_on_path(self.current_position)
        self.vis_pop(self.pop)

    '''txt파일로부터 전역 경로 생성'''
    def read_waypoints_from_file(self, file_path):
        waypoints = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                x = x * self.scale
                y = y * self.scale
                waypoints.append([-y, x])
        return np.array(waypoints)

    '''기존 point의 경로를 통해 spline 매끄러운 경로 생성'''
    def interpolate_path(self, path, num_points=10000):
        x = path[:, 0]
        y = path[:, 1]
        
        t = np.linspace(0, 1, len(path))
        fx = interp1d(t, x, kind='linear')
        fy = interp1d(t, y, kind='linear')
        
        # Generate new points for the interpolated path
        t_new = np.linspace(0, 1, num_points)
        x_new = fx(t_new)
        y_new = fy(t_new)
        
        interpolated_path = np.vstack((x_new, y_new)).T
        return interpolated_path

    '''경유점 최신화(범위안에 목표지점이 들어올 경우 다음 목적지 추종)'''
    def find_point_on_path(self, current_position):
        if not self.lookahead_distance:
            return
        for i in range(self.current_index, len(self.interpolated_path)):
            waypoint = self.interpolated_path[i]
            distance = np.linalg.norm(waypoint - current_position)
            if distance >= self.lookahead_distance:
                self.current_index = i 
                return waypoint
        return self.interpolated_path[-1]

    '''(선박-목표지점) 단위 벡터 생성'''
    def control(self, current_position):
        goal_point = self.find_point_on_path(current_position)
        direction = goal_point - current_position 
        return direction / np.linalg.norm(direction)
    
    def navigation_callback(self, msg):
        self.navigation_data = msg

    def boat_callback(self, msg):
        self.boat_data = msg

    def update_boat_state(self):
        if not hasattr(self, 'previous_odom_msg'):
            self.previous_odom_msg = self.navigation_data
            return
        self.psi = self.navigation_data.psi
        self.u   = self.navigation_data.u
        self.x   = self.navigation_data.x
        self.y   = self.navigation_data.y
        
        self.current_position = np.array([self.x, self.y])

    def publish_desiredData(self):
        if self.boat_data is None:
            return
        '''목표 선수각 산출(psi_d)'''
        ship_position = np.array([self.navigation_data.x, self.navigation_data.y, 0])
        desired_x     = self.pop[0]
        desired_y     = self.pop[1] 
        vector_d      = np.array([desired_x - ship_position[0], desired_y - ship_position[1]])
        psi_d         = np.arctan2(vector_d[1], vector_d[0]) * (180 / math.pi)

        '''Guidance Message Creating'''
        desired_publisher = GuidanceType()
        desired_publisher.desired_psi = psi_d
        desired_publisher.desired_u   = self.desired_u
        desired_publisher.error_psi   = psi_d - self.navigation_data.psi
        desired_publisher.error_u     = self.desired_u - self.navigation_data.u
        desired_publisher.distance    = np.linalg.norm(vector_d)
        desired_publisher.x_waypoint  = desired_x
        desired_publisher.y_waypoint  = desired_y
        desired_publisher.goback_flag = 0
        self.desiredData_publisher.publish(desired_publisher)

        '''World Map // Ship Map - TF Visualization'''
        transform_position = TransformStamped()
        transform_position.header.stamp = self.get_clock().now().to_msg()
        transform_position.header.frame_id = 'map'
        transform_position.child_frame_id = 'ship'
        transform_position.transform.translation.x = ship_position[0]
        transform_position.transform.translation.y = ship_position[1]
        rotation = transform_position.transform.rotation
        rotation.x = self.boat_data.pose.pose.orientation.x
        rotation.y = self.boat_data.pose.pose.orientation.y
        rotation.z = self.boat_data.pose.pose.orientation.z
        rotation.w = self.boat_data.pose.pose.orientation.w

        self.tf_broadcaster.sendTransform(transform_position)
        self.visualize_arrow(ship_position, np.array([desired_x, desired_y]))
    
    '''Psi_d Visualization // Marker.arrow'''
    def visualize_arrow(self, start, end):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 2
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = start[0]
        marker.pose.position.y = start[1]
        marker.pose.position.z = 0.0

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = np.sin(np.arctan2(dy, dx) / 2.0)
        marker.pose.orientation.w = np.cos(np.arctan2(dy, dx) / 2.0)

        marker.scale.x = np.sqrt(dx**2 + dy**2)  # Arrow Length
        marker.scale.y = 0.1                      # Arrow width
        marker.scale.z = 0.1                     # Arrow head width

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.psi_d_publisher.publish(marker)

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in self.interpolated_path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.publisher_.publish(path_msg)

    '''목표 경유점 시각화'''
    def vis_pop(self, pop):
        pop_vis = Marker()
        pop_vis.header.frame_id = "map"
        pop_vis.header.stamp = self.get_clock().now().to_msg()
        pop_vis.id = 1
        pop_vis.type = Marker.SPHERE
        pop_vis.action = Marker.ADD
        pop_vis.pose.position.x = pop[0]
        pop_vis.pose.position.y = pop[1]
        pop_vis.pose.position.z = 0.0
        pop_vis.scale.x = 0.2
        pop_vis.scale.y = 0.2
        pop_vis.scale.z = 0.2
        pop_vis.color.a = 1.0
        pop_vis.color.r = 1.0
        pop_vis.color.g = 0.0
        pop_vis.color.b = 0.0
        pop_vis.lifetime.sec = 0
        pop_vis.lifetime.nanosec = int(1e8)
        self.pop_publisher.publish(pop_vis)

def main(args=None):
    rclpy.init(args=args)
    path_plan_node = PurePursuitController()
    rclpy.spin(path_plan_node)
    path_plan_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
