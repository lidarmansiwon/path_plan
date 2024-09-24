import rclpy, os, math, tf2_ros
import numpy as np
import matplotlib.pyplot as plt
import os
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TransformStamped,Point, PointStamped
from nav_msgs.msg import Path, Odometry,OccupancyGrid
from ament_index_python.packages import get_package_share_directory
from mk3_msgs.msg import GuidanceType, NavigationType, WaypointType,PsiToWP
from visualization_msgs.msg import Marker, MarkerArray
from typing import List, Tuple
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from mk3_control.lib.gnc_tool import ssa
from pass_2024.navigation.ekf_navigationTool import *


class Pose:
    x              : float = 0.0
    y              : float = 0.0
    linear_velocity: float = 0.0
    psi            : float = 0.0

class GlobalPathPlanner(Node):
    def __init__(self):
        super().__init__('global_path_planner')
        self.declare_parameter('boundary_cost', 2.0)
        self.declare_parameter('lookahead_distance', 0.6)
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('desired_u', 0.8)
        
        self.boundary_cost = self.get_parameter('boundary_cost').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.scale              = self.get_parameter('scale').value
        self.desired_u          = self.get_parameter('desired_u').value

        package_share_directory = get_package_share_directory('path_plan')
        
        file_path_1             = os.path.join(package_share_directory, 'path', 'path1.txt')
        file_path_2             = os.path.join(package_share_directory, 'path', 'path2.txt')
        file_path_3             = os.path.join(package_share_directory, 'path', 'path3.txt')
        file_path_4             = os.path.join(package_share_directory, 'path', 'path4.txt')
        file_path_5             = os.path.join(package_share_directory, 'path', 'path5.txt')
        self.path1              = self.read_waypoints_from_file(file_path_1)
        self.path2              = self.read_waypoints_from_file(file_path_2)
        self.path3              = self.read_waypoints_from_file(file_path_3)
        self.path4              = self.read_waypoints_from_file(file_path_4)
        self.path5              = self.read_waypoints_from_file(file_path_5)

        self.paths              = [self.path1,self.path2,self.path3,self.path4,self.path5]
        qos_profile = qos_profile_sensor_data
        self.path_state_subscription      = self.create_subscription(
            Bool,
            '/path_state',
            self.path_state_callback,
            10
        )
        
        self.navi_subscription      = self.create_subscription(
            NavigationType,
            '/navigation',
            self.navigation_callback,
            qos_profile
        )
        
        self.imu_subscription   = self.create_subscription(
            Imu,
            '/ouster/imu',
            self.boat_callback,
            qos_profile = qos_profile_sensor_data
        )
        
        self.obstacle_subscription  = self.create_subscription(
            OccupancyGrid,
            '/height_grid',
            self.occupancy_grid_callback,
            10
        )
        
        self.point_subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',  # RViz에서 Publish된 Point
            self.point_callback,
            10
        )        
        self.current_state: Pose = Pose()

        self.data_check          = 0
        self.current_index       = 0
        self.start_check         = False
        self.navigation_data     = None
        self.local_map           = None
        self.boat_data           = None
        self.path_state          = True
        self.select_mode         = True  # True for selection, False for updating
        self.selected_index      = None
        self.pop                 = np.zeros(2)
        self.current_position    = np.array([0.0, 0.0])
        self.error_psi_d         = 0

        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.wayPoint_publisher    = self.create_publisher(WaypointType, '/gpp/waypoint', qos_profile = qos_profile_sensor_data)
        self.psiToWP_publisher     = self.create_publisher(PsiToWP, '/gpp/WPpsi', qos_profile = qos_profile_sensor_data)
        self.publisher_            = self.create_publisher(Path, '/gpp/planned_path', 10)
        self.pop_publisher         = self.create_publisher(Marker, '/gpp/pop',10)
        self.next_WP_publisher     = self.create_publisher(Marker, '/gpp/next_WP',10)
        # self.desiredData_publisher = self.create_publisher(GuidanceType, '/guidance', 10)
        self.ob_publisher          = self.create_publisher(MarkerArray, '/close_objects', 10)
        self.publisher = self.create_publisher(MarkerArray, '/objects', 10)

        self.waypoints_file_path = os.path.join(package_share_directory, 'path', 'path1.txt')
        self.current_path        = self.path1
        self.last_modified_time  = os.path.getmtime(self.waypoints_file_path)
        
        self.marker_id = 0

        self.current_index = 0
        self.timer = self.create_timer(0.01, self.process)

    def process(self):
        self.publish_path(self.current_path)
        
        ''' Checking the time when text1.txt was modified --> renewal the path simultaneously  '''
        current_modified_time = os.path.getmtime(self.waypoints_file_path)
        
        if current_modified_time != self.last_modified_time:
            self.get_logger().error("test1.txt파일의 waypoint가 최신화되었음 --> 경유점 변경 완료")
            self.last_modified_time = current_modified_time
            self.current_path = self.read_waypoints_from_file(self.waypoints_file_path)
            self.publish_path(self.current_path)
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        if (self.navigation_data is None or (self.local_map is None)):
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
        
        self.update_boat_state()  
        self.publish_desiredData()
        self.pop, self.nextWP = self.find_point_on_path(self.current_position)  ## POP -->> NED
        self.publish_wayPoint(self.pop[0],self.pop[1], self.nextWP[0],self.nextWP[1])
        self.vis_pop(self.pop)
        self.vis_nextWP(self.nextWP)
    
    def point_callback(self, msg: PointStamped):
        # 클릭한 좌표 가져오기
        clicked_x = msg.point.x
        clicked_y = msg.point.y
        clicked_point = np.array([clicked_x, -clicked_y])

        if self.select_mode:
            # 첫 번째 클릭: 경로 상에서 가장 가까운 좌표 찾기
            nearest_index, nearest_point = self.find_nearest_waypoint(clicked_point)

            if nearest_point is not None:
                # 선택된 경유점을 저장
                self.selected_index = nearest_index
                self.get_logger().info(f"선택한 Waypoint : {nearest_index}번째 {nearest_point}")
                self.get_logger().warn("Waypoint를 변경할 위치를 선택하십시오. (Publish Point 활용)")
                self.select_mode = False  # Switch to update mode

        else:
            # 두 번째 클릭: 선택된 경유점을 새로운 클릭된 좌표로 업데이트
            if self.selected_index is not None:
                self.current_path[self.selected_index] = clicked_point
                self.get_logger().info(f"업데이트된 {self.selected_index}번째 Waypoint // 변경된 위치: {clicked_point}")
                self.selected_index = None  # Reset the selected index
                self.select_mode = True  # Switch back to selection mode

                # 업데이트된 경로 다시 Publish
                self.publish_path(self.current_path)
                
                self.write_waypoints_to_file(self.current_path)
                
            else:
                self.get_logger().warn("No waypoint selected to update. Please click to select a waypoint first.")
        
    def write_waypoints_to_file(self, waypoints):
        try:
            with open(self.waypoints_file_path, 'w') as file:
                for waypoint in waypoints:
                    file.write(f"{waypoint[0]} {waypoint[1]}\n")

            self.get_logger().warn(f"txt파일 저장 경로 : {self.waypoints_file_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to write waypoints to file: {e}")

        
    def find_nearest_waypoint(self, clicked_point):
        """
        경로 상의 좌표 중에서 클릭한 좌표와 가장 가까운 좌표를 찾는다.
        """
        min_distance = float('inf')
        nearest_index = None
        nearest_point = None
        
        # 경로의 각 좌표와 클릭된 좌표의 거리를 계산
        for i, waypoint in enumerate(self.current_path):
            distance = np.linalg.norm(waypoint - clicked_point)
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
                nearest_point = waypoint

        self.get_logger().info(f"Nearest waypoint to clicked point is {nearest_index} at distance {min_distance}")
        return nearest_index, nearest_point

    def select_best_path(self):
        """
        Select the best path based on the lowest cost.
        """
        min_cost = float('inf')
        best_path = self.current_path
        for path in self.paths:
            cost = self.calculate_path_cost(path)
            if cost < min_cost:
                min_cost = cost
                best_path = path
        return best_path
    
    def calculate_path_cost(self, path):
        """
        Calculate the cost of a given path based on obstacle distance and alignment.
        """
        cost = 0.0

        # Find relevant waypoints within lookahead distance
        relevant_waypoints = self.find_relevant_waypoints(path)
        # print(relevant_waypoints)
        for waypoint in relevant_waypoints:
            cost += self.calculate_alignment_cost(waypoint)  # Cost based on alignment
            
        return cost
    
    def find_relevant_waypoints(self, path):
        """
        Find waypoints within a certain distance from the current position.
        """
        relevant_waypoints = []
        for waypoint in path:
            distance = np.linalg.norm(np.array(waypoint) - np.array(self.current_position))
            if distance <= self.boundary_cost:
                relevant_waypoints.append(waypoint)
        return relevant_waypoints


    def calculate_alignment_cost(self, waypoint):
        """
        Calculate the cost based on the alignment between the desired direction and the current heading.
        """
        goal_direction = np.array(waypoint) - self.current_position
        desired_psi = np.arctan2(goal_direction[1], goal_direction[0])
        psi_error = abs(self.psi - desired_psi)
        return psi_error

    '''txt파일로부터 전역 경로 생성'''
    def read_waypoints_from_file(self, file_path):
        if not self.scale:
            return
        waypoints = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                x = round(x,4)
                y = round(y,4)
                waypoints.append([x, y])
        return np.array(waypoints)

    '''경유점 최신화(범위안에 목표지점이 들어올 경우 다음 목적지 추종)'''
    ''' Lookahead distance Method'''
    # def find_point_on_path(self, current_position):
    #     if not self.lookahead_distance:
    #         return 
        
    #     for i in range(self.current_index, len(self.current_path)):
    #         waypoint = self.current_path[i]
    #         distance = np.linalg.norm(waypoint - current_position)

    #         if distance >= self.lookahead_distance:
    #             self.current_index = i 
    #             # 현재 경유점
    #             current_waypoint = waypoint
    #             # 다음 경유점
    #             if i + 1 < len(self.current_path):
    #                 next_waypoint = self.current_path[i + 1]              
    #             else:
    #                 next_waypoint = current_waypoint
    #             return current_waypoint, next_waypoint
            

    #             # current_waypoint = waypoint
        
    #     # 경로의 끝에 도달했을 때
    #     return self.current_path[-1], self.current_path[-1]
    
    
    def find_point_on_path(self, current_position):


        for i in range(self.current_index, len(self.current_path) - 1):
            waypoint = self.current_path[i]
            next_waypoint = self.current_path[i + 1]

            v1 = next_waypoint - waypoint  # A -> B 벡터
            v2 = waypoint - current_position  # P -> A 벡터

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            theta = np.arccos(cos_theta) * 180 / np.pi  # 라디안 -> 도

            distance = np.linalg.norm(waypoint - current_position)  

            if theta > 90:
                self.get_logger().info("Choose Next Waypoint")
                self.current_index = i + 1  # 경유점 갱신
                continue
            elif distance < self.lookahead_distance:
                self.current_index = i + 1  # 경유점 갱신
                continue                

            current_waypoint = waypoint
            next_waypoint = self.current_path[i + 1] if i + 1 < len(self.current_path) else current_waypoint
            return current_waypoint, next_waypoint

        return self.current_path[-1], self.current_path[-1]  # 경로 끝에 도달

    
    '''(선박-목표지점) 벡터 생성'''
    '''
    goat_point : 최신화된 경유점
    direction  : 선박으로부터 목표지점으로 이어지는 벡터
    return     : 단위 벡터 반환
    '''

    def path_state_callback(self, msg):
        self.path_state_data = msg
        self.path_state      = self.path_state_data.data

    def navigation_callback(self, msg):
        self.navigation_data = msg

    def boat_callback(self, msg):
        self.boat_data       = msg

    def occupancy_grid_callback(self, occupancy_grid_msg):
        self.local_map = occupancy_grid_msg
   
    def boat_to_map(self, boat_pos):
        """
        Converts boat position in boat coordinate system to map coordinate system.
        """
        self.radian_psi = np.deg2rad(self.psi)
        boat_x, boat_y = boat_pos
        map_x = self.current_position[0] + boat_x * np.cos(self.radian_psi) - boat_y * np.sin(self.radian_psi)
        map_y = self.current_position[1] + boat_x * np.sin(self.radian_psi) + boat_y * np.cos(self.radian_psi)
        return (map_x, map_y)
    
    def map_to_boat(self, map_pos):
        """
        Converts map position in map coordinate system to boat coordinate system.
        """
        self.radian_psi = np.deg2rad(self.psi)
        map_x, map_y = map_pos
        delta_x = map_x - self.current_position[0]
        delta_y = map_y - self.current_position[1]

        boat_x = -(delta_x * np.cos(self.radian_psi) - delta_y * np.sin(self.radian_psi))
        boat_y = -(delta_x * np.sin(self.radian_psi) + delta_y * np.cos(self.radian_psi))

        return (boat_x, boat_y)

    def update_boat_state(self):
        self.psi = self.navigation_data.psi
        self.u   = self.navigation_data.u
        self.x   = self.navigation_data.x
        self.y   = self.navigation_data.y
        self.current_position   = np.array([self.x,self.y])

    def publish_path(self,current_path):
        path_msg = Path()
        path_msg.header.frame_id = 'camera_init'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for point in current_path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = -point[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.publisher_.publish(path_msg)
        
    def publish_wayPoint(self,x,y,x2,y2):
        wayPoint_publisher   = WaypointType()
        wayPoint_publisher.x = x
        wayPoint_publisher.y = y
        wayPoint_publisher.x2 = x2
        wayPoint_publisher.y2 = y2
        self.wayPoint_publisher.publish(wayPoint_publisher)

    def publish_desiredData(self):
        if self.boat_data is None:
            return
        '''목표 선수각 산출(psi_d)'''
        '''
        path --> world Axis --> Change Axis to NED after get PSI
        '''
        ship_position = np.array([self.navigation_data.x, self.navigation_data.y, 0])
        desired_x     = self.pop[0]
        desired_y     = self.pop[1]
        # desired_point = (desired_x,desired_y)
        # print(desired_point,"desire")
        # trans_desired_points = self.map_to_boat(desired_point)
        vector_d      = np.array([desired_x - ship_position[0], desired_y - ship_position[1]])
        psi_d         = np.arctan2(vector_d[1], vector_d[0]) * (180/math.pi)
        # psi_d         = -np.arctan2(self.pop[1] , self.pop[0]) * (180/math.pi) 
        self.error_psi_d = ssa(psi_d - self.navigation_data.psi)
        '''Gudiance Message Creating'''
        '''
        desired_psi : 전역 좌표계 기준 psi_d
        desired_u   : x축 방향 목표 속도 (파라미터)
        error_psi   : 전역 좌표계 기준 error_psi (psi_d - psi)
        distance    : 선박과 목표지점 사이의 거리
        x_waypoint  : 목표지점 (x축)
        y_waypoint  : 목표지점 (y축)
        '''
        WPpsi             = PsiToWP()
        WPpsi.psi = round(self.error_psi_d, 2)
        self.psiToWP_publisher.publish(WPpsi)


    '''목표 경유점 시각화'''
    def vis_pop(self, pop):
        pop_vis = Marker()
        pop_vis.header.frame_id = "camera_init"
        pop_vis.header.stamp = self.get_clock().now().to_msg()
        pop_vis.id = 1
        pop_vis.type = Marker.SPHERE
        pop_vis.action = Marker.ADD
        pop_vis.pose.position.x = pop[0]
        pop_vis.pose.position.y = -pop[1]
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
        
    def vis_nextWP(self, WP):
        NEXY_WP_vis = Marker()
        NEXY_WP_vis.header.frame_id = "camera_init"
        NEXY_WP_vis.header.stamp = self.get_clock().now().to_msg()
        NEXY_WP_vis.id = 2
        NEXY_WP_vis.type = Marker.SPHERE
        NEXY_WP_vis.action = Marker.ADD
        NEXY_WP_vis.pose.position.x = WP[0]
        NEXY_WP_vis.pose.position.y = -WP[1]
        NEXY_WP_vis.pose.position.z = 0.0
        NEXY_WP_vis.scale.x = 0.2
        NEXY_WP_vis.scale.y = 0.2
        NEXY_WP_vis.scale.z = 0.2
        NEXY_WP_vis.color.a = 1.0
        NEXY_WP_vis.color.r = 1.0
        NEXY_WP_vis.color.g = 0.0
        NEXY_WP_vis.color.b = 0.0
        NEXY_WP_vis.lifetime.sec = 0
        NEXY_WP_vis.lifetime.nanosec = int(1e8)
        self.next_WP_publisher.publish(NEXY_WP_vis)
        
def main(args=None):
    rclpy.init(args=args)
    global_path_planner = GlobalPathPlanner()
    rclpy.spin(global_path_planner)
    global_path_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()