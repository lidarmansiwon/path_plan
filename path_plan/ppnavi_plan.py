import rclpy, os, math, tf2_ros
import numpy as np
import matplotlib.pyplot as plt
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped,Point
from nav_msgs.msg import Path, Odometry,OccupancyGrid
from ament_index_python.packages import get_package_share_directory
from mk3_msgs.msg import GuidanceType, NavigationType
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

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('ppnavi_plan')
        self.declare_parameter('boundary_cost', 2.0)
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('desired_u', 0.8)

        self.boundary_cost       = self.get_parameter('boundary_cost').value
        self.lookahead_distance  = self.get_parameter('lookahead_distance').value
        self.scale               = self.get_parameter('scale').value
        self.desired_u           = self.get_parameter('desired_u').value

        package_share_directory  = get_package_share_directory('path_plan')
        
        package_share_directory  = get_package_share_directory('path_plan')
        file_path_1              = os.path.join(package_share_directory, 'path', 'path1.txt')
        file_path_2              = os.path.join(package_share_directory, 'path', 'path2.txt')
        file_path_3              = os.path.join(package_share_directory, 'path', 'path3.txt')
        file_path_4              = os.path.join(package_share_directory, 'path', 'path4.txt')
        file_path_5              = os.path.join(package_share_directory, 'path', 'path5.txt')
        self.path1               = self.read_waypoints_from_file(file_path_1)
        self.path2               = self.read_waypoints_from_file(file_path_2)
        self.path3               = self.read_waypoints_from_file(file_path_3)
        self.path4               = self.read_waypoints_from_file(file_path_4)
        self.path5               = self.read_waypoints_from_file(file_path_5)

        self.current_path        = self.path1
        print("get Path!!")
        self.paths               = [self.path1, self.path2, self.path3, self.path4, self.path5]

        self.navi_subscription     = self.create_subscription(NavigationType, '/navigation',  self.navigation_callback,     10)
        self.imu_subscription      = self.create_subscription(Imu,            '/agent3/imu',  self.boat_callback,           qos_profile = qos_profile_sensor_data)
        self.obstacle_subscription = self.create_subscription(OccupancyGrid,  '/height_grid', self.occupancy_grid_callback, 10)
        self.current_state: Pose   = Pose()

        self.data_check            = 0
        self.current_index         = 0
        self.start_check           = False
        self.navigation_data       = None
        self.local_map             = None
        self.boat_data             = None
        self.pop                   = np.zeros(2)
        self.current_position      = np.array([0.0, 0.0])
        self.error_psi_d           = 0
        self.marker_id             = 0
        self.current_index         = 0
        self.timer                 = self.create_timer(0.01, self.process)

        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.publisher_            = self.create_publisher(Path,         '/gpp/planned_path', 10)
        self.pop_publisher         = self.create_publisher(Marker,       '/gpp/pop', 10)
        self.trans_pop_publisher   = self.create_publisher(Marker,       '/gpp/trans_waypoint', 10)
        self.psi_d_publisher       = self.create_publisher(Marker,       '/gpp/psi_d', 10)
        self.desiredData_publisher = self.create_publisher(GuidanceType, '/guidance', 10)
        self.ob_publisher          = self.create_publisher(MarkerArray,  '/close_objects', 10)
        self.publisher             = self.create_publisher(MarkerArray,  '/objects', 10)



    def process(self):
        self.publish_path(self.current_path)
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
        # closest_distance = self.check_current_path_for_obstacles()
        
        # if closest_distance < 0.5:
        #     # 장애물과의 거리가 임계값 이하인 경우 대안 경로 찾기
        #     # print('Current Path is too close with obstacle < 1.0, switching path...')
        #     self.current_path = self.select_best_path()
        # else:
        #     # print("Continue following the current path")
        #     pass
        
        self.publish_desiredData()

        
        self.pop = self.find_point_on_path(self.current_position)
        self.vis_pop(self.pop)
    
    def check_current_path_for_obstacles(self):
        """
        Check the current path for obstacles and return the closest distance.
        """
        closest_distance = float('inf')
        if self.local_map:
            obs_list = self.obstacle_search()
            closest_obstacle, distance= self.ob_check_on_path(obs_list, self.current_path)
            closest_distance = distance
            # print(closest_distance, closest_obstacle,"distance, closet_ob")
        return closest_distance

    def select_best_path(self):
        """
        Select the best path based on the lowest cost.
        """
        min_cost = float('inf')
        best_path = self.current_path
        for path in self.paths:
            cost = self.calculate_path_cost(path)
            # print(cost,"cost")
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
            closest_distance = self.calculate_distance_to_obstacles(waypoint)
            cost += max(0, 1.0 - closest_distance)  # Cost based on obstacle distance
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

    def calculate_distance_to_obstacles(self, waypoint):
        """
        Calculate the minimum distance from a waypoint to all obstacles.
        """
        min_distance = float('inf')
        for obs in self.obstacle_search():
            map_obs = self.boat_to_map(obs)
            distance = np.linalg.norm(np.array(waypoint) - np.array(map_obs))
            if distance < min_distance:
                min_distance = distance
        return min_distance

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
    def find_point_on_path(self, current_position):
        if not self.lookahead_distance:
            return
        for i in range(self.current_index, len(self.current_path)):
            waypoint = self.current_path[i]
            # mtobway  = self.map_to_boat(waypoint)
            distance = np.linalg.norm(waypoint - current_position)
            if distance >= self.lookahead_distance:
                self.current_index = i 
                
                # Check if the waypoint is already passed
                # if mtobway[0] < 0.3 or abs(mtobway[1]) > 2.0:
                #     self.current_index+=1
                # if abs(self.error_psi_d) > 50.0:
                #     self.current_index+=1

                return waypoint
        return self.current_path[-1]
    
    '''(선박-목표지점) 벡터 생성'''
    '''
    goat_point : 최신화된 경유점
    direction  : 선박으로부터 목표지점으로 이어지는 벡터
    return     : 단위 벡터 반환
    '''
    def control(self, current_position):
        goal_point = self.find_point_on_path(current_position)
        direction = goal_point - current_position 
        return direction / np.linalg.norm(direction)
    
    def navigation_callback(self, msg):
        self.navigation_data = msg

    def boat_callback(self, msg):
        self.boat_data       = msg

    def occupancy_grid_callback(self, occupancy_grid_msg):
        self.local_map = occupancy_grid_msg

    def obstacle_search(self) -> List[Tuple[float, float]]:
        obs_list = []

        if self.local_map is None:
            return obs_list
        
        resolution = self.local_map.info.resolution
        width = self.local_map.info.width
        height = self.local_map.info.height
        origin_x = self.local_map.info.origin.position.x
        origin_y = self.local_map.info.origin.position.y
    

        for i in range(height):  ## 여기서 만드는 object 좌표화 값들 (NWU)값? --> 일단 object 좌표들 base_link에 맞춰줌 
            for j in range(width):
                index = i * width + j

                if self.local_map.data[index] == 120:
                    
                    cell_x = (origin_x - j * resolution)
                    cell_y = (origin_y - i * resolution) ## Occupancy_grid_map 에서 grid->info.origin.position.x = position_x + length_x / 2; 이렇게 하기 때문에, origin을 빼줘야 y축이 0.0 이 된다. 

                    ob     = (cell_x,cell_y)
                    bToMob = self.boat_to_map(ob)
                    # obs_list.append((cell_x, cell_y))
                    obs_list.append(bToMob)
                        

        self.before_object(obs_list)

        return obs_list
    
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

    def ob_check_on_path(self, obs_list, path):
        """
        Checks for the nearest obstacle on the path.
        """
        closest_distance = float('inf')
        closest_obstacle = None

        for obs in obs_list:
            # Convert obstacle position to map coordinates
            for path_point in path:
                path_x, path_y = path_point
                dist = np.sqrt((obs[0] - path_x) ** 2 + (obs[1] - path_y) ** 2)

                if dist < closest_distance:
                    closest_distance = dist
                    closest_obstacle = obs

        return closest_obstacle, closest_distance

    def update_boat_state(self):
        # if not hasattr(self, 'previous_odom_msg'):
        #     self.previous_odom_msg = self.navigation_data
        #     print("dd")
        #     return
        self.psi = self.navigation_data.psi
        self.u   = self.navigation_data.u
        self.x   = self.navigation_data.x
        self.y   = self.navigation_data.y
        self.current_position   = np.array([self.x,self.y])

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
        desired_publisher             = GuidanceType()
        desired_publisher.desired_psi = round(psi_d, 2)
        # desired_publisher.desired_psi = 0.00
        desired_publisher.desired_u   = round(self.desired_u, 2)
        desired_publisher.error_psi   = psi_d - self.navigation_data.psi
        desired_publisher.error_psi   = round(self.error_psi_d, 2)
        # desired_publisher.error_psi   = 0.00
        desired_publisher.error_u     = round(self.desired_u - self.navigation_data.u, 2)
        desired_publisher.distance    = round(math.sqrt(vector_d[0]**2 + vector_d[1]**2), 2)
        desired_publisher.x_waypoint  = round(desired_x, 2)
        desired_publisher.y_waypoint  = round(desired_y, 2)
        desired_publisher.goback_flag = 0
        self.desiredData_publisher.publish(desired_publisher)

        # self.tf_broadcaster.sendTransform(transform_position)
        # self.visualize_arrow(ship_position, np.array([desired_x, desired_y]))
    
    '''Psi_d Visualization // Marker.arrow'''
    # def visualize_arrow(self, start, end):
    #     marker = Marker()
    #     marker.header.frame_id = "camera_init"
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.id = 2
    #     marker.type = Marker.ARROW
    #     marker.action = Marker.ADD
    #     # marker.pose.position.x = start[0]
    #     # marker.pose.position.y = -start[1]  # ship position Axis is NED. So need (-)
    #     marker.pose.position.x = 0.0
    #     marker.pose.position.y = 0.0
    #     marker.pose.position.z = 0.0
    #     start_point = Point(x=0.0,y=0.0,z=0.2)
    #     marker.points.append(start_point)
    #     point_x = 0.8*(np.cos(deg2radian(self.error_psi_d)))
    #     point_y = -0.8*(np.sin(deg2radian(self.error_psi_d)))
    #     end_point = Point(x=point_x,y=point_y,z=0.2)
    #     marker.points.append(end_point)
    #     # dx = end[0] - start[0]
    #     # dy = -end[1] + start[1]

    #     # marker.pose.orientation.x = 0.0
    #     # marker.pose.orientation.y = 0.0
    #     # marker.pose.orientation.z = np.sin(deg2radian(-self.error_psi_d))
    #     # marker.pose.orientation.w = np.cos(deg2radian(-self.error_psi_d))

    #     # marker.scale.x = np.sqrt(dx**2 + dy**2)  # Arrow Length
    #     marker.scale.x = 0.08  # Arrow Length
    #     marker.scale.y = 0.2                      # Arrow width
    #     marker.scale.z = 0.08                    # Arrow head width

    #     marker.color.a = 1.0
    #     marker.color.r = 1.0
    #     marker.color.g = 1.0
    #     marker.color.b = 0.0

    #     self.psi_d_publisher.publish(marker)

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

    def before_object(self,obs_list):
        markers = MarkerArray()
        for obs in obs_list:
            marker = Marker()
            marker.header.frame_id = "camera_init"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = self.marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obs[0]
            marker.pose.position.y = obs[1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            markers.markers.append(marker)
            self.marker_id += 1
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = int(1e8)

        # print(obs_list)        
        self.publisher.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    ppnavi_plan = PurePursuitController()
    rclpy.spin(ppnavi_plan)
    ppnavi_plan.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()