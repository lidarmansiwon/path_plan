import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from mk3_msgs.msg import GuidanceType, NavigationType
from visualization_msgs.msg import Marker, MarkerArray

class Pose:
    x: float = 0.0
    y: float = 0.0
    linear_velocity: float = 0.0
    psi: float = 0.0


class PurePursuitController(Node):

    def __init__(self):
        super().__init__('path_plan_node')
        self.lookahead_distance = 0.3  # Lookahead distance
        self.current_index = 0
        package_share_directory = get_package_share_directory('path_plan')
        file_path = os.path.join(package_share_directory, 'path', 'path.txt')
        self.path = self.read_waypoints_from_file(file_path)
        self.navigation_data     = None
        self.current_state: Pose = Pose()
        self.current_position   = np.array([0.0, 0.0])
        self.publisher_         = self.create_publisher(Path, 'planned_path', 10)
        self.pop_publisher      = self.create_publisher(Marker, 'pop',10)

        self.timer = self.create_timer(0.01, self.process)

        self.navi_subscription = self.create_subscription(
            NavigationType,
            '/navigation',
            self.navigation_callback,
            10
        )
        self.data_check  = 0
        self.start_check = False

    def process(self):
        self.publish_path()
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

        self.update_boat_state()
        self.pop = self.find_point_on_path(self.current_position)
        print(self.pop, self.current_position)
        self.vis_pop(self.pop)


        

    def read_waypoints_from_file(self, file_path):
        waypoints = []
        scale = 1.0
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())

                x = x*scale
                y = (y)*scale 


                waypoints.append([-x, y])

        return np.array(waypoints)


    def find_point_on_path(self, current_position):
        for i in range(self.current_index, len(self.path)):
            waypoint = self.path[i]
            distance = np.linalg.norm(waypoint - current_position)
            if distance >= self.lookahead_distance:
                self.current_index = i  ## 최신화를 통하여 이전에 확인한 웨이 포인트로는 돌아가지 않게 함. 항상 앞의 좌표들만 확인.
                return waypoint
        return self.path[-1]

    def control(self, current_position):
        goal_point = self.find_goal_point(current_position)
        direction = goal_point - current_position
        return direction / np.linalg.norm(direction)
    
    def navigation_callback(self, msg):
        self.navigation_data    = msg

    def update_boat_state(self):
        if not hasattr(self, 'previous_odom_msg'):
            self.previous_odom_msg = self.navigation_data
            return
        self.psi                = self.navigation_data.psi
        self.u                  = self.navigation_data.u
        self.x                  = self.navigation_data.x
        self.y                  = self.navigation_data.y
        # self.current_state      = Pose(self.x, self.y, self.u, self.psi)
        
        self.current_position   = np.array([self.x,self.y])

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in self.path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.publisher_.publish(path_msg)

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