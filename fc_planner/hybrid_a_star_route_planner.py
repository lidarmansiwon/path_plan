"""
Free Space Hybrid A Star Route Planner

Reference:
PythonRobotics A* grid planning (author: Atsushi Sakai(@Atsushi_twi) / Nikos Kanargias (nkana@tee.gr))
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py
"""

import math

import matplotlib.pyplot as plt

from free_space_map import FreeSpaceMap
from pose import Pose


class Node:
    def __init__(
            self, pose, cost, steering, parent_node_index,
    ):
        self.pose = pose  ## 현재 노드의 위치 및 방향을 나타내는 객체
        self.discrete_x = round(pose.x)
        self.discrete_y = round(pose.y)
        self.cost = cost  ## 현재 노드로 오기까지의 누적 비용
        self.steering = steering    ## 현재 노드로 오기까지의 조향 각도
        self.parent_node_index = parent_node_index  ## 부모 노드의 인덱스


class HybridAStarRoutePlanner:
    def __init__(self):
        self.free_space_map: FreeSpaceMap = None  ## FreeSpaceMap에 대한 클래스 인스턴스를 해당 self 변수가 받게 됨

        # Motion Model
        self.wheelbase = 0.3  ## 경로 생성 시 차량의 회전 반경 계산에 사용됨, 차량의 회전 및 이동시 경로의 각도를 결정하는데 영향을 줌
        steering_degree_inputs = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60] ##  [-40, -20, -10, 0, 10, 20, 40] 도의 조향 각도 입력 값
        self.steering_inputs = [math.radians(x) for x in steering_degree_inputs]
        self.chord_lengths = [0.3, 0.5, 0.8, 1, 2]  ## 차량 이동 거리 설정. 경로 생성 시 각 노드에서 다음 노드를 생성할 때 사용됨

        self.goal_node = None

    def search_route(self, free_space_map: FreeSpaceMap, show_process=False):
        self.free_space_map = free_space_map
        start_pose = self.free_space_map.get_drop_off_spot()
        print(start_pose,"start_pose")
        goal_pose = self.free_space_map.get_goal_state()
        print(f"Start Hybrid A Star Route Planner (start {start_pose.x, start_pose.y}, end {goal_pose.x, goal_pose.y})")

        start_node = Node(start_pose, 0, 0, -1)
        self.goal_node = Node(goal_pose, 0, 0, -1)

        open_set = {self.free_space_map.get_grid_index(start_node.discrete_x, start_node.discrete_y): start_node}  
        closed_set = {}

        while open_set:
            current_node_index = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calculate_heuristic_cost(open_set[o]),
            )
            current_node = open_set[current_node_index]
            if show_process:
                self.plot_process(current_node, closed_set)
            if self.calculate_distance_to_end(current_node.pose) < 0.5:
                print("Find Goal")
                self.goal_node = current_node
                rx, ry = self.process_route(closed_set)
                self.free_space_map.plot_route(rx, ry)


                return rx, ry

            # Remove the item from the open set
            del open_set[current_node_index]

            # Add it to the closed set
            closed_set[current_node_index] = current_node

            next_nodes = [
                self.calculate_next_node(
                    current_node, current_node_index, velocity, steering
                )
                for steering in self.steering_inputs
                for velocity in self.chord_lengths
            ]
            for next_node in next_nodes:  ## 노드가 장애물과 충돌하는지 안하는지 확인
                if self.free_space_map.is_not_on_obstacle((next_node.discrete_x, next_node.discrete_y)):
                    next_node_index = self.free_space_map.get_grid_index(next_node.discrete_x, next_node.discrete_y)
                    if next_node_index in closed_set:
                        continue

                    if next_node_index not in open_set:
                        open_set[next_node_index] = next_node  # discovered a new node
                    else:
                        if open_set[next_node_index].cost > next_node.cost:
                            # This path is the best until now. record it
                            open_set[next_node_index] = next_node

        print("Cannot find Route")
        return [], []

    def process_route(self, closed_set):
        rx = [self.goal_node.pose.x]
        ry = [self.goal_node.pose.y]
        parent_node = self.goal_node.parent_node_index
        while parent_node != -1:
            n = closed_set[parent_node]
            rx.append(n.pose.x)
            ry.append(n.pose.y)
            parent_node = n.parent_node_index
        return rx, ry

    def calculate_next_node(self, current, current_node_index, chord_length, steering):
        theta = self.change_radians_range(
            current.pose.theta + chord_length * math.tan(steering) / float(self.wheelbase)
        )
        x = current.pose.x + chord_length * math.cos(theta)
        y = current.pose.y + chord_length * math.sin(theta)

        return Node(
            Pose(x, y, theta),
            current.cost + chord_length,
            steering,
            current_node_index,
        )

    def calculate_heuristic_cost(self, node):
        distance_cost = self.calculate_distance_to_end(node.pose)
        angle_cost = abs(self.change_radians_range(node.pose.theta - self.goal_node.pose.theta)) * 10.0 ## 최종 목적지에 대한 각도값과 노드에 대한 각도값을 최대한 유사하게 가져가게 
        steering_cost = abs(node.steering) * 1.0  ## 방향 전환을 최소화하게 하기 위함

        cost = distance_cost + angle_cost + steering_cost
        return float(cost)

    def calculate_distance_to_end(self, pose):
        distance = math.sqrt(
            (pose.x - self.goal_node.pose.x) ** 2 + (pose.y - self.goal_node.pose.y) ** 2
        )
        return distance

    @staticmethod
    # Imitation Code: https://stackoverflow.com/a/29237626
    # Return radians range from -pi to pi
    def change_radians_range(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def plot_process(self, current_node, closed_set):
        # show graph
        self.free_space_map.plot_process(current_node)
        if len(closed_set.keys()) % 10 == 0:
            plt.pause(0.001)


def main():
    # free_space_map = FreeSpaceMap("modified_map.pgm")
    free_space_map = FreeSpaceMap("test4.pgm")
    # start and goal pose  (5262, 5781, 0)
    # free_space_map.set_drop_off_spot(5262, 5781, 0)
    # free_space_map.set_goal_state(7695, 5508, 90)
    free_space_map.set_drop_off_spot(494, 487, 90)
    free_space_map.set_goal_state(330, 447, -85)
    free_space_map.plot_map()
    hybrid_a_star_route_planner = HybridAStarRoutePlanner()
    hybrid_a_star_route_planner.search_route(free_space_map, True) ## 경로 생성을 진행되는 부분, True: 시각화, False: 시각화 제외 --> 시각화를 하면 계산이 느려짐 속도 차이 많이남
    free_space_map.plot_show()


if __name__ == "__main__":
    main()
