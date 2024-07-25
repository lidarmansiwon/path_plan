"""
Free Space Map
"""

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from pose import Pose


class FreeSpaceMap:
    def __init__(self, map_file_name):  ### 라이다 슬램을 기반하여 생성된 gridmap을 planning에 사용할 수 있도록 도와주는 class, visualize도 구현됨
        self.free_space = 255           ### 254는 흰색, 205는 회색
        self.resolution = 0.1  # [m]  픽셀과 미터사이의 변환을 다음과 같이 할 수 있음 --> 경로 생성에 사용되는 변수
        self.actual_resolution = 0.05  # 실제 맵 해상도 
        self.scale_factor =  self.actual_resolution / self.resolution
        self.resolution_inverse = int(1 / self.resolution)

        self.parking_width = self._m_to_px(1)   ## 사용되는 값은 미터인데 실제 사용되는 값은 픽셀이기 때문에 이렇게 함
        self.parking_length = self._m_to_px(1)

        self.vehicle_width = self._m_to_px(0.7)
        self.vehicle_front_length = self._m_to_px(0.5)
        self.vehicle_rear_length = self._m_to_px(0.5)
        self.vehicle_length = self.vehicle_front_length + self.vehicle_rear_length

        self.translation = (-49.4, -48.7)
        self.rotation_angle = np.radians(0)

        map_file_path = os.path.join(os.path.dirname(__file__), map_file_name)  ## 입력 받은 파일 이름
        with open(map_file_path, "rb") as map_image: 
            self.map = plt.imread(map_image)      ## self.map으로 받음. np.array로서 픽셀에 해당하는 색상값 가짐 // plt.imread 는 이미지 읽어오는 함수

        self.lot_width = self._px_to_m(self.map.shape[1])
        print(self.lot_width,"wigth",self.map.shape[1])
        self.lot_height = self._px_to_m(self.map.shape[0])
        print(self.lot_height,"height",self.map.shape[0])

        self.drop_off_spot = ()  ## 각각 세가지 값들 입력 받게 됨
        self.goal_state = ()
        self.parking_space = ()

        fig = plt.figure()
        self.ax = fig.add_subplot(111)

    def get_grid_index(self, x, y):
        return x + y * self.lot_width

    def is_not_on_obstacle(self, current_node):  ## 노드의 값을 주면 해당 값이 범위안에 있는지, 장애물과 곂침 판단
        is_in = 0 < current_node[0] < self.lot_width and 0 < current_node[1] < self.lot_height
        print(current_node[0],self.lot_width,current_node[1],self.lot_height,"check")
        return is_in and not self._is_on_obstacle(current_node)

    def _is_on_obstacle(self, current_node):
        print("kk")
        for x in range(self._m_to_px(current_node[0]), self._m_to_px(current_node[0] + 1)):
            for y in range(self._m_to_px(current_node[1]), self._m_to_px(current_node[1] + 1)):
                if self.map[y, x] != self.free_space:
                    return True

    def set_drop_off_spot(self, x, y, theta):
        self.drop_off_spot = (x, y, theta)

    def get_drop_off_spot(self):
        return Pose(
            self._px_to_m(self.drop_off_spot[0]),
            self._px_to_m(self.drop_off_spot[1]),
            np.radians(self.drop_off_spot[2])
        )

    def set_goal_state(self, x, y, theta):
        self.goal_state = (x, y, theta)

    def set_goal_state_pose(self, pose):
        self.goal_state = (self._m_to_px(pose.x), self._m_to_px(pose.y), np.degrees(pose.theta))

    def get_goal_state(self):
        return Pose(
            self._px_to_m(self.goal_state[0]),
            self._px_to_m(self.goal_state[1]),
            np.radians(self.goal_state[2])
        )

    def set_parking_space(self, x, y, theta):
        self.parking_space = (x, y, theta)

    def get_parking_space(self):
        return Pose(
            self._px_to_m(self.parking_space[0]),
            self._px_to_m(self.parking_space[1]),
            np.radians(self.parking_space[2])
        )

    def plot_map(self):  ## visualization 부분 
        plt.imshow(self.map, cmap="gray")
        print("polot")
        if self.drop_off_spot:
            self._draw_vehicle_pose(self.drop_off_spot, "blue")

        if self.goal_state:
            self._draw_vehicle_pose(self.goal_state, "yellow")

        if self.parking_space:
            self._draw_parking_space(self.parking_space, "red")

        if self.drop_off_spot and self.goal_state:
            print("ssss")
            self.ax.set(
                xlim=[
                    min(self.drop_off_spot[0], self.goal_state[0]) - 800,
                    max(self.drop_off_spot[0], self.goal_state[0]) + 800
                ],
                ylim=[
                    min(self.drop_off_spot[1], self.goal_state[1]) - 1000,
                    max(self.drop_off_spot[1], self.goal_state[1]) + 1000
                ]
            )
        else:
            print("else")
            self.ax.set(
                xlim=[0, self.map.shape[1]], ylim=[0, self.map.shape[0]]
            )

    def plot_process(self, current_node):
        # show graph
        plt.plot(self._m_to_px(current_node.discrete_x), self._m_to_px(current_node.discrete_y), "xc")
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

    def plot_route(self, rx, ry):  ## 최종적으로 경로 생성하면 시각화하는 부분
        plt.plot([self._m_to_px(x) for x in rx], [self._m_to_px(y) for y in ry], "-r")
        plt.pause(0.001)
        print(rx,ry)
        
        # 경로 변환
        transformed_rx, transformed_ry = self.transform_path(rx, ry, self.translation, self.rotation_angle, self.scale_factor)
        
        # 변환된 경로 저장
        self.save_path_to_file(transformed_rx, transformed_ry, 'transformed_path.txt')
        print("경로 변환 및 저장이 완료되었습니다.")
        plt.plot([self._m_to_px(x) for x in transformed_rx], [self._m_to_px(y) for y in transformed_ry], "-y")
        plt.pause(0.001)



    def plot_show(self):
        if self.drop_off_spot:
            self._draw_vehicle_pose(self.drop_off_spot, "blue")

        if self.goal_state:
            self._draw_vehicle_pose(self.goal_state, "yellow")

        if self.parking_space:
            self._draw_parking_space(self.parking_space, "red")

        plt.show()

    def _draw_parking_space(self, pose, color):
        rotation = np.radians(pose[2] - 90)
        x = pose[0] - self.parking_width / 2 * np.cos(rotation) + self.vehicle_rear_length * np.sin(rotation)
        y = pose[1] - self.parking_width / 2 * np.sin(rotation) - self.vehicle_rear_length * np.cos(rotation)
        parking_space = patches.Rectangle((x, y), self.parking_width, self.parking_length, color=color, alpha=0.50)
        t = transforms.Affine2D().rotate_deg_around(
            pose[0], pose[1], (pose[2] - 90)
        ) + self.ax.transData
        parking_space.set_transform(t)
        self.ax.add_patch(parking_space)

    def _draw_vehicle_pose(self, pose, color):
        rotation = np.radians(pose[2])
        x = pose[0] - self.vehicle_rear_length * np.cos(rotation) + self.vehicle_width / 2 * np.sin(rotation)
        y = pose[1] - self.vehicle_rear_length / 2 * np.sin(rotation) - self.vehicle_width / 2 * np.cos(rotation)
        parking_space = patches.Rectangle((x, y), self.vehicle_length, self.vehicle_width, color=color, alpha=0.50)
        t = transforms.Affine2D().rotate_deg_around(
            pose[0], pose[1], pose[2]
        ) + self.ax.transData
        parking_space.set_transform(t)
        self.ax.add_patch(parking_space)

    def _m_to_px(self, m):   ## 미터 --> 픽셀
        return int(m * self.resolution_inverse)

    def _px_to_m(self, px):  ## 픽셀 --> 미터
        return px * self.resolution

    def transform_path(self, rx, ry, translation, rotation_angle, scale_factor):
        transformed_rx = []
        transformed_ry = []

        for x, y in zip(rx, ry):
            # 회전 변환
            new_x = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
            new_y = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
            # 평행 이동
            new_x += translation[0]
            new_y += translation[1]
            # 해상도 변환
            new_x *= scale_factor
            new_y *= scale_factor
            transformed_rx.append(new_x)
            transformed_ry.append(new_y)

        return transformed_rx, transformed_ry

    # 경로를 텍스트 파일에 저장하는 함수
    def save_path_to_file(self, rx, ry, filename):
        with open(filename, 'w') as file:
            for x, y in zip(reversed(rx), reversed(ry)):
                file.write(f"{x} {y}\n")



if __name__ == "__main__":
    free_space_map = FreeSpaceMap("test4.pgm")
    free_space_map.set_drop_off_spot(494, 487, 90)
    free_space_map.set_parking_space(330, 447, -85)
    free_space_map.plot_map()
    free_space_map.plot_show()
