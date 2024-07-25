#!/usr/bin/env python3

import numpy as np
import math

from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

def x_rotation(angular):
    '''
    [ X축 회전 변환 행렬 ]
    angular : x축 기준 회전 각도
    '''
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angular), -np.sin(angular)],
                   [0, np.sin(angular), np.cos(angular)]])
    
    return Rx

def y_rotation(angular):
    '''
    [ Y축 회전 변환 행렬 ]
    angular : y축 기준 회전 각도
    '''
    Ry = np.array([[np.cos(angular), 0, np.sin(angular)],
                   [0, 1, 0],
                   [-np.sin(angular), 0, np.cos(angular)]])
    
    return Ry

def z_rotation(angular):
    '''
    [ Z축 회전 변환 행렬 ]
    angular : z축 기준 회전 각도
    '''
    Rz = np.array([[np.cos(angular), -np.sin(angular), 0],
                   [np.sin(angular), np.cos(angular), 0],
                   [0, 0, 1]])
    
    return Rz

def axis_coordinate(x_angle, y_angle, z_angle, position):
    '''
    [ 3차원 회전 변환 행렬 계산 함수 ]
    x_angle : x축 기준 회전 각도
    y_angle : y축 기준 회전 각도
    z_angle : z축 기준 회전 각도
    '''
    Rx = x_rotation(x_angle)
    Ry = y_rotation(y_angle)
    Rz = z_rotation(z_angle)
    
    Rt = np.dot(Rz, np.dot(Ry, Rx))
    rotate_position = np.dot(Rt, position)

    return rotate_position

def distance_with_tuple(a, b):
    return math.sqrt((a[1]-a[0])**2 + (b[1]-b[0])**2)

def distance_with_two(a, b):
    return math.sqrt(a**2 + b**2)