import math
import time

import numpy as np


from nav_msgs.msg import Odometry
import rclpy
from geometry_msgs.msg import Twist, Point
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
import rclpy.qos
import tf_transformations
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from math import pi, atan2


class TurtleBot3:
    def __init__(self):
        # qos = QoSProfile(depth=10)
        # self.node = rclpy.create_node('turtlebot3_DDQN_node')
        self.lidar_msg = LaserScan()
        self.odom_msg = Odometry()
        # set your desired goal:
        self.goal_x, self.goal_y = -1.868001, -3.722999  # this is for simulation change for real robot
        self.charger_x, self.charger_y = -1.932, -1.701999  # this is for simulation change for real robot

        # linear velocity is costant set your value
        self.linear_velocity = 0.2  # to comment

        # ang_vel is in rad/s, so we rotate 5 deg/s [action0, action1, action2]
        self.angular_velocity = [0.0, -2.20, 2.20]  # to comment

        # self.r = rclpy.spin_once(self.node,timeout_sec=0.25)

        print("Robot initialized")

    def SetLaser(self, msg):
        self.lidar_msg = msg

    def SetOdom(self, msg):
        self.odom_msg = msg

    def stop_tb(self):
        self.pub.publish(Twist())

    def get_odom(self):
        # read odometry pose from self.odom_msg (for domuentation check http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        # save in point variable the position
        # save in rot variable the rotation

        point = self.odom_msg.pose.pose.position
        rot = self.odom_msg.pose.pose.orientation
        self.rot_ = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

        return point, np.rad2deg(self.rot_[2]) / 180

    def get_scan(self):
        scan_val = []
        # read lidar msg from self.lidar_msg and save in scan variable
        scan = self.lidar_msg.ranges
        rays = 7

        for i in range(len(scan)):  # cast limit values (0, Inf) to usable floats
            if scan[i] == float("Inf"):
                scan[i] = 3.5
            elif math.isnan(scan[i]):
                scan[i] = 0
            scan[i] = scan[i] if scan[i] <= 1 else 1.0

        # get the rays like training if the network accept 3 (7) rays you have to keep the same number
        # I suggesto to create a list of desire angles and after get the rays from the ranges list
        # save the result in scan_val list
        # if(len(scan) > 180 ): return [0,0]

        desired = [
            -90,
            -60,
            -30,
            0,
            30,
            60,
            90,
        ]
        for desire in desired:
            scan_val.append(scan[desire])

        # acceptable = scan[0:91] + scan[ -90: -1]
        # print(len(acceptable))
        # total_ray_count = len(acceptable)
        # step = total_ray_count // (rays +1)
        # for i in range(0, rays):
        #     scan_val.append(min(acceptable[i * step: (i+1) *step]))

        print(scan_val)

        return scan_val

    def get_goal_info(self, tb3_pos):
        delta_y = self.goal_y - tb3_pos.y
        delta_x = self.goal_x - tb3_pos.x

        distance = np.sqrt(delta_x**2 + delta_y**2)
        heading = math.atan2(delta_y, delta_x) - self.rot_[2]
        # Map to [-np.pi, np.pi]
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        return distance / 5, heading

    def get_charger_info(self, tb3_pos):
        delta_y = self.charger_y - tb3_pos.y
        delta_x = self.charger_x - tb3_pos.x

        distance = np.sqrt(delta_x**2 + delta_y**2)
        heading = math.atan2(delta_y, delta_x) - self.rot_[2]
        # Map to [-np.pi, np.pi]
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        
        return distance / 5,  heading

    def move(self, action, pub):
        twist = Twist()
        
        linear_vel = min(float(action[0]), 0.2)
        angular_vel = -float(action[1])

        twist.linear.x = linear_vel
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_vel
        
        print(f"Action -> linear vel: {linear_vel}, angular vel: {angular_vel}")

        pub.publish(twist)
