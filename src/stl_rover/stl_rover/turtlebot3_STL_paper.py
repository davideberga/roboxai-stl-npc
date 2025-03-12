import math
import os
from time import sleep

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from stl_rover.agent import Agent
from stl_rover.turtlebot3 import TurtleBot3
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import rclpy.qos
import numpy as np
import time
import torch


class turtlebot3DQN(Node):
    def __init__(self):
        super().__init__("stl_rover")

        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.turtlebot3 = TurtleBot3()
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.callback_lidar, rclpy.qos.qos_profile_sensor_data)
        self.sub1 = self.create_subscription(Odometry, "/odom", self.callback_odom, 10)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.verbose = True
        self.agent = Agent(self.verbose, 'model_final_paper.ckpt', True,  self.device)
        self.battery = 5
        self.hold_time = 0.6
        
        self.action_sequence = []
        
        self.n_episode_test = 1
        self.timer_period = 1 # 0.25 seconds
        time.sleep(1)
        self.timer = self.create_timer(self.timer_period, self.control_loop)

    def callback_lidar(self, msg):
        self.turtlebot3.SetLaser(msg)

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)

    def control_loop(self):
        # Get the current state information.
        pos, _ = self.turtlebot3.get_odom()
        goal_x, goal_y = self.turtlebot3.get_goal()
        charger_x, charger_y = self.turtlebot3.get_charger()
        heading = self.turtlebot3.get_yaw_radiants()
        
        def remap(x):
            return abs(x)

        # Build the state vector.
        state = np.array((remap(pos.x) , remap(pos.y)  , remap(goal_x), remap(goal_y), remap(charger_x), remap(charger_y), self.battery, self.hold_time))
        state = self.agent.normalize_state(state)
        
        state_torch = torch.tensor([state], dtype=torch.float32).to(self.device)
        state_torch = torch.round(state_torch, decimals=3)

        # If there is no pending sequence, plan a new one.
        if len(self.action_sequence) == 0:
            print(state_torch)
            linear_vel, angular_vel = self.agent.plan_absolute_theta(state_torch, heading, self.timer_period)
            linear_vel = linear_vel.tolist()
            angular_vel = angular_vel.tolist()
            
            self.action_sequence = list(zip(linear_vel, angular_vel))
            self.get_logger().info(f"New action sequence planned: {len(self.action_sequence)} actions")

        # Execute the next planned action
        v, theta = self.action_sequence.pop(0)
        
        # Goal and charger distances
        dist_goal, _ = self.turtlebot3.get_goal_info(pos)
        dist_charger, _ = self.turtlebot3.get_charger_info(pos)

        if dist_goal < 0.05:
            self.get_logger().info("Goal reached")
            self.turtlebot3.stop(self.pub)
            rclpy.shutdown()
            return
        
        if self.battery < 0:
            self.get_logger().info("Battery ended")
            self.turtlebot3.stop(self.pub)
            rclpy.shutdown()
            return

        # Update battery and hold time based on charger distance.
        if dist_charger < 0.12:
            self.battery = min(self.battery + 0.5, 5)
            self.hold_time = max(0, self.hold_time - 0.3)
            self.get_logger().info("Recharging")
        else:
            self.battery -= 0.01

        if self.hold_time < 0.1:
            self.hold_time = 1
            
        # Execute the action.
        self.turtlebot3.move(v, theta, self.pub)


def main(args=None):
    rclpy.init(args=args)

    turtlebot3_DQN = turtlebot3DQN()

    rclpy.spin(turtlebot3_DQN)

    turtlebot3_DQN.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
