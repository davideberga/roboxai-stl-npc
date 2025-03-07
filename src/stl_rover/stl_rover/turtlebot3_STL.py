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
        self.agent = Agent(self.verbose, self.device)
        self.battery = 1
        self.hold_time = 1
        
        self.action_sequence = []
        
        self.n_episode_test = 1
        self.timer_period = 0.25 # 0.25 seconds
        time.sleep(1)
        self.timer = self.create_timer(self.timer_period, self.control_loop_one)

    def callback_lidar(self, msg):
        self.turtlebot3.SetLaser(msg)

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)

    def control_loop(self):
        # Get the current state information.
        pos, rot = self.turtlebot3.get_odom()
        dist, heading = self.turtlebot3.get_goal_info(pos)
        dist_charger, heading_charger = self.turtlebot3.get_charger_info(pos)
        scan = np.array(self.turtlebot3.get_scan())
        scan = scan - 0.7
        scan = np.clip(scan, a_min=0, a_max=1.0)
        scan = scan[::-1]

        # Build the state vector.
        state = np.concatenate((
            scan,
            [heading, dist, heading_charger, dist_charger, self.battery, self.hold_time]
        ))
        state = self.agent.normalize_state(state)
        state_torch = torch.tensor([state], dtype=torch.float32).to(self.device)

        # If there is no pending sequence, plan a new one.
        # Note: We do not execute an action in the same tick when planning,
        # ensuring a 0.25-second delay before executing the first action.
        if not self.action_sequence:
            linear_vel, angular_vel = self.agent.plan(state_torch, self.timer_period)
            # Convert tensor outputs to lists, if necessary.
            if isinstance(linear_vel, torch.Tensor):
                linear_vel = linear_vel.tolist()[0]
            if isinstance(angular_vel, torch.Tensor):
                angular_vel = angular_vel.tolist()[0]
            self.action_sequence = list(zip(linear_vel, angular_vel))
            self.get_logger().info(f"New action sequence planned: {len(self.action_sequence)} actions")
            return  # Exit this tick; next tick will execute the first action.

        # Pop and execute the next action from the sequence.
        v, theta = self.action_sequence.pop(0)

        # Check if the goal has been reached.
        if dist < 0.05:
            self.get_logger().info("Goal reached")
            self.turtlebot3.stop(self.pub)
            rclpy.shutdown()
            return

        # Update battery and hold time based on charger distance.
        if dist_charger < 0.1:
            self.battery = min(self.battery + 0.5, 5)
            self.hold_time = max(0, self.hold_time - 0.3)
        else:
            self.battery -= 0.001

        if self.hold_time < 0.1:
            self.hold_time = 1

        if dist_charger < 0.1:
            self.battery = 5
            self.get_logger().info("Recharged")

        # Execute the action.
        self.turtlebot3.move(v, theta, self.pub)
        # self.get_logger().info(f"Executing action: linear_vel = {v}, angular_vel = {theta}")

        
    def control_loop_one(self):
        pos, rot = self.turtlebot3.get_odom()

        dist, heading = self.turtlebot3.get_goal_info(pos)

        dist_charger, heading_charger = self.turtlebot3.get_charger_info(pos)
        if dist < 0.05:
            print("Goal reached")
            self.turtlebot3.stop(self.pub)
            quit(0)

        scan = self.turtlebot3.get_scan()
        # print(f"Scan: {scan}")
        scan = np.array(scan)
        scan = scan[::-1]
        scan -= 0.07
        scan = np.clip(scan, a_min=0, a_max=1.0)

        if dist_charger < 0.1:
                self.battery = min(self.battery + 0.5, 5)
                self.hold_time = max(0, self.hold_time - 0.3)
        else:
            self.battery -= 0.01

        if self.hold_time < 0.1:
            self.hold_time = 1
        
        print(self.battery)

        if dist_charger < 0.1:
            self.battery = 5
            print("Recharged")

        state = np.concatenate((scan, [heading, dist, heading_charger, dist_charger, self.battery, self.hold_time]))
        state = self.agent.normalize_state(state)
        print(state)
        state_torch = torch.tensor([state], dtype=torch.float32).to(self.device)
        linear_vel, angular_vel = self.agent.plan_one(state_torch, self.timer_period)

        # The the first action planned
        self.turtlebot3.move(linear_vel, angular_vel, self.pub)

        # for step_planned in planning:
        #     self.battery = self.battery - 0.01
        #     self.turtlebot3.move(step_planned, self.pub)
        #     time.sleep(1)


def main(args=None):
    rclpy.init(args=args)

    turtlebot3_DQN = turtlebot3DQN()

    rclpy.spin(turtlebot3_DQN)

    turtlebot3_DQN.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
