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
        self.charger_time = 1
        # self.turtlebot3 = TurtleBot3()

        # number of attempts, the network could be fail if you set a number greater than 1 the robot try again to reach the goal
        self.n_episode_test = 1
        self.timer_period = 0.25  # 0.25 seconds
        time.sleep(1)
        self.timer = self.create_timer(self.timer_period, self.control_loop)

    def callback_lidar(self, msg):
        self.turtlebot3.SetLaser(msg)

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)

    def control_loop(self):
        pos, rot = self.turtlebot3.get_odom()

        dist, heading = self.turtlebot3.get_goal_info(pos)

        dist_charger, heading_charger = self.turtlebot3.get_charger_info(pos)

        # print(f"Dist: {dist}, Head: {heading}")

        # if the robot reach the goal or to close at the obstalce, stop
        if dist < 0.05:
            print("Goal reached")
            self.turtlebot3.stop(self.pub)
            quit(0)

        scan = self.turtlebot3.get_scan()
        # print(f"Scan: {scan}")

        self.battery = self.battery - 0.01

        if dist_charger < 0.1:
            self.battery = 1
            print("Recharged")

        # Update scan to match the stl rule in training
        scan = np.array(scan)
        scan += 0.1
        scan = np.clip(scan, a_min=0, a_max=1.0)
        print(scan)

        state = np.concatenate((scan, [heading, dist, heading_charger, dist_charger, self.battery, self.charger_time]))

        # print(state)

        # # print(state)
        # print([heading, dist ])
        state = self.agent.normalize_state(state)
        state_torch = torch.tensor([state], dtype=torch.float32).to(self.device)
        linear_vel, angular_vel = self.agent.plan(state_torch, self.timer_period)

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
