import os
from time import sleep; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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


class turtlebot3DQN(Node):

    def __init__(self):
        super().__init__('stl_rover')

        
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.turtlebot3 = TurtleBot3()
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.callback_lidar, rclpy.qos.qos_profile_sensor_data)
        self.sub1 = self.create_subscription(Odometry, "/odom", self.callback_odom, 10)

        self.verbose = True
        self.agent = Agent(self.verbose)
        #self.turtlebot3 = TurtleBot3()
        
        # number of attempts, the network could be fail if you set a number greater than 1 the robot try again to reach the goal
        self.n_episode_test = 1
        timer_period = 0.25  # 0.25 seconds
        time.sleep(1)
        self.timer = self.create_timer(timer_period, self.control_loop)
        

    def callback_lidar(self, msg):
        self.turtlebot3.SetLaser(msg)

    def callback_odom(self, msg):
        self.turtlebot3.SetOdom(msg)

    def control_loop(self):
                
        pos, rot = self.turtlebot3.get_odom()
        
        dist, heading = self.turtlebot3.get_goal_info(pos)
        
        #print(f"Dist: {dist}, Head: {heading}")
        
        # if the robot reach the goal or to close at the obstalce, stop
        if dist < 0.1:
            print("Goal reached")
            self.turtlebot3.move(-1, self.pub)
            quit()
            
        
        scan = self.turtlebot3.get_scan()
        # print(f"Scan: {scan[3]}")

        state = np.concatenate((scan, [heading, dist ]))
        # print(state)
        print([heading, dist ])
        state = self.agent.normalize_state(state)
        action = self.agent.select_action(state)
        print("Action -> " + str(action))
        self.turtlebot3.move(action, self.pub)



def main(args=None):
    rclpy.init(args=args)

    turtlebot3_DQN = turtlebot3DQN()
    
    rclpy.spin(turtlebot3_DQN)

    turtlebot3_DQN.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    


    

