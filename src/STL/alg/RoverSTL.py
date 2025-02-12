import os
from typing import List

from .stl_network import RoverSTLPolicy
from .log_utils import create_logger
from torch.optim import Adam
import numpy as np
import time
import warnings
import torch
import random

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def update_head_distance_after_motion(d, alpha, v, theta, t=1):
    """
    Calculate the new relative distance and angle of a point from the robot after motion.

    Parameters:
    d (float): Initial distance of the point from the robot
    alpha (float): Initial angle of the point relative to the robot (in radians)
    v (float): Velocity of the robot
    theta (float): Motion direction of the robot relative to its heading (in radians)
    t (float): Time duration of motion

    Returns:
    tuple: (new distance, new angle in radians)
    """
    # Initial relative position of the point
    x_p, y_p = d * np.cos(alpha), d * np.sin(alpha)

    # Robot displacement
    s = v * t  # Distance traveled by the robot
    delta_x_r, delta_y_r = s * np.cos(theta), s * np.sin(theta)

    # New relative position of the point
    x_p_new, y_p_new = x_p - delta_x_r, y_p - delta_y_r
    d_new = np.sqrt(x_p_new**2 + y_p_new**2)
    alpha_new = np.arctan2(y_p_new, x_p_new)

    return d_new, alpha_new


class RoverSTL:
    def __init__(self, env, args):
        self.seed = args.seed
        if self.seed is None:
            self.seed = np.random.randint(0, 1000)
        self.seed_everything(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.env = env
        self.run_name = f"{args.alg}__{args.tag if args.tag != '' else ''}__{args.seed}__{int(time.time())}"
        self.input_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space

        # Training hyperparameters
        self.memory_size = args.memory_size
        self.gamma = args.gamma
        self.epoch = args.n_epochs
        self.batch_size = args.batch_size
        self.eps_decay = args.eps_decay
        self.tau = args.tau
        self.layers = args.n_layers
        self.nodes = args.layer_size

        self.rover_policy = RoverSTLPolicy().to(self.device)
        self.optimizer = Adam(self.rover_policy.parameters(), lr=args.lr)
        
        
        self.rover_vmax = 0
        self.rover_vmin = 1
        self.lidar_angles = [
            -np.pi / 2,
            -np.pi / 3,
            -np.pi / 6,
            0,
            np.pi / 6,
            np.pi / 3,
            np.pi / 2,
        ]

    def loop(self, args):
        # Initialize the logger
        logger = create_logger(self.run_name, args)
        logger_dict = {"reward": [], "success": [], "step": [], "cost": []}

        # Reset to start a new training
        state = self.env.reset()

        # Iterate the training loop over multiple episodes
        for step in range(args.n_total_epochs):
            state_torch = torch.tensor([state]).float().to(self.device).detach()
            # tensor([8.5000e-01, 9.8150e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 9.8150e-01,
            #     8.5000e-01, 4.5211e-01, 6.2520e-01, 6.2964e-01, 5.4233e-01, 5.0000e+02,
            #     5.0000e+00], dtype=torch.float64)
            u = self.rover_policy(state_torch)
            seg = self.dynamics(state_torch.cpu(), u.detach().cpu().numpy(), include_first=True)
            print(seg)
            return

            # # Initialize the values for the logger
            # logger_dict["reward"].append(0)
            # logger_dict["success"].append(0)
            # logger_dict["step"].append(0)
            # logger_dict["cost"].append(0)

            # # Main loop of the current episode
            # while True:
            #     # Select the action, perform the action and save the returns in the memory buffer
            #     action, action_prob = self.get_action(state)
            #     new_state, reward, done, info = self.env.step(action)
            #     self.memory_buffer.append(
            #         [state, action, action_prob, reward, new_state, done]
            #     )

            #     # Update the dictionaries for the logger and the trajectory
            #     logger_dict["reward"][-1] += reward
            #     logger_dict["step"][-1] += 1
            #     logger_dict["success"][-1] = 1 if info["target_reached"] else 0

            #     # Call the update rule of the algorithm
            #     self.network_update_rule(done)

            #     # Exit if terminal state and eventually update the state
            #     if done:
            #         break
            #     state = new_state

            # # after each episode log and print information
            # last_n = min(len(logger_dict["reward"]), 100)
            # reward_last_100 = logger_dict["reward"][-last_n:]
            # cost_last_100 = logger_dict["cost"][-last_n:]
            # step_last_100 = logger_dict["step"][-last_n:]
            # success_last_100 = logger_dict["success"][-last_n:]

            # record = {
            #     "Episode": episode,
            #     "Step": int(np.mean(step_last_100)),
            #     "Avg_Cost": int(np.mean(cost_last_100) * 100),
            #     "Avg_Success": int(np.mean(success_last_100) * 100),
            #     "Avg_Reward": np.mean(reward_last_100),
            # }
            # logger.write(record)

            # print(f"(DDQN) Ep: {episode:5}", end=" ")
            # print(
            #     f"reward: {logger_dict['reward'][-1]:5.2f} (last_100: {np.mean(reward_last_100):5.2f})",
            #     end=" ",
            # )
            # print(f"cost_last_100: {int(np.mean(cost_last_100))}", end=" ")
            # print(f"step_last_100 {int(np.mean(step_last_100)):3d}", end=" ")
            # if "eps_greedy" in self.__dict__.keys():
            #     print(f"eps: {self.eps_greedy:3.2f}", end=" ")
            # if "sigma" in self.__dict__.keys():
            #     print(f"sigma: {self.sigma:3.2f}", end=" ")
            # print(f"success_last_100 {int(np.mean(success_last_100) * 100):4d}%")

            # if args.wandb_log:
            #     wandb.log(record)

            # # save model if we reach avg_success greater than 78%
            # if int(np.mean(success_last_100) * 100) >= 79:
            #     self.actor.save(
            #         f"models/DDQN_id{self.seed}_ep{episode}_success{int(np.mean(success_last_100) * 100)}.h5"
            #     )

    def dynamics(self, x0, es_trajectories, include_first=False):
        """Estimate planning of T steps ahead

        Args:
            x0 (float[]): the planning start at this state given by the environment
            control (float[][]): batched trajectories planned by nn of T steps ahead composed by robot actions (v, theta) (n, T)
            include_first (bool, optional): if include the first state. Defaults to False.

        Returns:
            _type_: _description_
        """

        t = es_trajectories.shape[1]  # Extract actions predicted
        x = x0.clone()
        segs = [x0] if include_first else []
        
        for ti in range(t):
            new_x = self.dynamics_per_step(x, es_trajectories[:, ti])
            segs.append(new_x)
            x = new_x

        return torch.stack(segs, dim=1)

    def dynamics_per_step(self, x, es_trajectories):
        """Computes how the system state changes in one time step given the current state x
        and a single es_trajectories  u

        Args:
            x (_type_): _description_
            es_trajectories (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_x = torch.zeros_like(x)

        # The new state must have these values estimated
        # [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]

        predicted_velocity = es_trajectories[:, 0]
        predicted_theta = es_trajectories[:, 1]
        # Normalize velocity to the rover safe range
        predicted_velocity = (
            predicted_velocity * (self.rover_vmax - self.rover_vmin) + self.rover_vmin
        )

        # Estimate new lidar distances
        lidar_distances = x[:, :7]
        es_lidar_distances, _ = update_head_distance_after_motion(
            lidar_distances, self.lidar_angles, predicted_velocity, predicted_theta
        )
        
        # Estimate new target head and distance
        es_target_dist, es_target_head = update_head_distance_after_motion(
            x[:, 8], x[:, 7], predicted_velocity, predicted_theta
        )
        
        # Estimate new nearest charger head and distance
        es_charger_dist, es_charger_head = update_head_distance_after_motion(
            x[:, 10], x[:, 9], predicted_velocity, predicted_theta
        )
        
        # Estimate new battery level and hold time
        es_charger_time = 5
        es_battery_time = x[:, 11] -1
        # If near charger:
        if es_charger_dist < 0.1:
            es_battery_time = x[:, 11] + 10
            es_charger_time = x[:, 12] - 1
             

        new_x[:, 0:7] = es_lidar_distances
        new_x[:, 7] = es_target_head
        new_x[:, 8] = es_target_dist
        new_x[:, 9] = es_charger_head
        new_x[:, 10] = es_charger_dist
        new_x[:, 11] = es_battery_time
        new_x[:, 12] = es_charger_time
         
        return new_x

    def generateSTL():
        avoid_func = lambda y, y1, y2: Always(
            0,
            args.nt,
            And(
                AP(lambda x: -pts_in_poly(x[..., :2], y, args, obses_1=y1, obses_2=y2)),
                AP(
                    lambda x: args.seg_gain
                    * -seg_int_poly(x[..., :2], y, args, device, obses_1=y1, obses_2=y2)
                ),
            ),
        )

        avoids = []
        for obs, obs1, obs2 in zip(objs[1:], objs_t1[1:], objs_t2[1:]):
            avoids.append(avoid_func(obs, obs1, obs2))
        if args.norm_ap:
            at_dest = AP(
                lambda x: args.close_thres
                - torch.norm(x[..., 0:2] - x[..., 2:4], dim=-1)
            )
            at_charger = AP(
                lambda x: args.close_thres
                - torch.norm(x[..., 0:2] - x[..., 4:6], dim=-1)
            )
        else:
            at_dest = AP(
                lambda x: -((x[..., 0] - x[..., 2]) ** 2)
                - (x[..., 1] - x[..., 3]) ** 2
                + args.close_thres**2
            )
            at_charger = AP(
                lambda x: -((x[..., 0] - x[..., 4]) ** 2)
                - (x[..., 1] - x[..., 5]) ** 2
                + args.close_thres**2
            )

        battery_limit = args.dt * args.nt

        reach0 = Imply(
            AP(lambda x: x[..., 6] - battery_limit), Eventually(0, args.nt, at_dest)
        )
        battery = Always(0, args.nt, AP(lambda x: x[..., 6]))

        reaches = [reach0]
        emergency = Imply(
            AP(lambda x: battery_limit - x[..., 6]), Eventually(0, args.nt, at_charger)
        )
        if args.hold_t > 0:
            if args.norm_ap:
                stand_by = AP(
                    lambda x: 0.1 - torch.norm(x[..., 0:2] - x[..., 0:1, 0:2], dim=-1),
                    comment="Stand by",
                )
            else:
                stand_by = AP(
                    lambda x: 0.1**2
                    - (x[..., 0] - x[..., 0:1, 0]) ** 2
                    - (x[..., 1] - x[..., 0:1, 1]) ** 2,
                    comment="Stand by",
                )
            enough_stay = AP(lambda x: -x[..., 7], comment="Stay>%d" % (args.hold_t))
            hold_cond = Imply(
                at_charger, Always(0, args.hold_t, Or(stand_by, enough_stay))
            )
            hold_cond = [hold_cond]
        else:
            hold_cond = []
        stl = ListAnd([in_map] + avoids + reaches + hold_cond + [battery, emergency])
        return stl

    def seed_everything(self, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
