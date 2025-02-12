import os
from typing import List

from .utils import EtaEstimator
from .lib_stl_core import AP, Always, Eventually, Imply, ListAnd, Or
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


def update_head_distance_after_motion(d, alpha, v, theta, device, t=1):
    """
    Calculate the new relative distance and angle of a point from the robot after motion,
    using PyTorch tensors.

    Parameters:
        d (torch.Tensor): Initial distance of the point from the robot.
        alpha (torch.Tensor): Initial angle of the point relative to the robot (in radians).
        v (torch.Tensor): Velocity of the robot.
        theta (torch.Tensor): Motion direction of the robot relative to its heading (in radians).
        t (float or torch.Tensor, optional): Time duration of motion. Default is 1.

    Returns:
        tuple: (new_distance, new_angle) as torch.Tensors.
    """

    theta = torch.as_tensor(theta, dtype=torch.float32).to(device)
    t = torch.as_tensor(t, dtype=torch.float32).to(device)

    # Ensure v and theta have the right dimensions (batch_size, 1)

    # Initial relative position of the point
    alpha = alpha.squeeze()

    x_p = d * torch.cos(alpha)
    y_p = d * torch.sin(alpha)

    # Robot displacement
    s = v * t  # Distance traveled by the robot
    delta_x_r = s * torch.cos(theta)
    delta_y_r = s * torch.sin(theta)

    x_p_new = x_p - delta_x_r
    y_p_new = y_p - delta_y_r

    d_new = torch.sqrt(x_p_new**2 + y_p_new**2)
    alpha_new = torch.atan2(y_p_new, x_p_new)

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
        self.smoothing_factor = 500.0
        self.print_freq = 5

        self.predict_steps_ahead = 10
        self.rover_policy = RoverSTLPolicy().to(self.device)
        self.optimizer = Adam(self.rover_policy.parameters(), lr=args.lr)
        self.relu = torch.nn.ReLU()
        self.sample_batch = 10

        # Task specific
        self.safe_distance = 0.1
        self.enough_close_to = 0.01
        self.wait_for_charging = 5

        self.rover_vmax = 0
        self.rover_vmin = 1
        angles = np.array([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2])
        self.lidar_angles = torch.tensor(np.tile(angles, (self.sample_batch, 1))).unsqueeze(0).to(self.device)

    def loop(self, args):
        # Initialize the logger
        eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
        logger = create_logger(self.run_name, args)
        logger_dict = {"reward": [], "success": [], "step": [], "cost": []}

        # Reset to start a new training

        battery_limit = self.predict_steps_ahead
        stl = self.generateSTL(self.predict_steps_ahead, battery_limit)

        # Iterate the training loop over multiple episodes
        for step in range(args.n_total_epochs):
            eta.update()
            states = []
            for sample in range(self.sample_batch):
                state = self.env.reset()  # Assume state is a list or a NumPy array
                states.append(state)  # Append the state itself

            # Create a tensor from the list of states
            state_torch = torch.tensor(states, dtype=torch.float32).to(self.device).detach()

            # tensor([8.5000e-01, 9.8150e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 9.8150e-01,
            #     8.5000e-01, 4.5211e-01, 6.2520e-01, 6.2964e-01, 5.4233e-01, 5.0000e+02,
            #     5.0000e+00], dtype=torch.float64)
            control = self.rover_policy(state_torch)
            estimated_next_states = self.dynamics(state_torch, control, include_first=False)

            # STL score on the estimated next states
            score = stl(estimated_next_states, self.smoothing_factor)[:, :1]
            acc = (stl(estimated_next_states, self.smoothing_factor, d={"hard": True})[:, :1] >= 0).float()
            acc_avg = torch.mean(acc)

            small_charge = (state_torch[..., 11:12] <= battery_limit).float()
            # Initial distance - final distance
            # At the end of the planned steps the distance should be decreased
            dist_charger = state_torch[ :, 10] - estimated_next_states[:, :, 10]
            dist_target = state_torch[ :, 8] - estimated_next_states[:, :, 8]

            # TODO: Check this loss
            dist_target_charger_loss = torch.mean((dist_charger * small_charge + dist_target * (1 - small_charge)) * acc)
            # head_target_loss = - torch.abs(state_torch[:, -1, 7] -  estimated_next_states[:, -1, 7])

            loss = torch.mean(self.relu(0.5 - score)) + dist_target_charger_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 5 == 0:
                print(
                    "%s > %03d  loss:%.3f acc:%.3f dist:%.3f dT:%s T:%s ETA:%s"
                    % ("STL TRAINING ", step, loss.item(), acc_avg.item(), dist_target_charger_loss.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str())
                )

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
        predicted_velocity = predicted_velocity * (self.rover_vmax - self.rover_vmin) + self.rover_vmin

        # Estimate new lidar distances
        lidar_distances = x[:, :7]
        es_lidar_distances, _ = update_head_distance_after_motion(lidar_distances, self.lidar_angles, predicted_velocity.view(-1,1), predicted_theta.view(-1, 1), self.device)

        # Estimate new target head and distance
        es_target_dist, es_target_head = update_head_distance_after_motion(x[:, 8], x[:, 7], predicted_velocity, predicted_theta, self.device)

        # Estimate new nearest charger head and distance
        es_charger_dist, es_charger_head = update_head_distance_after_motion(x[:, 10], x[:, 9], predicted_velocity, predicted_theta, self.device)

        # Estimate new battery level and hold time
        es_charger_time = 5
        es_battery_time = x[:, 11] - 1

        # Create a mask where the condition is met
        mask = es_charger_dist < 0.1

        # Update the battery time and charger time where the mask is True
        es_battery_time = torch.where(mask, x[:, 11] + 10, x[:, 11])
        es_charger_time = torch.where(mask, x[:, 12] - 1, x[:, 12])

        new_x[:, 0:7] = es_lidar_distances
        new_x[:, 7] = es_target_head
        new_x[:, 8] = es_target_dist
        new_x[:, 9] = es_charger_head
        new_x[:, 10] = es_charger_dist
        new_x[:, 11] = es_battery_time
        new_x[:, 12] = es_charger_time

        return new_x

    def lidar_obs_avoidance_robustness(self, x):
        def smooth_min(lidar_values, alpha=10.0):
            # Computes a smooth approximation of the minimum.
            # To ensure smooth differentiability
            return -(1 / alpha) * torch.log(torch.sum(torch.exp(-alpha * lidar_values), dim=-1))

        lidar_values = x[..., 0:7]
        min_lidar = smooth_min(lidar_values)
        return min_lidar - self.safe_distance

    def generateSTL(self, steps_ahead: int, battery_limit: int):
        avoid = Always(0, steps_ahead, AP(lambda x: self.lidar_obs_avoidance_robustness(x), comment="Lidar safety"))
        at_dest = AP(lambda x: self.enough_close_to - x[..., 8], comment="Distance to destination")
        at_charger = AP(lambda x: self.enough_close_to - x[..., 10], comment="Distance to charger")

        if_enough_battery_go_destiantion = Imply(AP(lambda x: x[..., 11] - battery_limit), Eventually(0, steps_ahead, at_dest))
        if_low_battery_go_charger = Imply(AP(lambda x: battery_limit - x[..., 11]), Eventually(0, steps_ahead, at_charger))
        always_have_battery = Always(0, steps_ahead, AP(lambda x: x[..., 11]))

        stand_by = AP(lambda x: x[..., 10] - self.enough_close_to, comment="Stand by: agent remains close to charger")
        enough_stay = AP(lambda x: -x[..., 12], comment=f"Stay>{self.wait_for_charging} steps")
        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        return ListAnd(
            [
                avoid,
                if_enough_battery_go_destiantion,
                charging,
                always_have_battery,
                if_low_battery_go_charger,
            ]
        )

    def seed_everything(self, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
