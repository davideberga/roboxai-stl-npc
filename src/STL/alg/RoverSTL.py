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


def update_head_distance_after_motion(d, alpha, v, theta, device, t=0.005):
    """
    Simulate Unity-like motion (translation then rotation) in Python using PyTorch.
    
    Parameters:
        d (torch.Tensor): Initial distance of the point from the robot.
        alpha (torch.Tensor): Initial angle of the point relative to the robot (in radians).
                              Here, alpha=0 means the point is straight ahead (positive y direction).
        v (torch.Tensor): Velocity (units per second).
        turn_angle (torch.Tensor): Rotation applied (in radians) over time t.
        device: The torch device.
        t (float or torch.Tensor): Time duration (default is 1).
    
    Returns:
        tuple: (new_distance, new_angle) as torch.Tensors, where new_angle is relative to the new forward.
    """
    
    # Convert parameters to tensors on the proper device.
    theta = torch.as_tensor(theta, dtype=torch.float32).to(device)
    t = torch.as_tensor(t, dtype=torch.float32).to(device)
    
    # Squeeze in case alpha has extra dimensions.
    alpha = alpha.squeeze()

    # Convert polar to Cartesian using Unity's convention: 
    #   x is lateral (right), z (here y) is forward.
    x_p = d * torch.sin(alpha)  # lateral component
    y_p = d * torch.cos(alpha)  # forward component (corresponds to Unity's z-axis)

    # Robot's forward motion: in Unity, the translation is along the z-axis.
    s = v * t  # distance traveled
    # If theta = 0, we expect pure forward motion.
    delta_x_r = s * torch.sin(theta)
    delta_y_r = s * torch.cos(theta)

    # Subtract the robot's displacement from the point's position.
    # (Think of the point moving backward relative to the robot.)
    x_p_new = x_p - delta_x_r
    y_p_new = y_p - delta_y_r

    # Compute the new distance and heading.
    d_new = torch.sqrt(x_p_new**2 + y_p_new**2)
    # Since forward is along y, measure the angle with:
    new_alpha = torch.atan2(x_p_new, y_p_new)  # 0 means straight ahead

    # Optionally, clamp the distance if desired.
    d_new = torch.clamp(d_new, 0, 1)
    
    return d_new, new_alpha


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
        self.sample_batch = 10000

        # Task specific
        self.safe_distance = 0.1
        self.enough_close_to = 0.01
        self.wait_for_charging = 1

        self.rover_vmax = 0
        self.rover_vmin = 1
        angles = np.array([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2])
        self.lidar_angles = torch.tensor(np.tile(angles, (self.sample_batch, 1))).unsqueeze(0).to(self.device)

    # def training_loop(self, args):
    #     # Initialize the logger
    #     self.sample_batch = 1
    #     eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
    #     logger = create_logger(self.run_name, args)
    #     logger_dict = {"reward": [], "success": [], "step": [], "cost": []}

    #     # Reset to start a new training
    #     state = self.env.reset()

    #     battery_limit = 0.1
    #     stl = self.generateSTL(self.predict_steps_ahead, battery_limit=0.1)

    #     # Iterate the training loop over multiple episodes
    #     for step in range(args.n_total_epochs):
    #         eta.update()
    #         # states = []
    #         # for sample in range(self.sample_batch):
    #         #     state = self.env.reset()
    #         #     states.append(state)

    #         # np.savez("states.npz", states=np.array(states))
    #         # return

    #         # loaded = np.load("states.npz")
    #         # print("loaded")
    #         # Create a tensor from the list of states
    #         # df = int(self.sample_batch)
    #         # print(loaded["states"][a * df:(a +1) *df])
    #         # state_torch = torch.tensor(loaded["states"][a * df : (a + 1) * df], dtype=torch.float32).to(self.device).detach()
            
    #         self.optimizer.zero_grad()
            
    #         state_torch = torch.tensor([state], dtype=torch.float32).to(self.device).detach()
    #         control = self.rover_policy(state_torch)
    #         control_numpy = control.cpu().detach().numpy()
            
            
    #         first_control_trajectory = control_numpy[0]
    #         planned_actual_states = []
    #         for c in first_control_trajectory:
    #             state, _, _ , _ = self.env.step(c)
    #             time.sleep(0.25)
    #             planned_actual_states.append(state)

                
    #         estimated =  self.dynamics(state_torch, control, include_first=False)[0][5] 
    #         actual_states_torch = torch.tensor([planned_actual_states], dtype=torch.float32, device=self.device).detach()
    #         done = actual_states_torch[0][5]
            
    #         difference = torch.mean(torch.pow(estimated - done, 2))
    #         print(f"Difference {difference}")
            
    #         continue

    #         # print(state_torch)
    #         # print("loaded to cuda")

            

    #         # tensor([8.5000e-01, 9.8150e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 9.8150e-01,
    #         #     8.5000e-01, 4.5211e-01, 6.2520e-01, 6.2964e-01, 5.4233e-01, 5.0000e+02,
    #         #     5.0000e+00], dtype=torch.float64)
    #         # control = self.rover_policy(state_torch)
    #         # estimated_next_states = self.dynamics(state_torch, control, include_first=False)

    #         # STL score on the estimated next states
    #         score = stl(planned_actual_states_torch, self.smoothing_factor)[:, :1]
    #         acc = (stl(planned_actual_states_torch, self.smoothing_factor, d={"hard": True})[:, :1] >= 0).float()
    #         acc_avg = torch.mean(acc)

    #         small_charge = (state_torch[..., 11:12] <= battery_limit).float()
    #         # Initial distance - final distance
    #         # At the end of the planned steps the distance should be decreased
    #         dist_charger = state_torch[:, 10] - planned_actual_states_torch[:, 9, 10]
    #         dist_target = state_torch[:, 8] - planned_actual_states_torch[:, 9, 8]

    #         # TODO: Check this loss
    #         dist_target_charger_loss = torch.mean((dist_charger * small_charge + dist_target * (1 - small_charge)) * acc)
    #         # head_target_loss = - torch.abs(state_torch[:, -1, 7] -  estimated_next_states[:, -1, 7])

    #         loss = torch.mean(self.relu(0.5 - score)) + dist_target_charger_loss
    #         loss.backward()
    #         self.optimizer.step()

    #         print(
    #             "%s > %03d  loss:%.3f acc:%.20f dist:%.3f dT:%s T:%s ETA:%s"
    #             % ("STL TRAINING ", step, loss.item(), acc_avg.item(), dist_target_charger_loss.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str())
    #         )
    
    
    def generate_dataset(self, args):
        for _ in range(50):
            states = []
            for sample in range(50000):
                state = self.env.reset()
                states.append(state)

            np.savez(f"states-{int(time.time())}.npz", states=np.array(states))
        return
            
    def training_loop(self, args):
        # Initialize the logger
        eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
        logger = create_logger(self.run_name, args)
        logger_dict = {"reward": [], "success": [], "step": [], "cost": []}

        # Reset to start a new training
        state = self.env.reset()

        battery_limit = 0.7
        stl = self.generateSTL(self.predict_steps_ahead, battery_limit=battery_limit)

        # Iterate the training loop over multiple episodes
        for step in range(args.n_total_epochs):
            eta.update()
            # states = []
            # for sample in range(self.sample_batch):
            #     state = self.env.reset()
            #     states.append(state)

            # np.savez("states.npz", states=np.array(states))
            # return

            loaded = np.load("states.npz")
            print("loaded")
            # Create a tensor from the list of states
            for a in range(5):
                df = int(self.sample_batch)
                print(loaded["states"][a * df:(a +1) *df][0])
                state_torch = torch.tensor(loaded["states"][a * df : (a + 1) * df], dtype=torch.float32).to(self.device).detach()
                
                self.optimizer.zero_grad()
                
                # state_torch = torch.tensor([state], dtype=torch.float32).to(self.device).detach()
                # control = self.rover_policy(state_torch)
                # control_numpy = control.cpu().detach().numpy()
                # print(state_torch[0])
                
                
                # first_control_trajectory = control_numpy[0]
                # planned_actual_states = []
                # for c in first_control_trajectory:
                #     state, _, _ , _ = self.env.step(control_numpy[0][0])
                #     print("Action done")
                #     planned_actual_states.append(state)
                
                # print("Estimated: ")
                # self.print_beautified_state(self.dynamics(state_torch, control, include_first=False)[0][0])   
                
                
                # planned_actual_states_torch = torch.tensor([planned_actual_states], dtype=torch.float32, device=self.device).detach()
                # print("Actual: ")
                # self.print_beautified_state(planned_actual_states_torch[0][0])
                # return 

                # print(state_torch)
                # print("loaded to cuda")

                

                # tensor([8.5000e-01, 9.8150e-01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 9.8150e-01,
                #     8.5000e-01, 4.5211e-01, 6.2520e-01, 6.2964e-01, 5.4233e-01, 5.0000e+02,
                #     5.0000e+00], dtype=torch.float64)
                control = self.rover_policy(state_torch)
                estimated_next_states = self.dynamics(state_torch, control, include_first=False)

                # STL score on the estimated next states
                score = stl(estimated_next_states, self.smoothing_factor)[:, :1]
                acc = (stl(estimated_next_states, self.smoothing_factor, d={"hard": True})[:, :1] >= 0).float()
                acc_avg = torch.mean(acc)
                
                if acc_avg > 0.8:
                    self.rover_policy.save('exap_model.pth')
                    print(f"Saving with: {acc_avg}")
                    return 

                small_charge = (state_torch[..., 11:12] <= battery_limit).float()
                # Initial distance - final distance
                # At the end of the planned steps the distance should be decreased
                dist_charger = state_torch[:, 10] - estimated_next_states[:, 9, 10]
                dist_target = state_torch[:, 8] - estimated_next_states[:, 9, 8]

                # TODO: Check this loss
                dist_target_charger_loss = torch.mean((dist_charger * small_charge + dist_target * (1 - small_charge)) * acc)
                # head_target_loss = - torch.abs(state_torch[:, -1, 7] -  estimated_next_states[:, -1, 7])

                loss = torch.mean(self.relu(0.5 - score)) + dist_target_charger_loss
                loss.backward()
                self.optimizer.step()

                print(
                    "%s > %03d  loss:%.3f acc:%.20f dist:%.3f dT:%s T:%s ETA:%s"
                    % ("STL TRAINING ", step, loss.item(), acc_avg.item(), dist_target_charger_loss.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str())
                )
            
    def print_beautified_state(self, x0):
        x0 = x0.cpu().detach().tolist()
        print(f"LIDAR: {x0[:7]}")
        print(f"TARGET : {x0[7:9]}")
        print(f"CHARGER : {x0[9:11]}")
        print(f"BATTERY  : {x0[11:]}")

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
        es_lidar_distances, _ = update_head_distance_after_motion(lidar_distances, self.lidar_angles, predicted_velocity.view(-1, 1), predicted_theta.view(-1, 1), self.device)

        # Estimate new target head and distance
        es_target_dist, es_target_head = update_head_distance_after_motion(x[:, 8], x[:, 7], predicted_velocity, predicted_theta, self.device)

        # Estimate new nearest charger head and distance
        es_charger_dist, es_charger_head = update_head_distance_after_motion(x[:, 10], x[:, 9], predicted_velocity, predicted_theta, self.device)

        # Estimate new battery level and hold time
        es_charger_time = 1
        es_battery_time = x[:, 11] - 0.01

        # Create a mask where the condition is met
        mask = es_charger_dist < 0.01

        # Update the battery time and charger time where the mask is True
        es_battery_time = torch.where(mask, x[:, 11] + 0.1, es_battery_time)
        es_charger_time = torch.where(mask, x[:, 12] - 0.01, es_charger_time)

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

    def generateSTL(self, steps_ahead: int, battery_limit: float):
        avoid = Always(0, steps_ahead, AP(lambda x: self.lidar_obs_avoidance_robustness(x), comment="Lidar safety"))
        at_dest = AP(lambda x: self.enough_close_to - x[..., 8], comment="Distance to destination")
        at_charger = AP(lambda x: self.enough_close_to - x[..., 10], comment="Distance to charger")

        if_enough_battery_go_destiantion = Imply(AP(lambda x: x[..., 11] - battery_limit), Eventually(0, steps_ahead, at_dest))
        if_low_battery_go_charger = Imply(AP(lambda x: battery_limit - x[..., 11]), Eventually(0, steps_ahead, at_charger))
        always_have_battery = Always(0, steps_ahead, AP(lambda x: x[..., 11]))

        stand_by = AP(lambda x: x[..., 10] - self.enough_close_to, comment="Stand by: agent remains close to charger")
        enough_stay = AP(lambda x: -x[..., 12], comment=f"Stay>{self.wait_for_charging} steps")
        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        return ListAnd([avoid]) #, charging, if_enough_battery_go_destiantion, if_low_battery_go_charger])

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
