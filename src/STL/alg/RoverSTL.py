import os
from typing import List

from alg.dynamics import DynamicsSimulator
from .utils import EtaEstimator
from .lib_stl_core import AP, Always, Eventually, Imply, ListAnd, Or
from .stl_network import RoverSTLPolicy
from .log_utils import create_logger
from torch.optim import Adam, Adagrad
import numpy as np
import time
import warnings
import torch
import random

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class RoverSTL:
    def __init__(self, env, args):
        self.seed = args.seed
        if self.seed is None:
            self.seed = np.random.randint(0, 1000)
        self.seed_everything(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.env = env
        self.run_name = (
            f"{args.alg}__{args.tag if args.tag != '' else ''}__{args.seed}__{int(time.time())}"
        )

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
        self.sample_batch = 1000

        self.visit_same_states_for = 100

        # Task specific
        self.safe_distance = 0.12  # From unity
        self.enough_close_to = 0.05
        self.wait_for_charging = 1
        self.battery_limit = 0.4

        self.rover_vmax = 0
        self.rover_vmin = 1
        self.beam_angles = torch.tensor(
            [
                -torch.pi / 2,
                -torch.pi / 3,
                -torch.pi / 4,
                0.0,
                torch.pi / 4,
                torch.pi / 3,
                torch.pi / 2,
            ]
        ).to(self.device)

        self.rover_policy = RoverSTLPolicy(self.predict_steps_ahead).to(self.device)
        self.optimizer = Adam(self.rover_policy.parameters(), lr=args.lr)
        self.relu = torch.nn.ReLU()
        self.simulator = DynamicsSimulator(
            wait_for_charging=self.wait_for_charging,
            area_h=10,
            area_w=10,
            squared_area=True,
            beam_angles=self.beam_angles,
            device=self.device,
        )

    def to_obs_tensor(self, world_objects):
        tmp = []
        for obstacle in world_objects:
            tmp.append(
                [
                    obstacle["center"][0],
                    obstacle["center"][1],
                    obstacle["width"],
                    obstacle["height"],
                ]
            )
        return torch.tensor(tmp).float().to(self.device)

    def generate_env_with_starting_states(
        self, area_width: int, area_height: int, n_objects: int, n_states: int
    ):
        obstacles = [
            {"center": [1.5, 1.5], "width": 3.0, "height": 3.0},
            {"center": [8.5, 1.5], "width": 3.0, "height": 3.0},
            {"center": [5, 8.5], "width": 4.0, "height": 3.0},
            {"center": [5, 5], "width": 1.0, "height": 1.0},
        ]
        obstacles = obstacles + self.simulator.walls()
        obstacles_tensor = self.to_obs_tensor(obstacles)

        min_size = 0.5
        max_size = 4.0
        target_size = 0.2
        charger_size = 0.2

        return self.simulator.generate_random_environments(
            n_states,
            None,
            min_size,
            max_size,
            target_size,
            charger_size,
            0.4,
            obstacles=obstacles_tensor,
            num_chargers=1,
            max_attempts=1,
        )

    def training_loop(self, args):
        # Initialize the logger
        eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
        logger = create_logger(self.run_name, args)
        logger_dict = {"reward": [], "success": [], "step": [], "cost": []}

        stl = self.generateSTL(self.predict_steps_ahead, battery_limit=self.battery_limit)

        area_width = 10
        area_height = 10
        n_objects = 5
        n_states = 500

        for step in range(args.n_total_epochs):
            eta.update()

            world_objects, states, robot_poses, targets, chargers = (
                self.generate_env_with_starting_states(
                    area_width, area_height, n_objects, self.sample_batch
                )
            )

            # Mini loop to reuse the generated states multiple times
            complete_loss, stl_accuracy, task_loss = [], [], []
            for same in range(self.visit_same_states_for):
                control = self.rover_policy(states)

                estimated_next_states = self.dynamics(
                    world_objects[0],
                    states,
                    robot_poses,
                    targets,
                    chargers,
                    control,
                    include_first=True,
                )
                score = stl(estimated_next_states, self.smoothing_factor)[:, :1]
                acc = (
                    stl(estimated_next_states, self.smoothing_factor, d={"hard": True})[:, :1] >= 0
                ).float()
                acc_avg = torch.mean(acc)

                small_charge = (states[..., 11:12] <= self.battery_limit).float()
                # Initial distance - final distance
                # At the end of the planned steps the distance should be decreased
                dist_charger = estimated_next_states[:, :, 10]
                dist_target = estimated_next_states[:, :, 8]

                # TODO: Check this loss
                dist_target_charger_loss = torch.mean(
                    (dist_charger * small_charge + dist_target * (1 - small_charge)) * acc
                )
                dist_target_charger_loss = dist_target_charger_loss
                # head_target_loss = - torch.abs(state_torch[:, -1, 7] -  estimated_next_states[:, -1, 7])

                # old_params = {name: param.clone().detach() for name, param in self.rover_policy.named_parameters()}

                loss = torch.mean(self.relu(0.5 - score)) + dist_target_charger_loss

                complete_loss.append(loss.item())
                stl_accuracy.append(acc_avg.item())
                task_loss.append(dist_target_charger_loss.item())

                self.optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    loss.backward()
                self.optimizer.step()

                # # naive method to prevent overfitting
                indices = torch.randperm(states.shape[0])
                states = states[indices]
                robot_poses = robot_poses[indices]
                targets = targets[indices]
                chargers = chargers[indices]

            mean_acc = np.mean(stl_accuracy)
            if mean_acc > 0.1:
                self.rover_policy.save(
                    f"model_testing/model_correct_dynamics_training_{mean_acc}_{step}.pth"
                )
                print(f"Saving with: {mean_acc}")

            print(
                "%s > %04d loss:%.3f acc:%.20f dist:%.3f dT:%s T:%s ETA:%s"
                % (
                    "STL TRAINING ",
                    step,
                    np.mean(complete_loss),
                    np.mean(stl_accuracy),
                    np.mean(task_loss),
                    eta.interval_str(),
                    eta.elapsed_str(),
                    eta.eta_str(),
                )
            )

    def print_beautified_state(self, x0):
        x0 = x0.cpu().detach().tolist()
        print(f"LIDAR: {x0[:7]}")
        print(f"TARGET : {x0[7:9]}")
        print(f"CHARGER : {x0[9:11]}")
        print(f"BATTERY  : {x0[11:]}")

    def dynamics(
        self,
        world_objects,
        states,
        robot_poses,
        targets,
        chargers,
        es_trajectories,
        include_first=False,
    ):
        """Estimate planning of T steps ahead"""

        t = es_trajectories.shape[1]  # Extract actions predicted
        x = states.clone()
        poses = robot_poses.clone()
        tgs = targets.clone()
        chrs = chargers.clone()

        segs = [states] if include_first else []

        for ti in range(t):
            new_x, new_poses = self.dynamics_per_step(
                x, world_objects, poses, tgs, chrs, es_trajectories[:, ti]
            )
            segs.append(new_x)
            x = new_x
            poses = new_poses

        return torch.stack(segs, dim=1)

    def dynamics_per_step(self, x, world_objects, poses, tgs, chrs, es_trajectories):
        """Computes how the system state changes in one time step given the current state x
        and a single es_trajectories  u
        """

        # The new state must have these values estimated
        # [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]

        predicted_velocity = es_trajectories[:, 0]
        predicted_theta = es_trajectories[:, 1]

        return self.simulator.update_state_batch(
            x,
            predicted_velocity,
            predicted_theta,
            poses,
            world_objects,
            tgs,
            chrs,
        )

    def lidar_obs_avoidance_robustness(self, x):
        def smooth_min(lidar_values, alpha=500.0):
            # Computes a smooth approximation of the minimum.
            # To ensure smooth differentiability
            return -(1 / alpha) * torch.logsumexp(-alpha * lidar_values, dim=-1)

        lidar_values = x[..., 0:7]
        min_lidar = smooth_min(lidar_values)
        # print(f"Smooth min { min_lidar}")
        # print(f"Actual min { torch.min(lidar_values, dim=-1)}")
        # print(f"Lidar robustness: ${min_lidar - self.safe_distance}")

        # Rescale lidar values to give them more importance
        return min_lidar - self.safe_distance

    def generateSTL(self, steps_ahead: int, battery_limit: float):
        def debug_print(label, func, x):
            value = func(x)
            # print(f"{label}: {value}")
            return value

        # avoid = Always(0, steps_ahead, AP(lambda x: debug_print("Lidar safety", self.lidar_obs_avoidance_robustness, x), comment="Lidar safety"))
        # at_dest = AP(lambda x: debug_print("Distance to destination", lambda x: (self.enough_close_to**2) - (x[..., 8] ** 2), x), comment="Distance to destination")
        # at_charger = AP(lambda x: debug_print("Distance to charger", lambda x: (self.enough_close_to**2) - (x[..., 10] ** 2), x), comment="Distance to charger")

        # if_enough_battery_go_destiantion = Imply(AP(lambda x: debug_print("Battery level > limit", lambda x: x[..., 11] - battery_limit, x)), Eventually(0, steps_ahead, at_dest))
        # if_low_battery_go_charger = Imply(AP(lambda x: debug_print("Battery level < limit", lambda x: battery_limit - x[..., 11], x)), Eventually(0, steps_ahead, at_charger))

        # always_have_battery = Always(0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x)))

        # stand_by = AP(lambda x: debug_print("Stand by (distance from charger)", lambda x: (x[..., 10]**2) - (self.enough_close_to**2), x), comment="Stand by: agent remains close to charger")
        # enough_stay = AP(lambda x: debug_print(f"Stay > {self.wait_for_charging} steps", lambda x: -x[..., 12], x), comment=f"Stay>{self.wait_for_charging} steps")

        # charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        avoid = Always(
            0,
            steps_ahead,
            AP(
                lambda x: debug_print("Lidar safety", self.lidar_obs_avoidance_robustness, x),
                comment="Lidar safety",
            ),
        )
        at_dest = AP(
            lambda x: debug_print(
                "Distance to destination", lambda x: self.enough_close_to - x[..., 8], x
            ),
            comment="Distance to destination",
        )
        at_charger = AP(
            lambda x: debug_print(
                "Distance to charger", lambda x: self.enough_close_to - x[..., 10], x
            ),
            comment="Distance to charger",
        )

        if_enough_battery_go_destiantion = Imply(
            AP(
                lambda x: debug_print(
                    "Battery level > limit", lambda x: x[..., 11] - battery_limit, x
                )
            ),
            Eventually(0, steps_ahead, at_dest),
        )
        if_low_battery_go_charger = Imply(
            AP(
                lambda x: debug_print(
                    "Battery level < limit", lambda x: battery_limit - x[..., 11], x
                )
            ),
            Eventually(0, steps_ahead, at_charger),
        )

        always_have_battery = Always(
            0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x))
        )

        stand_by = AP(
            lambda x: debug_print(
                "Stand by (distance from charger)", lambda x: x[..., 10] - self.enough_close_to, x
            ),
            comment="Stand by: agent remains close to charger",
        )
        enough_stay = AP(
            lambda x: debug_print(
                f"Stay > {self.wait_for_charging} steps", lambda x: -x[..., 12], x
            ),
            comment=f"Stay>{self.wait_for_charging} steps",
        )

        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        return ListAnd(
            [avoid, always_have_battery, if_low_battery_go_charger, charging]
        )  # ListAnd([avoid])# if_enough_battery_go_destiantion, always_have_battery, if_low_battery_go_charger]) # missing charging

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
