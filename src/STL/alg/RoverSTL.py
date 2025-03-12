import os
from typing import List

from matplotlib import pyplot as plt

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
        self.run_name = f"{args.alg}__{args.tag if args.tag != '' else ''}__{args.seed}__{int(time.time())}"

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
        self.sample_batch = 5000
        self.visit_same_states_for = 500

        # Task specific
        self.safe_distance = 0.05  # 0.12 From unity
        self.enough_close_to = 0.08  # Adapted from the paper
        self.wait_for_charging = 3
        self.battery_limit = 2

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
            steps_ahead=self.predict_steps_ahead,
            area_h=10,
            area_w=10,
            squared_area=True,
            beam_angles=self.beam_angles,
            device=self.device,
            close_thres=self.enough_close_to,
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

    def generate_env_with_starting_states(self, area_width: int, area_height: int, n_objects: int, n_states: int):
        obstacles = [
            {"center": [0.75, 0.75], "width": 1.5, "height": 1.5},
            {"center": [4.25, 0.75], "width": 1.5, "height": 1.5},
            {"center": [2.5, 4.25], "width": 2, "height": 1.5},
            {"center": [2.5, 2.5], "width": 0.5, "height": 0.5},
        ]
        obstacles = obstacles + self.simulator.walls()
        obstacles_tensor = self.to_obs_tensor(obstacles)

        min_size = 0.5
        max_size = 4.0
        target_size = 0.2
        charger_size = 0.2

        return self.simulator.generate_random_environments_v2(
            n_states,
            None,
            min_size,
            max_size,
            target_size,
            charger_size,
            0.2,
            obstacles=obstacles_tensor,
            num_chargers=1,
            max_attempts=1,
        )

    def training_loop(self, args):
        # Initialize the logger
        eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
        stl, avoid, always_have_battery, if_low_battery_go_charger, charging, if_enough_battery_go_destiantion = self.generateSTL(self.predict_steps_ahead, battery_limit=self.battery_limit)

        area_width = 10
        area_height = 10
        n_objects = 5

        _, obstacles, _, _ = self.simulator.generate_objects()
        states, obstacles_t, robot_poses, targets, chargers = self.simulator.initialize_x(self.sample_batch, obstacles)
        world_objects_val, states_val, robot_poses_val, targets_val, chargers_val = self.generate_env_with_starting_states(area_width, area_height, n_objects, self.sample_batch // 10)
        obstacles_t = obstacles_t[1:]

        best_accuracy = 0
        for step in range(args.n_total_epochs):
            eta.update()

            if step != 0 and step % self.visit_same_states_for == 0:
                states, obstacles_t, robot_poses, targets, chargers = self.simulator.initialize_x(self.sample_batch, obstacles)
                obstacles_t = obstacles_t[1:]

            states = states.detach()
            control = self.rover_policy(states)

            estimated_next_states, pos_array = self.dynamics(
                obstacles_t,
                states,
                robot_poses,
                targets,
                chargers,
                control,
                include_first=True,
            )
            score = stl(estimated_next_states, self.smoothing_factor)[:, :1]
            acc = (stl(estimated_next_states, self.smoothing_factor, d={"hard": True})[:, :1] >= 0).float()
            acc_avg = torch.mean(acc)
            # no mask
            acc = 1
            small_charge = (states[..., 11:12] <= self.battery_limit).float()
            # print(states[..., 11:12])
            # print(f"{torch.sum(small_charge)} / {states[..., 11:12].shape[0]}")

            import torch.nn.functional as F

            # charger_decrease = F.leaky_relu(estimated_next_states[:, self.predict_steps_ahead, 10] - estimated_next_states[:, 1, 10])
            # dist_decrease = F.leaky_relu(estimated_next_states[:, self.predict_steps_ahead, 8] - estimated_next_states[:, 1, 8])
            # dist_charger = charger_decrease + estimated_next_states[:, self.predict_steps_ahead, 10] ** 2
            # dist_target = dist_decrease + estimated_next_states[:, self.predict_steps_ahead, 8] ** 2
            
            # Seems to work
            # charger_decrease = F.relu(estimated_next_states[:, self.predict_steps_ahead, 10] - estimated_next_states[:, 1, 10])
            # dist_decrease = F.relu(estimated_next_states[:, self.predict_steps_ahead, 8] - estimated_next_states[:, 1, 8])
            # dist_charger = charger_decrease + estimated_next_states[:, self.predict_steps_ahead, 10] ** 2
            # dist_target = dist_decrease + estimated_next_states[:, self.predict_steps_ahead, 8] ** 2
            
            # Including all steps
            charger_decrease = F.relu(estimated_next_states[:, :, 10] - states.unsqueeze(dim=1)[:, :, 10])
            dist_decrease = F.relu(estimated_next_states[:, :, 8] -  states.unsqueeze(dim=1)[:, :, 8])
            dist_charger = charger_decrease + estimated_next_states[:, :, 10] ** 2
            dist_target = dist_decrease + estimated_next_states[:, :, 8] ** 2

            # Old loss
            # dist_charger = estimated_next_states[:, :, 10]
            # dist_target = estimated_next_states[:, :, 8]
            dist_target_charger_loss = torch.mean((dist_charger * small_charge + dist_target * (1 - small_charge)) * acc)
            dist_target_charger_loss = dist_target_charger_loss * 0.1

            loss = torch.mean(F.relu(0.5 - score)) + dist_target_charger_loss

            if acc_avg.item() > best_accuracy:
                print(f"Increased accuracy: {acc_avg.item()}")
                # self.rover_policy.save(f"model_testing/env_gen_random_from_paper_{acc_avg.item()}_{step}.pth")
                best_accuracy = acc_avg.item()

            self.optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()

            # # naive method to prevent overfitting

            # TODO: Print best accuracy in the previouss loop
            # TODO: more chargers in training
            # TODO: more steps ahead
            # TODO: remove squared from the distance
            # TODO: smaller map
            # TODO: scale or not velocity

            if step % self.visit_same_states_for == 0:
                print(
                    "%s > %04d loss:%.3f acc:%.20f dist:%.3f dT:%s T:%s ETA:%s"
                    % ("STL TRAINING ", step, loss.item(), acc_avg.item(), dist_target_charger_loss.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str())
                )
                
                

                print(
                    "Avoid: %.3f, Battery: %.3f, Dest: %.3f, Charger: %.3f, Charging: %.3f"
                    % (
                        torch.mean(avoid(estimated_next_states, self.smoothing_factor)[:, :1]).item(),
                        torch.mean(always_have_battery(estimated_next_states, self.smoothing_factor)[:, :1]).item(),
                        torch.mean(if_enough_battery_go_destiantion(estimated_next_states, self.smoothing_factor)[:, :1]).item(),
                        torch.mean(if_low_battery_go_charger(estimated_next_states, self.smoothing_factor)[:, :1]).item(),
                        torch.mean(charging(estimated_next_states, self.smoothing_factor)[:, :1]).item(),
                    )
                )
                
                print(f"Saving with: {acc_avg.item()}")
                self.rover_policy.save(f"model_testing/model-closeness-beta-increased_{acc_avg.item()}_{step}.pth")

                col = 3 * 2
                row = 3 * 2
                bloat = 0.5

                f, ax_list = plt.subplots(
                    row,
                    col,
                    figsize=(16, 12),
                    gridspec_kw={
                        "height_ratios": [2, 1] * (row // 2),
                        "width_ratios": [1] * col,
                    },
                )

                extra = torch.full_like(targets, 0.4)  # Create zeros of required shape
                targets_expanded = torch.cat([targets, extra], dim=1)
                extra = torch.full_like(chargers, 0.4)
                chargers_expanded = torch.cat([chargers, extra], dim=1).unsqueeze(1)

                for i in range(row // 2):
                    for j in range(col):
                        idx = (i * 2) * col + j
                        ax = ax_list[i * 2, j]
                        p = np.array(pos_array).transpose(1, 0, 2)

                        self.simulator.visualize_environment_v2(
                            robot_poses[idx], lidar_scan=states[idx][0:7], world_objects=obstacles_t, target=targets_expanded[idx], chargers=chargers_expanded[idx], poses=p[idx], ax=ax
                        )

                figname = "%s/iter_%05d.png" % ("./images/", step)
                plt.savefig(figname, bbox_inches="tight", pad_inches=0.1)
                plt.close()

            # indices = torch.randperm(states.shape[0])
            # states = states[indices]
            # robot_poses = robot_poses[indices]
            # targets = targets[indices]
            # chargers = chargers[indices]

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
        poses_arr = []

        for ti in range(t):
            new_x, new_poses = self.dynamics_per_step(x, world_objects, poses, tgs, chrs, es_trajectories[:, ti])
            segs.append(new_x)
            x = new_x
            poses_arr.append(new_poses.detach().cpu().numpy())
            poses = new_poses

        return torch.stack(segs, dim=1), poses_arr

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
        return (min_lidar - self.safe_distance) * 100

    def generateSTL(self, steps_ahead: int, battery_limit: float):
        def debug_print(label, func, x):
            value = func(x)
            # print(f"{label}: {value}")
            return value

        avoid0 = Always(0, steps_ahead, AP(lambda x: (x[..., 0] - self.safe_distance) * 100))
        avoid1 = Always(0, steps_ahead, AP(lambda x: (x[..., 1] - self.safe_distance) * 100))
        avoid2 = Always(0, steps_ahead, AP(lambda x: (x[..., 2] - self.safe_distance) * 100))
        avoid3 = Always(0, steps_ahead, AP(lambda x: (x[..., 3] - self.safe_distance) * 100))
        avoid4 = Always(0, steps_ahead, AP(lambda x: (x[..., 4] - self.safe_distance) * 100))
        avoid5 = Always(0, steps_ahead, AP(lambda x: (x[..., 5] - self.safe_distance) * 100))
        avoid6 = Always(0, steps_ahead, AP(lambda x: (x[..., 6] - self.safe_distance) * 100))

        avoid_list = [avoid0, avoid1, avoid2, avoid3, avoid4, avoid5, avoid6]

        avoid = ListAnd(avoid_list)

        at_dest = AP(lambda x: debug_print("Distance to destination", lambda x: self.enough_close_to - x[..., 8], x), comment="Distance to destination")
        at_charger = AP(lambda x: debug_print("Distance to charger", lambda x: self.enough_close_to - x[..., 10], x), comment="Distance to charger")

        if_enough_battery_go_destiantion = Imply(AP(lambda x: debug_print("Battery level > limit", lambda x: x[..., 11] - battery_limit, x)), Eventually(0, steps_ahead, at_dest))
        if_low_battery_go_charger = Imply(AP(lambda x: debug_print("Battery level < limit", lambda x: battery_limit - x[..., 11], x)), Eventually(0, steps_ahead, at_charger))

        always_have_battery = Always(0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x)))

        stand_by = AP(lambda x: debug_print("Stand by (distance from charger)", lambda x: self.enough_close_to - x[..., 10], x), comment="Stand by: agent remains close to charger")
        enough_stay = AP(lambda x: debug_print(f"Stay > {self.wait_for_charging} steps", lambda x: -x[..., 12], x), comment=f"Stay>{self.wait_for_charging} steps")
        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        # stl = if_enough_battery_go_destiantion
        return (
            ListAnd(
                [
                    avoid,
                    always_have_battery,
                    if_low_battery_go_charger,
                    charging,
                    if_enough_battery_go_destiantion,
                ]
            ),
            avoid,
            always_have_battery,
            if_low_battery_go_charger,
            charging,
            if_enough_battery_go_destiantion,
        )

        avoid = Always(0, steps_ahead, AP(lambda x: debug_print("Lidar safety", self.lidar_obs_avoidance_robustness, x), comment="Lidar safety"))

        at_dest = AP(lambda x: debug_print("Distance to destination", lambda x: (self.enough_close_to - x[..., 8]) * 10, x), comment="Distance to destination")
        at_charger = AP(lambda x: debug_print("Distance to charger", lambda x: (self.enough_close_to - x[..., 10]) * 10, x), comment="Distance to charger")

        if_enough_battery_go_destiantion = Imply(AP(lambda x: debug_print("Battery level > limit", lambda x: x[..., 11] - battery_limit, x)), Eventually(0, steps_ahead, at_dest))
        if_low_battery_go_charger = Imply(AP(lambda x: debug_print("Battery level < limit", lambda x: battery_limit - x[..., 11], x)), Eventually(0, steps_ahead, at_charger))

        always_have_battery = Always(0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x)))

        stand_by = AP(lambda x: debug_print("Stand by (distance from charger)", lambda x: (self.enough_close_to - x[..., 10]) * 10, x), comment="Stand by: agent remains close to charger")
        enough_stay = AP(lambda x: debug_print(f"Stay > {self.wait_for_charging} steps", lambda x: -x[..., 12], x), comment=f"Stay>{self.wait_for_charging} steps")

        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        return ListAnd(
            [avoid, always_have_battery, if_low_battery_go_charger, charging, if_enough_battery_go_destiantion]
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
