import os
from typing import List

from matplotlib import pyplot as plt

from alg.dynamics import DynamicsSimulator
from .utils import EtaEstimator
from .lib_stl_core import AP, Always, Eventually, Imply, ListAnd, Or
from .stl_network import RoverSTLPolicy
from torch.optim import Adam
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

    def training_loop(self, args):

        eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
        stl, avoid, always_have_battery, if_low_battery_go_charger, charging, if_enough_battery_go_destiantion = self.generateSTL(self.predict_steps_ahead, battery_limit=self.battery_limit)

        _, obstacles, _, _ = self.simulator.generate_objects()
        states, obstacles_t, robot_poses, targets, chargers = self.simulator.initialize_x(self.sample_batch, obstacles)
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

            import torch.nn.functional as F

            # Including all steps
            # no mask
            acc = 1
            small_charge = (states[..., 11:12] <= self.battery_limit).float()
            charger_decrease = F.relu(estimated_next_states[:, :, 10] - states.unsqueeze(dim=1)[:, :, 10])
            dist_decrease = F.relu(estimated_next_states[:, :, 8] - states.unsqueeze(dim=1)[:, :, 8])
            dist_charger = charger_decrease + estimated_next_states[:, :, 10] ** 2
            dist_target = dist_decrease + estimated_next_states[:, :, 8] ** 2

            dist_target_charger_loss = torch.mean((dist_charger * small_charge + dist_target * (1 - small_charge)) * acc)
            dist_target_charger_loss = dist_target_charger_loss * 0.1

            loss = torch.mean(F.relu(0.5 - score)) + dist_target_charger_loss

            if acc_avg.item() > best_accuracy:
                print(f"Increased accuracy: {acc_avg.item()}")
                best_accuracy = acc_avg.item()

            self.optimizer.zero_grad()
            with torch.autograd.detect_anomaly():
                loss.backward()
            self.optimizer.step()

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

                # ---- PLOT training result ---

                col = 3 * 2
                row = 3 * 2

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
        and a single es_trajectories  tgs
        """
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

    def generateSTL(self, steps_ahead: int, battery_limit: float):
        avoid0 = Always(0, steps_ahead, AP(lambda x: (x[..., 0] - self.safe_distance) * 100))
        avoid1 = Always(0, steps_ahead, AP(lambda x: (x[..., 1] - self.safe_distance) * 100))
        avoid2 = Always(0, steps_ahead, AP(lambda x: (x[..., 2] - self.safe_distance) * 100))
        avoid3 = Always(0, steps_ahead, AP(lambda x: (x[..., 3] - self.safe_distance) * 100))
        avoid4 = Always(0, steps_ahead, AP(lambda x: (x[..., 4] - self.safe_distance) * 100))
        avoid5 = Always(0, steps_ahead, AP(lambda x: (x[..., 5] - self.safe_distance) * 100))
        avoid6 = Always(0, steps_ahead, AP(lambda x: (x[..., 6] - self.safe_distance) * 100))

        avoid_list = [avoid0, avoid1, avoid2, avoid3, avoid4, avoid5, avoid6]

        avoid = ListAnd(avoid_list)

        at_dest = AP(lambda x: self.enough_close_to - x[..., 8], comment="Distance to destination")
        at_charger = AP(lambda x: self.enough_close_to - x[..., 10], comment="Distance to charger")

        if_enough_battery_go_destiantion = Imply(AP(lambda x: x[..., 11] - battery_limit), Eventually(0, steps_ahead, at_dest))
        if_low_battery_go_charger = Imply(AP(lambda x: battery_limit - x[..., 11]), Eventually(0, steps_ahead, at_charger))

        always_have_battery = Always(0, steps_ahead, AP(lambda x: lambda x: x[..., 11]))

        stand_by = AP(lambda x: self.enough_close_to - x[..., 10], comment="Stand by: agent remains close to charger")
        enough_stay = AP(lambda x:  -x[..., 12], comment=f"Stay>{self.wait_for_charging} steps")
        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

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

    def seed_everything(self, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
