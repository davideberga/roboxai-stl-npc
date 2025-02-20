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
        self.rover_policy = RoverSTLPolicy(self.predict_steps_ahead).to(self.device)
        self.optimizer = Adam(self.rover_policy.parameters(), lr=args.lr)
        self.relu = torch.nn.ReLU()
        self.simulator = DynamicsSimulator()
        self.sample_batch = 10000

        # Task specific
        self.safe_distance = 0.05
        self.enough_close_to = 0.05
        self.wait_for_charging = 1

        self.rover_vmax = 0
        self.rover_vmin = 1
        angles = np.array([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2])
        self.lidar_angles = torch.tensor(np.tile(angles, (self.sample_batch, 1))).unsqueeze(0).to(self.device)

        self.beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2]).to(self.device)

    def generate_dataset(self, args):
        area_width = 10
        area_height = 10
        n_objects = 5
        n_states = 50000
        for i in range(5):
            print(f"Starting {i}")
            world_objects, states, robot_poses, targets, chargers = self.generate_env_with_starting_states(area_width, area_height, n_objects, n_states)
            print(world_objects)
            np.savez(
                f"dataset/dynamics-states-{int(time.time())}.npz",
                world_objects=world_objects,
                states=states.cpu().numpy(),
                robot_poses=robot_poses.cpu().numpy(),
                targets=targets.cpu().numpy(),
                chargers=chargers.cpu().numpy(),
            )
        return

    def generate_env_with_starting_states_batch(self, area_width: int, area_height: int, n_objects: int, n_states: int):
        """
        Generate an environment with obstacles (world_objects), a target and charger, and a batch
        of valid random starting states.

        Returns:
            world_objects : list of obstacles
            states        : torch.Tensor of shape (n_states, 11) where each state is:
                            [LIDAR_1, ..., LIDAR_7, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER]
            robot_poses   : torch.Tensor of shape (n_states, 3) for the robot's [x, y, heading]
            targets       : torch.Tensor of shape (n_states, 4), repeated target (as [cx, cy, width, height])
            chargers      : torch.Tensor of shape (n_states, 4), repeated charger (as [cx, cy, width, height])
        """
        sim = DynamicsSimulator()

        # Environment parameters.
        min_size = 0.5  # minimum obstacle size.
        max_size = 2.0  # maximum obstacle size.
        target_size = 0.2  # target square size.
        charger_size = 0.2  # charger square size.
        robot_radius = 0.3  # robot's radius.

        # Generate the static environment (world_objects, target, charger, and an initial pose which we ignore)
        world_objects, target, charger, _ = sim.generate_random_environment(
            n_objects,
            area_width,
            area_height,
            min_size,
            max_size,
            target_size,
            charger_size,
            robot_radius,
            obstacles=None,  # Use random obstacles.
            max_attempts=1000,
        )

        # Generate a batch of valid random robot starting poses.
        robot_poses = self.simulator.generate_random_robot_poses(n_states, world_objects, area_width, area_height, robot_radius, target, charger, max_attempts=4)

        # Instead of concatenating in a loop, we accumulate the state vectors in a list.
        state_list = []
        for pose in robot_poses:
            # Convert pose (a tensor of shape (3,)) to a tuple for the helper functions.
            pose_tuple = (pose[0].item(), pose[1].item(), pose[2].item())
            # Compute the lidar scan (assumed shape e.g. (7,))
            lidar_scan = sim.simulate_lidar_scan(pose_tuple, self.beam_angles, world_objects, max_range=10.0)
            # Compute relative destination estimates.
            target_norm, target_angle = sim.estimate_destination(pose_tuple, target, max_distance=10.0)
            charger_norm, charger_angle = sim.estimate_destination(pose_tuple, charger, max_distance=10.0)

            # Build state vector:
            # [LIDAR readings (7,), HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, 1, 1]
            state = torch.cat([lidar_scan, target_angle.reshape(1), target_norm.reshape(1), charger_angle.reshape(1), charger_norm.reshape(1), torch.tensor([1.0]), torch.tensor([1.0])]).float()
            state_list.append(state)

        states = torch.stack(state_list, dim=0)  # shape: (n_states, 11)

        # Prepare repeated target and charger tensors.
        targets = target.reshape(1, -1).repeat(robot_poses.shape[0], 1)
        chargers = charger.reshape(1, -1).repeat(robot_poses.shape[0], 1)

        # Move tensors to the desired device.
        return (world_objects, states.to(self.device).detach(), robot_poses.to(self.device), targets.to(self.device), chargers.to(self.device))

    def generate_env_with_starting_states(self, area_width: int, area_height: int, n_objects: int, n_states: int):
        sim = DynamicsSimulator()

        min_size = 0.5  # minimum obstacle size.
        max_size = 2.0  # maximum obstacle size.
        target_size = 0.2  # size of target square.
        charger_size = 0.2  # size of charger square.
        robot_radius = 0.3  # robot's radius.
        world_objects, target, charger, robot_pose = sim.generate_random_environment(
            n_objects,
            area_width,
            area_height,
            min_size,
            max_size,
            target_size,
            charger_size,
            robot_radius,
            obstacles=None,  # or pass a list of obstacles to override random generation.
            max_attempts=1000,
        )

        states = torch.empty((0,))
        robot_poses = torch.empty((0,))
        targets = torch.empty((0,))
        chargers = torch.empty((0,))

        for _ in range(n_states):
            lidar_scan = sim.simulate_lidar_scan(robot_pose, self.beam_angles, world_objects, max_range=10.0)
            target_distance, target_angle = sim.estimate_destination(robot_pose, target, max_distance=10.0)
            charger_distance, charger_angle = sim.estimate_destination(robot_pose, charger, max_distance=10.0)

            new_state = (
                torch.cat((lidar_scan, target_angle.reshape(1), target_distance.reshape(1), charger_angle.reshape(1), charger_distance.reshape(1), torch.tensor([1]), torch.tensor([1])))
                .float()
                .unsqueeze(0)
            )

            states = torch.cat((states, new_state), dim=0) if states.numel() > 0 else new_state
            robot_pose = torch.tensor(robot_pose)
            robot_poses = torch.cat((robot_poses, robot_pose.reshape(1, -1))) if robot_poses.numel() > 0 else robot_pose.reshape(1, -1)
            targets = torch.cat((targets, target.reshape(1, -1))) if targets.numel() > 0 else target.reshape(1, -1)
            chargers = torch.cat((chargers, charger.reshape(1, -1))) if chargers.numel() > 0 else charger.reshape(1, -1)

            # Obstacles fixed, randomize others
            _, target, charger, robot_pose = sim.generate_random_environment(
                n_objects,
                area_width,
                area_height,
                min_size,
                max_size,
                target_size,
                charger_size,
                robot_radius,
                obstacles=world_objects,
                max_attempts=1000,
            )

        return world_objects, states.to(self.device).detach(), robot_poses.to(self.device), targets.to(self.device), chargers.to(self.device)

    def training_loop(self, args):
        # Initialize the logger
        eta = EtaEstimator(0, args.n_total_epochs, self.print_freq)
        logger = create_logger(self.run_name, args)
        logger_dict = {"reward": [], "success": [], "step": [], "cost": []}

        battery_limit = 0.1
        stl = self.generateSTL(self.predict_steps_ahead, battery_limit=battery_limit)

        area_width = 10
        area_height = 10
        n_objects = 5
        n_states = 500

        for step in range(args.n_total_epochs):
            eta.update()
            
            import glob

            files = glob.glob("dataset/*.npz")  # Change `.txt` to your desired extension
            for path in files:
                dataset = np.load(path, allow_pickle=True)
                print('Loaded')
                for a in range(200):
                    offset = int(len(dataset["states"])/200)
                    world_objects = dataset["world_objects"]
                    states = torch.tensor(dataset["states"][a * offset: (a+1) * offset]).to(self.device).detach()
                    robot_poses = torch.tensor(dataset["robot_poses"][a * offset: (a+1) * offset]).to(self.device)
                    targets = torch.tensor(dataset["targets"][a * offset: (a+1) * offset]).to(self.device)
                    chargers = torch.tensor(dataset["chargers"][a * offset: (a+1) * offset]).to(self.device)

                    # world_objects, states, robot_poses, targets, chargers = self.generate_env_with_starting_states(area_width, area_height, n_objects, n_states)

                    # print(states)

                    control = self.rover_policy(states)
                    # print("Control requires grad:", control.requires_grad)
                    estimated_next_states = self.dynamics(world_objects, states, robot_poses, targets, chargers, control, include_first=True)

                    # print(estimated_next_states)


                    score = stl(estimated_next_states, self.smoothing_factor)[:, :1]
                    acc = (stl(estimated_next_states, self.smoothing_factor, d={"hard": True})[:, :1] >= 0).float()
                    acc_avg = torch.mean(acc)

                    if acc_avg > 0.1:
                        self.rover_policy.save(f"exap_model_only_score_{acc_avg.item()}.pth")
                        print(f"Saving with: {acc_avg.item()}")

                    small_charge = (states[..., 11:12] <= battery_limit).float()
                    # Initial distance - final distance
                    # At the end of the planned steps the distance should be decreased
                    dist_charger = estimated_next_states[:, 9, 10]
                    dist_target = estimated_next_states[:, 9, 8]

                    # TODO: Check this loss
                    dist_target_charger_loss = torch.mean((dist_charger * small_charge + dist_target * (1 - small_charge)) * acc)
                    # dist_target_charger_loss = dist_target_charger_loss * 0.01
                    # head_target_loss = - torch.abs(state_torch[:, -1, 7] -  estimated_next_states[:, -1, 7])

                    # old_params = {name: param.clone().detach() for name, param in self.rover_policy.named_parameters()}
                    
                    loss = torch.mean(self.relu(0.5 - score)) # + dist_target_charger_loss

                    loss.backward()
                    # Update parameters
                    self.optimizer.step()
                    
                    
                    # print("\nParameter updates (difference after update):")
                    # for name, param in self.rover_policy.named_parameters():
                    #     update = old_params[name] - param.data
                    #     print(f"{name}: {update}")

                    
                    
                    # ray_origin = torch.tensor([1.0, 1.0], requires_grad=True)
                    # ray_direction = torch.tensor([1.0, 0.0], requires_grad=True)  # assume normalized
                    # # Use one of your obstacles
                    # dummy_rect = {"center": [5.0, 5.0], "width": 2.0, "height": 2.0}
                    # t = self.simulator.ray_rect_intersection(ray_origin, ray_direction, dummy_rect, max_range=10.0)
                    # t.backward()
                    # print("Gradient of intersection time w.r.t. ray_origin:", ray_origin.grad)
                    # print("Gradient of intersection time w.r.t. ray_direction:", ray_direction.grad)

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

    def dynamics(self, world_objects, states, robot_poses, targets, chargers, es_trajectories, include_first=False):
        """Estimate planning of T steps ahead

        Args:
            x0 (float[]): the planning start at this state given by the environment
            control (float[][]): batched trajectories planned by nn of T steps ahead composed by robot actions (v, theta) (n, T)
            include_first (bool, optional): if include the first state. Defaults to False.

        Returns:
            _type_: _description_
        """

        t = es_trajectories.shape[1]  # Extract actions predicted
        x = states.clone()
        poses = robot_poses.clone()
        tgs = targets.clone()
        chrs = chargers.clone()

        segs = [states] if include_first else []

        for ti in range(t):
            new_x, new_poses = self.dynamics_per_step(x, world_objects, poses, tgs, chrs, es_trajectories[:, ti]) 
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

        return self.simulator.update_state_batch(x, predicted_velocity, predicted_theta, poses, self.beam_angles, world_objects, tgs, chrs, device=self.device)

    def lidar_obs_avoidance_robustness(self, x):
        def smooth_min(lidar_values, alpha=10.0):
            # Computes a smooth approximation of the minimum.
            # To ensure smooth differentiability
            return -(1 / alpha) * torch.log(torch.sum(torch.exp(-alpha * lidar_values), dim=-1))

        lidar_values = x[..., 0:7]
        min_lidar = smooth_min(lidar_values)
        # print(f"Lidar robustness: ${min_lidar - self.safe_distance}")
        return min_lidar - self.safe_distance
    
    

    def generateSTL(self, steps_ahead: int, battery_limit: float):
        
        def debug_print(label, func, x):
            value = func(x)
            # print(f"{label}: {value}")
            return value

        avoid = Always(0, steps_ahead, AP(lambda x: debug_print("Lidar safety", self.lidar_obs_avoidance_robustness, x), comment="Lidar safety"))


        at_dest = AP(lambda x: debug_print("Distance to destination", lambda x: (self.enough_close_to - x[..., 8]) / self.enough_close_to, x), comment="Distance to destination")
        at_charger = AP(lambda x: debug_print("Distance to charger", lambda x: (self.enough_close_to - x[..., 10]) / self.enough_close_to, x), comment="Distance to charger")

        if_enough_battery_go_destiantion = Imply(
            AP(lambda x: debug_print("Battery level > limit", lambda x: x[..., 11] - battery_limit, x)),
            Eventually(0, steps_ahead, at_dest)
        )

        if_low_battery_go_charger = Imply(
            AP(lambda x: debug_print("Battery level < limit", lambda x: battery_limit - x[..., 11], x)),
            Eventually(0, steps_ahead, at_charger)
        )

        always_have_battery = Always(0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x)))

        stand_by = AP(lambda x: debug_print("Stand by (distance from charger)", lambda x: x[..., 10] - self.enough_close_to, x), comment="Stand by: agent remains close to charger")
        enough_stay = AP(lambda x: debug_print(f"Stay > {self.wait_for_charging} steps", lambda x: -x[..., 12], x), comment=f"Stay>{self.wait_for_charging} steps")

        charging = Imply(at_charger, Always(0, self.wait_for_charging, Or(stand_by, enough_stay)))

        return ListAnd([avoid, always_have_battery, charging, if_low_battery_go_charger, if_enough_battery_go_destiantion])


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