from collections import deque
import numpy as np
import tensorflow as tf

from .dynamics import DynamicsSimulator
from .lib_stl_core import AP, Always, Eventually, Imply, ListAnd, Or
from ament_index_python.packages import get_package_share_directory
import torch
import torch.nn as nn
import numpy as np

enough_close_to = 0.08
safe_distance = 0.12
wait_for_charging = 3

def build_relu_nn(input_dim, output_dim, hiddens, activation_fn, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)


class RoverSTLPolicy(nn.Module):
    def __init__(self, steps_ahead: int):
        super(RoverSTLPolicy, self).__init__()
        # input  [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]
        # lidar_scan + 6 elements,
        # - normalized in [0, 1] ==> heading (first) + distance (second) of the TARGET
        # - normalized in [0, 1] ==> heading (first) + distance (second) of the NEAREST CHARGER
        # - Battery time (number of steps) [0, 10000], Charger hold time (number of steps) [0,5]
        #
        # output (rover v theta; astro v theta)
        self.steps_ahead = steps_ahead
        self.dropout = nn.Dropout(0.5)
        self.net = build_relu_nn( 7 + 2 + 2 + 2, self.steps_ahead * 2, [256, 256, 256], activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        control = self.net(x).reshape(num_samples, self.steps_ahead, -1)
        control0 = torch.clip(control[..., 0], 0, 1)
        control1 = torch.clip(control[..., 1], -np.pi, np.pi)
        control_final = torch.stack([control0, control1], dim=-1)
        return control_final
    
    def save(self, path: str):
        torch.save(self.net.state_dict(), path)
        
    def load_eval(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))
        
    def load_eval_paper(self, path: str):
        checkpoint = torch.load(path)
        state_dict_parsed = {k.replace("net.", ""): v for k, v in checkpoint.items()}
        self.net.load_state_dict(state_dict_parsed)
        
def generateSTL(steps_ahead: int, battery_limit: float):
    def debug_print(label, func, x):
        value = func(x)
        # print(f"{label}: {value}")
        return value

    avoid0 = Always(0, steps_ahead, AP(lambda x: (x[..., 0] - safe_distance) * 100))
    avoid1 = Always(0, steps_ahead, AP(lambda x: (x[..., 1] - safe_distance) * 100))
    avoid2 = Always(0, steps_ahead, AP(lambda x: (x[..., 2] - safe_distance) * 100))
    avoid3 = Always(0, steps_ahead, AP(lambda x: (x[..., 3] - safe_distance) * 100))
    avoid4 = Always(0, steps_ahead, AP(lambda x: (x[..., 4] - safe_distance) * 100))
    avoid5 = Always(0, steps_ahead, AP(lambda x: (x[..., 5] - safe_distance) * 100))
    avoid6 = Always(0, steps_ahead, AP(lambda x: (x[..., 6] - safe_distance) * 100))

    avoid_list = [avoid0, avoid1, avoid2, avoid3, avoid4, avoid5, avoid6]

    avoid = ListAnd(avoid_list)

    at_dest = AP(lambda x: debug_print("Distance to destination", lambda x: enough_close_to - x[..., 8], x), comment="Distance to destination")
    at_charger = AP(lambda x: debug_print("Distance to charger", lambda x: enough_close_to - x[..., 10], x), comment="Distance to charger")

    if_enough_battery_go_destiantion = Imply(AP(lambda x: debug_print("Battery level > limit", lambda x: x[..., 11] - battery_limit, x)), Eventually(0, steps_ahead, at_dest))
    if_low_battery_go_charger = Imply(AP(lambda x: debug_print("Battery level < limit", lambda x: battery_limit - x[..., 11], x)), Eventually(0, steps_ahead, at_charger))

    always_have_battery = Always(0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x)))

    stand_by = AP(lambda x: debug_print("Stand by (distance from charger)", lambda x: enough_close_to - x[..., 10], x), comment="Stand by: agent remains close to charger")
    enough_stay = AP(lambda x: debug_print(f"Stay > {wait_for_charging} steps", lambda x: -x[..., 12], x), comment=f"Stay>{wait_for_charging} steps")
    charging = Imply(at_charger, Always(0, wait_for_charging, Or(stand_by, enough_stay)))

    return ListAnd(
        [avoid, always_have_battery, if_low_battery_go_charger, charging, if_enough_battery_go_destiantion]
    )  # ListAnd([avoid])# if_enough_battery_go_destiantion, always_have_battery, if_low_battery_go_charger]) # missing charging


def dynamics(
    sim,
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
        new_x, new_poses = dynamics_per_step(sim, x, world_objects, poses, tgs, chrs, es_trajectories[:, ti])
        segs.append(new_x)
        x = new_x
        poses = new_poses

    return torch.stack(segs, dim=1)


def dynamics_per_step(sim, x, world_objects, poses, tgs, chrs, es_trajectories):
    """Computes how the system state changes in one time step given the current state x
    and a single es_trajectories  u
    """

    # The new state must have these values estimated
    # [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]

    predicted_velocity = es_trajectories[:, 0]
    predicted_theta = es_trajectories[:, 1]

    return sim.update_state_batch(
        x,
        predicted_velocity,
        predicted_theta,
        poses,
        world_objects,
        tgs,
        chrs,
    )


class Agent:
    def __init__(self, verbose, device):

        package_dir = get_package_share_directory("stl_rover")
        model_path = package_dir + "/model_trained/model_0.9167999625205994_102500.pth"
        # load weights of pretrained model
        beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2]).to(device)
        self.sim: DynamicsSimulator = DynamicsSimulator(wait_for_charging=4, steps_ahead=100, area_h=10, area_w=10, squared_area=True, beam_angles=beam_angles, device=device, close_thres=0.05)
        self.stl = generateSTL(steps_ahead=10, battery_limit=2)
        self.model = RoverSTLPolicy(10).to(device)
        self.model.load_eval(model_path)
        self.model.eval()
        
        if verbose:
            print('==================================================')
            print('  We are using this model for the testing phase:  ')
            print('==================================================')

    def plan(self, state, delta_t: float):
        control = self.model(state)
        # Take the first planned state
        #print(control[0])
        
        # estimated = dynamics(self.sim, world_objects, state, robot_pose, target, charger, control)
        # stl_score = self.stl(estimated, 500, d={"hard": False})[:, :1]
        # stl_max_i = torch.argmax(stl_score, dim=0)
        # safe_control = control[stl_max_i : stl_max_i + 1]

        # for ctl in safe_control[0]:
        #     v = ctl[0] * 10 * 0.5
        #     theta = ctl[1].unsqueeze(0)

        control = control[0].detach().cpu().numpy()
        linear_velocity = control[:5, 0] * 0.2 # 10 * delta_t
        angular_velocity = control[:5, 1] / delta_t
        return linear_velocity, angular_velocity
    
    def plan_one(self, state, delta_t: float):
        control = self.model(state)
        # Take the first planned state
        control = control[0][5].detach().cpu().numpy()
        linear_velocity = control[0] * 2 * delta_t
        angular_velocity = control[1] / delta_t
        return linear_velocity, angular_velocity

    def normalize_state(self, state):
        return np.around(state,3)
