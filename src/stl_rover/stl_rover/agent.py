from collections import deque
import numpy as np
import tensorflow as tf

from ament_index_python.packages import get_package_share_directory
import torch
import torch.nn as nn
import numpy as np

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
        
class PolicyPaper(nn.Module):
    def __init__(self):
        super(PolicyPaper, self).__init__()
        # input  (rover xy; dest xy; charger xy; battery t; hold_t)
        # output (rover v theta)
        self.net = build_relu_nn(
            2 + 2 + 2 + 2, 2 * 10, [256, 256, 256], activation_fn=nn.ReLU
        )

    def forward(self, x):
        num_samples = x.shape[0]
        u = self.net(x).reshape(num_samples, 10, -1)
        u0 = torch.clip(u[..., 0], 0, 1)
        u1 = torch.clip(u[..., 1], -np.pi, np.pi)
        uu = torch.stack([u0, u1], dim=-1)
        return uu
    
    def load_eval_paper(self, path: str):
        checkpoint = torch.load(path)
        state_dict_parsed = {k.replace("net.", ""): v for k, v in checkpoint.items()}
        self.net.load_state_dict(state_dict_parsed)
        
        


class DifferentialController:
    def __init__(self, k_p, k_d, dt):
        self.k_p = k_p
        self.k_d = k_d
        self.dt = dt
        self.prev_error = 0.0
        
    def normalize_angle(self, angle):
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def compute_control(self, desired_angle, current_angle):
        # Compute the normalized error between desired and current headings.
        error = self.normalize_angle(desired_angle - current_angle)
        # Compute the derivative term as the difference in error divided by the time step.
        derivative = (error - self.prev_error) / self.dt
        # Save the current error for the next iteration.
        self.prev_error = error
        # Compute the angular control output.
        control_output = self.k_p * error + self.k_d * derivative
        return control_output

class Agent:
    def __init__(self, verbose, model_name: str, is_paper: bool,  device):

        package_dir = get_package_share_directory("stl_rover")
        # model_path = package_dir + f"/model_trained/model_0.9865999817848206_172000.pth"
        model_path = package_dir + f"/model_trained/{model_name}"
        # load weights of pretrained model
        if not is_paper:
            self.model = RoverSTLPolicy(10).to(device)
            self.model.load_eval(model_path)
        else:
            self.model = PolicyPaper().to(device)
            self.model.load_eval_paper(model_path)
        self.model.eval()
        
        
        if verbose:
            print('==================================================')
            print('  We are using this model for the testing phase:  ')
            print('==================================================')

    def plan(self, state, delta_t: float):
        control = self.model(state)

        control = control[0].detach().cpu().numpy()
 
        linear_velocity = control[:, 0] * 0.2 * delta_t
        angular_velocity = control[4:7, 1] / delta_t
        return linear_velocity, angular_velocity
    
    def plan_absolute_theta(self, state, heading, delta_t: float):
        control = self.model(state)
        control = control[0].detach().cpu().numpy()
        linear_velocity = control[:, 0] * 10 * 0.5 #  * delta_t
        # angle_diff = control[:, 1] - heading
        # normalized_angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        # angular_velocity = 1 * normalized_angle_diff

        return linear_velocity, control[:, 1]
    
    def plan_absolute_theta_our(self, state, heading, delta_t: float):
        control = self.model(state)
        control = control[0].detach().cpu().numpy()
        linear_velocity = control[:, 0] * 10 * delta_t
        # angle_diff = control[:, 1] - heading
        # normalized_angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        # angular_velocity = 1 * normalized_angle_diff

        return linear_velocity, control[:, 1]
    
    def plan_one(self, state, delta_t: float):
        control = self.model(state)
        # Take the first planned state
        control = control[0][3].detach().cpu().numpy()
        linear_velocity = control[0] * 5 * delta_t
        angular_velocity = control[1] #/ delta_t
        return linear_velocity, angular_velocity
    

    def normalize_state(self, state):
        return np.around(state,3)
