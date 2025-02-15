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
    def __init__(self):
        super(RoverSTLPolicy, self).__init__()
        # input  [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]
        # lidar_scan + 6 elements,
        # - normalized in [0, 1] ==> heading (first) + distance (second) of the TARGET
        # - normalized in [0, 1] ==> heading (first) + distance (second) of the NEAREST CHARGER
        # - Battery time (number of steps) [0, 10000], Charger hold time (number of steps) [0,5]
        #
        # output (rover v theta; astro v theta)
        self.net = build_relu_nn( 7 + 2 + 2 + 2, 20, [256, 256, 256], activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        control = self.net(x).reshape(num_samples, 10, -1)
        control0 = torch.tanh(control[..., 0]) * 0.5 + 0.5
        control1 = torch.tanh(control[..., 1]) * np.pi
        control_final = torch.stack([control0, control1], dim=-1)
        return control_final
    
    def save(self, path: str):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))


class Agent:
    def __init__(self, verbose, device):

        package_dir = get_package_share_directory("stl_rover")
        model_path = package_dir + "/model_trained/exap_model.pth"
        # load weights of pretrained model
        self.model = RoverSTLPolicy().to(device)
        self.model.load(model_path)
        self.model.eval()
        
        if verbose:
            print('==================================================')
            print('  We are using this model for the testing phase:  ')
            print('==================================================')

    def plan(self, state):
        return self.model(state)

    def normalize_state(self, state):
        return np.around(state,3)
