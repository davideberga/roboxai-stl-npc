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