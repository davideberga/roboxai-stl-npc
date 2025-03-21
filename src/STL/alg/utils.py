import time
import torch
import numpy as np
import random
import os

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)

class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq, num_workers=1):
        self.start_iter = start_iter
        num_workers = 1 if num_workers is None else num_workers
        self.end_iter = end_iter//num_workers
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0
        self.num_workers = num_workers

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        if self.curr_iter % (max(1,self.check_freq//self.num_workers)) == 0:
            self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
            self.eta_t = self.interval * (self.end_iter - self.curr_iter)
    
    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)
    
    
def stable_softmin(a, b, beta):
        x = -beta * a
        y = -beta * b
        stacked = torch.stack([x, y], dim=0)
        return -(1 / beta) * torch.logsumexp(stacked, dim=0)

def stable_softmax(a, b, beta):
    x = beta * a
    y = beta * b
    stacked = torch.stack([x, y], dim=0)
    return (1 / beta) * torch.logsumexp(stacked, dim=0)

def soft_step_hard(x):
    hard = (x>=0).float()
    soft = (torch.tanh(500 * x) + 1)/2
    return soft + (hard - soft).detach()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
