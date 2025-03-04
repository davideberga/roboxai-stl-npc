"""Parser

"""
import argparse
import os
import multiprocessing as mp
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()

    # Logging        
    parser.add_argument("--wandb_log", type=lambda x: bool(strtobool(x)), default=False, help="Wandb log")

    # Wandb
    parser.add_argument("--wandb_project_name", type=str, default="MobileRoboticsTest", help="Wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default="luca0", help="Entity of wandb's project")    
    parser.add_argument("--wandb_mode", type=str, default="online", help="{online, offline} wandb mode")
    parser.add_argument("--wandb_code", type=lambda x: bool(strtobool(x)), default=True, help="Wandb's save code")
    parser.add_argument("--wandb_notes", type=str, default='', help="Experiment notes")

    # Metrics
    parser.add_argument("--last_n", type=int, default=100, help="Average metrics over this time horizon")
    parser.add_argument("--metrics", type=object, default=["Episode", "Step", "Avg_Success", "Avg_Reward", "Avg_Cost"], help="List of desired metrics")
     
    # Episode length
    parser.add_argument("--n_total_epochs", type=int, default=250000, help="Experiment total timesteps")
   
    # Algorithm
    parser.add_argument("--alg", type=str, default='DDQN', help="Training algorithm")
    parser.add_argument("--tag", type=str, default='', help="Training tag")
    parser.add_argument("--seed", type=int, default=940, help="Experiment seed") 

    # Algorithm hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps", type=float, default=1.0, help="ε-greedy starting value")
    parser.add_argument("--eps_min", type=float, default=0.05, help="ε-greedy final value")
    parser.add_argument("--eps_decay", type=float, default=0.9995, help="ε-greedy decay value")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging factor")

    # Buffer
    parser.add_argument("--memory_size", type=int, default=5000, help="Size of the memory buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of each minibatch")

    # Network
    parser.add_argument("--n_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--layer_size", type=int, default=32, help="Number nodes for each layer")
    parser.add_argument("--activation", type=str, default='relu', help="Activation function")

    # Update
    parser.add_argument("--n_epochs", type=int, default=40, help="Number of epochs for policy update")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")

    # Cuda 
    parser.add_argument("--cuda", type=str, default="0", help="-1 to disable cuda")

    return parser.parse_args()