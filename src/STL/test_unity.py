import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys

sys.path.append("./")
import tensorflow as tf
import numpy as np
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rover_navigation_test import RoverNavigationTest
from alg.stl_network import PolicyPaper
from alg.utils import seed_everything
import torch

seed = 42
seed_everything(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_degrees(angle):
    return (angle + 180) % 360 - 180


def get_plan(state, policy):
    print("Planning...")

    state = torch.tensor([state]).to(device).float()
    # state = torch.round(state, decimals=3)
    print(state)
    control = policy(state)
    control = control[0].detach().cpu().numpy()
    linear_velocity = control[:, 0] * 10  * 0.2 # * 10 * 0.6
    theta = control[:, 1]

    plan = []

    linear_vel = linear_velocity.tolist()
    theta = theta.tolist()

    for v, t in zip(linear_vel, theta):
        plan.append((v, normalize_degrees(t * (180 / 3.14))))
        # plan.append((None, normalize_degrees(t * (180 / 3.14))))
        # plan.append((v, None))
    print(f"Planned {len(plan)} actions")
    return plan


def main(env, policy_network, iterations=100):
    goal, crash = 0, 0

    for ep in range(iterations):
        state = env.reset()

        planned_actions = get_plan(state, policy_network)

        while True:
            if len(planned_actions) < 1:
                # Replanning
                planned_actions = get_plan(state, policy_network)

            v, t = planned_actions.pop(0)
            state, reward, done, info = env.step([v, t])
            print(f"Executing {(v, t)}")
            if done:
                print(f"Episode: {ep}")
                break

        if info["target_reached"]:
            print( f"{ep:3}: Goal!" )
            goal += 1

        elif info["collision"]:
            print( f"{np.round(state, 4)} => {v}")
            print( f"{ep:3}: Crash!" )
            crash += 1

        # else:
        # print( f"{ep:3}: Time Out!" )

    return goal, crash, iterations


if __name__ == "__main__":
    policy_paper = PolicyPaper().to(device).float()
    policy_paper.load_eval_paper("model_testing/model_final_paper.ckpt")
    policy_paper.eval()

    try:
        env = RoverNavigationTest(env_type="test", seed=seed, worker_id=0, is_env_for_paper=True)
        success = main(env, policy_paper)
        # print('\n======================================')
        # print(f'\nSuccess: {success[0]}/{success[2]}\nCrash: {success[1]}/{success[2]}\n')
        # print('======================================\n')

    finally:
        print("Test paper finished!")
