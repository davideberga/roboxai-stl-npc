import sys, os


sys.path.append("./")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rover_navigation_test import RoverNavigationTest
from alg.stl_network import PolicyPaper, RoverSTLPolicy
from alg.utils import seed_everything
from alg.RoverSTL import RoverSTL
import torch

seed = 42
seed_everything(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

rover_model = RoverSTL(None, type("c", (object,), {"seed": seed, "n_epochs": 2, "lr": 0.0001})())
stl, _, _, _, _, _ = rover_model.generateSTL(10, 2)
_, obstacles, _, _ = rover_model.simulator.generate_objects()
objs = torch.tensor(rover_model.simulator.transform_objects(obstacles)).float().to(device)
objs[:, 2] -= 0.5
objs[:, 3] -= 0.5


def normalize_degrees(angle):
    return (angle + 180) % 360 - 180


def get_plan(state, policy):
    state = torch.tensor([state]).to(device).float()
    control = policy(state)
    control = control[0].detach().cpu().numpy()
    linear_velocity = control[:, 0] * 10 * 0.2  # * 10 * 0.6
    theta = control[:, 1]

    plan = []
    linear_vel = linear_velocity.tolist()
    theta = theta.tolist()
    for v, t in zip(linear_vel, theta):
        plan.append((v, normalize_degrees(t * (180 / 3.14))))
    return plan


def get_plan_x(state, policy, state_complete):
    state = torch.tensor([state]).to(device).float()
    control = policy(state)

    positions = state_complete[11:17]

    robot_pose = torch.concat((torch.tensor(positions[0:2]), torch.tensor([0]))).unsqueeze(0).to(device)
    target = torch.tensor(positions[2:4]).unsqueeze(0).to(device)
    charger = torch.tensor(positions[4:]).unsqueeze(0).to(device)

    estimated, _ = rover_model.dynamics(objs, state, robot_pose, target, charger, control)
    stl_score = stl(estimated, 500, d={"hard": False})[:, :1]
    stl_max_i = torch.argmax(stl_score, dim=0)
    safe_control = control[stl_max_i : stl_max_i + 1]

    plan = []
    for ctl in safe_control[0].cpu().detach().numpy():
        v = ctl[0] * 10 * 0.2
        t = ctl[1]
        if(abs(v) > 0):
            plan.append((v, normalize_degrees(t * (180 / 3.14))))
    return plan


def get_plan_stl(state, policy):
    state = torch.tensor([state]).to(device).float()
    control = policy(state)
    control = control[0].detach().cpu().numpy()
    linear_velocity = control[:, 0] * 10 * 0.2  # * 10 * 0.6
    theta = control[:, 1]

    plan = []
    linear_vel = linear_velocity.tolist()
    theta = theta.tolist()
    for v, t in zip(linear_vel, theta):
        plan.append((v, normalize_degrees(t * (180 / 3.14))))
    print(plan)
    return plan


def main(env, policy_network, iterations=100, is_paper=False):
    goal, crash = 0, 0

    # First reset
    state, state_complete = env.reset()

    saved_episodes = []

    for ep in range(iterations):
        planned_actions = get_plan(state, policy_network) if is_paper else get_plan_x(state, policy_network, state_complete)

        # An episode is complete when 3 goals are reached or collision or battery finished occur
        goal = 0
        collision = 0
        battery = 0

        episode = []

        state, state_complete = env.reset(battery_reset=True)

        while True:
            if len(planned_actions) < 1:
                planned_actions = get_plan(state, policy_network) if is_paper else  get_plan_x(state, policy_network, state_complete)

            v, t = planned_actions.pop(0)
            state, reward, done, info, state_complete = env.step([v, t])

            if info["target_reached"]:
                goal = 1
            if info["collision"]:
                collision = 1
            if info["battery_ended"]:
                battery = 1

            episode.append(np.concatenate((state_complete, np.array([v, t, goal, collision, battery]))))

            if done:
                break

        print(f" Episode {ep}")
        saved_episodes.append(np.array(episode))

    return np.array(saved_episodes)


if __name__ == "__main__":
    ENV_TYPE = "test"

    policy_paper = PolicyPaper().to(device).float()
    policy_paper.load_eval_paper("model_testing/model_final_paper.ckpt")
    policy_paper.eval()

    try:
        env = RoverNavigationTest(env_type=ENV_TYPE, seed=seed, worker_id=0, is_env_for_paper=True)
        saved_episodes = main(env, policy_paper, is_paper=True)
        np.savez("test-result/paper.result.npz", episodes=saved_episodes)
        env.close()
    finally:
        print("Test paper finished!")

    policy_our = RoverSTLPolicy(10).to(device).float()
    policy_our.load_eval("model_testing/model-closeness-beta-increased_0.9581999778747559_157500.pth")
    policy_our.eval()

    try:
        env = RoverNavigationTest(env_type=ENV_TYPE, seed=seed, worker_id=0, is_env_for_paper=False)
        saved_episodes = main(env, policy_our)
        np.savez("test-result/our.result.npz", episodes=saved_episodes)
        env.close()
    finally:
        print("Test our finished!")
