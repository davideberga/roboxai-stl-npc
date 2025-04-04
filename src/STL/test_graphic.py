import torch
from alg.RoverSTL import RoverSTL
from alg.lib_stl_core import AP, Always, Eventually, Imply, ListAnd, Or
from alg.dynamics import DynamicsSimulator
from alg.stl_network import PolicyPaper, RoverSTLPolicy
from alg.utils import seed_everything
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2]).to(device)


def get_control(state, policy):
    control = policy(state)
    return control


def paper_state(pose, target, charger, battery, hold_time):
    return torch.tensor([pose[0][0], pose[0][1], target[0][0], target[0][1], charger[0][0], charger[0][1], battery, hold_time]).unsqueeze(0).to(device)


def main(model, iterations=1000, is_paper=False):
    area_width = 10
    area_height = 10
    steps_ahead = 10
    saved_episodes = []
    max_steps = 200
    
    random_battery = np.random.uniform(0.08, 5, iterations).tolist()

    # Initialize model
    sim: DynamicsSimulator = DynamicsSimulator(wait_for_charging=4, steps_ahead=10, area_h=10, area_w=10, squared_area=True, beam_angles=beam_angles, device=device, close_thres=0.05)
    rover_model = RoverSTL(None, type("c", (object,), {"seed": 20, "n_epochs": 2, "lr": 0.0001})())

    # Initialize simulator
    _, obstacles, _, _ = sim.generate_objects()
    state, obstacles_t, robot_pose, target, charger = sim.initialize_x(1, obstacles)
    battery = 5
    hold_time = 1
    state_paper = paper_state(robot_pose, target, charger, battery, hold_time)
    obstacles_t = obstacles_t[1:]
    world_objects = obstacles_t

    # fig, ax = plt.subplots(figsize=(8, 8))
    # writer = animation.FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])

    # with writer.saving(fig, f"test_{'paper' if is_paper else 'our'}.mp4", dpi=100):
    for ep in range(iterations):
        done = False
        step_counter = 0
        while not done:
            episode = []
            control = get_control(state_paper, model) if is_paper else get_control(state, model)

            new_state = None

            for ctl in control[0]:
                v = ctl[0] * 10 * 0.5
                theta = ctl[1].unsqueeze(0)

                new_state, new_pose, collision_detected = sim.update_state_batch_figure(state, v, theta, robot_pose, world_objects, target, charger, collision_enabled=True)
                robot_pose = new_pose
                if new_state[..., 10] < 0.1:
                    battery = min(battery + 0.5, 5)
                    hold_time = max(0, hold_time - 0.3)
                else:
                    battery -= 0.01

                if hold_time < 0.1:
                    hold_time = 1
                    
                new_state[..., 11] = battery
                new_state[..., 12] = hold_time
                
                # ax.clear()
                # new_lidar = sim.simulate_lidar_scan(new_pose, world_objects)
                # extra = torch.full_like(target, 0.4)  # Create zeros of required shape
                # targets_expanded = torch.cat([target, extra], dim=1)
                # extra = torch.full_like(charger, 0.4)
                # chargers_expanded = torch.cat([charger, extra], dim=1).unsqueeze(1)
                # sim.visualize_environment(
                #     new_pose.squeeze(),
                #     new_lidar.squeeze(),
                #     world_objects,
                #     targets_expanded[0],
                #     chargers_expanded[0],
                #     battery_level=new_state[..., 11].item(),
                #     ax=ax,
                # )

                # writer.grab_frame()


                step_counter += 1

                goal_reached = 0
                battery_finished = 0
                collision = 0
                
                if new_state[..., 8].item() < 0.1:
                    done = True
                    goal_reached = 1

                if new_state[..., 11].item() <= 0:
                    done = True
                    battery_finished = 1

                if step_counter > 100 or collision_detected:
                    done = True
                    collision = 1
                

                new_state_arr = new_state[..., :11].detach().cpu().numpy().reshape(1, -1)
                new_pose_arr = new_pose[..., :2].detach().cpu().numpy().reshape(1, -1)
                target_arr = target.detach().cpu().numpy().reshape(1, -1)
                charger_arr = charger.detach().cpu().numpy().reshape(1, -1)
                remaining_state_arr = new_state[..., 11:].detach().cpu().numpy().reshape(1, -1)
                ctl_arr = ctl.detach().cpu().numpy().reshape(1, -1)
                end = np.array([goal_reached, collision, battery_finished]).reshape(1, -1)
                concatenated_data = np.concatenate([new_state_arr, new_pose_arr, target_arr, charger_arr, remaining_state_arr, ctl_arr, end], axis=1)

                episode.append(np.squeeze(concatenated_data))
                
                
                
                if done: break
                
            state_paper = paper_state(robot_pose, target, charger, battery, hold_time)
            state = new_state
                
        print(f"Episode {ep}")
        saved_episodes.append(np.array(episode))

        # Reset
        state, _, robot_pose, target, charger = sim.initialize_x(1, obstacles)
        battery = random_battery[ep]
        hold_time = 1
        state_paper = paper_state(robot_pose, target, charger, battery, hold_time)

    # plt.close(fig)
    return np.array(saved_episodes)


if __name__ == "__main__":
    ENV_TYPE = "test"

    seed_everything(seed)
    policy_paper = PolicyPaper().to(device).float()
    policy_paper.load_eval_paper("model_testing/model_final_paper.ckpt")
    policy_paper.eval()

    try:
        saved_episodes = main(policy_paper, is_paper=True)
        np.savez("test-result/paper-figure.result.npz", episodes=saved_episodes)
    finally:
        print("Test paper finished!")

    seed_everything(seed)
    policy_our = RoverSTLPolicy(10).to(device).float()
    policy_our.load_eval("model_testing/model-closeness-beta-increased_0.9581999778747559_157500.pth")
    policy_our.eval()

    try:
        saved_episodes = main(policy_our)
        np.savez("test-result/our-figure.result.npz", episodes=saved_episodes)
    finally:
        print("Test our finished!")
        
    seed_everything(seed)
    policy_no_avoid = RoverSTLPolicy(10).to(device).float()
    policy_no_avoid.load_eval("model_testing/model-no-avoid_1.0_92000.pth")
    policy_no_avoid.eval()

    try:
        saved_episodes = main(policy_no_avoid)
        np.savez("test-result/no_avoid-figure.result.npz", episodes=saved_episodes)
    finally:
        print("Test our finished!")
