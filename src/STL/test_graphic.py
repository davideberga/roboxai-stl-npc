import torch
from alg.RoverSTL import RoverSTL
from alg.lib_stl_core import AP, Always, Eventually, Imply, ListAnd, Or
from alg.dynamics import DynamicsSimulator
from alg.stl_network import RoverSTLPolicy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from alg.utils import soft_step_hard

device = "cuda" if torch.cuda.is_available() else "cpu"
beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2]).to(device)

enough_close_to = 0.05
safe_distance = 0.05
wait_for_charging = 3



def to_obs_tensor(world_objects):
    tmp = []
    for obstacle in world_objects:
        tmp.append([obstacle["center"][0], obstacle["center"][1], obstacle["width"], obstacle["height"]])
    return torch.tensor(tmp).float().to(device)

if __name__ == "__main__":
    # Define fixed area dimensions.
    area_width = 10
    area_height = 10
    steps_ahead = 10

    # Initialize model
    sim: DynamicsSimulator = DynamicsSimulator(wait_for_charging=4, steps_ahead=100, area_h=10, area_w=10, squared_area=True, beam_angles=beam_angles, device=device, close_thres=0.05)
    model = RoverSTLPolicy(steps_ahead).to(device)
    rover_model = RoverSTL(None, type('c', (object,), {'seed': 20, 'n_epochs': 2, 'lr': 0.0001})())
    stl, _, _, _, _, _ = rover_model.generateSTL(10, 2)
    # model.load_eval("model_testing/model_correct_dynamics_training_0.9039800465106964_395.pth")
    # model.load_eval("model_testing/model_correct_dynamics_training_0.798820035457611_22.pth")
    # 172000
    # model.load_eval("model_testing/model_0.9145999550819397_170000.pth")
    # model.load_eval_paper("model_testing/model_10000.ckpt")
    model.load_eval_paper("model_testing/model-closeness-beta-increased_0.9581999778747559_157500.pth")
    model.eval()
    
    _, obstacles, _, _ = sim.generate_objects()
    state, obstacles_t, robot_pose, target, charger = sim.initialize_x(1, obstacles)
    obstacles_t = obstacles_t[1:]
    
    world_objects = obstacles_t
    robot_radius = 0.3
    # robot_pose = robot_pose[0]
    battery = 5
    hold_time = 1
    state[..., 11] = 5
    state[..., 12] = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    writer = animation.FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])


    with writer.saving(fig, "simulation_video.mp4", dpi=100):
        for _ in range(5):
            # Init change of target, chargers
            # target = target[0]
            # charger = charger[0]

            episode_finished = False
            for frame in range(100):
                # Define 7 lidar beam angles (radians) relative to robot forward.

                control = model(state)

                v = control[0][0]
                theta = control[0][0][1]

                estimated, _ = rover_model.dynamics(world_objects, state, robot_pose, target, charger, control)
                stl_score = stl(estimated, 500, d={"hard": False})[:, :1]
                stl_max_i = torch.argmax(stl_score, dim=0)
                safe_control = control[stl_max_i : stl_max_i + 1]
                print(stl_max_i)
                # print("---------------------------------")
                # print(world_objects)
                # print(state)
                # print(robot_pose)
                # print(target)
                # print(charger)
                # print(control)
                
                # print(len(safe_control))
                
                # robot_pose = torch.tensor([[5, 6.2, 1.57]]).to(device)

                for ctl in safe_control[0]:
                    # 0.5 for our model
                    v = ctl[0] * 10 * 0.5
                    if not v > 0: continue
                    theta = ctl[1].unsqueeze(0)

                    new_state, new_pose = sim.update_state_batch(
                        state,
                        v,
                        theta,
                        robot_pose,
                        world_objects,
                        target,
                        charger,
                        collision_enabled=True
                    )

                    # print(new_pose[0].cpu().tolist())

                    new_lidar = sim.simulate_lidar_scan(new_pose, world_objects)
                    target_distance, target_angle = sim.estimate_destination(new_pose, target)
                    charger_distance, charger_angle = sim.estimate_destination(new_pose, charger)

                    # near_charger = (torch.tanh(1000 * (0.05 * (0.8 - new_state[..., 10]))) + 1) / 2
                    # # Update the battery
                    # new_state[..., 11] = (new_state[..., 11].unsqueeze(1) - 0.01) * (1 - near_charger) + 1 * near_charger
                    # new_state[..., 12] = state[..., 12].unsqueeze(1) - 0.2 * near_charger

                    # Manual handling
                    
                    
                    # print(new_state[..., :])
                    
                    # near_charger = soft_step_hard(0.05 * (enough_close_to - charger_distance))
                    # # near_charger = (torch.tanh(500 * (0.05 * (self.enough_close_to_charger - nearest_dists))) + 1) / 2
                    # battery = (new_state[:, 11].unsqueeze(1) - 0.2) * (1 - near_charger.unsqueeze(1)) + 5 * near_charger.unsqueeze(1)
                    # hold_time = new_state[:, 12].unsqueeze(1) - 0.2 * near_charger.unsqueeze(1)
                    
                    if new_state[..., 10].item() < 0.1:
                        battery = min(battery + 0.5, 5)
                        hold_time = max(0, hold_time - 0.3)
                    else:
                        battery -= 0.05

                    if hold_time < 0.1:
                        hold_time = 1

                    
                    
                    new_state[..., 11] = battery
                    new_state[..., 12] = hold_time
                    
                    # Visualize the initial environment.
                    ax.clear()
                    extra = torch.full_like(target, 0.4)  # Create zeros of required shape
                    targets_expanded = torch.cat([target, extra], dim=1)
                    extra = torch.full_like(charger, 0.4)
                    chargers_expanded = torch.cat([charger, extra], dim=1).unsqueeze(1)
                    sim.visualize_environment(
                        new_pose.squeeze(),
                        new_lidar.squeeze(),
                        world_objects,
                        targets_expanded[0],
                        chargers_expanded[0],
                        battery_level=new_state[..., 11].item(),
                        ax=ax,
                    )

                    writer.grab_frame()

                    if target_distance.item() < 0.1:
                        print("Goal reached")
                        episode_finished = True
                        break
                        # plt.close(fig)
                        # exit(0)

                    if new_state[..., 11].item() <= 0:
                        print("Battery finished")
                        episode_finished = True
                        break
                        # plt.close(fig)
                        # exit(0)

                    # new_state[0, 11] = torch.clamp(new_state[0, 11], 0, 25 * 0.2)
                    # new_state[0, 12] = torch.clamp(new_state[0, 12], -0.2, 3  * 0.2)
                    state = new_state
                    robot_pose = new_pose

                if episode_finished:
                    break

            # Randomize new env
            _, _, _, target, _ = sim.initialize_x(1, obstacles)
            obstacles_t = obstacles_t[1:]

        # Update the robot pose and lidar scan for the next iteration.
    plt.close(fig)
