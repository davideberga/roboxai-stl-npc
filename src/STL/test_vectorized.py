import torch
from alg.dynamics import DynamicsSimulator
from alg.stl_network import RoverSTLPolicy
from alg.RoverSTL import RoverSTL
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2]).to(device)


def generate_env_with_starting_states(area_width: int, area_height: int, n_objects: int, n_states: int):
    sim = DynamicsSimulator()
    obstacles = [
        {"center": [1.5, 1.5], "width": 3.0, "height": 3.0},
        {"center": [8.5, 1.5], "width": 3.0, "height": 3.0},
        {"center": [5, 8.5], "width": 4.0, "height": 3.0},
        {"center": [5, 5], "width": 1.0, "height": 1.0},
    ]
    obstacles = obstacles + sim.walls(10)
    obstacles_tensor = to_obs_tensor(obstacles)

    min_size = 0.5
    max_size = 4.0
    target_size = 0.2
    charger_size = 0.2

    return sim.generate_random_environments(
        n_states, None, area_width, area_height, min_size, max_size, target_size, charger_size, 0.4, beam_angles=beam_angles, obstacles=obstacles_tensor, max_attempts=100
    )


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
    sim: DynamicsSimulator = DynamicsSimulator()
    model = RoverSTLPolicy(steps_ahead).to(device)
    model.load_eval("model_testing/model_correct_dynamics_training_0.9039800465106964_395.pth")
    model.eval()

    area_width = 10
    area_height = 10
    n_objects = 3
    n_states = 1

    # Create first random env
    world_objects, state, robot_pose, target, charger = generate_env_with_starting_states(area_width, area_height, n_objects, n_states)
    world_objects = world_objects[0]
    robot_radius = 0.3
    robot_pose = robot_pose[0]
    battery = 1

    fig, ax = plt.subplots(figsize=(8, 8))
    writer = animation.FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    
    with writer.saving(fig, "simulation_video.mp4", dpi=100):
        for _ in range(5):
            # Init change of target, chargers
            target = target[0]
            charger = charger[0]
            
            for frame in range(100):
                # Define 7 lidar beam angles (radians) relative to robot forward.

                control = model(state)

                v = control[0][0][0]
                theta = control[0][0][1]

                new_state, new_pose = sim.update_state_batch_vectorized(state, v, theta, robot_pose.unsqueeze(0), beam_angles, world_objects, target.unsqueeze(0), charger.unsqueeze(0), device)
                
                

                # print(new_pose[0].cpu().tolist())

                new_lidar = sim.simulate_lidar_scan_vectorized(new_pose, beam_angles, world_objects)
                target_distance, target_angle = sim.estimate_destination_vectorized(new_pose, target.unsqueeze(0))
                charger_distance, charger_angle = sim.estimate_destination_vectorized(new_pose, charger.unsqueeze(0))
                battery -= 0.01
                new_state[..., 11] = battery
                
                # near_charger = (torch.tanh(1000 * (0.05 * (0.05 - new_state[..., 10]))) + 1) / 2
                # # Update the battery
                # new_state[..., 11] = (new_state[..., 11].unsqueeze(1) - 0.01) * (1 - near_charger) + 1 * near_charger
                # new_state[..., 12] = state[..., 12].unsqueeze(1) - 0.2 * near_charger

                #new_state[..., 11] -= battery
                
                # Visualize the initial environment.
                ax.clear()
                sim.visualize_environment(
                    new_pose.squeeze(),
                    beam_angles,
                    new_lidar.squeeze(),
                    world_objects,
                    target.squeeze(),
                    charger.squeeze(),
                    area_width,
                    area_height,
                    max_range=5.0,
                    battery_level=new_state[..., 11].item(),
                    ax=ax,
                )
                
                writer.grab_frame()

                if target_distance.item() < 0.05:
                    print("Goal reached")
                    break
                
                if new_state[..., 11].item() <= 0:
                    print("Battery finished")
                    break

                state = new_state
                robot_pose = new_pose.squeeze()
                
            # Randomize new env
            _, _, _, target, charger = generate_env_with_starting_states(area_width, area_height, n_objects, n_states)

            

        # Update the robot pose and lidar scan for the next iteration.
    plt.close(fig)
