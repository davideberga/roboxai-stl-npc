import torch
from alg.dynamics import DynamicsSimulator
from alg.stl_network import RoverSTLPolicy
import random
import numpy as np

seed = 42 

np.random.seed(seed)  
random.seed(seed)  
torch.manual_seed(seed)  
torch.cuda.manual_seed_all(seed) 

def generate_env_with_starting_states(area_width: int, area_height: int, n_objects: int, n_states: int, beam_angles, device):
    sim = DynamicsSimulator()

    min_size = 0.5  # minimum obstacle size.
    max_size = 2.0  # maximum obstacle size.
    target_size = 0.2  # size of target square.
    charger_size = 0.2  # size of charger square.
    robot_radius = 0.3  # robot's radius.
    world_objects, target, charger, robot_pose = sim.generate_random_environment(
        n_objects,
        area_width,
        area_height,
        min_size,
        max_size,
        target_size,
        charger_size,
        robot_radius,
        obstacles=None,  # or pass a list of obstacles to override random generation.
        max_attempts=1000,
    )

    target = target.to(device)
    charger = charger.to(device)
    robot_pose = robot_pose.to(device)

    states = torch.empty((0,))
    robot_poses = torch.empty((0,))
    targets = torch.empty((0,))
    chargers = torch.empty((0,))

    for _ in range(n_states):
        lidar_scan = sim.simulate_lidar_scan(robot_pose, beam_angles, world_objects, max_range=10.0)
        target_distance, target_angle = sim.estimate_destination(robot_pose, target, max_distance=10.0)
        charger_distance, charger_angle = sim.estimate_destination(robot_pose, charger, max_distance=10.0)

        new_state = (
            torch.cat(
                (lidar_scan, target_angle.reshape(1), target_distance.reshape(1), charger_angle.reshape(1), charger_distance.reshape(1), torch.tensor([1]).to(device), torch.tensor([1]).to(device))
            )
            .float()
            .unsqueeze(0)
        )

        states = torch.cat((states, new_state), dim=0) if states.numel() > 0 else new_state
        robot_pose = torch.tensor(robot_pose)
        robot_poses = torch.cat((robot_poses, robot_pose.reshape(1, -1))) if robot_poses.numel() > 0 else robot_pose.reshape(1, -1)
        targets = torch.cat((targets, target.reshape(1, -1))) if targets.numel() > 0 else target.reshape(1, -1)
        chargers = torch.cat((chargers, charger.reshape(1, -1))) if chargers.numel() > 0 else charger.reshape(1, -1)

        # Obstacles fixed, randomize others
        _, target, charger, robot_pose = sim.generate_random_environment(
            n_objects,
            area_width,
            area_height,
            min_size,
            max_size,
            target_size,
            charger_size,
            robot_radius,
            obstacles=world_objects,
            max_attempts=1000,
        )

    return world_objects, states.to(device).detach(), robot_poses.to(device), targets.to(device), chargers.to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define fixed area dimensions.
    area_width = 10
    area_height = 10

    steps_ahead = 10

    sim = DynamicsSimulator()
    model = RoverSTLPolicy(steps_ahead).to(device)
    model.load_eval("exap_model_0.1680000126361847.pth")
    model.eval()

    beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2]).to(device)

    area_width = 10
    area_height = 10
    n_objects = 3
    n_states = 1

    world_objects, state, robot_pose, target, charger = generate_env_with_starting_states(area_width, area_height, n_objects, n_states, beam_angles, device)

    robot_radius = 0.3

    robot_pose = robot_pose[0]
    target = target[0]
    charger = charger[0]
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Create a figure and axes.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create an FFMpeg writer (make sure you have ffmpeg installed).
    writer = animation.FFMpegWriter(
        fps=5,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"]
    )
    
    with writer.saving(fig, "simulation_video.mp4", dpi=100):
        for frame in range(100):  
            # Define 7 lidar beam angles (radians) relative to robot forward.

            control = model(state)
            v = torch.max(control[0][0][0], torch.tensor(0.2))
            theta =  control[0][0][1]
            robot_pose, lidar_scan = sim.predict_lidar_scan_from_motion(
                robot_pose, v, theta, beam_angles, world_objects, area_width=area_width, area_height=area_height, robot_radius=robot_radius, 
                max_range=10.0, use_perfection=False
            )

            target_distance, target_angle = sim.estimate_destination(robot_pose, target, max_distance=10.0)
            charger_distance, charger_angle = sim.estimate_destination(robot_pose, charger, max_distance=10.0)

            # Visualize the initial environment.
            ax.clear()
            sim.visualize_environment(robot_pose, beam_angles, lidar_scan, world_objects, target, charger, area_width, area_height, max_range=10.0, ax=ax)
            
            
            
            if state[0][11].item() < 0:
                print("Battery finished")
                break

            if target_distance < 0.05:
                print("Goal reached")
                break

            state = torch.cat([lidar_scan, target_angle.unsqueeze(0), target_distance.unsqueeze(0), target_angle.unsqueeze(0), target_distance.unsqueeze(0), state[0][11].unsqueeze(0) - 0.01, state[0][12].unsqueeze(0)])
            print(f"Battery: {state[11].item()}")
            state = state.unsqueeze(0)
            
            writer.grab_frame()

        # Update the robot pose and lidar scan for the next iteration.
    plt.close(fig)