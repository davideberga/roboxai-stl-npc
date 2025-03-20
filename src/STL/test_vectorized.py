import torch
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


def generate_env_with_starting_states(sim, area_width: int, area_height: int, n_objects: int, n_states: int):
    obstacles = [
        {"center": [1.5, 1.5], "width": 3.0, "height": 3.0},
        {"center": [8.5, 1.5], "width": 3.0, "height": 3.0},
        {"center": [5, 8.5], "width": 4.0, "height": 3.0},
        {"center": [5, 5], "width": 1.0, "height": 1.0},
    ]
    obstacles = obstacles + sim.walls()
    obstacles_tensor = to_obs_tensor(obstacles)

    min_size = 0.5
    max_size = 4.0
    target_size = 0.2
    charger_size = 0.2

    return sim.generate_random_environments_v2(
        n_states,
        None,
        min_size,
        max_size,
        target_size,
        charger_size,
        0.05,
        obstacles=obstacles_tensor,
        num_chargers=4,
        max_attempts=100,
    )


def to_obs_tensor(world_objects):
    tmp = []
    for obstacle in world_objects:
        tmp.append([obstacle["center"][0], obstacle["center"][1], obstacle["width"], obstacle["height"]])
    return torch.tensor(tmp).float().to(device)


def dynamics(
    sim,
    world_objects,
    states,
    robot_poses,
    targets,
    chargers,
    es_trajectories,
    include_first=False,
):
    """Estimate planning of T steps ahead"""

    t = es_trajectories.shape[1]  # Extract actions predicted
    x = states.clone()
    poses = robot_poses.clone()
    tgs = targets.clone()
    chrs = chargers.clone()

    segs = [states] if include_first else []

    for ti in range(t):
        new_x, new_poses = dynamics_per_step(sim, x, world_objects, poses, tgs, chrs, es_trajectories[:, ti])
        segs.append(new_x)
        x = new_x
        poses = new_poses

    return torch.stack(segs, dim=1)


def dynamics_per_step(sim, x, world_objects, poses, tgs, chrs, es_trajectories):
    """Computes how the system state changes in one time step given the current state x
    and a single es_trajectories  u
    """

    # The new state must have these values estimated
    # [LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, LIDAR, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER, B_TIME, C_TIME]

    predicted_velocity = es_trajectories[:, 0]
    predicted_theta = es_trajectories[:, 1]

    return sim.update_state_batch(
        x,
        predicted_velocity,
        predicted_theta,
        poses,
        world_objects,
        tgs,
        chrs,
    )


def lidar_obs_avoidance_robustness(x):
    def smooth_min(lidar_values, alpha=500.0):
        # Computes a smooth approximation of the minimum.
        # To ensure smooth differentiability
        return -(1 / alpha) * torch.logsumexp(-alpha * lidar_values, dim=-1)

    lidar_values = x[..., 0:7]
    min_lidar = smooth_min(lidar_values)
    # print(f"Smooth min { min_lidar}")
    # print(f"Actual min { torch.min(lidar_values, dim=-1)}")
    # print(f"Lidar robustness: ${min_lidar - self.safe_distance}")

    # Rescale lidar values to give them more importance
    return (min_lidar - safe_distance) * 100


def generateSTL(steps_ahead: int, battery_limit: float):
    def debug_print(label, func, x):
        value = func(x)
        # print(f"{label}: {value}")
        return value

    avoid0 = Always(0, steps_ahead, AP(lambda x: (x[..., 0] - safe_distance) * 100))
    avoid1 = Always(0, steps_ahead, AP(lambda x: (x[..., 1] - safe_distance) * 100))
    avoid2 = Always(0, steps_ahead, AP(lambda x: (x[..., 2] - safe_distance) * 100))
    avoid3 = Always(0, steps_ahead, AP(lambda x: (x[..., 3] - safe_distance) * 100))
    avoid4 = Always(0, steps_ahead, AP(lambda x: (x[..., 4] - safe_distance) * 100))
    avoid5 = Always(0, steps_ahead, AP(lambda x: (x[..., 5] - safe_distance) * 100))
    avoid6 = Always(0, steps_ahead, AP(lambda x: (x[..., 6] - safe_distance) * 100))

    avoid_list = [avoid0, avoid1, avoid2, avoid3, avoid4, avoid5, avoid6]

    avoid = ListAnd(avoid_list)

    at_dest = AP(lambda x: debug_print("Distance to destination", lambda x: enough_close_to - x[..., 8], x), comment="Distance to destination")
    at_charger = AP(lambda x: debug_print("Distance to charger", lambda x: enough_close_to - x[..., 10], x), comment="Distance to charger")

    if_enough_battery_go_destiantion = Imply(AP(lambda x: debug_print("Battery level > limit", lambda x: x[..., 11] - battery_limit, x)), Eventually(0, steps_ahead, at_dest))
    if_low_battery_go_charger = Imply(AP(lambda x: debug_print("Battery level < limit", lambda x: battery_limit - x[..., 11], x)), Eventually(0, steps_ahead, at_charger))

    always_have_battery = Always(0, steps_ahead, AP(lambda x: debug_print("Battery level", lambda x: x[..., 11], x)))

    stand_by = AP(lambda x: debug_print("Stand by (distance from charger)", lambda x: enough_close_to - x[..., 10], x), comment="Stand by: agent remains close to charger")
    enough_stay = AP(lambda x: debug_print(f"Stay > {wait_for_charging} steps", lambda x: -x[..., 12], x), comment=f"Stay>{wait_for_charging} steps")
    charging = Imply(at_charger, Always(0, wait_for_charging, Or(stand_by, enough_stay)))

    return ListAnd(
        [avoid, always_have_battery, if_low_battery_go_charger, charging, if_enough_battery_go_destiantion]
    )  # ListAnd([avoid])# if_enough_battery_go_destiantion, always_have_battery, if_low_battery_go_charger]) # missing charging

    return ListAnd(
        [
            avoid,
            if_enough_battery_go_destiantion,
            charging,
            always_have_battery,
            if_low_battery_go_charger,
        ]
    )


if __name__ == "__main__":
    # Define fixed area dimensions.
    area_width = 10
    area_height = 10
    steps_ahead = 10

    # Initialize model
    sim: DynamicsSimulator = DynamicsSimulator(wait_for_charging=4, steps_ahead=100, area_h=10, area_w=10, squared_area=True, beam_angles=beam_angles, device=device, close_thres=0.05)
    model = RoverSTLPolicy(steps_ahead).to(device)
    # model.load_eval("model_testing/model_correct_dynamics_training_0.9039800465106964_395.pth")
    # model.load_eval("model_testing/model_correct_dynamics_training_0.798820035457611_22.pth")
    # 172000
    model.load_eval("model_testing/model_0.9145999550819397_170000.pth")
    # model.load_eval_paper("model_testing/model_10000.ckpt")
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

    stl = generateSTL(10, 2.0)

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

                # estimated = dynamics(sim, world_objects, state, robot_pose, target, charger, control)
                # stl_score = stl(estimated, 500, d={"hard": False})[:, :1]
                # stl_max_i = torch.argmax(stl_score, dim=0)
                # safe_control = control[stl_max_i : stl_max_i + 1]
                
                # print(len(safe_control))
                
                # robot_pose = torch.tensor([[5, 6.2, 1.57]]).to(device)

                for ctl in control[0]:
                    # 0.5 for our model
                    v = ctl[0] * 10 * 0.5
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
                    
                    near_charger = soft_step_hard(0.05 * (enough_close_to - charger_distance))
                    # near_charger = (torch.tanh(500 * (0.05 * (self.enough_close_to_charger - nearest_dists))) + 1) / 2
                    battery = (new_state[:, 11].unsqueeze(1) - 0.2) * (1 - near_charger.unsqueeze(1)) + 5 * near_charger.unsqueeze(1)
                    hold_time = new_state[:, 12].unsqueeze(1) - 0.2 * near_charger.unsqueeze(1)
                    
                    
                    # new_state[..., 11] = battery
                    # new_state[..., 12] = hold_time
                    
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
