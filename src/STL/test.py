import torch
from alg.dynamics import DynamicsSimulator
from alg.stl_network import RoverSTLPolicy




if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define fixed area dimensions.
    area_width = 10
    area_height = 10
    
    steps_ahead = 10

    sim = DynamicsSimulator()
    model = RoverSTLPolicy(steps_ahead).to(device)
    model.load_eval('exap_model_0.1680000126361847.pth')
    model.eval()
    
    n_objects = 5
    min_size = 0.5
    max_size = 2.0 
    target_size = 1.0      
    charger_size = 1.0     
    robot_radius = 0.3     
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

    # Run a sequence of simulation steps.
    for a in range(10):
        # Define 7 lidar beam angles (radians) relative to robot forward.
        beam_angles = torch.tensor([
            -torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0,
             torch.pi / 4, torch.pi / 3, torch.pi / 2
        ])

        # Simulate initial lidar scan (normalized).
        lidar_scan = sim.simulate_lidar_scan(robot_pose, beam_angles, world_objects, max_range=10.0)

        # Visualize the initial environment.
        sim.visualize_environment(robot_pose, beam_angles, lidar_scan, world_objects, target, charger, area_width, area_height, max_range=10.0)

        # Estimate the relative (normalized) distance and angle from the robot to the target and charger.
        if target is not None:
            target_distance, target_angle = sim.estimate_destination(robot_pose, target, max_distance=10.0)
            print(f"Target: normalized distance = {target_distance:.2f}, angle = {torch.rad2deg(target_angle):.2f}°")
        if charger is not None:
            charger_distance, charger_angle = sim.estimate_destination(robot_pose, charger, max_distance=10.0)
            print(f"Charger: normalized distance = {charger_distance:.2f}, angle = {torch.rad2deg(charger_angle):.2f}°")

        # Simulate a robot motion: move forward and rotate.
        v = 1.0
        theta_motion = torch.deg2rad(torch.tensor(20.0))
        new_pose, new_lidar_scan = sim.predict_lidar_scan_from_motion(
            robot_pose, v, theta_motion, beam_angles, world_objects,
            area_width=area_width, area_height=area_height,
            robot_radius=robot_radius, max_range=10.0
        )
        print("New robot pose:", new_pose)

        # Visualize the environment after the robot motion.
        sim.visualize_environment(new_pose, beam_angles, new_lidar_scan, world_objects, target, charger, area_width, area_height, max_range=10.0)

        # Update the robot pose and lidar scan for the next iteration.
        robot_pose = new_pose
        lidar_scan = new_lidar_scan