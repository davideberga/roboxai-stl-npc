import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from shapely.geometry import box, Point
from shapely.prepared import prep
from shapely.ops import triangulate
import concurrent.futures
import numpy as np
import multiprocessing as mp


class DynamicsSimulator:
    def __init__(self):
        self.beta = 20  # 1
        self.beta2 = 100  # 5
        self.epsilon = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rover_max_velocity = 0.2
        self.rover_min_velocity = 0.0

        self.enough_close_to_charger = 0.05
        self.battery_charge = 1
        self.dt = 0.50

    def predict_lidar_scan_from_motion(self, old_pose, v, theta_motion, beam_angles, world_objects, area_width, area_height, robot_radius, max_range=10.0, use_perfection=False):
        """
        Given the robot's old pose (tensor of shape (3,)) and a motion command (scalar v and theta_motion),
        compute the new pose and simulate the lidar scan.
        If the new pose results in a collision or falls outside the environment, the move is rejected and
        the old pose (with its corresponding lidar scan) is returned.

        Parameters:
            old_pose    : torch.Tensor of shape (3,) representing [x, y, heading].
            v           : torch.Tensor or scalar (linear velocity).
            theta_motion: torch.Tensor or scalar (rotation angle in radians).
            beam_angles : torch.Tensor of lidar beam angles (relative to robot's forward).
            world_objects: List of obstacles (each as a dictionary).
            area_width  : Width of the environment.
            area_height : Height of the environment.
            robot_radius: Robot's collision radius.
            max_range   : Maximum lidar range.

        Returns:
            new_pose: torch.Tensor of shape (3,) [x, y, heading].
            new_scan: torch.Tensor of normalized lidar distances.
        """
        # Unpack old_pose (assumed to be a tensor).
        x, y, heading = old_pose[0], old_pose[1], old_pose[2]

        invalid = False
        if x.item() < robot_radius or x.item() > area_width - robot_radius or y.item() < robot_radius or y.item() > area_height - robot_radius:
            invalid = True

        # Check for collision with any obstacle.
        for obj in world_objects:
            if self.circle_rect_collision((x.item(), y.item()), robot_radius, obj):
                invalid = True
                break

        if invalid:
            return old_pose, self.simulate_lidar_scan(old_pose, beam_angles, world_objects, max_range) if not use_perfection else self.simulate_perfect_lidar_scan(
                old_pose, beam_angles, world_objects, max_range
            )

        # Compute new pose using tensor operations.
        new_x = x + v * torch.cos(heading)
        new_y = y + v * torch.sin(heading)
        new_heading = heading + theta_motion
        new_pose = torch.stack([new_x, new_y, new_heading])

        # Compute the new lidar scan.
        new_scan = (
            self.simulate_lidar_scan(new_pose, beam_angles, world_objects, max_range) if not use_perfection else self.simulate_perfect_lidar_scan(new_pose, beam_angles, world_objects, max_range)
        )

        return new_pose, new_scan

    def get_bounds(self, rect):
        """
        Return the bounding box (min_x, max_x, min_y, max_y) of a rectangle.
        Works for both dictionary and torch.Tensor representations.
        """
        if isinstance(rect, torch.Tensor):
            cx, cy, w, h = rect[0].item(), rect[1].item(), rect[2].item(), rect[3].item()
        else:
            cx, cy = rect["center"]
            w, h = rect["width"], rect["height"]
        return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)

    def check_overlap(self, rect1, rect2):
        """Check if two axis-aligned rectangles overlap."""
        r1_minx, r1_maxx, r1_miny, r1_maxy = self.get_bounds(rect1)
        r2_minx, r2_maxx, r2_miny, r2_maxy = self.get_bounds(rect2)
        if r1_maxx < r2_minx or r1_minx > r2_maxx or r1_maxy < r2_miny or r1_miny > r2_maxy:
            return False
        else:
            return True

    def circle_rect_collision(self, circle_center, circle_radius, rect):
        """
        Check whether a circle (robot) collides with an axis-aligned rectangle,
        using PyTorch for the distance calculation.
        Supports both dictionary and torch.Tensor representations for the rectangle.
        """
        cx, cy = circle_center
        if isinstance(rect, torch.Tensor):
            rx, ry, w, h = rect[0], rect[1].item(), rect[2].item(), rect[3].item()
        else:
            rx, ry = rect["center"]
            w, h = rect["width"], rect["height"]
        rect_min_x = rx - w / 2
        rect_max_x = rx + w / 2
        rect_min_y = ry - h / 2
        rect_max_y = ry + h / 2

        circle_center_tensor = torch.tensor([cx, cy]).float().to(self.device)
        closest_x = torch.clamp(circle_center_tensor[0], min=rect_min_x, max=rect_max_x)
        closest_y = torch.clamp(circle_center_tensor[1], min=rect_min_y, max=rect_max_y)
        distance = torch.sqrt((circle_center_tensor[0] - closest_x) ** 2 + (circle_center_tensor[1] - closest_y) ** 2)
        return distance.item() < circle_radius

    def generate_random_environments(
        self,
        num_envs,
        n_objects,
        area_width,
        area_height,
        min_size,
        max_size,
        target_size,
        charger_size,
        robot_radius,
        beam_angles,
        obstacles=None,
        max_attempts=1000,
        max_range=10.0,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Generate Obstacles (cached if static) ---
        if obstacles is None:
            # [Obstacle generation code remains similar, possibly vectorized]
            attempts = 0
            world_objects = []
            while len(world_objects) < n_objects and attempts < max_attempts:
                sizes = torch.rand(n_objects, 2) * (max_size - min_size) + min_size
                positions = torch.rand(n_objects, 2) * (torch.tensor([area_width, area_height]) - sizes)
                centers = positions + sizes / 2
                new_objects = torch.cat([centers, sizes], dim=1).to(device)
                if not world_objects:
                    world_objects = new_objects
                else:
                    for obj in new_objects:
                        if not any(self.check_overlap(obj, w_obj) for w_obj in world_objects):
                            world_objects.append(obj)
                attempts += 1
            if attempts >= max_attempts:
                print("Warning: Maximum attempts reached while generating obstacles.")
            world_objects = torch.stack(world_objects)
        else:
            world_objects = torch.tensor(obstacles, dtype=torch.float32).to(device)

        # --- Compute Free Space (Do this once) ---
        free_space = box(0, 0, area_width, area_height)
        world_objs_np = world_objects.cpu().numpy()
        for obj in world_objs_np:
            cx, cy, w, h = obj
            obs_poly = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            free_space = free_space.difference(obs_poly)

        # Buffer free space for each object
        free_target_space = free_space.buffer(-target_size / 2)
        free_charger_space = free_space.buffer(-charger_size / 2)
        free_robot_space = free_space.buffer(-robot_radius)

        # Use prepared geometries for faster "contains" checks
        prep_target = prep(free_target_space)
        prep_charger = prep(free_charger_space)
        prep_robot = prep(free_robot_space)

        # --- Batch Sampling Function ---
        def sample_point(poly, prepared_poly, attempts, batch_size=1000):
            minx, miny, maxx, maxy = poly.bounds
            for _ in range(attempts):
                xs = np.random.uniform(minx, maxx, batch_size)
                ys = np.random.uniform(miny, maxy, batch_size)
                for x, y in zip(xs, ys):
                    if prepared_poly.contains(Point(x, y)):
                        return x, y
            return None, None

        # --- Generate Environments ---
        targets, chargers, robot_poses, states = [], [], [], []
        for _ in range(num_envs):
            # Sample target
            tx, ty = sample_point(free_target_space, prep_target, max_attempts)
            target = torch.tensor([tx, ty, target_size, target_size], device=device) if tx is not None else torch.tensor([float("nan")] * 4, device=device)
            targets.append(target)

            # Sample charger
            cx_val, cy_val = sample_point(free_charger_space, prep_charger, max_attempts)
            charger = torch.tensor([cx_val, cy_val, charger_size, charger_size], device=device) if cx_val is not None else torch.tensor([float("nan")] * 4, device=device)
            chargers.append(charger)

            # Sample robot pose
            rx, ry = sample_point(free_robot_space, prep_robot, max_attempts)
            if rx is None:
                robot_pose = torch.tensor([float("nan")] * 3, device=device)
            else:
                heading = np.random.uniform(0, 2 * np.pi)
                robot_pose = torch.tensor([rx, ry, heading], device=device)
            robot_poses.append(robot_pose)

            # Compute LiDAR scan values, distances, etc.
            scan = self.simulate_lidar_scan_vectorized(robot_pose.unsqueeze(0), beam_angles, world_objects, max_range)
            dist_target, angle_target = self.estimate_destination_vectorized(robot_pose.unsqueeze(0), target.unsqueeze(0), max_range)
            dist_charger, angle_charger = self.estimate_destination_vectorized(robot_pose.unsqueeze(0), charger.unsqueeze(0), max_range)
            es_battery_time = torch.full_like(dist_target, fill_value=1.0)
            es_charger_time = torch.full_like(dist_target, fill_value=1.0)

            state = torch.cat([scan, angle_target, dist_target, angle_charger, dist_charger, es_battery_time, es_charger_time], dim=-1)
            states.append(state.squeeze())

        return (
            world_objects.unsqueeze(0).float().expand(num_envs, -1, -1),
            torch.stack(states).detach().float(),
            torch.stack(robot_poses).float(),
            torch.stack(targets).float(),
            torch.stack(chargers).float(),
        )

    def generate_random_environment(self, n_objects, area_width, area_height, min_size, max_size, target_size, charger_size, robot_radius, obstacles=None, max_attempts=1000):
        """
        Generate a random environment within a fixed area that includes:
            - Random obstacles (if obstacles is None)
            - A target destination (yellow square as a torch.Tensor: [cx, cy, width, height])
            - A charger destination (green square as a torch.Tensor: [cx, cy, width, height])
            - A robot starting pose (blue circle) that does not collide with any object.

        Returns:
            world_objects: Tensor of shape (n_objects, 4) with [cx, cy, width, height].
            target       : Tensor [cx, cy, width, height].
            charger      : Tensor [cx, cy, width, height].
            robot_pose   : Tensor [x, y, heading].
        """
        world_objects = []
        if obstacles is not None:
            world_objects = obstacles
        else:
            attempts = 0
            while len(world_objects) < n_objects and attempts < max_attempts:
                w = torch.rand(1).item() * (max_size - min_size) + min_size
                h = torch.rand(1).item() * (max_size - min_size) + min_size
                cx = torch.rand(1).item() * (area_width - w) + w / 2
                cy = torch.rand(1).item() * (area_height - h) + h / 2
                new_rect = {"center": [cx, cy], "width": w, "height": h}
                if any(self.check_overlap(new_rect, obj) for obj in world_objects):
                    attempts += 1
                    continue
                world_objects.append(new_rect)
                attempts += 1
            if attempts >= max_attempts:
                print("Warning: maximum attempts reached while generating obstacles.")

        def generate_square(square_size):
            for _ in range(max_attempts):
                cx = torch.rand(1) * (area_width - square_size) + square_size / 2
                cy = torch.rand(1) * (area_height - square_size) + square_size / 2
                square = torch.tensor([cx.item(), cy.item(), square_size, square_size], dtype=torch.float32)
                if any(self.check_overlap(square, obj) for obj in world_objects):
                    continue
                return square
            return torch.tensor([float("nan")] * 4)  # Return NaN tensor if placement fails

        target = generate_square(target_size)
        charger = generate_square(charger_size)

        if torch.isnan(target).any() or torch.isnan(charger).any():
            print("Warning: Could not place target or charger without overlap.")

        robot_pose = torch.tensor([float("nan"), float("nan"), float("nan")], dtype=torch.float32)
        for _ in range(max_attempts):
            rx = torch.rand(1) * (area_width - 2 * robot_radius) + robot_radius
            ry = torch.rand(1) * (area_height - 2 * robot_radius) + robot_radius
            heading = torch.rand(1) * 2 * torch.pi
            collision = False

            # Check collision with obstacles, target, and charger
            for obj in world_objects:
                if self.circle_rect_collision((rx.item(), ry.item()), robot_radius, obj):
                    collision = True
                    break
            if target is not None and self.circle_rect_collision((rx.item(), ry.item()), robot_radius, target):
                collision = True
            if charger is not None and self.circle_rect_collision((rx.item(), ry.item()), robot_radius, charger):
                collision = True
            if not collision:
                robot_pose = torch.tensor([rx.item(), ry.item(), heading.item()], dtype=torch.float32)
                break

        if torch.isnan(robot_pose).any():
            print("Warning: Could not place robot without collision after maximum attempts.")

        return world_objects, target, charger, robot_pose

    def softmin2(self, a, b, beta=100.0):
        """
        Differentiable approximation to min(a, b).
        A high beta makes the approximation closer to the true min.
        """
        vals = torch.stack([a, b])
        weights = torch.softmax(-beta * vals, dim=0)
        return torch.sum(weights * vals)

    def softmax2(self, a, b, beta=100.0):
        """
        Differentiable approximation to max(a, b).
        A high beta makes the approximation closer to the true max.
        """
        vals = torch.stack([a, b])
        weights = torch.softmax(beta * vals, dim=0)
        return torch.sum(weights * vals)

    def softmin3(self, x, y, beta=10.0):
        return -torch.log(torch.exp(-beta * x) + torch.exp(-beta * y)) / beta

    # A differentiable version of estimate_destination.
    # Assumes dest is a tensor of shape (4,) with [cx, cy, width, height],
    # and robot_pose is a tensor of shape (3,) with [x, y, heading].

    def walls(self, map_width: float):
        a = 0.2
        x_min, x_max, y_min, y_max = 0, map_width, 0, map_width
        map_walls = [
            {"center": [x_min + a, (y_min + y_max) / 2], "width": 0.4, "height": y_max - y_min},  # Left wall
            {"center": [x_max - a, (y_min + y_max) / 2], "width": 0.4, "height": y_max - y_min},  # Right wall
            {"center": [(x_min + x_max) / 2, y_max - a], "width": x_max - x_min, "height": 0.4},  # Top wall
            {"center": [(x_min + x_max) / 2, y_min + a], "width": x_max - x_min, "height": 0.4},  # Bottom wall
        ]
        return map_walls

    def simulate_lidar_scan(self, robot_pose, beam_angles, world_objects, max_range=10.0):
        """
        Simulate a lidar scan from a given robot pose using differentiable ray–casting.
        Distances are normalized to [0, 1] by dividing by max_range.
        """
        ray_origin = robot_pose[:2]  # (2,)
        heading = robot_pose[2]
        scan_vals = []

        if not isinstance(beam_angles, torch.Tensor):
            beam_angles = torch.tensor(beam_angles, dtype=beam_angles.dtype, device=beam_angles.device)
        for beam in beam_angles:
            global_angle = heading + beam
            ray_direction = torch.stack((torch.cos(global_angle), torch.sin(global_angle)))
            ray_direction = ray_direction / torch.norm(ray_direction)
            min_distance = max_range * torch.ones(1, dtype=beam_angles.dtype, device=beam_angles.device)
            for rect in world_objects:
                t = self.ray_rect_intersection(ray_origin, ray_direction, rect, max_range)
                min_distance = torch.min(min_distance, t)
            scan_vals.append(min_distance)
        return torch.stack(scan_vals).squeeze() / max_range

    def ray_rect_intersection(self, ray_origin, ray_direction, rect, max_range, beta=10.0, beta2=10.0, epsilon=1e-6):
        """
        Differentiable approximation to the ray–axis-aligned rectangle intersection.

        Instead of using hard if/else conditions, we use softmin/softmax functions and
        sigmoids to blend between cases.

        Parameters:
        ray_origin : (2,) tensor for ray start.
        ray_direction : (2,) tensor for ray direction (assumed normalized).
        rect : dictionary with keys "center" (a list [cx, cy]), "width", "height".
        max_range : scalar maximum range.
        beta, beta2 : scalars controlling sharpness (higher means sharper).
        epsilon : small constant to avoid division by zero.

        Returns:
        intersection_time : a differentiable scalar tensor approximating the intersection distance.
        """
        # Convert rectangle center to a tensor.
        center = torch.tensor(rect["center"], dtype=ray_direction.dtype, device=ray_direction.device)
        w = rect["width"]
        h = rect["height"]
        min_x = center[0] - w / 2.0
        max_x = center[0] + w / 2.0
        min_y = center[1] - h / 2.0
        max_y = center[1] + h / 2.0

        # Compute candidate intersection distances along x.
        tx1 = (min_x - ray_origin[0]) / (ray_direction[0] + epsilon)
        tx2 = (max_x - ray_origin[0]) / (ray_direction[0] + epsilon)
        tmin_x = self.softmin2(tx1, tx2, beta)
        tmax_x = self.softmax2(tx1, tx2, beta)

        # And along y.
        ty1 = (min_y - ray_origin[1]) / (ray_direction[1] + epsilon)
        ty2 = (max_y - ray_origin[1]) / (ray_direction[1] + epsilon)
        tmin_y = self.softmin2(ty1, ty2, beta)
        tmax_y = self.softmax2(ty1, ty2, beta)

        # Overall candidate intersection times.
        tmin = self.softmax2(tmin_x, tmin_y, beta)  # approximates max(tmin_x, tmin_y)
        tmax = self.softmin2(tmax_x, tmax_y, beta)  # approximates min(tmax_x, tmax_y)

        # Create a differentiable “validity” indicator.
        # valid_indicator is near 1 if tmax is positive and tmax > tmin.
        valid_indicator = torch.sigmoid(beta2 * tmax) * torch.sigmoid(beta2 * (tmax - tmin))

        # Create an indicator for the ray starting inside the rectangle (tmin < 0).
        inside_indicator = 1.0 - torch.sigmoid(beta2 * tmin)
        # Blend between using tmin (if outside) or tmax (if inside)
        t_intermediate = (1 - inside_indicator) * tmin + inside_indicator * tmax

        # Finally, if the intersection is not valid, return max_range.
        intersection_time = valid_indicator * t_intermediate + (1 - valid_indicator) * max_range
        return intersection_time

    def estimate_destination(self, robot_pose, dest, max_distance=10.0):
        rx, ry, rtheta = robot_pose[0], robot_pose[1], robot_pose[2]
        # Use the first two elements as the destination center.
        cx, cy = dest[0], dest[1]
        dx = cx - rx
        dy = cy - ry
        distance = torch.sqrt(dx**2 + dy**2)
        normalized_distance = torch.clamp(distance / max_distance, max=1.0)
        angle = torch.atan2(dy, dx) - rtheta
        # Normalize angle to [-pi, pi]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        return normalized_distance, angle

    def simulate_perfect_lidar_scan(self, robot_pose, beam_angles, world_objects, max_range=10.0):
        """
        Simulate a perfect lidar scan from a given robot pose using exact ray casting.
        Distances are normalized to [0, 1] by dividing by max_range.
        """
        ray_origin = robot_pose[:2]  # (x, y)
        heading = robot_pose[2]
        scan_vals = []

        # Ensure beam_angles is a tensor.
        if not isinstance(beam_angles, torch.Tensor):
            beam_angles = torch.tensor(beam_angles, dtype=torch.float32, device=robot_pose.device)

        for beam in beam_angles:
            global_angle = heading + beam
            # Compute the ray direction as a unit vector.
            ray_direction = torch.stack((torch.cos(global_angle), torch.sin(global_angle)))
            ray_direction = ray_direction / torch.norm(ray_direction)

            # Initialize minimum distance as a torch scalar.
            min_distance = torch.tensor(max_range, dtype=ray_direction.dtype, device=ray_direction.device)
            for rect in world_objects:
                t = self.perfect_ray_rect_intersection(ray_origin, ray_direction, rect, max_range)
                # Update minimum distance using torch.min.
                min_distance = torch.min(min_distance, t)
            scan_vals.append(min_distance)

        # Stack the results and normalize by max_range.
        return torch.stack(scan_vals) / max_range

    def perfect_ray_rect_intersection(self, ray_origin, ray_direction, rect, max_range, epsilon=1e-6):
        """
        Computes the exact intersection distance between a ray and an axis-aligned rectangle
        using PyTorch operations.

        Parameters:
            ray_origin    : (2,) tensor for the ray start.
            ray_direction : (2,) normalized tensor for the ray direction.
            rect          : Dictionary with keys "center" ([cx, cy]), "width", and "height".
            max_range     : Scalar maximum range.
            epsilon       : Small constant to avoid division by zero.

        Returns:
            A scalar tensor representing the intersection distance along the ray,
            or max_range if no valid intersection exists.

        Note:
            If the ray starts inside the rectangle (t_enter < 0), the exit distance (t_exit) is returned.
        """
        # Extract rectangle parameters.
        center = torch.tensor(rect["center"], dtype=ray_direction.dtype, device=ray_direction.device)
        w = rect["width"]
        h = rect["height"]
        min_x = center[0] - w / 2.0
        max_x = center[0] + w / 2.0
        min_y = center[1] - h / 2.0
        max_y = center[1] + h / 2.0

        # Compute intersections with vertical (x) slabs.
        tx1 = (min_x - ray_origin[0]) / ray_direction[0]
        tx2 = (max_x - ray_origin[0]) / ray_direction[0]
        tx_min = torch.where(torch.abs(ray_direction[0]) < epsilon, torch.tensor(-float("inf"), dtype=ray_direction.dtype, device=ray_direction.device), torch.min(tx1, tx2))
        tx_max = torch.where(torch.abs(ray_direction[0]) < epsilon, torch.tensor(float("inf"), dtype=ray_direction.dtype, device=ray_direction.device), torch.max(tx1, tx2))

        # Compute intersections with horizontal (y) slabs.
        ty1 = (min_y - ray_origin[1]) / ray_direction[1]
        ty2 = (max_y - ray_origin[1]) / ray_direction[1]
        ty_min = torch.where(torch.abs(ray_direction[1]) < epsilon, torch.tensor(-float("inf"), dtype=ray_direction.dtype, device=ray_direction.device), torch.min(ty1, ty2))
        ty_max = torch.where(torch.abs(ray_direction[1]) < epsilon, torch.tensor(float("inf"), dtype=ray_direction.dtype, device=ray_direction.device), torch.max(ty1, ty2))

        # Entry and exit times.
        t_enter = torch.max(tx_min, ty_min)
        t_exit = torch.min(tx_max, ty_max)

        # Valid intersection exists if t_exit >= 0 and t_enter <= t_exit.
        valid = (t_exit >= 0) & (t_enter <= t_exit)
        # If the ray starts inside the rectangle, use t_exit; otherwise use t_enter.
        t_intersect = torch.where(valid, torch.where(t_enter < 0, t_exit, t_enter), torch.tensor(max_range, dtype=ray_origin.dtype, device=ray_origin.device))
        # Clamp the distance to max_range.
        t_intersect = torch.clamp(t_intersect, max=max_range)

        return t_intersect

    # The updated batch state update function.
    def update_state_batch(self, state, v, theta, robot_pose, beam_angles, world_objects, target, charger, device, max_range=10.0, use_perfection=False):
        """
        Given a batch of current states and commands, update the state after a robot move.
        All operations are implemented with PyTorch tensor operations so that the update is differentiable.

        Parameters:
            state       : torch.Tensor of shape (B, 11)
            v           : torch.Tensor of shape (B,) (linear velocities)
            theta       : torch.Tensor of shape (B,) (angular changes)
            robot_pose  : torch.Tensor of shape (B, 3) with [x, y, heading] for each example.
            beam_angles : torch.Tensor of shape (num_beams,) (lidar beam angles)
            world_objects: List of obstacles (each as a dictionary); assumed common to all batch examples.
            target      : torch.Tensor of shape (B, 4) representing [cx, cy, width, height] for each example.
            charger     : torch.Tensor of shape (B, 4) representing [cx, cy, width, height] for each example.
            max_range   : Maximum lidar range.

        Returns:
            new_state : torch.Tensor of shape (B, 11)
                        [lidar (7,), head_target, dist_target, head_charger, dist_charger, battery_time, charger_time].
            new_pose  : torch.Tensor of shape (B, 3) representing updated [x, y, heading].
        """
        # Update robot poses (all tensor operations)
        new_x = robot_pose[:, 0] + v * torch.cos(robot_pose[:, 2])
        new_y = robot_pose[:, 1] + v * torch.sin(robot_pose[:, 2])
        new_heading = robot_pose[:, 2] + theta
        new_pose = torch.stack([new_x, new_y, new_heading], dim=1)  # (B, 3)

        B = new_pose.shape[0]
        lidar_list = []
        target_angle_list = []
        target_norm_list = []
        charger_angle_list = []
        charger_norm_list = []

        # Loop over batch elements.
        for i in range(B):
            pose_i = new_pose[i]  # keep as tensor (3,) for differentiability
            new_scan = (
                self.simulate_lidar_scan(pose_i, beam_angles, world_objects, max_range) if not use_perfection else self.simulate_perfect_lidar_scan(pose_i, beam_angles, world_objects, max_range)
            )
            lidar_list.append(new_scan)
            # Here target[i] and charger[i] are tensors of shape (4,)
            t_norm, t_angle = self.estimate_destination(pose_i, target[i])
            target_angle_list.append(t_angle)
            target_norm_list.append(t_norm)
            c_norm, c_angle = self.estimate_destination(pose_i, charger[i])
            charger_angle_list.append(c_angle)
            charger_norm_list.append(c_norm)

        new_lidar = torch.stack(lidar_list, dim=0).to(device)  # (B, num_beams)
        t_angle_tensor = torch.stack(target_angle_list, dim=0).unsqueeze(1)  # (B, 1)
        t_norm_tensor = torch.stack(target_norm_list, dim=0).unsqueeze(1)  # (B, 1)
        c_angle_tensor = torch.stack(charger_angle_list, dim=0).unsqueeze(1)  # (B, 1)
        c_norm_tensor = torch.stack(charger_norm_list, dim=0).unsqueeze(1)  # (B, 1)

        # Assume state[:, 11] is battery time and state[:, 12] is charger time.
        es_battery_time = torch.max(state[:, 11] - 0.1, torch.full_like(state[:, 11], 0))
        es_charger_time = state[:, 12]
        mask = c_norm_tensor < 0.05
        mask_not_at_charger = c_norm_tensor > 0.05

        es_battery_time = torch.where(mask, torch.min(state[:, 11] + 1, torch.full_like(state[:, 11], 1)), es_battery_time)
        es_charger_time = torch.where(mask, torch.max(state[:, 12] - 1, torch.full_like(state[:, 12], 0)), es_charger_time)
        es_charger_time = torch.where(mask_not_at_charger, 1, es_charger_time)

        new_state = torch.cat([new_lidar, t_angle_tensor, t_norm_tensor, c_angle_tensor, c_norm_tensor, es_battery_time[0].unsqueeze(1), es_charger_time[0].unsqueeze(1)], dim=1)

        return new_state, new_pose

    def generate_random_robot_poses(self, n, world_objects, area_width, area_height, robot_radius, target, charger, max_attempts=1000):
        """
        Generate a batch of n valid robot poses (each: [x, y, heading]) in the given environment.
        This helper assumes world_objects, target and charger are fixed.
        """
        poses = []
        attempts = 0
        # We try to generate until we have n poses or we run out of attempts.
        while len(poses) < n and attempts < max_attempts * n:
            # Sample candidate pose in a vectorized manner
            rx = torch.rand(1).item() * (area_width - 2 * robot_radius) + robot_radius
            ry = torch.rand(1).item() * (area_height - 2 * robot_radius) + robot_radius
            heading = torch.rand(1).item() * 2 * torch.pi
            collision = False

            # Check collision with obstacles
            for obj in world_objects:
                if self.circle_rect_collision((rx, ry), robot_radius, obj):
                    collision = True
                    break
            # Check collision with target and charger (if provided)
            if target is not None and self.circle_rect_collision((rx, ry), robot_radius, target):
                collision = True
            if charger is not None and self.circle_rect_collision((rx, ry), robot_radius, charger):
                collision = True

            if not collision:
                poses.append([rx, ry, heading])
            attempts += 1

        if len(poses) < n:
            print("Warning: Could not generate enough valid robot poses.")
        return torch.tensor(poses[:n], dtype=torch.float32)

    def update_state(self, state, v, theta, robot_pose, beam_angles, world_objects, target, charger, max_range=10.0):
        """
        Given the current state vector:
        [LIDAR_1, ..., LIDAR_7, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER],
        update the state after a robot move with linear velocity v and angular velocity theta.
        The world objects, robot pose, target, and charger positions are used to compute the new sensor readings.
        All operations are implemented in PyTorch and remain differentiable.

        Parameters:
            state       : torch.Tensor of shape (11,)
                        (Not used directly in the update, but provided for consistency with the system state.)
            v           : torch.Tensor (scalar) representing the linear velocity.
            theta       : torch.Tensor (scalar) representing the angular change.
            robot_pose  : torch.Tensor of shape (3,) representing [x, y, heading].
            beam_angles : torch.Tensor of lidar beam angles (relative to robot forward).
            world_objects: List of obstacles (each as a dictionary or tensor).
            target      : torch.Tensor representing the target [cx, cy, width, height].
            charger     : torch.Tensor representing the charger [cx, cy, width, height].
            max_range   : Maximum range for the lidar and destination estimation.

        Returns:
            new_state : torch.Tensor of shape (11,)
                        [new_lidar readings (7,), new_head_target, new_dist_target,
                        new_head_charger, new_dist_charger].
            new_pose  : torch.Tensor of shape (3,) representing the updated [x, y, heading].
        """
        # Update robot pose (all in torch so gradients flow)
        new_x = robot_pose[0] + v * torch.cos(robot_pose[2])
        new_y = robot_pose[1] + v * torch.sin(robot_pose[2])
        new_heading = robot_pose[2] + theta
        new_pose = torch.stack([new_x, new_y, new_heading])

        # Recompute the lidar scan at the new pose.
        # Note: For differentiability, ensure that simulate_lidar_scan is implemented without breaking gradients.
        new_lidar = self.simulate_lidar_scan((new_pose[0], new_pose[1], new_pose[2]), beam_angles, world_objects, max_range)
        # new_lidar is assumed to be a tensor of shape (7,)

        # Recompute relative destination estimates.
        # estimate_destination returns (normalized_distance, angle) where:
        #    normalized_distance ∈ [0,1] and angle ∈ [-pi, pi].
        # In the state vector the order is: [HEAD, DIST], so we swap the returned order.
        target_norm_dist, target_angle = self.estimate_destination((new_pose[0], new_pose[1], new_pose[2]), target, max_range)
        charger_norm_dist, charger_angle = self.estimate_destination((new_pose[0], new_pose[1], new_pose[2]), charger, max_range)

        es_charger_time = 1
        es_battery_time = state[:, 11] - 0.01

        # Create a mask where the condition is met
        mask = charger_norm_dist < 0.01

        # Update the battery time and charger time where the mask is True
        es_battery_time = torch.where(mask, state[:, 11] + 0.2, es_battery_time)
        es_charger_time = torch.where(mask, state[:, 12] - 0.2, es_charger_time)

        # Build the new state vector.
        # The expected order is:
        # [LIDAR_1, ..., LIDAR_7, HEAD_TARGET, DIST_TARGET, HEAD_N_CHARGER, DIST_N_CHARGER]
        new_state = torch.cat(
            [
                new_lidar,  # (7,)
                target_angle.reshape(1),  # HEAD_TARGET
                target_norm_dist.reshape(1),  # DIST_TARGET
                charger_angle.reshape(1),  # HEAD_N_CHARGER
                charger_norm_dist.reshape(1),  # DIST_N_CHARGER
                es_battery_time.reshape(1),
                es_charger_time.reshape(1),
            ]
        )

        return new_state, new_pose

    def stable_softmin2(self, a, b, beta):
        x = -beta * a
        y = -beta * b
        stacked = torch.stack([x, y], dim=0)
        return -(1 / beta) * torch.logsumexp(stacked, dim=0)

    def stable_softmax2(self, a, b, beta):
        x = beta * a
        y = beta * b
        stacked = torch.stack([x, y], dim=0)
        return (1 / beta) * torch.logsumexp(stacked, dim=0)

    def ray_rect_intersection_vectorized(self, ray_origins, ray_directions, world_objects, max_range):
        """
        Vectorized differentiable approximation to the ray–axis-aligned rectangle intersection.

        Parameters:
            ray_origins   : Tensor of shape (B, num_beams, 2) with ray start positions.
            ray_directions: Tensor of shape (B, num_beams, 2) (assumed normalized).
            world_objects : Tensor of shape (N, 4) where each row is [cx, cy, width, height].
            max_range     : Scalar maximum range.

        Returns:
            intersections : Tensor of shape (B, num_beams, N) with approximated intersection distances.
        """
        beta = self.beta
        beta2 = self.beta2
        epsilon = self.epsilon

        # Compute rectangle boundaries.
        centers = world_objects[:, :2]  # (N, 2)
        widths = world_objects[:, 2].unsqueeze(1)  # (N, 1)
        heights = world_objects[:, 3].unsqueeze(1)  # (N, 1)
        min_xy = centers - torch.cat([widths / 2, heights / 2], dim=1)  # (N, 2)
        max_xy = centers + torch.cat([widths / 2, heights / 2], dim=1)  # (N, 2)
        min_x = min_xy[:, 0]  # (N,)
        min_y = min_xy[:, 1]  # (N,)
        max_x = max_xy[:, 0]  # (N,)
        max_y = max_xy[:, 1]  # (N,)

        B, num_beams, _ = ray_origins.shape
        N = world_objects.shape[0]

        # Expand ray origins and directions to shape (B, num_beams, N, 2).
        ray_origins_exp = ray_origins.unsqueeze(2).expand(B, num_beams, N, 2)
        ray_directions_exp = ray_directions.unsqueeze(2).expand(B, num_beams, N, 2)

        # Expand rectangle boundaries to shape (B, num_beams, N).
        min_x_exp = min_x.view(1, 1, N).expand(B, num_beams, N)
        max_x_exp = max_x.view(1, 1, N).expand(B, num_beams, N)
        min_y_exp = min_y.view(1, 1, N).expand(B, num_beams, N)
        max_y_exp = max_y.view(1, 1, N).expand(B, num_beams, N)

        # Extract ray components.
        ray_origin_x = ray_origins_exp[..., 0]
        ray_origin_y = ray_origins_exp[..., 1]
        ray_dir_x = ray_directions_exp[..., 0]
        ray_dir_y = ray_directions_exp[..., 1]

        # Compute candidate intersection times with the x sides.
        tx1 = (min_x_exp - ray_origin_x) / (ray_dir_x + epsilon)
        tx2 = (max_x_exp - ray_origin_x) / (ray_dir_x + epsilon)
        # print(tx1, tx2)
        tmin_x = self.stable_softmin2(tx1, tx2, beta)
        tmax_x = self.stable_softmax2(tx1, tx2, beta)

        # Compute candidate intersection times with the y sides.
        ty1 = (min_y_exp - ray_origin_y) / (ray_dir_y + epsilon)
        ty2 = (max_y_exp - ray_origin_y) / (ray_dir_y + epsilon)
        tmin_y = self.stable_softmin2(ty1, ty2, beta)
        tmax_y = self.stable_softmax2(ty1, ty2, beta)

        # Overall candidate intersection times:
        # tmin approximates max(tmin_x, tmin_y) and tmax approximates min(tmax_x, tmax_y).
        tmin = self.stable_softmax2(tmin_x, tmin_y, beta)
        tmax = self.stable_softmin2(tmax_x, tmax_y, beta)

        # Validity indicator: near 1 if tmax > 0 and tmax > tmin.
        valid_indicator = torch.sigmoid(beta2 * tmax) * torch.sigmoid(beta2 * (tmax - tmin))
        # Indicator for ray starting inside the rectangle (tmin < 0).
        inside_indicator = 1.0 - torch.sigmoid(beta2 * tmin)
        # Blend between using tmin (if outside) or tmax (if inside).
        t_intermediate = (1 - inside_indicator) * tmin + inside_indicator * tmax

        # If no valid intersection, return max_range.
        intersections = valid_indicator * t_intermediate + (1 - valid_indicator) * max_range
        return intersections

    def simulate_lidar_scan_vectorized(self, robot_pose, beam_angles, world_objects, max_range=10.0, use_perfection=False):
        B = robot_pose.shape[0]
        num_beams = beam_angles.shape[0]
        global_angles = robot_pose[:, 2].unsqueeze(1) + beam_angles.unsqueeze(0)
        ray_dirs = torch.stack([torch.cos(global_angles), torch.sin(global_angles)], dim=-1)
        ray_origins = robot_pose[:, :2].unsqueeze(1).expand(B, num_beams, 2)

        intersections = self.ray_rect_intersection_vectorized(ray_origins, ray_dirs, world_objects, max_range)
        # Choose the minimum intersection distance for each ray.
        min_intersections, _ = intersections.min(dim=-1)
        # min_intersections = torch.nan_to_num(min_intersections, nan=max_range)
        # Replace NaNs with max_range (i.e., no detection).
        # Normalize scan values to [0, 1]
        scan = min_intersections / max_range
        return scan

    def estimate_destination_vectorized(self, robot_pose, dest, max_distance=10.0):
        """
        Vectorized computation of the relative distance and angle from the robot to a destination.

        Parameters:
            robot_pose : Tensor of shape (B, 3) with [x, y, heading].
            dest       : Tensor of shape (B, 4) representing destination [cx, cy, width, height]
                         (only cx and cy are used).
            max_distance: Scalar maximum distance for normalization.

        Returns:
            normalized_distance: Tensor of shape (B, 1) with values in [0, 1].
            angle              : Tensor of shape (B, 1) representing the relative heading.
        """
        rx = robot_pose[:, 0].unsqueeze(1)
        ry = robot_pose[:, 1].unsqueeze(1)
        rtheta = robot_pose[:, 2].unsqueeze(1)
        cx = dest[:, 0].unsqueeze(1)
        cy = dest[:, 1].unsqueeze(1)
        dx = cx - rx
        dy = cy - ry
        distance = torch.sqrt(dx**2 + dy**2)
        normalized_distance = torch.clamp(distance / max_distance, max=1.0)
        angle = torch.atan2(dy, dx) - rtheta
        # Normalize angle to [-pi, pi].
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        return normalized_distance, angle

    def update_state_batch_vectorized(self, state, v, theta, robot_pose, beam_angles, world_objects, target, charger, device, max_range=10.0, use_perfection=False):
        """
        Fully vectorized update of the robot state for a batch of moves.

        Parameters:
            state       : Tensor of shape (B, 13) (assumed to include sensor readings and timers;
                        battery_time is at state[:, 11] and charger_time at state[:, 12]).
            v           : Tensor of shape (B,) (linear velocities).
            theta       : Tensor of shape (B,) (angular changes).
            robot_pose  : Tensor of shape (B, 3) with [x, y, heading].
            beam_angles : Tensor of shape (num_beams,) (lidar beam angles).
            world_objects: Tensor of shape (N, 4) representing obstacles as [cx, cy, width, height].
            target      : Tensor of shape (B, 4) representing destination [cx, cy, width, height].
            charger     : Tensor of shape (B, 4) representing charger [cx, cy, width, height].
            device      : torch.device to which the new state should be moved.
            max_range   : Maximum lidar range.
            use_perfection: Boolean flag to choose a perfect scan simulation (if desired).

        Returns:
            new_state : Tensor of shape (B, 11) containing:
                        [lidar readings, head_target, dist_target, head_charger, dist_charger, battery_time, charger_time].
            new_pose  : Tensor of shape (B, 3) representing the updated [x, y, heading].
        """
        # Update robot pose.

        # No rescale velocity
        v = v * (self.rover_max_velocity - self.rover_min_velocity) + self.rover_min_velocity
        new_x = robot_pose[:, 0] + v * torch.cos(robot_pose[:, 2])
        new_y = robot_pose[:, 1] + v * torch.sin(robot_pose[:, 2])
        new_heading = robot_pose[:, 2] + theta
        new_pose = torch.stack([new_x, new_y, new_heading], dim=1)  # (B, 3)

        # --- Collision Check ---
        # For each new position, check against all world_objects.
        # A collision occurs if:
        #   abs(new_x - cx) <= width/2  and  abs(new_y - cy) <= height/2.
        # Expand new_pose coordinates to compare with each obstacle.
        x_exp = new_pose[:, 0].unsqueeze(1)  # (B, 1)
        y_exp = new_pose[:, 1].unsqueeze(1)  # (B, 1)
        # Expand world object parameters.
        obs_cx = world_objects[:, 0].unsqueeze(0)  # (1, N)
        obs_cy = world_objects[:, 1].unsqueeze(0)  # (1, N)
        obs_w = world_objects[:, 2].unsqueeze(0)  # (1, N)
        obs_h = world_objects[:, 3].unsqueeze(0)  # (1, N)

        collision_mask = (torch.abs(x_exp - obs_cx) <= obs_w / 2) & (torch.abs(y_exp - obs_cy) <= obs_h / 2)
        collision_any = collision_mask.any(dim=1)
        new_pose[collision_any] = robot_pose[collision_any]

        new_scan = self.simulate_lidar_scan_vectorized(new_pose, beam_angles, world_objects, max_range=5.0)
        t_norm, t_angle = self.estimate_destination_vectorized(new_pose, target, max_distance=10)
        c_norm, c_angle = self.estimate_destination_vectorized(new_pose, charger, max_distance=10)

        # ADAPTED From the paper code
        near_charger = (torch.tanh(500 * (0.05 * (self.enough_close_to_charger - c_norm))) + 1) / 2
        # Update the battery
        es_battery_time = (state[:, 11].unsqueeze(1) - 0.05) * (1 - near_charger) + self.battery_charge * near_charger
        es_charger_time = state[:, 12].unsqueeze(1) - 0.2 * near_charger

        new_state = torch.cat(
            [
                new_scan,
                t_angle,
                t_norm,
                c_angle,
                c_norm,
                es_battery_time,
                es_charger_time,
            ],
            dim=1,
        ).to(device)

        return new_state, new_pose

    def visualize_environment(self, robot_pose, beam_angles, lidar_scan, world_objects, target, charger, area_width, area_height, max_range=10.0, battery_level=1.0, ax=None):
        """
        Visualizes the environment with obstacles, target, charger, robot, and lidar scan.
        Displays lidar scan distances on the rays and the battery level outside of the map.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(0, area_width)
        ax.set_ylim(0, area_height)

        # Draw obstacles.
        for obj in world_objects:
            if isinstance(obj, dict):
                cx, cy = obj["center"]
                w, h = obj["width"], obj["height"]
            else:
                cx, cy, w, h = obj[0].item(), obj[1].item(), obj[2].item(), obj[3].item()
            lower_left = (cx - w / 2, cy - h / 2)
            rect_patch = patches.Rectangle(lower_left, w, h, linewidth=1, edgecolor="black", facecolor="gray", alpha=0.5)
            ax.add_patch(rect_patch)

        # Draw target.
        if target is not None:
            cx, cy, w, h = target[:4].tolist()
            lower_left = (cx - w / 2, cy - h / 2)
            target_patch = patches.Rectangle(lower_left, w, h, linewidth=2, edgecolor="orange", facecolor="yellow", alpha=0.8, label="Target")
            ax.add_patch(target_patch)

        # Draw charger.
        if charger is not None:
            cx, cy, w, h = charger[:4].tolist()
            lower_left = (cx - w / 2, cy - h / 2)
            charger_patch = patches.Rectangle(lower_left, w, h, linewidth=2, edgecolor="green", facecolor="lightgreen", alpha=0.8, label="Charger")
            ax.add_patch(charger_patch)

        # Draw robot.
        rx, ry, rtheta = robot_pose[:3].tolist()
        ax.plot(rx, ry, "bo", markersize=8, label="Robot")
        arrow_length = 0.5
        ax.arrow(rx, ry, arrow_length * math.cos(rtheta), arrow_length * math.sin(rtheta), head_width=0.2, head_length=0.2, fc="blue", ec="blue")

        # Ensure beam_angles is a torch tensor.
        if not isinstance(beam_angles, torch.Tensor):
            beam_angles = torch.tensor(beam_angles, dtype=torch.float32)

        # Draw lidar beams and distances.
        for beam, dist in zip(beam_angles, lidar_scan):
            beam_val = beam.item()
            global_angle = rtheta + beam_val
            norm_dist = max(0.0, min(dist.item(), 1.0))
            actual_dist = norm_dist * max_range
            end_x = rx + actual_dist * math.cos(global_angle)
            end_y = ry + actual_dist * math.sin(global_angle)
            style = "-" if norm_dist < 1.0 else "--"
            ax.plot([rx, end_x], [ry, end_y], style, color="red", linewidth=1)
            ax.plot(end_x, end_y, "ro", markersize=3)
            text_offset = 0.2
            text_x = end_x + text_offset * math.cos(global_angle)
            text_y = end_y + text_offset * math.sin(global_angle)
            ax.text(text_x, text_y, f"{dist:.2f}", fontsize=8, color="black", ha="center", va="center")

        # Display battery level outside of the map
        ax.text(
            area_width / 2,
            area_height + 1,
            f"Battery: {battery_level * 100:.1f}%",
            fontsize=12,
            color="blue",
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

        ax.set_aspect("equal")
        ax.set_title("Environment with Obstacles, Target, Charger, Robot, Lidar Scan, and Battery Level")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        ax.grid(True)


# ----- Main Example Usage -----
if __name__ == "__main__":
    # Define fixed area dimensions.
    area_width = 10
    area_height = 10

    sim = DynamicsSimulator()

    # Generate a random environment:
    # Parameters for environment generation.
    n_objects = 5  # number of obstacles to generate if not provided.
    min_size = 0.5  # minimum obstacle size.
    max_size = 2.0  # maximum obstacle size.
    target_size = 1.0  # size of target square.
    charger_size = 1.0  # size of charger square.
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

    # Run a sequence of simulation steps.
    for a in range(10):
        # Define 7 lidar beam angles (radians) relative to robot forward.
        beam_angles = torch.tensor([-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, 0.0, torch.pi / 4, torch.pi / 3, torch.pi / 2])

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
            robot_pose, v, theta_motion, beam_angles, world_objects, area_width=area_width, area_height=area_height, robot_radius=robot_radius, max_range=10.0
        )
        print("New robot pose:", new_pose)

        # Visualize the environment after the robot motion.
        sim.visualize_environment(new_pose, beam_angles, new_lidar_scan, world_objects, target, charger, area_width, area_height, max_range=10.0)

        # Update the robot pose and lidar scan for the next iteration.
        robot_pose = new_pose
        lidar_scan = new_lidar_scan
