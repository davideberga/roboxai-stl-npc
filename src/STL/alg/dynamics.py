import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


class DynamicsSimulator:
    def __init__(self):
        # Stack to record any invalid move attempts (new_pose, new_lidar_scan)
        self.invalid_attempts = []

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

        # Compute new pose using tensor operations.
        new_x = x + v * torch.cos(heading)
        new_y = y + v * torch.sin(heading)
        new_heading = heading + theta_motion
        new_pose = torch.stack([new_x, new_y, new_heading])

        # Compute the new lidar scan.
        new_scan = self.simulate_lidar_scan(new_pose, beam_angles, world_objects, max_range) if not use_perfection else self.simulate_perfect_lidar_scan(new_pose, beam_angles, world_objects, max_range)

        # Check for border violations.
        invalid = False
        if new_x.item() < robot_radius or new_x.item() > area_width - robot_radius or new_y.item() < robot_radius or new_y.item() > area_height - robot_radius:
            invalid = True

        # Check for collision with any obstacle.
        for obj in world_objects:
            if self.circle_rect_collision((new_x.item(), new_y.item()), robot_radius, obj):
                invalid = True
                break

        # If the move is invalid, revert to the old pose.
        if invalid:
            self.invalid_attempts.append((new_pose, new_scan))
            new_pose = old_pose
            new_scan = self.simulate_lidar_scan(old_pose, beam_angles, world_objects, max_range)

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

        circle_center_tensor = torch.tensor([cx, cy], dtype=torch.float32)
        closest_x = torch.clamp(circle_center_tensor[0], min=rect_min_x, max=rect_max_x)
        closest_y = torch.clamp(circle_center_tensor[1], min=rect_min_y, max=rect_max_y)
        distance = torch.sqrt((circle_center_tensor[0] - closest_x) ** 2 + (circle_center_tensor[1] - closest_y) ** 2)
        return distance.item() < circle_radius

    import torch

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

    # A differentiable version of estimate_destination.
    # Assumes dest is a tensor of shape (4,) with [cx, cy, width, height],
    # and robot_pose is a tensor of shape (3,) with [x, y, heading].
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

    def ray_rect_intersection(self, ray_origin, ray_direction, rect, max_range, beta=1.0, beta2=1.0, epsilon=1e-6):
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
            t_norm, t_angle = self.estimate_destination(pose_i, target[i], max_range)
            target_angle_list.append(t_angle)
            target_norm_list.append(t_norm)
            c_norm, c_angle = self.estimate_destination(pose_i, charger[i], max_range)
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
        mask = c_norm_tensor < 0.1
        mask_not_at_charger = c_norm_tensor > 0.1

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

    def visualize_environment(self, robot_pose, beam_angles, lidar_scan, world_objects, target, charger, area_width, area_height, max_range=10.0, ax=None):
        """
        Draw the environment including obstacles (gray), target (yellow square),
        charger (green square), robot (blue circle with arrow), and lidar beams.

        The lidar_scan is assumed to be normalized (in [0,1]); it is multiplied by max_range
        for drawing. The function accepts an optional axis (ax); if not provided, a new figure
        is created.
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
            if isinstance(target, torch.Tensor):
                cx, cy, w, h = target[0].item(), target[1].item(), target[2].item(), target[3].item()
            else:
                cx, cy = target["center"]
                w, h = target["width"], target["height"]
            lower_left = (cx - w / 2, cy - h / 2)
            target_patch = patches.Rectangle(lower_left, w, h, linewidth=2, edgecolor="orange", facecolor="yellow", alpha=0.8, label="Target")
            ax.add_patch(target_patch)

        # Draw charger.
        if charger is not None:
            if isinstance(charger, torch.Tensor):
                cx, cy, w, h = charger[0].item(), charger[1].item(), charger[2].item(), charger[3].item()
            else:
                cx, cy = charger["center"]
                w, h = charger["width"], charger["height"]
            lower_left = (cx - w / 2, cy - h / 2)
            charger_patch = patches.Rectangle(lower_left, w, h, linewidth=2, edgecolor="green", facecolor="lightgreen", alpha=0.8, label="Charger")
            ax.add_patch(charger_patch)

        # Draw robot.
        if isinstance(robot_pose, torch.Tensor):
            rx, ry, rtheta = robot_pose[0].item(), robot_pose[1].item(), robot_pose[2].item()
        else:
            rx, ry, rtheta = robot_pose
        ax.plot(rx, ry, "bo", markersize=8, label="Robot")
        arrow_length = 0.5
        ax.arrow(rx, ry, arrow_length * math.cos(rtheta), arrow_length * math.sin(rtheta), head_width=0.2, head_length=0.2, fc="blue", ec="blue")

        # Ensure beam_angles is a torch tensor.
        if not isinstance(beam_angles, torch.Tensor):
            beam_angles = torch.tensor(beam_angles, dtype=torch.float32)

        # Draw lidar beams.
        for beam, dist in zip(beam_angles, lidar_scan):
            # Convert beam angle to a float.
            beam_val = beam.item() if isinstance(beam, torch.Tensor) else beam
            # Compute the global angle: robot heading + beam relative angle.
            global_angle = rtheta + beam_val

            # Clamp the normalized distance to [0,1] and compute the actual distance.
            norm_dist = max(0.0, min(dist.item(), 1.0))
            actual_dist = norm_dist * max_range

            # Compute the end point using math.cos and math.sin.
            end_x = rx + actual_dist * math.cos(global_angle)
            end_y = ry + actual_dist * math.sin(global_angle)

            # Use a dashed line if the beam reading is 1 (i.e. no hit).
            style = "-" if norm_dist < 1.0 else "--"
            ax.plot([rx, end_x], [ry, end_y], style, color="red", linewidth=1)
            ax.plot(end_x, end_y, "ro", markersize=3)

        ax.set_aspect("equal")
        ax.set_title("Environment with Obstacles, Target, Charger, Robot, and Lidar Scan")
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
