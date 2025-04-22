from shapely import LineString
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from shapely.geometry import box, Point
from shapely.prepared import prep
import numpy as np

from .utils import rand_choice_tensor, soft_step_hard, stable_softmax, stable_softmin, uniform_tensor


class DynamicsSimulator:
    def __init__(self, wait_for_charging: int, steps_ahead: int, area_h: float, area_w: float, squared_area: bool, beam_angles: torch.Tensor, device: str, close_thres):
        # Math/torch config
        self.beta = 1
        self.beta2 = 5
        self.epsilon = 1e-4
        self.device = device
        self.steps_ahead = steps_ahead

        # Rover config
        self.rover_max_velocity = 10
        self.rover_min_velocity = 0.0
        self.beam_angles = beam_angles

        # Enviroment config
        self.area_h = area_h
        self.area_w = area_w
        self.max_range_destination = area_h if squared_area else max(area_h, area_w)
        self.max_range_lidar = area_h / 2 if squared_area else min(area_h, area_w)

        # Task config
        self.hold_t = wait_for_charging
        self.close_thres = close_thres
        self.enough_close_to_charger = close_thres
        self.battery_charge = 5
        self.dt = 0.2

    def walls(self):
        a = 0.1
        x_min, x_max, y_min, y_max = 0, self.area_h, 0, self.area_w
        map_walls = [
            {
                "center": [x_min + a, (y_min + y_max) / 2],
                "width": 3.2,
                "height": y_max - y_min,
            },  # Left wall
            {
                "center": [x_max - a, (y_min + y_max) / 2],
                "width": 3.2,
                "height": y_max - y_min,
            },  # Right wall
            {
                "center": [(x_min + x_max) / 2, y_max - a],
                "width": x_max - x_min,
                "height": 3.2,
            },  # Top wall
            {
                "center": [(x_min + x_max) / 2, y_min + a],
                "width": x_max - x_min,
                "height": 3.2,
            },  # Bottom wall
        ]
        return map_walls

    def ray_rect_intersection(self, ray_origins, ray_directions, world_objects, max_range):
        """
        Vectorized differentiable approximation to the ray–axis-aligned rectangle intersection.
        """
        # Compute rectangle boundaries.
        centers = world_objects[:, :2]
        widths = world_objects[:, 2].unsqueeze(1)
        heights = world_objects[:, 3].unsqueeze(1)
        min_xy = centers - torch.cat([widths / 2, heights / 2], dim=1)
        max_xy = centers + torch.cat([widths / 2, heights / 2], dim=1)
        min_x = min_xy[:, 0]
        min_y = min_xy[:, 1]
        max_x = max_xy[:, 0]
        max_y = max_xy[:, 1]

        B, num_beams, _ = ray_origins.shape
        N = world_objects.shape[0]

        ray_origins_exp = ray_origins.unsqueeze(2).expand(B, num_beams, N, 2)
        ray_directions_exp = ray_directions.unsqueeze(2).expand(B, num_beams, N, 2)

        min_x_exp = min_x.view(1, 1, N).expand(B, num_beams, N)
        max_x_exp = max_x.view(1, 1, N).expand(B, num_beams, N)
        min_y_exp = min_y.view(1, 1, N).expand(B, num_beams, N)
        max_y_exp = max_y.view(1, 1, N).expand(B, num_beams, N)

        # Extract ray components.
        ray_origin_x = ray_origins_exp[..., 0]
        ray_origin_y = ray_origins_exp[..., 1]
        ray_dir_x = ray_directions_exp[..., 0]
        ray_dir_y = ray_directions_exp[..., 1]

        tx1 = (min_x_exp - ray_origin_x) / (ray_dir_x + self.epsilon)
        tx2 = (max_x_exp - ray_origin_x) / (ray_dir_x + self.epsilon)

        tmin_x = stable_softmin(tx1, tx2, self.beta)
        tmax_x = stable_softmax(tx1, tx2, self.beta)

        ty1 = (min_y_exp - ray_origin_y) / (ray_dir_y + self.epsilon)
        ty2 = (max_y_exp - ray_origin_y) / (ray_dir_y + self.epsilon)
        tmin_y = stable_softmin(ty1, ty2, self.beta)
        tmax_y = stable_softmax(ty1, ty2, self.beta)

        tmin = stable_softmax(tmin_x, tmin_y, self.beta)
        tmax = stable_softmin(tmax_x, tmax_y, self.beta)

        valid_indicator = torch.sigmoid(self.beta2 * tmax) * torch.sigmoid(self.beta2 * (tmax - tmin))
        inside_indicator = 1.0 - torch.sigmoid(self.beta2 * tmin)
        t_intermediate = (1 - inside_indicator) * tmin + inside_indicator * tmax

        intersections = valid_indicator * t_intermediate + (1 - valid_indicator) * max_range
        return intersections

    def simulate_lidar_scan(self, robot_pose, world_objects):
        B = robot_pose.shape[0]
        num_beams = self.beam_angles.shape[0]
        global_angles = robot_pose[:, 2].unsqueeze(1) + self.beam_angles.unsqueeze(0)
        ray_dirs = torch.stack([torch.cos(global_angles), torch.sin(global_angles)], dim=-1)
        ray_origins = robot_pose[:, :2].unsqueeze(1).expand(B, num_beams, 2)

        import torch.nn.functional as F

        def soft_min_intersections(intersections, beta_soft=2.0):
            # intersections has shape (B, num_beams, N)
            weights = F.softmax(-beta_soft * intersections, dim=-1)
            soft_min = torch.sum(weights * intersections, dim=-1)
            return soft_min

        intersections = self.ray_rect_intersection(ray_origins, ray_dirs, world_objects, self.max_range_lidar)

        # Compute soft-minimum intersection distance for each ray
        min_intersections = soft_min_intersections(intersections, beta_soft=2.0)

        # Check if the robot is inside any object (Soft differentiable approach)
        centers = world_objects[:, :2]
        widths = world_objects[:, 2].unsqueeze(1)
        heights = world_objects[:, 3].unsqueeze(1)
        min_xy = centers - torch.cat([widths / 2, heights / 2], dim=1)
        max_xy = centers + torch.cat([widths / 2, heights / 2], dim=1)

        robot_pos = robot_pose[:, :2].unsqueeze(1)  # Shape (B, 1, 2)
        beta_inside = 10.0  # Higher beta -> sharper transition
        inside_x = torch.sigmoid(beta_inside * (robot_pos[..., 0] - min_xy[:, 0])) * torch.sigmoid(-beta_inside * (robot_pos[..., 0] - max_xy[:, 0]))
        inside_y = torch.sigmoid(beta_inside * (robot_pos[..., 1] - min_xy[:, 1])) * torch.sigmoid(-beta_inside * (robot_pos[..., 1] - max_xy[:, 1]))

        inside_soft = (inside_x * inside_y).sum(dim=-1)  # Shape (B,)

        # Smooth transition instead of hard switch
        epsilon = 1e-3
        alpha_inside = 10.0  # Controls smoothness of transition
        blending_factor = torch.sigmoid(alpha_inside * (inside_soft - 0.5))  # Smooth transition in [0,1]

        scan = blending_factor.unsqueeze(1) * epsilon + (1 - blending_factor.unsqueeze(1)) * (min_intersections / self.max_range_lidar)

        return scan  #  min_intersections / self.max_range_lidar

    def estimate_destination(self, robot_pose, dest):
        """
        Vectorized computation of the relative distance and angle from the rover to a destination.
        """
        rx = robot_pose[:, 0].unsqueeze(1)
        ry = robot_pose[:, 1].unsqueeze(1)
        rtheta = robot_pose[:, 2].unsqueeze(1)
        cx = dest[:, 0].unsqueeze(1)
        cy = dest[:, 1].unsqueeze(1)

        dx = cx - rx
        dy = cy - ry
        distance = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1)
        normalized_distance = torch.clamp(distance / self.max_range_destination, max=1.0)
        angle = torch.atan2(dy, dx + self.epsilon) # - rtheta
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        # old angle = torch.atan2(torch.sin(angle), torch.cos(angle) + self.epsilon)
        return normalized_distance, angle

    def update_state_batch(self, state, v, theta, robot_pose, world_objects, target, chargers, collision_enabled=False):
        """
        Fully vectorized update of the robot state for a batch of moves.
        """
        # --- Rescale velocity ---
        if not collision_enabled:
            v = v * (self.rover_max_velocity - self.rover_min_velocity) + self.rover_min_velocity

        # --- Update robot pose linearly ---
        # Predict angle displacement
        # new_x = robot_pose[:, 0] + (v * torch.cos(robot_pose[:, 2] + theta) * self.dt)
        # new_y = robot_pose[:, 1] + (v * torch.sin(robot_pose[:, 2] + theta) * self.dt)
        # new_heading = robot_pose[:, 2] + theta
        
        new_x = robot_pose[:, 0] + (v * torch.cos(theta) * self.dt)
        new_y = robot_pose[:, 1] + (v * torch.sin(theta) * self.dt)
        new_heading = theta

        new_pose = torch.stack([new_x, new_y, new_heading], dim=1)

        if collision_enabled:
            x_exp = new_pose[:, 0].unsqueeze(1)
            y_exp = new_pose[:, 1].unsqueeze(1)
            obs_cx = world_objects[:, 0].unsqueeze(0)
            obs_cy = world_objects[:, 1].unsqueeze(0)
            obs_w = world_objects[:, 2].unsqueeze(0)
            obs_h = world_objects[:, 3].unsqueeze(0)

            collision_mask = (torch.abs(x_exp - obs_cx) <= obs_w / 2) & (torch.abs(y_exp - obs_cy) <= obs_h / 2)
            collision_any = collision_mask.any(dim=1)
            
            if collision_any.shape[0] == 1 and collision_any[0]:
                return None, None

            # If a collision is detected, revert the pose to the previous one.
            new_pose[collision_any] = robot_pose[collision_any]
            

        new_scan = self.simulate_lidar_scan(new_pose, world_objects)
        t_norm, t_angle = self.estimate_destination(new_pose, target)

        robot_pos = new_pose[:, :2].unsqueeze(1)
        charger_centers = chargers[..., :2]
        diff = charger_centers - robot_pos
        dists = torch.norm(diff, dim=2)
        angles = torch.atan2(diff[..., 1], diff[..., 0] + self.epsilon)

        nearest_dists, min_idx = dists.min(dim=1)
        batch_indices = torch.arange(new_pose.shape[0], device=self.device)
        nearest_angles = angles[batch_indices, min_idx]

        nearest_dists = nearest_dists / self.max_range_destination

        # ADAPTED from the paper code
        battery_charge = 5
        near_charger = soft_step_hard(0.05 * (self.enough_close_to_charger - nearest_dists))
        # near_charger = (torch.tanh(500 * (0.05 * (self.enough_close_to_charger - nearest_dists))) + 1) / 2
        es_battery_time = (state[:, 11].unsqueeze(1) - self.dt) * (1 - near_charger.unsqueeze(1)) + battery_charge * near_charger.unsqueeze(1)
        es_charger_time = state[:, 12].unsqueeze(1) - self.dt * near_charger.unsqueeze(1)

        new_state = torch.cat(
            [
                new_scan,
                t_angle,
                t_norm,
                nearest_angles.unsqueeze(1),
                nearest_dists.unsqueeze(1),
                es_battery_time,
                es_charger_time,
            ],
            dim=1,
        ).to(self.device)

        return new_state, new_pose
    
    def update_state_batch_figure(self, state, v, theta, robot_pose, world_objects, target, chargers, collision_enabled=False):
        """
        Fully vectorized update of the robot state for a batch of moves.
        """
        # --- Rescale velocity ---
        if not collision_enabled:
            v = v * (self.rover_max_velocity - self.rover_min_velocity) + self.rover_min_velocity

        # --- Update robot pose linearly ---
        # Predict angle displacement
        # new_x = robot_pose[:, 0] + (v * torch.cos(robot_pose[:, 2] + theta) * self.dt)
        # new_y = robot_pose[:, 1] + (v * torch.sin(robot_pose[:, 2] + theta) * self.dt)
        # new_heading = robot_pose[:, 2] + theta
        
        new_x = robot_pose[:, 0] + (v * torch.cos(theta) * self.dt)
        new_y = robot_pose[:, 1] + (v * torch.sin(theta) * self.dt)
        new_heading = theta

        new_pose = torch.stack([new_x, new_y, new_heading], dim=1)

        collision_detected = False
        if collision_enabled:
            x_exp = new_pose[:, 0].unsqueeze(1)
            y_exp = new_pose[:, 1].unsqueeze(1)
            obs_cx = world_objects[:, 0].unsqueeze(0)
            obs_cy = world_objects[:, 1].unsqueeze(0)
            obs_w = world_objects[:, 2].unsqueeze(0)
            obs_h = world_objects[:, 3].unsqueeze(0)

            collision_mask = (torch.abs(x_exp - obs_cx) <= obs_w / 2) & (torch.abs(y_exp - obs_cy) <= obs_h / 2)
            collision_any = collision_mask.any(dim=1)
            
            if collision_any.shape[0] == 1:
                collision_detected = collision_any[0]

            # If a collision is detected, revert the pose to the previous one.
            new_pose[collision_any] = robot_pose[collision_any]
            

        new_scan = self.simulate_lidar_scan(new_pose, world_objects)
        t_norm, t_angle = self.estimate_destination(new_pose, target)

        robot_pos = new_pose[:, :2].unsqueeze(1)
        charger_centers = chargers[..., :2]
        diff = charger_centers - robot_pos
        dists = torch.norm(diff, dim=2)
        angles = torch.atan2(diff[..., 1], diff[..., 0] + self.epsilon)

        nearest_dists, min_idx = dists.min(dim=1)
        batch_indices = torch.arange(new_pose.shape[0], device=self.device)
        nearest_angles = angles[batch_indices, min_idx]

        nearest_dists = nearest_dists / self.max_range_destination

        # ADAPTED from the paper code
        battery_charge = 5
        near_charger = soft_step_hard(0.05 * (self.enough_close_to_charger - nearest_dists))
        # near_charger = (torch.tanh(500 * (0.05 * (self.enough_close_to_charger - nearest_dists))) + 1) / 2
        es_battery_time = (state[:, 11].unsqueeze(1) - self.dt) * (1 - near_charger.unsqueeze(1)) + battery_charge * near_charger.unsqueeze(1)
        es_charger_time = state[:, 12].unsqueeze(1) - self.dt * near_charger.unsqueeze(1)

        new_state = torch.cat(
            [
                new_scan,
                t_angle,
                t_norm,
                nearest_angles.unsqueeze(1),
                nearest_dists.unsqueeze(1),
                es_battery_time,
                es_charger_time,
            ],
            dim=1,
        ).to(self.device)

        return new_state, new_pose, collision_detected

    def visualize_environment(self, robot_pose, lidar_scan, world_objects, target, chargers, battery_level=1.0, ax=None):
        """
        Visualizes the environment with obstacles, target, charger, robot, and lidar scan.
        Displays lidar scan distances on the rays and the battery level outside of the map.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(0, self.area_w)
        ax.set_ylim(0, self.area_h)

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
            target_patch = patches.Rectangle(
                lower_left,
                w,
                h,
                linewidth=2,
                edgecolor="orange",
                facecolor="yellow",
                alpha=0.8,
                label="Target",
            )
            ax.add_patch(target_patch)

        # Draw charger.
        if chargers is not None:
            for charger in chargers:
                cx, cy, w, h = charger[:4].tolist()
                lower_left = (cx - w / 2, cy - h / 2)
                charger_patch = patches.Rectangle(
                    lower_left,
                    w,
                    h,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="lightgreen",
                    alpha=0.8,
                    label="Charger",
                )
                ax.add_patch(charger_patch)

        # Draw robot.
        rx, ry, rtheta = robot_pose[:3].tolist()
        ax.plot(rx, ry, "bo", markersize=8, label="Robot")
        arrow_length = 0.5
        ax.arrow(
            rx,
            ry,
            arrow_length * math.cos(rtheta),
            arrow_length * math.sin(rtheta),
            head_width=0.2,
            head_length=0.2,
            fc="blue",
            ec="blue",
        )

        beam_angles = self.beam_angles
        if not isinstance(self.beam_angles, torch.Tensor):
            beam_angles = torch.tensor(self.beam_angles, dtype=torch.float32)

        # Draw lidar beams and distances.
        for beam, dist in zip(beam_angles, lidar_scan):
            beam_val = beam.item()
            global_angle = rtheta + beam_val
            norm_dist = max(0.0, min(dist.item(), 1.0))
            actual_dist = norm_dist * self.max_range_lidar
            end_x = rx + actual_dist * math.cos(global_angle)
            end_y = ry + actual_dist * math.sin(global_angle)
            style = "-" if norm_dist < 1.0 else "--"
            ax.plot([rx, end_x], [ry, end_y], style, color="red", linewidth=1)
            ax.plot(end_x, end_y, "ro", markersize=3)
            text_offset = 0.2
            text_x = end_x + text_offset * math.cos(global_angle)
            text_y = end_y + text_offset * math.sin(global_angle)
            ax.text(text_x, text_y, f"{dist:.2f}", fontsize=8, color="black", ha="center", va="center")

        ax.text(
            self.area_w / 2,
            self.area_h / 2,
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

    def visualize_environment_v2(self, robot_pose, lidar_scan, world_objects, target, chargers, poses, ax=None):
        """
        Visualizes the environment with obstacles, target, charger, robot, and lidar scan.
        Displays lidar scan distances on the rays and the battery level outside of the map.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(-1, self.area_w + 1)
        ax.set_ylim(-1, self.area_h + 1)

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
            target_patch = patches.Rectangle(
                lower_left,
                w,
                h,
                linewidth=2,
                edgecolor="orange",
                facecolor="yellow",
                alpha=0.8,
                label="Target",
            )
            ax.add_patch(target_patch)

        # Draw charger.
        if chargers is not None:
            for charger in chargers:
                cx, cy, w, h = charger[:4].tolist()
                lower_left = (cx - w / 2, cy - h / 2)
                charger_patch = patches.Rectangle(
                    lower_left,
                    w,
                    h,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="lightgreen",
                    alpha=0.8,
                    label="Charger",
                )
                ax.add_patch(charger_patch)

        # Draw robot.
        rx, ry, rtheta = robot_pose[:3].tolist()
        ax.plot(rx, ry, "bo", markersize=8, label="Robot")
        arrow_length = 0.5
        ax.arrow(
            rx,
            ry,
            arrow_length * math.cos(rtheta),
            arrow_length * math.sin(rtheta),
            head_width=0.2,
            head_length=0.2,
            fc="blue",
            ec="blue",
        )

        beam_angles = self.beam_angles
        if not isinstance(self.beam_angles, torch.Tensor):
            beam_angles = torch.tensor(self.beam_angles, dtype=torch.float32)

        # Draw lidar beams and distances.
        for beam, dist in zip(beam_angles, lidar_scan):
            beam_val = beam.item()
            global_angle = rtheta + beam_val
            norm_dist = max(0.0, min(dist.item(), 1.0))
            actual_dist = norm_dist * self.max_range_lidar
            end_x = rx + actual_dist * math.cos(global_angle)
            end_y = ry + actual_dist * math.sin(global_angle)
            style = "-" if norm_dist < 1.0 else "--"
            ax.plot([rx, end_x], [ry, end_y], style, color="red", linewidth=1)
            ax.plot(end_x, end_y, "ro", markersize=3)
            text_offset = 0.2
            text_x = end_x + text_offset * math.cos(global_angle)
            text_y = end_y + text_offset * math.sin(global_angle)
            ax.text(text_x, text_y, f"{dist:.2f}", fontsize=8, color="black", ha="center", va="center")

        ax.plot(
            poses[:, 0],
            poses[:, 1],
            color="blue",
            linewidth=2,
            alpha=0.5,
            zorder=10,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

    def initialize_x_cycle(self, n, test=False):
        charger_x = uniform_tensor(0, 10, (n, 1))
        charger_y = uniform_tensor(0, 10, (n, 1))

        closeness = 10 if test else 0.8
        MAX_BATTERY_N = 25
        battery_t = rand_choice_tensor([self.dt * nn for nn in range(MAX_BATTERY_N + 1)], (n, 1))
        rover_theta = uniform_tensor(-np.pi, np.pi, (n, 1))
        rover_rho = uniform_tensor(0, 1, (n, 1)) * (battery_t * self.rover_max_velocity)
        rover_rho = torch.clamp(rover_rho, closeness, 14.14)

        rover_x = charger_x + rover_rho * torch.cos(rover_theta)
        rover_y = charger_y + rover_rho * torch.sin(rover_theta)

        dest_x = uniform_tensor(0, 10, (n, 1))
        dest_y = uniform_tensor(0, 10, (n, 1))

        # place hold case
        ratio = 0.25 if not test else 0.0
        rand_mask = uniform_tensor(0, 1, (n, 1))
        rand = rand_mask > 1 - ratio
        ego_rho = uniform_tensor(0, closeness, (n, 1))
        rover_x[rand] = (charger_x + ego_rho * torch.cos(rover_theta))[rand]
        rover_y[rand] = (charger_y + ego_rho * torch.sin(rover_theta))[rand]
        # battery_t[rand] = np.random.random() * (2 - 0.2) + 0.2
        battery_t[rand] = self.dt * MAX_BATTERY_N  # np.random.random() * (2.5 - 1.5) + 1.5

        hold_t = 0 * dest_x + self.dt * self.hold_t
        hold_t[rand] = rand_choice_tensor([self.dt * nn for nn in range(self.hold_t + 1)], (n, 1))[rand]

        return torch.cat(
            [rover_x, rover_y, dest_x, dest_y, charger_x, charger_y, battery_t, hold_t],
            dim=1,
        ), rover_theta

    def generate_objects(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        objs_np.append(np.array([[0.0, 0.0], [obs_w, 0], [obs_w, obs_w], [0, obs_w]]))  # first obstacle
        objs_np.append(objs_np[1] + np.array([[5 - obs_w / 2, 10 - obs_w]]))  # second obstacle (top-center)
        objs_np.append(objs_np[1] + np.array([[10 - obs_w, 0]]))  # third obstacle (bottom-right)
        objs_np.append(objs_np[1] / 2 + np.array([[5 - obs_w / 4, 5 - obs_w / 4]]))  # forth obstacle (center-center, shrinking)

        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar
        walls_w = obs_w
        objs_np.append(np.array([[0.0, -10], [-walls_w, -10], [-walls_w, 20], [0, 20]]))
        objs_np.append(np.array([[0.0, 0.0], [10, 0], [10, -walls_w], [0, -walls_w]]))
        objs_np.append(np.array([[10.0 + walls_w, -10], [10, -10], [10, 20], [10 + walls_w, 20]]))
        objs_np.append(np.array([[0.0, 10], [10, 10], [10, 10.0 + walls_w], [0.0, 10 + walls_w]]))

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2
    
    def generate_objects_different(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        objs_np.append(np.array([[0.0, 10.0], [obs_w, 10], [obs_w, 10 - obs_w], [0, 10 - obs_w]]))  # first obstacle
        objs_np.append(np.array([[10-obs_w, 10.0], [10, 10], [10, 10 - obs_w], [10-obs_w, 10 - obs_w]])) 
        objs_np.append(np.array([[5 - obs_w / 2, obs_w], [5 + obs_w / 2, 0], [5 + obs_w / 2, obs_w], [5 - obs_w / 2, 0]]))
        objs_np.append(np.array([[5 - obs_w / 4, 6], [5 + obs_w / 4, obs_w], [5 + obs_w / 4, obs_w], [5 - obs_w / 4, 5]]))  
        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar
        walls_w = obs_w
        objs_np.append(np.array([[0.0, -10], [-walls_w, -10], [-walls_w, 20], [0, 20]]))
        objs_np.append(np.array([[0.0, 0.0], [10, 0], [10, -walls_w], [0, -walls_w]]))
        objs_np.append(np.array([[10.0 + walls_w, -10], [10, -10], [10, 20], [10 + walls_w, 20]]))
        objs_np.append(np.array([[0.0, 10], [10, 10], [10, 10.0 + walls_w], [0.0, 10 + walls_w]]))

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2
    
    def generate_objects_different_v2(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        objs_np.append(np.array([[0.0, 10.0], [obs_w, 10], [obs_w, 10 - obs_w], [0, 10 - obs_w]]))  # first obstacle
        objs_np.append(np.array([[10-obs_w, 10.0], [10, 10], [10, 10 - obs_w], [10-obs_w, 10 - obs_w]])) 
        objs_np.append(np.array([[5 - obs_w / 2, obs_w], [5 + obs_w / 2, 0], [5 + obs_w / 2, obs_w], [5 - obs_w / 2, 0]]))
        objs_np.append(np.array([[5 - obs_w / 3, 4], [5 + obs_w / 3, 5], [5 + obs_w / 3, 5], [5 - obs_w / 3, 5]])) 
        objs_np.append(np.array([[8 - obs_w / 3, 6], [8 + obs_w / 3, 6], [8 + obs_w / 3, 4], [8 - obs_w / 3, 4]])) 
        objs_np.append(np.array([[2 - obs_w / 3, 6], [2 + obs_w / 3, 6], [2 + obs_w / 3, 4], [2 - obs_w / 3, 4]])) 
        # objs_np.append(np.array([[5 - obs_w / 4, 4], [5 + obs_w / 4, obs_w], [5 + obs_w / 4, obs_w], [5 - obs_w / 4, 5]]))  
        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar
        walls_w = obs_w
        objs_np.append(np.array([[0.0, -10], [-walls_w, -10], [-walls_w, 20], [0, 20]]))
        objs_np.append(np.array([[0.0, 0.0], [10, 0], [10, -walls_w], [0, -walls_w]]))
        objs_np.append(np.array([[10.0 + walls_w, -10], [10, -10], [10, 20], [10 + walls_w, 20]]))
        objs_np.append(np.array([[0.0, 10], [10, 10], [10, 10.0 + walls_w], [0.0, 10 + walls_w]]))

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2
    
    def generate_no_obstacles(self):
        obs_w = 3.0
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        def to_torch(x, device):
            return torch.from_numpy(x).float().to(device)

        # Set walls for lidar
        walls_w = obs_w
        objs_np.append(np.array([[0.0, -10], [-walls_w, -10], [-walls_w, 20], [0, 20]]))
        objs_np.append(np.array([[0.0, 0.0], [10, 0], [10, -walls_w], [0, -walls_w]]))
        objs_np.append(np.array([[10.0 + walls_w, -10], [10, -10], [10, 20], [10 + walls_w, 20]]))
        objs_np.append(np.array([[0.0, 10], [10, 10], [10, 10.0 + walls_w], [0.0, 10 + walls_w]]))

        objs = [to_torch(ele, self.device) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

        return objs_np, objs, objs_t1, objs_t2
    
    def generate_random_objects(self,
                                num_samples: int,
                                obj_count: int = 8,
                                size_range: tuple = (3, 4)):
        samples = []
        map_corners = torch.tensor([
            [0.0, 0.0],
            [self.area_w, 0.0],
            [self.area_w, self.area_h],
            [0.0, self.area_h]
        ], dtype=torch.float32, device=self.device)

        # Precompute wall polygons
        wall_polys = []
        walls_w = 3.0
        wall_polys.append(torch.tensor([[0.0, -10], [-walls_w, -10], [-walls_w, 20], [0, 20]], dtype=torch.float32, device=self.device))
        wall_polys.append(torch.tensor([[0.0, 0.0], [10, 0], [10, -walls_w], [0, -walls_w]], dtype=torch.float32, device=self.device))
        wall_polys.append(torch.tensor([[10.0 + walls_w, -10], [10, -10], [10, 20], [10 + walls_w, 20]], dtype=torch.float32, device=self.device))
        wall_polys.append(torch.tensor([[0.0, 10], [10, 10], [10, 10.0 + walls_w], [0.0, 10 + walls_w]], dtype=torch.float32, device=self.device))

        min_size, max_size = size_range
        for _ in range(num_samples):
            polys = [map_corners] + wall_polys.copy()
            # Random obstacle sizes
            widths = torch.empty(obj_count, device=self.device).uniform_(min_size, max_size)
            heights = torch.empty(obj_count, device=self.device).uniform_(min_size, max_size)
            # Random centers within valid range
            rand = torch.rand(obj_count, device=self.device)
            cx = (widths / 2) + rand * (self.area_w - widths)
            rand = torch.rand(obj_count, device=self.device)
            cy = (heights / 2) + rand * (self.area_h - heights)
            # Build obstacle polygons
            for w_i, h_i, cx_i, cy_i in zip(widths, heights, cx, cy):
                half_w = w_i / 2
                half_h = h_i / 2
                corners = torch.stack([
                    torch.tensor([cx_i - half_w, cy_i - half_h], device=self.device),
                    torch.tensor([cx_i + half_w, cy_i - half_h], device=self.device),
                    torch.tensor([cx_i + half_w, cy_i + half_h], device=self.device),
                    torch.tensor([cx_i - half_w, cy_i + half_h], device=self.device)
                ], dim=0)
                polys.append(corners)

            samples.append(polys)

        return samples



    def transform_objects(self, objs):
        result = []
        for obj in objs:
            min_xy = obj.min(dim=0).values
            max_xy = obj.max(dim=0).values
            center = (min_xy + max_xy) / 2
            size = max_xy - min_xy
            result.append(torch.cat([center, size]))
        return torch.stack(result)

    def initialize_x(self, n, objs, test=False):
        x_list = []
        x_theta = []
        total_n = 0
        while total_n < n:
            x_init, thetas = self.initialize_x_cycle(n, test)
            valids = []
            for obj_i, obj in enumerate(objs):
                obs_cpu = obj.detach().cpu()
                xmin, xmax, ymin, ymax = (
                    torch.min(obs_cpu[:, 0]),
                    torch.max(obs_cpu[:, 0]),
                    torch.min(obs_cpu[:, 1]),
                    torch.max(obs_cpu[:, 1]),
                )

                for x, y in [
                    (x_init[:, 0], x_init[:, 1]),
                    (x_init[:, 2], x_init[:, 3]),
                    (x_init[:, 4], x_init[:, 5]),
                ]:
                    if obj_i == 0:  # in map
                        val = torch.logical_and(
                            (x - xmin) * (xmax - x) >= 0,
                            (y - ymin) * (ymax - y) >= 0,
                        )
                    else:  # avoid obstacles
                        val = torch.logical_not(
                            torch.logical_and(
                                (x - xmin) * (xmax - x) >= 0,
                                (y - ymin) * (ymax - y) >= 0,
                            )
                        )
                    valids.append(val)

            valids = torch.stack(valids, dim=-1)
            valids_indices = torch.where(torch.all(valids, dim=-1) == True)[0]
            x_val = x_init[valids_indices]
            total_n += x_val.shape[0]
            x_list.append(x_val)
            x_theta.append(thetas[valids_indices])

        x_list = torch.cat(x_list, dim=0)[:n]
        x_theta = torch.cat(x_theta, dim=0)[:n]

        tensor_objs_cx_cy_w_h = torch.tensor(self.transform_objects(objs)).float().to(self.device)

        # Reduce dimension of objects, distance from objects
        tensor_objs_cx_cy_w_h[:, 2] -= 0.5
        tensor_objs_cx_cy_w_h[:, 3] -= 0.5

        # Remove map from obstacles
        obstacles = tensor_objs_cx_cy_w_h[1:]

        robot_pose = torch.cat((x_list[:, :2], x_theta), dim=1).float().to(self.device)
        target_position = x_list[:, 2:4].float().to(self.device)
        charger_position = x_list[:, 4:6].float().to(self.device)
        battery_time_hold = x_list[:, 6:].float().to(self.device)

        scan = self.simulate_lidar_scan(robot_pose, obstacles)
        target_dist, target_angle = self.estimate_destination(robot_pose, target_position)
        charger_dist, charger_angle = self.estimate_destination(robot_pose, charger_position)

        new_state = (
            torch.cat(
                (
                    scan,
                    target_angle,
                    target_dist,
                    charger_angle,
                    charger_dist,
                    battery_time_hold,
                ),
                dim=1,
            )
            .float()
            .to(self.device)
        )

        return (
            new_state,
            tensor_objs_cx_cy_w_h,
            robot_pose,
            target_position,
            charger_position,
        )


    def initialize_x_hard(self, n, objs, test=False):
        x_list = []
        x_theta = []
        total_n = 0

        # Precompute obstacle bounding‐boxes (skip the first obj, which is the map)
        bboxes = []
        for obj in objs[1:]:
            obs_cpu = obj.detach().cpu()
            xmin, xmax = torch.min(obs_cpu[:, 0]), torch.max(obs_cpu[:, 0])
            ymin, ymax = torch.min(obs_cpu[:, 1]), torch.max(obs_cpu[:, 1])
            bboxes.append((xmin, xmax, ymin, ymax))

        while total_n < n:
            x_init, thetas = self.initialize_x_cycle(n, test)   # shape [n, …]

            # 1) your original “in‐map” and “outside‐obstacle” checks
            valids = []
            for obj_i, obj in enumerate(objs):
                obs_cpu = obj.detach().cpu()
                xmin, xmax = torch.min(obs_cpu[:, 0]), torch.max(obs_cpu[:, 0])
                ymin, ymax = torch.min(obs_cpu[:, 1]), torch.max(obs_cpu[:, 1])

                for dim in range(0, 6, 2):
                    x, y = x_init[:, dim], x_init[:, dim+1]
                    inside = (x - xmin) * (xmax - x) >= 0
                    inside &= (y - ymin) * (ymax - y) >= 0

                    if obj_i == 0:
                        # must lie inside the map
                        valids.append(inside)
                    else:
                        # must lie *outside* each obstacle
                        valids.append(~inside)

            valids = torch.stack(valids, dim=-1)
            base_mask = torch.all(valids, dim=-1)   # shape [n]

            # 2) NEW: straight‐line–collision check to the *target* (x_init[:,2:4])
            #    we only accept samples where the line from robot→target *does* intersect an obstacle
            robot_x, robot_y = x_init[:, 0], x_init[:, 1]
            target_x, target_y = x_init[:, 2], x_init[:, 3]

            # start with “no collision” everywhere
            blocked = torch.zeros_like(robot_x, dtype=torch.bool)

            # for each obstacle bbox, mark any segment that crosses it
            for xmin, xmax, ymin, ymax in bboxes:
                # trivial‐reject test:
                #   no intersection if both endpoints are entirely on one side of the rect
                reject = ((robot_x < xmin) & (target_x < xmin)) | \
                        ((robot_x > xmax) & (target_x > xmax)) | \
                        ((robot_y < ymin) & (target_y < ymin)) | \
                        ((robot_y > ymax) & (target_y > ymax))

                # if not rejected, it means the line *does* intersect this bbox
                blocked |= ~reject

            # combine both masks
            final_mask = base_mask & blocked
            valid_indices = torch.where(final_mask)[0]

            # collect
            x_list.append(x_init[valid_indices])
            x_theta.append(thetas[valid_indices])
            total_n += valid_indices.shape[0]

        # truncate to exactly n
        x_list = torch.cat(x_list, dim=0)[:n]
        x_theta = torch.cat(x_theta, dim=0)[:n]

        # … the rest of your code stays the same …
        tensor_objs_cx_cy_w_h = torch.tensor(self.transform_objects(objs)).float().to(self.device)
        tensor_objs_cx_cy_w_h[:, 2:] -= 0.5
        obstacles = tensor_objs_cx_cy_w_h[1:]

        robot_pose    = torch.cat((x_list[:, :2], x_theta), dim=1).float().to(self.device)
        target_position  = x_list[:, 2:4].float().to(self.device)
        charger_position = x_list[:, 4:6].float().to(self.device)
        battery_time_hold = x_list[:, 6:].float().to(self.device)

        scan              = self.simulate_lidar_scan(robot_pose, obstacles)
        target_dist, target_angle   = self.estimate_destination(robot_pose, target_position)
        charger_dist, charger_angle = self.estimate_destination(robot_pose, charger_position)

        new_state = torch.cat([
            scan,
            target_angle,
            target_dist,
            charger_angle,
            charger_dist,
            battery_time_hold,
        ], dim=1).to(self.device)

        return (
            new_state,
            tensor_objs_cx_cy_w_h,
            robot_pose,
            target_position,
            charger_position,
        )
