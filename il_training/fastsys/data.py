import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
from time import time
from io import BytesIO
import random
from matplotlib.colors import LinearSegmentedColormap
import math

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent
sys.path.append(str(root_path))

try:
    import io_tool as OSSIO
    oss_offine = OSSIO.OSSOffline.fromOssPath(oss_path='oss://your-bucket/')
    oss_online = OSSIO.OSSOnline.fromOssPath(oss_path='oss://your-bucket/')
    OSS_AVAILABLE = True
except ImportError:
    print("[WARNING] io_tool not found. OSS functionality disabled. This is OK for inference.")
    oss_offine = None
    oss_online = None
    OSS_AVAILABLE = False

def world_to_grid(target_points, grid_resolution=0.06, grid_size=127):
    """
    Convert target points from world coordinates to grid coordinates.
    Args:
        target_points: (B, 2) target points in world coordinates, origin at grid center
        grid_resolution: grid resolution in meters per cell
        grid_size: grid size (must be odd number)
    Returns:
        (B, 2) grid coordinates (x, y), range [0, grid_size-1]
    """
    device = target_points.device
    center = (grid_size - 1) // 2
    center_tensor = torch.tensor([center, center], device=device, dtype=torch.float32)

    grid_coords_float = target_points / grid_resolution + center_tensor
    grid_coords = torch.round(grid_coords_float).long()
    grid_coords = torch.clamp(grid_coords, min=0, max=grid_size-1)
    
    return grid_coords

def rotation_axis_counterclockwise(input):
    output = input.clone()
    output[:,0] = input[:,1]
    output[:,1] = -input[:,0]
    return output
    
def rotation_axis_clockwise(input):
    output = input.clone()
    output[:,0] = -input[:,1]
    output[:,1] = input[:,0]
    return output

def get_state(state, target, obstacle, action, device, lidar_range=4.0):
    '''
        state: (n, 6)
        target: (n, 2)
        obstacle: (n, 1, 127, 127)
        action: (n, 3)
    '''
    batch_raycast_occ_res = batched_vectorized_raycast(obstacle[:,0,:,:], state[:, :2], target[:, :2],
                                      start_angle=90, end_angle=90+360, angle_interval=2.5) 
    ray_cast = batch_raycast_occ_res[0]
    
    ray_cast_processed = ray_cast.unsqueeze(dim=1).unsqueeze(-1)
    ray_cast_processed = (lidar_range - ray_cast_processed).clamp_max(lidar_range).clamp_min(0.0)
    
    target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
    target_dir = target / target_norm
    zeros = torch.zeros(target.shape[0], 1, device=device, dtype=target.dtype)
    target_dir_3d = torch.cat([target_dir, zeros], dim=-1)
    target_dir_3d = rotation_axis_counterclockwise(target_dir_3d)
    velo_3d = torch.cat([state[:,3:5], zeros], dim=-1)
    yaw_vec_2d = torch.cat([torch.cos(state[:,2:3]), torch.sin(state[:,2:3])], dim=1)
    yaw_vec_3d = torch.cat([yaw_vec_2d, zeros], dim=-1)
    yaw_rate = state[:,5:6]
    state_processed = torch.cat([target_dir_3d, target_norm, velo_3d, yaw_vec_3d, yaw_rate], dim=-1)
    
    action_processed = None
    if action is not None:
        action_processed = action
    
    return state_processed, ray_cast_processed, action_processed




def vectorized_raycast(occ_grid, pos_meters, target, grid_resolution=0.06, occ_size=127, 
                       start_angle=-90, end_angle=270, angle_interval=1):
    """
    Vectorized raycast on occupancy grid.
    Args:
        occ_grid: (127, 127) binary occupancy grid, 1 indicates obstacle
        pos_meters: (2,) current position in world coordinates (origin at grid center)
        target: (2,) target point in world coordinates
        grid_resolution: physical size of each grid cell in meters
        occ_size: grid size (default 127x127)
    Returns:
        (360,) raycast features, each element is distance to nearest obstacle
    """
    t0 = time()
    
    device = occ_grid.device
    
    angles_deg = torch.arange(start_angle, end_angle, angle_interval, device=device)
    angles_rad = torch.deg2rad(angles_deg)
    dirs = torch.stack([torch.cos(angles_rad), torch.sin(angles_rad)], dim=1)

    center = (occ_size // 2, occ_size // 2)
    pos_grid = pos_meters / grid_resolution + torch.tensor(center, device=device).float()
    pos_grid = pos_grid.view(1, 2)

    max_dist = (occ_size ** 2 + occ_size ** 2) ** 0.5 * grid_resolution / 2
    end_points = pos_meters + dirs * max_dist
    end_points_grid = end_points / grid_resolution + torch.tensor(center, device=device).float()

    steps = torch.linspace(0, 1, steps=int(max_dist / grid_resolution), device=device)
    steps = steps.view(-1, 1, 1)
    
    line_points = pos_grid + steps * (end_points_grid - pos_grid)
    line_points = line_points.round().long()

    valid_mask = (line_points[..., 0] >= 0) & (line_points[..., 0] < occ_size) & \
                 (line_points[..., 1] >= 0) & (line_points[..., 1] < occ_size)
    
    occ_expanded = occ_grid.unsqueeze(0).expand(line_points.shape[0], -1, -1)
    x_coords = line_points[..., 0].clamp(0, occ_size-1)
    y_coords = line_points[..., 1].clamp(0, occ_size-1)
    hits = occ_expanded[torch.arange(line_points.shape[0], device=device)[:, None], x_coords, y_coords]

    hit_mask = (hits > 0) & valid_mask
    hit_indices = hit_mask.float().argmax(dim=0)

    step_dist = grid_resolution
    distances = hit_indices.float() * step_dist
    
    no_hit_mask = (hit_mask.sum(dim=0) == 0)
    distances[no_hit_mask] = max_dist

    hit_occ = torch.zeros_like(occ_grid, device=device, dtype=torch.float32)

    valid_hit = ~no_hit_mask
    
    if valid_hit.any():
        steps = hit_indices[valid_hit]
        rays = valid_hit.nonzero().squeeze(-1)
        
        hit_x = x_coords[steps, rays]
        hit_y = y_coords[steps, rays]
        
        hit_occ[hit_x, hit_y] = 1.
    target_occ = world_to_grid(target)
    hit_occ[target_occ[0], target_occ[1]] = 2.

    t1 = time()
    # print('t1-0: {}'.format(t1 - t0))
    
    return distances, hit_occ  # (360,)

def batched_vectorized_raycast(occ_grid, pos_meters, targets, grid_resolution=0.06, occ_size=127, 
                             start_angle=-90, end_angle=270, angle_interval=1):
    """
    Batched version of vectorized raycast.
    Args:
        occ_grid: (B, 127, 127) binary occupancy grids
        pos_meters: (B, 2) current positions in world coordinates
        targets: (B, 2) target point coordinates
    Returns:
        distances: (B, 360) obstacle distances for each angle
        hit_occ: (B, 127, 127) hit occupancy map for each sample
    """
    B = occ_grid.shape[0]
    device = occ_grid.device

    angles_deg = torch.arange(start_angle, end_angle, angle_interval, device=device)
    angles_rad = torch.deg2rad(angles_deg)
    dirs = torch.stack([torch.cos(angles_rad), torch.sin(angles_rad)], dim=1)
    dirs = dirs.unsqueeze(0)

    center = torch.tensor([occ_size//2, occ_size//2], device=device).float()
    pos_grid = pos_meters / grid_resolution + center
    pos_grid = pos_grid.view(B, 1, 1, 2)

    max_dist = (occ_size**2 * 2)**0.5 * grid_resolution / 2
    end_points = pos_meters.unsqueeze(1) + dirs * max_dist
    end_points_grid = end_points / grid_resolution + center.unsqueeze(0).unsqueeze(0)

    num_steps = int(max_dist / grid_resolution)
    steps = torch.linspace(0, 1, num_steps, device=device)
    steps = steps.view(1, 1, -1, 1)

    line_points = pos_grid + steps * (end_points_grid.unsqueeze(2) - pos_grid)
    line_points = line_points.round().long()
    line_points = line_points.permute(0, 2, 1, 3)

    valid_mask = (line_points[..., 0] >= 0) & (line_points[..., 0] < occ_size) & \
                 (line_points[..., 1] >= 0) & (line_points[..., 1] < occ_size)

    x_coords = line_points[..., 0].clamp(0, occ_size-1)
    y_coords = line_points[..., 1].clamp(0, occ_size-1)

    batch_idx = torch.arange(B, device=device)[:, None, None]
    hits = occ_grid[batch_idx, x_coords, y_coords]

    hit_mask = (hits > 0) & valid_mask
    hit_indices = hit_mask.float().argmax(dim=1)

    step_dist = grid_resolution
    distances = hit_indices.float() * step_dist
    no_hit_mask = hit_mask.sum(dim=1) == 0
    distances[no_hit_mask] = max_dist

    hit_occ = torch.zeros_like(occ_grid, dtype=torch.float32)
    
    valid_hit = ~no_hit_mask
    batch_idx, ray_idx = torch.where(valid_hit)
    
    if len(batch_idx) > 0:
        step_idx = hit_indices[batch_idx, ray_idx]
        x = x_coords[batch_idx, step_idx, ray_idx]
        y = y_coords[batch_idx, step_idx, ray_idx]
        
        hit_occ = hit_occ.index_put(
            indices=(batch_idx, x, y),
            values=torch.ones_like(x, dtype=torch.float32),
            accumulate=True
        )

    targets_grid = (targets / grid_resolution + center).long()
    targets_grid = targets_grid.clamp(0, occ_size-1)
    hit_occ[torch.arange(B, device=device), targets_grid[:, 0], targets_grid[:, 1]] = 2.0

    return distances, hit_occ

    
def visualize_raycast(occ_grid, pos_meters, ray_distances, ray_cast_occ, grid_resolution=0.06, occ_size=127,
                      start_angle=-90, end_angle=270, angle_interval=1):
    """
    Visualize occupancy map, robot position and raycast results.
    Args:
        occ_grid: torch.Tensor (127, 127) binary occupancy grid
        pos_meters: torch.Tensor (2,) robot position in world coordinates
        ray_distances: torch.Tensor (360,) raycast distance features
        grid_resolution: physical size of each grid cell in meters
        occ_size: grid size (default 127)
    """
    occ_np = occ_grid.cpu().numpy() if occ_grid.is_cuda else occ_grid.numpy()
    pos_np = pos_meters.cpu().numpy() if pos_meters.is_cuda else pos_meters.numpy()
    ray_distances_np = ray_distances.cpu().numpy() if ray_distances.is_cuda else ray_distances.numpy()
    ray_cast_occ_np = ray_cast_occ.cpu().numpy() if occ_grid.is_cuda else occ_grid.numpy()
    
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 10))
    
    ax1.imshow(ray_cast_occ_np.T, cmap='Greys', origin='lower', extent=[
        -occ_size, 
        occ_size,
        -occ_size,
        occ_size
    ])
    
    ax.imshow(occ_np.T, cmap='Greys', origin='lower', extent=[
        -occ_size//2 * grid_resolution, 
        occ_size//2 * grid_resolution,
        -occ_size//2 * grid_resolution,
        occ_size//2 * grid_resolution
    ])
    
    ax.plot(pos_np[0], pos_np[1], 'ro', markersize=8, label='Robot Position')
    
    angles_deg = np.arange(start_angle, end_angle, angle_interval)
    angles_rad = np.deg2rad(angles_deg)
    dirs = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
    
    end_points = pos_np + dirs * ray_distances_np[:, None]
    
    for i in range(int((end_angle - start_angle) / angle_interval)):
        ax.plot([pos_np[0], end_points[i, 0]], 
                [pos_np[1], end_points[i, 1]], 
                color='blue', alpha=0.2, linewidth=0.5)
    
    hit_points = end_points[ray_distances_np < (occ_size * grid_resolution * 1.414)]
    colors = [(0.0, 0.0, 0.0),
             (1., 1., 1.)]
    cmap = LinearSegmentedColormap.from_list("depth_green", colors)
    if len(hit_points) > 0:
        ax.scatter(hit_points[:, 0], hit_points[:, 1], cmap=cmap, s=10, label='Hit Points')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Raycast Visualization')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()

class EmbodiedDataset(Dataset):
    def __init__(self, file_list, min_action, max_action, oss_mode=False, output_norm=True):
        self.oss_mode = oss_mode
        self.file_list = file_list
        self.min_action = torch.tensor(min_action, dtype=torch.float32)
        self.max_action = torch.tensor(max_action, dtype=torch.float32)
        self.output_norm = output_norm
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = None
        try:
            if self.oss_mode:
                if not OSS_AVAILABLE:
                    raise RuntimeError("OSS mode requested but io_tool not available. Please check io_tool installation.")
                data = np.load(BytesIO(oss_online.get_object(self.file_list[idx]).read()))
            else:
                data = np.load(self.file_list[idx])
        except Exception as e:
            print(f"Warning: Failed to load {self.file_list[idx]}, error: {e}. Loading a random sample instead.")
            random_idx = np.random.randint(0, len(self.file_list))
            return self.__getitem__(random_idx)
        state = torch.tensor(data['state'], dtype=torch.float32)
        target = torch.tensor(data['target'], dtype=torch.float32)
        obstacle = torch.tensor(data['obstacle'], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        action = torch.tensor(data['action'], dtype=torch.float32)
        
        if self.output_norm:
            action_norm = (action - self.min_action) / (self.max_action - self.min_action)
            state[3:] = (state[3:] - self.min_action) / (self.max_action - self.min_action)
            return state, target, obstacle, action_norm

        action = action.clamp_max(self.max_action).clamp_min(self.min_action)
        return state, target, obstacle, action


def create_data_loaders(cfg, output_norm=True):
    
    oss_mode = cfg['data']['train_dir'].startswith('oss')
    
    if oss_mode and not OSS_AVAILABLE:
        raise RuntimeError("OSS mode requested but io_tool not available. Please check io_tool installation or use local file paths.")

    t0 = time()
    test_files = []
    if oss_mode: # TODO:
        scene_list = oss_online.list_objects_without_subdir(cfg['data']['test_dir'])[0]
        for scene in scene_list:
            scene_oss_path = cfg['data']['test_dir'] + '/' + scene
            file_name_list = oss_online.list_objects_without_subdir(scene_oss_path)[1]
            for file_name in file_name_list:
                test_files.append(scene_oss_path + '/' + file_name)
    else:
        for root, dirs, files in os.walk(cfg['data']['test_dir']):
            for f in files:
                if f.endswith('.npz'):
                    test_files.append(os.path.join(root, f))
    np.save('test', np.array(test_files))
    t1 = time()
    test_files = np.load('test.npy').tolist()
    test_files = [path for path in test_files if path.endswith('.npz')]
    print(f'len(test_files): {len(test_files)} cost time: {t1 - t0}')

    train_files = []
    if oss_mode: # TODO:
        scene_list = oss_online.list_objects_without_subdir(cfg['data']['train_dir'])[0]
        for scene in scene_list:
            scene_oss_path = cfg['data']['train_dir'] + '/' + scene
            file_name_list = oss_online.list_objects_without_subdir(scene_oss_path)[1]
            for file_name in file_name_list:
                train_files.append(scene_oss_path + '/' + file_name)
    else:
        for root, dirs, files in os.walk(cfg['data']['train_dir']):
            for f in files:
                if f.endswith('.npz'):
                    train_files.append(os.path.join(root, f))
    np.save('train', np.array(train_files))
    t2 = time()
    train_files = np.load('train.npy').tolist()
    train_files = [path for path in train_files if path.endswith('.npz')]
    print(f'len(train_files): {len(train_files)} cost time: {t2 - t1}')

    min_action = cfg['data']['min_action']
    max_action = cfg['data']['max_action']
    
    train_dataset = EmbodiedDataset(train_files, min_action, max_action, oss_mode, output_norm=output_norm)
    test_dataset = EmbodiedDataset(test_files, min_action, max_action, oss_mode, output_norm=output_norm)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        multiprocessing_context='spawn'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        multiprocessing_context='spawn'
    )
    
    return train_loader, test_loader

if __name__ == "__main__":

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Prepare data
    train_loader, test_loader = create_data_loaders(cfg)

    import torch
    from sklearn.cluster import KMeans
    import numpy as np

    all_actions = []
    for state, target, obstacle, action in train_loader:
        all_actions.append(action)

    all_actions = torch.cat(all_actions, dim=0)
    print("All actions concatenated shape:", all_actions.shape)

    actions_np = all_actions.cpu().numpy()

    kmeans = KMeans(n_clusters=32, random_state=0)
    kmeans.fit(actions_np)
    centers = kmeans.cluster_centers_

    print("32 cluster centers (actions):")
    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")

    np.save('./action_discrete.npy', centers)
    