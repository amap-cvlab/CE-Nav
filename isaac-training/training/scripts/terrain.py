from omni.isaac.orbit.terrains.height_field.utils import height_field_to_mesh
from omni.isaac.orbit.terrains.height_field import hf_terrains_cfg
from copy import deepcopy

import numpy as np

@height_field_to_mesh
def discrete_obstacles_terrain_with_large_border_terrain(difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    complex_flag = False
    
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)
    obstacles_history = []

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    # print("Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("height range: ", cfg.obstacle_height_range)
    probability_length = len(cfg.obstacle_height_probability)

    def good_distance(x, y, width, length, obstacles_hist, bad_range = [2, 10]):
        # lower_bound_pixels = bad_range[0] / cfg.horizontal_scale
        # upper_bound_pixels = bad_range[1] / cfg.horizontal_scale

        lower_bound_pixels = bad_range[0]
        upper_bound_pixels = bad_range[1]    
        # previous x, y, w, l. Calculate for closet points
        for (xp, yp, wp, lp) in obstacles_hist:
            dx = abs(xp - x) - width
            dy = abs(yp - y) - length
            if dx<0 and dy<0:
                continue
            distance = np.sqrt(dx**2 + dy**2)
            # print("distance: ", distance)
            # print("lower_bound_pixels: ", lower_bound_pixels)
            # print("upper_bound_pixels: ", upper_bound_pixels)
            # print("x: ", x)
            # print("y: ", y)
            if distance >= lower_bound_pixels and distance <= upper_bound_pixels:
                return False
        return True
    
    
    def good_distace_opt(x, y, width, length, obstacles_hist, bad_range = 10):
        # lower_bound_pixels = bad_range[0]
        # upper_bound_pixels = bad_range[1]    
        # previous x, y, w, l. Calculate for closet points
        for (xp, yp, wp, lp) in obstacles_hist:
            dx = abs(xp + wp / 2 - (x + width / 2)) - (width + wp) / 2
            dy = abs(yp + lp / 2 - (y + length / 2)) - (length + lp) / 2
            if dx<bad_range and dy<bad_range:
                return False
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= 30:
                return False
            # print('dx: {}'.format(dx))
            # print('dy: {}'.format(dy))
        return True

    if complex_flag:
        target_height = cfg.obstacle_height_range[-1]/cfg.vertical_scale
    
        # border obstacles
        # 0,0 y
        x_start = 0
        x_end = x_start + int(1 / cfg.horizontal_scale)
        y_start = int(5 / cfg.horizontal_scale)
        y_end = y_start + int(10 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        y_start = y_end + int(10 / cfg.horizontal_scale)
        y_end = y_start + int(10 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        # end,0 y
        x_end = obs_x_range[-1]
        x_start = x_end - int(1 / cfg.horizontal_scale)
        y_start = int(5 / cfg.horizontal_scale)
        y_end = y_start + int(10 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        y_start = y_end + int(10 / cfg.horizontal_scale)
        y_end = y_start + int(10 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        # 0, 0 x
        x_start = int(5 / cfg.horizontal_scale)
        x_end = x_start + int(10 / cfg.horizontal_scale)
        y_start = 0
        y_end = y_start + int(1 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        x_start = x_end + int(10 / cfg.horizontal_scale)
        x_end = x_start + int(10 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        # 0, end x
        x_start = int(5 / cfg.horizontal_scale)
        x_end = x_start + int(10 / cfg.horizontal_scale)
        y_end = obs_y_range[-1]
        y_start = y_end - int(1 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        x_start = x_end + int(10 / cfg.horizontal_scale)
        x_end = x_start + int(10 / cfg.horizontal_scale)
        hf_raw[x_start:x_end, y_start:y_end] = target_height
        
        
        # center obstacles
        size_range = [int(3 / cfg.horizontal_scale), int(6 / cfg.horizontal_scale)]
        larget_obstacle_num_sqrt = 3
        larget_obstacle_num = larget_obstacle_num_sqrt * larget_obstacle_num_sqrt
        larget_obstacle_obs_width_range = np.arange(size_range[0], size_range[1], 1)
        larget_obstacle_obs_length_range = np.arange(size_range[0], size_range[1], 1)
        
        num = 0
        for i in range(larget_obstacle_num):
            width_obstacle_idx = int(i / larget_obstacle_num_sqrt)
            length_obstacle_idx = i % larget_obstacle_num_sqrt
            
            height = cfg.obstacle_height_range[-1]/cfg.vertical_scale

            attempts = 0
            # print("Start choosing location!!")
            while attempts < 100000:
                width = int(np.random.choice(larget_obstacle_obs_width_range))
                length = int(np.random.choice(larget_obstacle_obs_length_range))
                # sample position
                obs_x_range_large = np.arange(0 + int(8 / cfg.horizontal_scale), width_pixels - int(8 / cfg.horizontal_scale), 4)
                obs_y_range_large = np.arange(0 + int(8 / cfg.horizontal_scale), length_pixels - int(8 / cfg.horizontal_scale), 4)
                # x_start = int(np.random.choice(obs_x_range_large))
                # y_start = int(np.random.choice(obs_y_range_large))
                x_start = int(np.random.choice(np.split(obs_x_range_large, larget_obstacle_num_sqrt)[width_obstacle_idx][5:-5]))
                y_start = int(np.random.choice(np.split(obs_y_range_large, larget_obstacle_num_sqrt)[length_obstacle_idx][5:-5]))
                if x_start + width > width_pixels:
                    x_start = width_pixels - width
                if y_start + length > length_pixels:
                    y_start = length_pixels - length
                
                
                if good_distace_opt(x_start, y_start, width, length, obstacles_history):
                    break
                elif obstacles_history == []:
                    break
                # print("attempts when generated: ", attempts)
                # print("obstacles_history: ", obstacles_history)
                attempts += 1
            print("attempts when generated: ", attempts)
            obstacles_history.append((x_start, y_start, width, length))
            num += 1
            # clip start position to the terrain
            # print("x_start: ", x_start)
            # print("y_start: ", y_start)
            # add to terrain
            hf_raw[x_start : x_start + width, y_start : y_start + length] = height

    num = 0
    for _ in range(cfg.num_obstacles):
        # print("Number of cylinders generated: ", num)
        # sample size        
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        elif cfg.obstacle_height_mode == "range":
            random_roll = np.random.choice(probability_length, 1, p=cfg.obstacle_height_probability)
            for n in range(probability_length):
                if random_roll == n:
                    height = np.random.uniform(cfg.obstacle_height_range[n]/cfg.vertical_scale, cfg.obstacle_height_range[n+1]/cfg.vertical_scale)
                    break

            # height = np.random.uniform(cfg.obstacle_height_range[0]/cfg.vertical_scale, cfg.obstacle_height_range[1]/cfg.vertical_scale)
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed' or 'range'.")
        
        attempts = 0
        # print("Start choosing location!!")
        while attempts < 100000:
            width = int(np.random.choice(obs_width_range))
            length = int(np.random.choice(obs_length_range))
            # sample position
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            if x_start + width > width_pixels:
                x_start = width_pixels - width
            if y_start + length > length_pixels:
                y_start = length_pixels - length
            
            if good_distance(x_start, y_start, width, length, obstacles_history):
                break
            elif obstacles_history == []:
                break
            # print("attempts when generated: ", attempts)
            # print("obstacles_history: ", obstacles_history)
            attempts += 1
        # print("attempts when generated: ", attempts)
        obstacles_history.append((x_start, y_start, width, length))
        num += 1
        # clip start position to the terrain
        # print("x_start: ", x_start)
        # print("y_start: ", y_start)
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    np.savetxt('/tmp/hf_raw.txt', np.rint(hf_raw).astype(np.int16))
    print('save /tmp/hf_raw.txt')
    return np.rint(hf_raw).astype(np.int16)

discrete_obstacles_terrain_hard_mesh = height_field_to_mesh(discrete_obstacles_terrain_with_large_border_terrain)

def discrete_obstacles_terrain_hard_height_map(difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg, size) -> np.ndarray:
    cfg_clone = deepcopy(cfg)
    cfg_clone.size = size
    return discrete_obstacles_terrain_with_large_border_terrain(difficulty, cfg_clone)