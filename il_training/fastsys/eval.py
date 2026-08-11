#!/usr/bin/env python3
"""
DWA避障模拟脚本
模拟10x10走廊场景，随机生成机器人位置、目标位置和障碍物
使用dwa_service中的cal_dwa_control方法进行避障规划
"""
import shutil
import numpy as np
import matplotlib
# 在多线程环境中使用非交互式后端，避免GUI警告
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from infer import PolicyInference
import yaml
from data import vectorized_raycast
import torch
from multiprocessing import Pool
from data import world_to_grid
import hydra

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加路径以导入dwa_service
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# try:
#     from communication_node.dwa_service import cal_dwa_control, calc_obstacle_cost, Config
# except ImportError:
#     # 如果上面的导入失败，尝试其他路径
#     sys.path.append(os.path.join(parent_dir, '..'))
#     from communication_node.dwa_service import cal_dwa_control, calc_obstacle_cost, Config


class DWASimulation:
    def __init__(self, eval_scene):
        # 地图参数
        if eval_scene == 'easy':
            self.map_size_x = 10.0  # x方向10米
            self.map_size_y = 15.0  # y方向15米
            self.resolution = 0.06  # 分辨率0.06米
        elif eval_scene == 'hard':
            self.map_size_x = 39.9  # x方向10米
            self.map_size_y = 39.9  # y方向15米
            self.resolution = 0.06  # 分辨率0.06米
            
        self.dt = 0.1
        self.grid_size_x = int(self.map_size_x / self.resolution)  # x方向栅格数
        self.grid_size_y = int(self.map_size_y / self.resolution)  # y方向栅格数
        self.scene_num = 0
        self.obstacle_num = 0

        # 初始化DWA配置
        # self.config = Config(1.5)  # 最大速度1.5m/s

        # 场景参数
        random.seed(0)
        np.random.seed(0)
        self.corridor_width_x = 1 + random.random() * 1.5  # 走廊宽度
        self.corridor_width_y = 1 + random.random() * 1.5  # 走廊宽度
        self.obstacle_density = 0.1  # 障碍物密度

        # viz
        self.robot_radius = 0.01

        self.y_start = int(6.0 / self.resolution)  # y=6米对应的网格
        self.y_end = int((6.0 + self.corridor_width_y) / self.resolution)  # y=7.5米对应的网格
        self.x_start = int(1.0 / self.resolution)  # x=1米对应的网格
        self.x_end = int((1.0 + self.corridor_width_x + random.random()) / self.resolution)  # x=2.5米对应的网格

    def generate_corridor_scenario(self):
        """生成T型走廊场景，走廊宽度2m，障碍物随机生成在走廊内部"""
        occupancy_grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.uint8)

        # 先设置所有区域为墙
        occupancy_grid[:, :] = 100

        # 设置T型走廊为通路
        occupancy_grid[self.x_start:self.x_end + 1, :] = 0
        occupancy_grid[self.x_start:, self.y_start:self.y_end + 1] = 0

        occupancy_grid[:self.x_start - random.randint(3, 10), :] = 0
        occupancy_grid[self.x_end + random.randint(3, 10):, self.y_end + random.randint(3, 10):] = 0
        occupancy_grid[self.x_end + random.randint(3, 10):, :self.y_start - random.randint(3, 10)] = 0

        # 随机生成障碍物（圆形和长方形）
        num_circular = random.randint(1,3)
        num_rect = random.randint(1,3)
        self.obstacle_num = num_circular + num_rect
        for _ in range(num_circular):
            self._add_random_circular_obstacle_in_T(occupancy_grid)
        for _ in range(num_rect):
            self._add_random_rect_obstacle_in_T(occupancy_grid)

        return occupancy_grid

    def _add_random_circular_obstacle_in_T(self, occupancy_grid):
        """在T型走廊内随机添加圆形障碍物，确保障碍物之间留有0.5米距离"""
        placed = False
        attempts = 0
        radius = 0.2 + random.random() * 0.2
        min_distance = 0.5  # 障碍物之间的最小距离

        while not placed and attempts < 100:
            attempts += 1
            if random.random() < 0.5:
                x = random.randint(self.x_start + int(radius / self.resolution),
                                   self.x_end - int(radius / self.resolution))
                y = random.randint(self.y_start + int(radius / self.resolution),
                                   self.y_end - int(radius / self.resolution))
            else:
                x = random.randint(self.x_start + int(radius / self.resolution),
                                   self.x_end - int(radius / self.resolution))
                if random.random() < 0.5:
                    y = random.randint(1 + int(radius / self.resolution), self.y_start - int(radius / self.resolution))
                else:
                    y = random.randint(self.y_end + int(radius / self.resolution),
                                       self.grid_size_y - int(radius / self.resolution) - 2)

            # 检查圆形区域是否空闲且与其他障碍物距离足够
            ok = True
            center_x = x * self.resolution
            center_y = y * self.resolution
            
            # 检查与现有障碍物的距离（只检查附近区域）
            check_range = int((min_distance + radius) / self.resolution) + 1
            min_x = max(0, x - check_range)
            max_x = min(self.grid_size_x, x + check_range)
            min_y = max(0, y - check_range)
            max_y = min(self.grid_size_y, y + check_range)
            
            for check_y in range(min_y, max_y):
                for check_x in range(min_x, max_x):
                    if occupancy_grid[check_x, check_y] > 1e-6:
                        # 计算障碍物中心
                        obstacle_center_x = check_x * self.resolution
                        obstacle_center_y = check_y * self.resolution
                        distance = math.sqrt((center_x - obstacle_center_x)**2 + (center_y - obstacle_center_y)**2)
                        if distance < min_distance:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                # 检查圆形区域是否空闲
                for dy in range(-int(radius / self.resolution), int(radius / self.resolution) + 1):
                    for dx in range(-int(radius / self.resolution), int(radius / self.resolution) + 1):
                        if dx * dx + dy * dy <= (radius / self.resolution) ** 2:
                            grid_x = x + dx
                            grid_y = y + dy
                            if 0 <= grid_x < self.grid_size_x and 0 <= grid_y < self.grid_size_y:
                                if occupancy_grid[grid_x, grid_y] != 0:
                                    ok = False
                                    break
                    if not ok:
                        break
                
                if ok:
                    # 放置障碍物
                    for dy in range(-int(radius / self.resolution), int(radius / self.resolution) + 1):
                        for dx in range(-int(radius / self.resolution), int(radius / self.resolution) + 1):
                            if dx * dx + dy * dy <= (radius / self.resolution) ** 2:
                                grid_x = x + dx
                                grid_y = y + dy
                                if 0 <= grid_x < self.grid_size_x and 0 <= grid_y < self.grid_size_y:
                                    occupancy_grid[grid_x, grid_y] = 100
                    placed = True

    def _add_random_rect_obstacle_in_T(self, occupancy_grid):
        """在T型走廊内随机添加长方形障碍物，确保障碍物之间留有0.5米距离"""
        placed = False
        attempts = 0
        width = 0.4 + random.random() * 0.6
        height = 0.4 + random.random() * 0.6
        width_grid = int(width / self.resolution)
        height_grid = int(height / self.resolution)
        min_distance = 0.5  # 障碍物之间的最小距离

        while not placed and attempts < 100:
            attempts += 1
            if random.random() < 0.5:
                x = random.randint(self.x_start + 1, self.x_end - width_grid - 1)
                y = random.randint(self.y_start + 1, self.y_end - height_grid - 1)
            else:
                x = random.randint(self.x_start + 1, self.x_end - width_grid - 1)
                if random.random() < 0.5:
                    y = random.randint(1, self.y_start - height_grid - 1)
                else:
                    y = random.randint(self.y_end + 1, self.grid_size_y - height_grid - 2)

            # 计算长方形中心
            center_x = (x + width_grid / 2) * self.resolution
            center_y = (y + height_grid / 2) * self.resolution
            
            # 检查与现有障碍物的距离（只检查附近区域）
            max_dim = max(width_grid, height_grid)
            check_range = int((min_distance + max_dim * self.resolution / 2) / self.resolution) + 1
            min_x = max(0, x - check_range)
            max_x = min(self.grid_size_x, x + width_grid + check_range)
            min_y = max(0, y - check_range)
            max_y = min(self.grid_size_y, y + height_grid + check_range)
            
            ok = True
            for check_y in range(min_y, max_y):
                for check_x in range(min_x, max_x):
                    if occupancy_grid[check_x, check_y] > 1e-6:
                        # 计算障碍物中心（假设为单个栅格的中心）
                        obstacle_center_x = check_x * self.resolution
                        obstacle_center_y = check_y * self.resolution
                        distance = math.sqrt((center_x - obstacle_center_x)**2 + (center_y - obstacle_center_y)**2)
                        if distance < min_distance:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                # 检查区域是否空闲
                for dy in range(height_grid):
                    for dx in range(width_grid):
                        grid_x = x + dx
                        grid_y = y + dy
                        if (0 <= grid_x < self.grid_size_x and 0 <= grid_y < self.grid_size_y and
                                occupancy_grid[grid_x, grid_y] != 0):
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    # 放置障碍物
                    for dy in range(height_grid):
                        for dx in range(width_grid):
                            grid_x = x + dx
                            grid_y = y + dy
                            if 0 <= grid_x < self.grid_size_x and 0 <= grid_y < self.grid_size_y:
                                occupancy_grid[grid_x, grid_y] = 100
                    placed = True

    def generate_random_positions(self, occupancy_grid):
        """随机生成机器人位置和目标位置"""

        pos_1 = self._find_free_position_in_T(occupancy_grid, 1, 1 + self.corridor_width_x, 0.5, 1.5)
        pos_2 = self._find_free_position_in_T(occupancy_grid, 1, 1 + self.corridor_width_x, 13.5, 14.5)
        pos_3 = self._find_free_position_in_T(occupancy_grid, 8.5, 9.5, 6, 6 + self.corridor_width_y)

        pos_1[2] = math.pi * 0.5 #random.uniform(math.pi*0.25, math.pi*0.75)
        pos_2[2] = math.pi * 1.5 #random.uniform(math.pi*1.25, math.pi*1.75)
        pos_3[2] = math.pi * 1.0 #random.uniform(math.pi*0.75, math.pi*1.25)

        # 从pos_1, pos_2, pos_3中随机选择robot_pos和goal_pos，且不重复
        pos_list = [pos_1, pos_2, pos_3]
        selected = random.sample(pos_list, 2)
        robot_pos = selected[0]
        goal_pos = selected[1]
        return robot_pos, goal_pos

    def _find_free_position_in_T(self, occupancy_grid, x_start, x_end, y_start, y_end, exclude_pos=None):
        """在T型走廊内找到空闲位置"""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = random.randint(int(x_start / self.resolution), int(x_end / self.resolution))
            y = random.randint(int(y_start / self.resolution), int(y_end / self.resolution))
            obstacle_dis = 0.3
            obstacle_dis_voxel_num = int(round(obstacle_dis / self.resolution))
            if x - obstacle_dis_voxel_num < 0 or x + obstacle_dis_voxel_num >= self.grid_size_x or \
                y - obstacle_dis_voxel_num < 0 or y + obstacle_dis_voxel_num >= self.grid_size_y:
                continue
            
            if np.sum(occupancy_grid[x-obstacle_dis_voxel_num:x+obstacle_dis_voxel_num,y-obstacle_dis_voxel_num:y+obstacle_dis_voxel_num]) > 1e-6:
                continue
            pos = [x * self.resolution, y * self.resolution, math.pi / 2]
            return pos

        return [(x_start + x_end) / 2 , (y_start + y_end) / 2, math.pi / 2]

    def _distance(self, pos1, pos2):
        """计算两点间距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def generate_obstacles_from_occ(self, occupancy_grid, robot_pos):
        """根据占用栅格生成障碍物数组
        以机器人为原点，y轴正向，x轴右侧
        计算范围：127x127正方形，分辨率0.06
        """
        obstacles = []
        
        # 机器人在地图中的位置
        robot_world_x = robot_pos[0]
        robot_world_y = robot_pos[1]
        robot_yaw = robot_pos[2]

        range_size = 127
        range_meters = range_size * self.resolution
        half_range = range_meters / 2
        
        # 遍历占用栅格，找到障碍物
        for y in range(self.grid_size_y):
            for x in range(self.grid_size_x):
                if occupancy_grid[x, y] > 1e-6:
                    obstacle_world_x = x * self.resolution
                    obstacle_world_y = y * self.resolution

                    dx = obstacle_world_x - robot_world_x
                    dy = obstacle_world_y - robot_world_y

                    cos_yaw = math.cos(robot_yaw)
                    sin_yaw = math.sin(robot_yaw)

                    robot_x = dx * sin_yaw - dy * cos_yaw
                    robot_y = dx * cos_yaw + dy * sin_yaw

                    if abs(robot_x) <= half_range and abs(robot_y) <= half_range:
                        grid_x = int(robot_x / self.resolution) + range_size // 2
                        grid_y = int(robot_y / self.resolution) + range_size // 2
                        
                        if 0 <= grid_x < range_size and 0 <= grid_y < range_size:
                            if random.random() < 0.5:
                                obstacles.append([robot_x, robot_y])
        
        return np.array(obstacles)

    def calculate_acceleration(self, speeds, dt):
        if len(speeds) < 2:
            raise ValueError("速度列表至少需要包含2个元素")
        
        # 计算瞬时加速度列表
        accelerations = [(speeds[i+1] - speeds[i]) / dt for i in range(len(speeds)-1)]
        accelerations = abs(np.array(accelerations))
        
        # 计算统计量
        avg_accel = sum(accelerations) / len(accelerations)
        max_accel = max(accelerations)  # 代数最大值版本
        
        return avg_accel, max_accel

    def simulate_dwa_navigation(self, robot_pos, goal_pos, occupancy_grid, save_dir, scenario_id, checkpoint, mode, cfg,
                                cfg_navrl, max_time=60.0, enable_realtime_plot=False, delete_failed_flag=True):
        """模拟DWA导航过程
        每次循环重新计算障碍物，更新机器人位置，记录轨迹
        到达目标或超时30秒时结束
        """
        # print(f"开始DWA导航模拟")
        # print(f"机器人初始位置: {robot_pos}")
        # print(f"目标位置: {goal_pos}")
        # print(f"最大运行时间: {max_time}秒")

        # 初始化机器人状态
        cur_vel = 0.0  # 初始速度
        cur_yaw_vel = 0.0  # 初始角速度
        
        trajectory = [robot_pos.copy()]  # 记录轨迹
        commands = []  # 记录命令
        timestamps = [0.0]  # 记录时间戳
        
        start_time = time.time()

        infer = None
        if mode == 'il':
            infer = PolicyInference(checkpoint, cfg)
        elif mode == 'navrl_il':
            from infer_policy import PolicyInference as NAVRLPolicyInference
            infer = NAVRLPolicyInference(checkpoint, cfg_navrl)
            
        
        # 初始化实时绘图
        if enable_realtime_plot:
            plt.ion()  # 启用交互模式
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 设置坐标轴范围
            ax.set_xlim(0, self.map_size_x)
            ax.set_ylim(0, self.map_size_y)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # 绘制占用栅格
            cmap = plt.cm.colors.ListedColormap(['white', 'black'])
            bounds = [0, 50, 100]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            occupancy_grid_for_plot = occupancy_grid.T
            ax.imshow(occupancy_grid_for_plot, cmap=cmap, norm=norm, origin='lower', 
                     extent=[0, self.map_size_x, 0, self.map_size_y], alpha=0.3)
            
            # 绘制目标位置
            goal_circle = patches.Circle((goal_pos[0], goal_pos[1]), 0.3, 
                                       fc='red', ec='black', alpha=0.7, label='目标')
            ax.add_patch(goal_circle)
            
            # 初始化机器人显示
            robot_circle = patches.Circle((robot_pos[0], robot_pos[1]), self.robot_radius,
                                        fc='green', ec='black', alpha=0.7, label='机器人')
            ax.add_patch(robot_circle)
            
            # 初始化机器人朝向箭头
            arrow_length = self.robot_radius * 1.5
            arrow_dx = arrow_length * math.cos(robot_pos[2])
            arrow_dy = arrow_length * math.sin(robot_pos[2])
            robot_arrow = ax.quiver(robot_pos[0], robot_pos[1], arrow_dx, arrow_dy,
                                   color='blue', alpha=0.8, scale=1, scale_units='xy',
                                   angles='xy', width=0.01, headwidth=3, headlength=4)
            
            # 初始化轨迹线
            trajectory_line, = ax.plot([], [], '-b', linewidth=2, alpha=0.8, label='实际轨迹')
            dwa_traj_line, = ax.plot([], [], '--g', linewidth=1, alpha=0.6, label='DWA预测轨迹')
            ax.legend(loc='upper right')
        
        step = 0
        timeout_exit = False  # 标记是否因为超时退出
        collision_flag = False
        vx_list = []
        vy_list = []
        vyaw_list = []

        while True:
            current_time = time.time() - start_time
            
            # 检查是否超时
            if step > 200: #1000
                # print(f"运行时间超过{max_time}秒，停止模拟")
                timeout_exit = True  # 标记为超时退出
                break
            
            step += 1
            # print(f"\n步骤 {step} (时间: {current_time:.1f}s):")
            # print(f"  当前机器人位置: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
            # print(f"  当前速度: {cur_vel:.3f} m/s, 角速度: {cur_yaw_vel:.3f} rad/s")

            obstacles = self.generate_obstacles_from_occ(occupancy_grid, robot_pos)

            dx = goal_pos[0] - robot_pos[0]
            dy = goal_pos[1] - robot_pos[1]

            cos_yaw = math.cos(robot_pos[2])
            sin_yaw = math.sin(robot_pos[2])

            robot_goal_x = dx * sin_yaw - dy * cos_yaw
            robot_goal_y = dx * cos_yaw + dy * sin_yaw
            robot_goal = [robot_goal_x, robot_goal_y]
            
            dist_to_goal = math.sqrt(robot_goal[0] ** 2 + robot_goal[1] ** 2)
            # print(f"  距离目标: {dist_to_goal:.3f} m")

            # 检查是否到达目标
            if dist_to_goal < 1:
                # print(f"到达目标！总步数: {step}, 总时间: {current_time:.1f}秒")
                break
            if collision_flag:
                break

            statue = np.array([0.0, 0.0, math.pi / 2, cur_vel, 0, cur_yaw_vel])
            target = np.array(robot_goal)
            obstacle_matrix = np.zeros((127, 127), dtype=np.float32)
            if len(obstacles) > 0:
                for obstacle in obstacles:
                    x_idx = int((obstacle[0] + 3.81) / self.resolution)
                    y_idx = int((obstacle[1] + 3.81) / self.resolution)

                    # 确保索引在有效范围内
                    if 0 <= x_idx < 127 and 0 <= y_idx < 127:
                        obstacle_matrix[x_idx, y_idx] = 1

            
            

            # plt.figure(figsize=(4, 4))
            # plt.imshow(obstacle_matrix.T, cmap='gray', origin='lower')
            # plt.title(f"Obstacle Matrix Step {step}")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(os.path.join('./eval/0/', f"obstacle_matrix_{step:04d}.png"))
            # plt.close()
            
            dwa_cmd = None
            if mode == 'il':
                dwa_cmd = infer.predict(statue, target, obstacle_matrix)
            elif mode == 'navrl_il':
                dwa_cmd = infer.predict(statue, target, obstacle_matrix)
            elif mode == 'dwa':
                
                # test dwa
                from dwa_service import Config, cal_dwa_control
                config = Config(1.5)
                
                # start_calc_time = time.time()
                dwa_cmd, dwa_traj = cal_dwa_control([0.0, 0.0], cur_vel, cur_yaw_vel, robot_goal, obstacles, config)
                # # calc_time = (time.time() - start_calc_time) * 1000

                # # print(f"  DWA计算时间: {calc_time:.2f} ms")
                # # print(f"  DWA命令: vx={dwa_cmd[0]:.3f}, vy={dwa_cmd[1]:.3f}, vyaw={dwa_cmd[2]:.3f}")

                # dwa_cost = calc_obstacle_cost(dwa_traj, obstacles, self.config)
                # print(f"  DWA避障代价: {dwa_cost:.3f}")

            # print(dwa_cmd)
            vx_robot = dwa_cmd[0]
            vy_robot = dwa_cmd[1]
            vyaw = dwa_cmd[2]
            vx_list.append(dwa_cmd[0])
            vy_list.append(dwa_cmd[1])
            vyaw_list.append(dwa_cmd[2])

            new_yaw = robot_pos[2] + vyaw * self.dt  # 更新角度
            new_x = robot_pos[0] + vx_robot * math.cos(robot_pos[2]) * self.dt - vy_robot * math.sin(robot_pos[2]) * self.dt  # 更新x位置
            new_y = robot_pos[1] + vx_robot * math.sin(robot_pos[2]) * self.dt + vy_robot * math.cos(robot_pos[2]) * self.dt  # 更新y位置
            
            # 更新机器人位置
            robot_pos = [new_x, new_y, new_yaw]

            xmin = max(round((new_x - self.robot_radius) / self.resolution), 0)
            xmax = min(round((new_x + self.robot_radius) / self.resolution), occupancy_grid.shape[0] - 1)
            ymin = max(round((new_y - self.robot_radius) / self.resolution), 0)
            ymax = min(round((new_y + self.robot_radius) / self.resolution), occupancy_grid.shape[1] - 1)
            ix = min(max(round(new_x / self.resolution), 0), occupancy_grid.shape[0] - 1)
            iy = min(max(round(new_y / self.resolution), 0), occupancy_grid.shape[1] - 1)
            # 获取候选区块的索引网格
            x_idxs = np.arange(xmin, xmax + 1)
            y_idxs = np.arange(ymin, ymax + 1)
            xx, yy = np.meshgrid(x_idxs, y_idxs, indexing='ij')
            # 计算距离中心栅格的欧氏距离矩阵
            dist = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
            # 筛选欧式距离小于等于r_cells的格子
            mask = dist <= round(self.robot_radius / self.resolution)
            # 提取这些格子的占用值
            values = occupancy_grid[xx[mask], yy[mask]]
            if np.any(values) > 1e-6:
                collision_flag = True
            # if occupancy_grid[min(max(int(round(new_x / self.resolution)), 0), occupancy_grid.shape[0] - 1), 
            #                   min(max(int(round(new_y / self.resolution)), 0), occupancy_grid.shape[1] - 1)] > 1e-6:
            #     collision_flag = True

            cur_vel = vx_robot  # 使用机器人坐标系x方向速度
            cur_yaw_vel = vyaw

            # # 卡在角落
            # if cur_vel < 0 and self.config.in_corner is False:
            #     self.config.in_corner = True
            #     self.config.min_speed = 0.4
            # if self.config.in_corner and vx_robot > 0:
            #     self.config.in_corner = False
            #     self.config.min_speed = 0.0
            #     self.config.near_left = None
            # if self.config.in_corner:
            #     cur_vel = max(cur_vel, 0.4)

            # 输出样本
            # if step % 2 == 1:
            action = np.array(dwa_cmd)
            

            if mode == 'dwa':
                file_name = f"scene_{self.scene_num}_{step}.npz"
                filepath = os.path.join(save_dir, file_name)
                np.savez_compressed(filepath,
                                    state=statue,
                                    target=target,
                                    obstacle=obstacle_matrix,
                                    action=action)


            trajectory.append(robot_pos.copy())
            commands.append(dwa_cmd.copy())
            timestamps.append(current_time)


            if enable_realtime_plot:
                ax.clear()

                ax.set_xlim(-15, 15)
                ax.set_ylim(-5, 15)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                if len(obstacles) > 0:
                    ax.plot(obstacles[:, 0], obstacles[:, 1], 'sk', markersize=2, alpha=0.7, label='障碍物')

                ax.plot(robot_goal[0], robot_goal[1], 'xr', markersize=15, linewidth=3, label='目标')

                robot = patches.Circle((0.0, 0.0), self.robot_radius,
                                     fc='g', ec='k', alpha=0.7, label='机器人')
                ax.add_patch(robot)

                arrow_length = self.robot_radius * 1.5
                robot_arrow = ax.quiver(0.0, 0.0, 0.0, arrow_length,
                                       color='blue', alpha=0.8, scale=1, scale_units='xy',
                                       angles='xy', width=0.01, headwidth=3, headlength=4)

                # if dwa_traj is not None and len(dwa_traj) > 0:
                #     ax.plot(dwa_traj[:, 0], dwa_traj[:, 1], '--g', linewidth=2, alpha=0.8, label='DWA预测轨迹')

                ax.legend(loc='upper right')

                title = f'DWA避障模拟 - 步骤 {step} (时间: {current_time:.1f}s)\n'
                title += f'robot coor: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) | target world coor: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})\n'
                title += f'target coor: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f})\n'
                title += f'velo: {cur_vel:.2f} {vy_robot:.2f} m/s | omg: {cur_yaw_vel:.2f} rad/s | dis: {dist_to_goal:.2f} m'
                ax.set_title(title, fontsize=10)

                ax.set_xlabel('X (m) - 机器人坐标系')
                ax.set_ylabel('Y (m) - 机器人坐标系')

                fig.canvas.draw()
                fig.canvas.flush_events()

                plt.pause(0.05)
                
                plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.png')) 

        # 输出统计信息
        total_time = time.time() - start_time
        print(f"\n{scenario_id}模拟完成:")
        print(f"  总步数: {step}")
        print(f"  总时间: {total_time:.1f}秒")
        # print(f"  轨迹点数: {len(trajectory)}")
        # print(f"  最终位置: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
        
        final_dist = math.sqrt((goal_pos[0] - robot_pos[0])**2 + (goal_pos[1] - robot_pos[1])**2)
        # print(f"  最终距离目标: {final_dist:.3f} m")

        if delete_failed_flag and (timeout_exit or collision_flag):
            print(f"  因为超时退出，删除保存目录: {save_dir}")
            try:
                shutil.rmtree(save_dir)
                if enable_realtime_plot:
                    plt.ioff()
                    plt.show()
                return trajectory, commands, timestamps
            except Exception as e:
                print(f"  删除目录失败: {e}")

        self.visualize_scenario(occupancy_grid, robot_pos, goal_pos, trajectory, save_dir)

        if enable_realtime_plot:
            plt.ioff()
            plt.show()

        failed_flag = timeout_exit or collision_flag
        mean_vx_acc = None
        max_vx_acc = None
        mean_vy_acc = None
        max_vy_acc = None
        mean_vyaw_acc = None
        max_vyaw_acc = None
        if not failed_flag:
            mean_vx_acc, max_vx_acc = self.calculate_acceleration(vx_list, self.dt)
            mean_vy_acc, max_vy_acc = self.calculate_acceleration(vy_list, self.dt)
            mean_vyaw_acc, max_vyaw_acc = self.calculate_acceleration(vyaw_list, self.dt)
        return trajectory, commands, timestamps, failed_flag, (mean_vx_acc, max_vx_acc, mean_vy_acc, max_vy_acc, mean_vyaw_acc, max_vyaw_acc)

    def visualize_scenario(self, occupancy_grid, robot_pos, goal_pos, trajectory=None, save_dir=None):
        """可视化场景并保存到指定目录"""
        fig, ax = plt.subplots(figsize=(12, 10))

        cmap = plt.cm.colors.ListedColormap(['white', 'black'])
        bounds = [0, 4, 100]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        occupancy_grid_for_plot = occupancy_grid.T
        
        ax.imshow(occupancy_grid_for_plot, cmap=cmap, norm=norm, origin='lower', 
                 extent=[0, self.map_size_x, 0, self.map_size_y])
        
        # 绘制机器人位置
        robot_circle = patches.Circle(robot_pos, 0.2,
                                    fc='green', ec='black', alpha=0.7, label='机器人')
        ax.add_patch(robot_circle)
        
        # 绘制目标位置
        goal_circle = patches.Circle(goal_pos, 0.3, 
                                   fc='red', ec='black', alpha=0.7, label='目标')
        ax.add_patch(goal_circle)
        
        # 绘制轨迹
        if trajectory and len(trajectory) > 1:
            traj_xy = [[pos[0], pos[1]] for pos in trajectory]
            traj_array = np.array(traj_xy)
            ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=2, alpha=0.8, label='DWA轨迹')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('DWA避障模拟场景 - T型走廊')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像到指定目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"scene_{self.scene_num}_obj_{self.obstacle_num}_trajectory.png"
            filepath = os.path.join(save_dir, filename)
            # 调整保存参数，避免布局警告
            plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
            # print(f"  轨迹图像已保存: {filepath}")
        
        plt.close(fig)


    def is_near_obstacle(self, robot_pos, occupancy_grid, resolution=0.06, check_grid_num=5):
        grid_x = int(round(robot_pos[0] / resolution))
        grid_y = int(round(robot_pos[1] / resolution))
        check_x_min = max(grid_x - check_grid_num, 0)
        check_x_max = min(grid_x + check_grid_num, occupancy_grid.shape[0])
        check_y_min = max(grid_y - check_grid_num, 0)
        check_y_max = min(grid_y + check_grid_num, occupancy_grid.shape[1])
        if np.sum(occupancy_grid[check_x_min:check_x_max, check_y_min:check_y_max]) > 1e-6:
            return True


    # change from isaac sim source code
    def discrete_obstacles_terrain_with_large_border_terrain(self, num_obstacles=500) -> np.ndarray:
        height = 5.0
        
        # switch parameters to discrete units
        # -- terrain
        width_pixels = int(round(self.map_size_x / self.resolution))
        length_pixels = int(round(self.map_size_x / self.resolution))
        # -- obstacles
        obs_width_min = int(0.4 / self.resolution)
        obs_width_max = int(1.1 / self.resolution)
        # -- center of the terrain
        platform_width = 0

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


        # print("Calculation Start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        num = 0
        for _ in range(num_obstacles):
            # print("Number of cylinders generated: ", num)
            # sample size        
            
            
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
        # print("Calculation End!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
        # clip the terrain to the platform
        x1 = (width_pixels - platform_width) // 2
        x2 = (width_pixels + platform_width) // 2
        y1 = (length_pixels - platform_width) // 2
        y2 = (length_pixels + platform_width) // 2
        hf_raw[x1:x2, y1:y2] = 0
        # round off the heights to the nearest vertical step
        return np.rint(hf_raw).astype(np.int16)

    def generate_random_positions_hard_scene(self, occupancy_grid):
        """随机生成机器人位置和目标位置"""

        target_dis = 15 # 目标距离
        edge_dis = 2 # 边缘距离
        target_random_thre = 1
        start_pos = self._find_free_position_in_T(occupancy_grid, target_dis + edge_dis, self.map_size_x - target_dis - edge_dis,
                                                target_dis + edge_dis, self.map_size_y - target_dis - edge_dis)
        
        # 生成随机角度（0到2π）
        theta = np.random.uniform(np.pi / 4, np.pi - np.pi / 4)
        # 生成随机半径（考虑均匀分布）
        # 使用平方根保证在环形区域内的均匀分布
        # min_radius = target_dis - target_random_thre
        # max_radius = target_dis + target_random_thre
        # r = np.sqrt(np.random.uniform(min_radius**2, max_radius**2))
        
        r = target_dis
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        
        end_pos = self._find_free_position_in_T(occupancy_grid, start_pos[0] + dx - target_random_thre, start_pos[0] + dx + target_random_thre,
                                                start_pos[1] + dy - target_random_thre, start_pos[1] + dy + target_random_thre)
        # end_pos = [start_pos[0] + dx, start_pos[1] + dy, start_pos[2]]
        return start_pos, end_pos

    def simulate_single_scenario(self, arg):
        (scenario_id, checkpoint, save_folder, delete_failed_flag, mode, eval_scene, cfg, cfg_navrl) = arg
        """模拟单个场景的封装函数"""
        # try:
        random.seed(scenario_id)
        np.random.seed(scenario_id)
        
        # # 生成场景
        # occupancy_grid = self.generate_corridor_scenario()
        # # 生成机器人位置（x,y,yaw）和目标位置(x,y)
        # robot_pos, goal_pos = self.generate_random_positions(occupancy_grid)
        
        occupancy_grid = None
        robot_pos, goal_pos = None, None
        for i in range(100):
            if eval_scene == 'easy':
                # 生成场景
                occupancy_grid = self.generate_corridor_scenario()
                # 生成机器人位置（x,y,yaw）和目标位置(x,y)
                robot_pos, goal_pos = self.generate_random_positions(occupancy_grid)
            elif eval_scene == 'hard':
                occupancy_grid = self.discrete_obstacles_terrain_with_large_border_terrain().astype(np.float32)
                robot_pos, goal_pos = self.generate_random_positions_hard_scene(occupancy_grid)
            else:
                print(f'unknown eval_scene {eval_scene}')
                raise ValueError
                
            if robot_pos is not None and goal_pos is not None:
                break
        
        
        self.scene_num = scenario_id
        save_dir = save_folder
        os.makedirs(save_dir, exist_ok=True)
        
        
        
        # 运行DWA导航
        trajectory, commands, timestamps, failed_flag, acc_res = self.simulate_dwa_navigation(
            robot_pos, goal_pos, occupancy_grid, save_dir, scenario_id, checkpoint, mode, cfg, cfg_navrl,
            delete_failed_flag=delete_failed_flag)

        
        # print(f"场景 {scenario_id + 1} 模拟完成")
        return scenario_id, True, None, failed_flag, acc_res
            
        # except Exception as e:
        #     print(f"场景 {scenario_id + 1} 模拟过程中发生错误: {e}")
        #     return scenario_id, False, str(e)

        
    def run_simulation(self, checkpoint, save_folder, cfg, cfg_navrl, num_scenarios=5, max_workers=4, 
                       delete_failed_flag=False, mode='il', eval_scene='easy'):
        '''
            mode: 'il', 'navrl_il'
        '''
        print(f"总场景数: {num_scenarios}")
        print(f"最大线程数: {max_workers}")
        print(f"{'=' * 50}")
        
        arg_list = []
        for i in range(num_scenarios):
            arg_list.append((i, checkpoint, save_folder, delete_failed_flag, mode, eval_scene, cfg, cfg_navrl))
        
        
        # for arg in arg_list:
        #     self.simulate_single_scenario(arg)
        # return 0
        
        with Pool(processes=max_workers) as pool:
            result = list(pool.imap(self.simulate_single_scenario, arg_list))
            failed_idx_list = [idx for idx, elem in enumerate(result) if elem[3]]
            success_num = num_scenarios - len(failed_idx_list)
            print('success ratio: {}/{}'.format(success_num, num_scenarios))
            print('failed_idx_list: {}'.format(failed_idx_list))
            np.savetxt(os.path.join(save_folder, 'failed_list.txt'), np.round(np.array(failed_idx_list)).astype(np.int32), fmt="%d")
            acc_res_list = [abs(np.array(elem[4])) for elem in result if not elem[3]]
            acc_res_mean = np.mean(np.array(acc_res_list), axis=0)
            print(f'acc_res_mean: {acc_res_mean}')
        
        return success_num

@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'isaac-training/training/cfg'), config_name="ppo", version_base=None)
def main(navrl_cfg):
    
    # simulator.simulate_single_scenario(0)
    
    # mode='il'
    # checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', '<run_id>', 'best_model.pt')
    # config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    # save_folder = 'eval'
    # navrl_cfg = None
    
    mode='dwa' # il | navrl_il | dwa   (dwa: 用DWA专家采集训练数据; navrl_il/il: 评测已训练策略)
    eval_scene='hard' # easy | hard
    checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'dynfji91', 'best_model.pt')
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_navrl_il.yaml')
    save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_results')
    
    simulator = DWASimulation(eval_scene)
    
    # def load_config():
    #     CFG_FILE_PATH = os.path.join(
    #         '../', 
    #         'isaac-training', 
    #         'training', 
    #         "cfg"
    #     )
        
    #     hydra.initialize(config_path=CFG_FILE_PATH, version_base=None)
    #     cfg = hydra.compose(config_name="ppo")
    #     hydra.core.global_hydra.GlobalHydra.instance().clear()
    #     return cfg
    # navrl_cfg = load_config()

    
    try:
        # simulator.run_simulation(checkpoint, save_folder, num_scenarios=100, max_workers=16, delete_failed_flag=False)
        simulator.run_simulation(checkpoint, save_folder, config_path, navrl_cfg, num_scenarios=100, max_workers=12, delete_failed_flag=False, mode=mode, eval_scene=eval_scene)
    except KeyboardInterrupt:
        print("\n模拟被用户中断")
    except Exception as e:
        print(f"模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
