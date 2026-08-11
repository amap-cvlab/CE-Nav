import matplotlib

#matplotlib.use('TkAgg')  # 在导入plt之前设置后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import random
import time


# 机器人参数
class Config:
    def __init__(self, max_speed):
        # 机器人参数
        self.max_speed = max_speed  # [m/s] 最大速度
        self.min_speed = 0  # [m/s] 最小速度
        self.max_yaw_rate = 90.0 * math.pi / 180.0  # [rad/s] 最大偏航率
        self.max_accel = 1.5  # [m/ss] 最大加速度
        self.min_accel = -5.0  # [m/ss] 最小加速度
        self.max_delta_yaw_rate = 300 * math.pi / 180.0  # [rad/ss] 最大偏航加速度
        self.v_resolution = 0.1  # [m/s] 速度分辨率
        self.yaw_rate_resolution = 10 * math.pi / 180.0  # [rad/s] 偏航率分辨率
        self.dt = 0.1  # [s] 时间间隔
        self.predict_time = 1  # [s] 预测时间
        self.to_goal_cost_gain = 1.0  # 目标代价增益
        self.speed_cost_gain = 15.0  # 速度代价增益
        self.obstacle_cost_gain = 0.5  # 障碍物代价增益
        self.robot_radius = 0.2  # [m] 机器人半径
        self.goal_tolerance = 0.4  # [m] 目标容忍度
        self.in_corner = False  # 机器人是否卡在角落
        self.near_left = None  # 机器人在角落处靠近左侧障碍物
        self.vel_rate = 1.0
        self.yaw_vel_rate = 1.0
        self.use_navrl = False


# 生成随机障碍物地图
def generate_map(width=20, height=20, num_obstacles=50):
    obstacles = []

    # 边界障碍物
    for x in range(width):
        obstacles.append([x, 0])
        obstacles.append([x, height - 1])
    for y in range(height):
        obstacles.append([0, y])
        obstacles.append([width - 1, y])

    # 随机障碍物
    for _ in range(num_obstacles):
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        obstacles.append([x, y])

    return np.array(obstacles)


# 运动模型
def motion_model(x, u, dt):
    """
    机器人运动模型
    :param x: 状态 [x(m), y(m), yaw(rad), vx(m/s), vy(m/s), omega(rad/s)]
    :param u: 控制输入 [vx(m/s), vy(m/s), omega(rad/s)]
    :param dt: 时间间隔
    :return: 新状态
    """
    x[2] += u[2] * dt  # 更新角度
    x[0] += u[0] * math.cos(x[2]) * dt - u[1] * math.sin(x[2]) * dt  # 更新x位置
    x[1] += u[0] * math.sin(x[2]) * dt + u[1] * math.cos(x[2]) * dt  # 更新y位置
    x[3] = u[0]  # 更新速度
    x[4] = u[1]  # 更新速度
    x[5] = u[2]  # 更新角速度
    return x


# 计算动态窗口
def calc_dynamic_window(x, heading_angle, config):
    """
    计算动态窗口
    :param heading_angle:
    :param x: 状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    :param config: 配置参数
    :return: 动态窗口 [min_v, max_v, min_yaw_rate, max_yaw_rate]
    """
    rate = 1
    if heading_angle > 60:
        rate = max(1 - (heading_angle - 60) / 30, 0.5)
    # 速度动态窗口
    vs = [config.min_speed, config.max_speed * rate,
          -config.max_yaw_rate, config.max_yaw_rate]

    # 基于当前速度和加速度的动态窗口
    vd = [x[3] + config.min_accel * config.dt,
          max(x[3] + config.max_accel * config.dt, 0.5 / config.vel_rate),
          x[5] - config.max_delta_yaw_rate * config.dt,
          x[5] + config.max_delta_yaw_rate * config.dt]

    # 最终动态窗口
    vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
          max(vs[2], vd[2]), min(vs[3], vd[3])]

    return vr


# 计算轨迹
def calc_trajectory(x_init, vx, vy, yaw_rate, config):
    """
    计算轨迹
    :param x_init: 初始状态
    :param vx: 速度
    :param vy: 速度
    :param yaw_rate: 偏航率
    :param config: 配置参数
    :return: 轨迹
    """
    x = np.array(x_init)
    traj = np.array(x)
    time = 0

    while time <= config.predict_time:
        x = motion_model(x, [vx, vy, yaw_rate], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj


# 计算障碍物代价
def calc_obstacle_cost(traj, obstacles, config):
    """
    计算障碍物代价
    :param traj: 轨迹
    :param obstacles: 障碍物列表
    :param config: 配置参数
    :return: 障碍物代价
    """
    # 处理空的障碍物数组
    if len(obstacles) == 0:
        return 0.0
    
    # 确保obstacles是二维数组
    if len(obstacles.shape) == 1:
        obstacles = obstacles.reshape(-1, 2)
    
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    dx = traj[:, 0] - ox[:, None]
    dy = traj[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if np.min(r) <= config.robot_radius:
        return float("inf")

    return 1.0 / np.min(r)  # 距离障碍物越近，代价越大


# 计算速度代价
def calc_speed_cost(traj, config):
    """
    计算速度代价
    :param traj: 轨迹
    :param config: 配置参数
    :return: 速度代价
    """
    return config.max_speed - traj[-1, 3]


# 评估轨迹
def evaluate_trajectory(traj, goal, obstacles, config):
    """
    改进版的轨迹评估函数
    :param traj: 轨迹
    :param goal: 目标位置
    :param obstacles: 障碍物列表
    :param config: 配置参数
    :return: 总代价
    """
    # 计算到终点的距离
    final_dist = math.hypot(traj[-1, 0] - goal[0], traj[-1, 1] - goal[1])

    # 目标代价（距离越近代价越小）
    to_goal_cost = config.to_goal_cost_gain * final_dist

    # 终点奖励（当非常接近目标时给予奖励）
    if final_dist < config.robot_radius * 2:
        to_goal_cost *= 0.1  # 大幅降低代价

    # 速度代价（鼓励保持适当速度）
    speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3])

    # 障碍物代价
    obstacle_cost = calc_obstacle_cost(traj, obstacles, config)
    if config.in_corner and obstacle_cost > 4:
        obstacle_cost = float("inf")
    obstacle_cost = config.obstacle_cost_gain * obstacle_cost

    return to_goal_cost + speed_cost + obstacle_cost


# DWA算法
def dwa_control(x, goal, obstacles, config):
    """
    DWA算法
    :param x: 状态 [x(m), y(m), yaw(rad), vx(m/s), vy(m/s), omega(rad/s)]
    :param goal: 目标位置 [x(m), y(m)]
    :param obstacles: 障碍物列表
    :param config: 配置参数
    :return: 最优控制 [vx(m/s), vy(m/s), omega(rad/s)], 最优轨迹
    """
    # 计算目标角度
    heading_angle = 0
    if x[0] == 0.0 and x[1] == 0.0 and config.use_navrl is False:
        heading_angle = abs(math.atan2(goal[0], goal[1]) / math.pi * 180.0)

    # 计算动态窗口
    vr = calc_dynamic_window(x, heading_angle, config)

    # 评估所有可能的轨迹
    min_cost = float("inf")
    best_u = [0.0, 0.0, 0.0]
    best_traj = np.array([x])

    # 遍历所有可能的速度和偏航率
    for vx in np.arange(vr[0], vr[1], config.v_resolution):
        used_yaw = False
        for yaw_rate in np.arange(vr[2], vr[3], config.yaw_rate_resolution):
            # 真机控制yaw速度太小不生效
            if abs(yaw_rate) < 0.3 / config.yaw_vel_rate:
                if used_yaw is False:
                    used_yaw = True
                    yaw_rate = 0
                else:
                    continue
            if abs(vx) != 0.0 and abs(vx) < 0.4 / config.vel_rate:
                vx = 0.0
            # 计算轨迹
            traj = calc_trajectory(x, vx, 0.0, yaw_rate, config)

            # 计算代价
            cost = evaluate_trajectory(traj, goal, obstacles, config)

            # 更新最优轨迹
            if cost < min_cost:
                min_cost = cost
                best_u = [vx, 0.0, yaw_rate]
                best_traj = traj

    if config.use_navrl:
        return best_u, best_traj

    if best_u[0] == 0.0 and best_u[1] == 0.0:
        best_u = cal_corner_vel(x, obstacles, config)
        best_traj = calc_trajectory(x, best_u[0], best_u[1], best_u[2], config)
    else:
        if vr[2] * vr[3] > 0:
            traj = calc_trajectory(x, best_u[0], 0.0, -best_u[2], config)
            best_cost = calc_obstacle_cost(best_traj, obstacles, config)
            try_cost = calc_obstacle_cost(traj, obstacles, config)
            # 更新最优轨迹
            if try_cost + 0.5 < best_cost:
                best_u = [max(best_u[0] - 0.05, 0.0), 0.0, 0.0]
                best_traj = calc_trajectory(x, best_u[0], best_u[1], best_u[2], config)

    if best_u[0] == 0.0 and abs(best_u[2]) < 0.45:
        best_u[2] = best_u[2] / abs(best_u[2]) * 0.45
        best_traj = calc_trajectory(x, best_u[0], best_u[1], best_u[2], config)

    return best_u, best_traj

def cal_corner_vel(x, obstacles, config):
    """
    计算角落情况下的速度
    :param x: 状态 [x(m), y(m), yaw(rad), vx(m/s), vy(m/s), omega(rad/s)]
    :param obstacles: 障碍物列表
    :param config: 配置参数
    :return: 控制输入 [vx(m/s), vy(m/s), omega(rad/s)]
    """
    # 机器人当前位置和朝向
    robot_x, robot_y = x[0], x[1]
    robot_yaw = x[2]
    
    # 计算正前方、正左侧、正右侧的方向向量
    front_direction = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    left_direction = np.array([np.cos(robot_yaw + np.pi/2), np.sin(robot_yaw + np.pi/2)])
    right_direction = np.array([np.cos(robot_yaw - np.pi/2), np.sin(robot_yaw - np.pi/2)])
    
    # 计算各个方向上的探测距离
    probe_distance = 0.2  # 探测距离
    
    # 计算探测点位置
    front_probe = np.array([robot_x, robot_y]) + front_direction * probe_distance
    left_probe = np.array([robot_x, robot_y]) + left_direction * probe_distance
    right_probe = np.array([robot_x, robot_y]) + right_direction * probe_distance
    
    # 计算各个探测点到最近障碍物的距
    front_dist = get_min_distance_to_obstacles(front_probe, obstacles)
    left_dist = get_min_distance_to_obstacles(left_probe, obstacles)
    right_dist = get_min_distance_to_obstacles(right_probe, obstacles)
    
    # 计算速度分量
    vx = -0.4
    vy = 0
    if front_dist > config.robot_radius:
        vx = -0.01
    if config.near_left is None:
        if abs(left_dist - right_dist) > 0.2:
            if left_dist > right_dist:
                config.near_left = False
                vy = 0.4
            else:
                config.near_left = True
                vy = -0.4
    else:
        if config.near_left:
            vy = -0.4
        else:
            vy = 0.4
    if left_dist < 0.1 and vy > 0:
        vy = -0.2
    if right_dist < 0.1 and vy < 0:
        vy = 0.2
    if abs(vy) < 0.3 and abs(vx) < 0.3:
        vx = -0.4
    
    return [vx, vy, 0.0]

def get_min_distance_to_obstacles(probe_point, obstacles):
    distances = np.linalg.norm(obstacles - probe_point, axis=1)
    return np.min(distances)

def cal_dwa_control(robot_pos, cur_vel, cur_yaw_vel, goal, obstacles, config):
    x = np.array([robot_pos[0], robot_pos[1], math.pi / 2.0, cur_vel, 0.0, cur_yaw_vel])
    best_cmd, best_traj = dwa_control(x, goal, obstacles, config)
    return best_cmd, best_traj

def cal_vla_traj(robot_pos, cur_vel, cur_yaw_vel, config):
    x = np.array([robot_pos[0], robot_pos[1], math.pi / 2.0, cur_vel, 0.0, cur_yaw_vel])
    return calc_trajectory(x, cur_vel, 0.0, cur_yaw_vel, config)


def main():
    # 初始化配置
    config = Config(1.5)

    # 初始状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([2.0, 2.0, math.pi / 2, 0.3, 0.0, 0.0])
    goal = np.array([2.0, 5.0])
    obstacles = generate_map(20, 20, 100)
    print(f"obstacles size : {obstacles.shape}")

    # 启用交互模式
    plt.ion()

    # 创建图形窗口
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.grid(True)

    # 绘制障碍物和目标
    ax.plot(obstacles[:, 0], obstacles[:, 1], 'sk', markersize=2)
    ax.plot(goal[0], goal[1], 'xr', markersize=15)

    # 初始化机器人显示
    robot = patches.Circle((x[0], x[1]), config.robot_radius,
                           fc='g', ec='k', alpha=0.5, label='Robot')
    ax.add_patch(robot)

    # 初始化机器人朝向箭头
    arrow_length = config.robot_radius * 1.5
    arrow_dx = arrow_length * math.cos(x[2])
    arrow_dy = arrow_length * math.sin(x[2])
    robot_arrow = ax.quiver(x[0], x[1], arrow_dx, arrow_dy,
                            color='red', alpha=0.8, scale=1, scale_units='xy',
                            angles='xy', width=0.01, headwidth=3, headlength=4)

    # 初始化轨迹线
    trajectory_line, = ax.plot([], [], '-b', linewidth=2, label='Trajectory')
    best_traj_line, = ax.plot([], [], '--g', linewidth=1, alpha=0.5, label='Best Trajectory')
    ax.legend(loc='upper right')

    # 初始化轨迹列表
    trajectory = []
    trajectory.append(x[:2].copy())  # 使用copy()来避免引用问题

    for i in range(1000):
        # DWA控制
        cur_time = time.time()
        x[4] = 0.0
        u, predicted_traj = dwa_control(x, goal, obstacles, config)
        print(f"cal time : {(time.time() - cur_time) * 1000}ms")

        if u[0] < 0 and config.in_corner is False:
            config.in_corner = True
            config.min_speed = 0.4
        if config.in_corner is True and u[0] > 0:
            config.in_corner = False
            config.min_speed = 0.0
            config.near_left = None
            u = [0.0, 0.0, 0.0]

        x = motion_model(x, u, config.dt)

        if config.in_corner:
            x[3] = max(x[3], 0.4)
            x[4] = 0.0

        # 添加新的轨迹点
        trajectory.append(x[:2].copy())  # 使用copy()来避免引用问题

        # 更新图形
        robot.center = (x[0], x[1])

        # 更新机器人朝向箭头
        arrow_dx = arrow_length * math.cos(x[2])
        arrow_dy = arrow_length * math.sin(x[2])
        robot_arrow.set_offsets([x[0], x[1]])
        robot_arrow.set_UVC(arrow_dx, arrow_dy)

        # 更新轨迹线
        trajectory_points = np.array(trajectory)

        # 更新轨迹线数据
        trajectory_line.set_data(trajectory_points[:, 0], trajectory_points[:, 1])
        best_traj_line.set_data(predicted_traj[:, 0], predicted_traj[:, 1])

        # 更新标题
        plt.title(f'Step {i}')

        # 强制更新图形
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 短暂暂停
        plt.pause(0.1)

        # 检查是否到达目标
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.goal_tolerance:
            print("Goal reached!")
            break

    # 保持图形窗口显示
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
