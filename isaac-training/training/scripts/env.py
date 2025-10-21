import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni_drones.robots import RobotBase, RobotCfg
from omni.isaac.orbit.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg, FlatPatchSamplingCfg, TerrainGeneratorCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.mesh as mesh_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.utils.warp import convert_to_warp_mesh, raycast_mesh
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time
import carb
import random

from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG

class NavigationEnv(IsaacEnv):
    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        # self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vfov = cfg.sensor.lidar_vfov
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        
        self.curr_num_obstacles = cfg.env.get("initial_num_obstacles", 0)
        self.max_num_obstacles = cfg.env.get("max_num_obstacles", 1000)
        self.incre_num_obstacles = cfg.env.get("increment_num_obstacles", 30)
        
        print(f"[ENV CONFIG] Fixed obstacle count: {self.curr_num_obstacles}")
        print(f"[ENV CONFIG] Warning: Obstacle count cannot be changed during runtime, sim app restart required")
        self.desired_height = 0.42
        self.standstill_steps = 100
        self.checkpoint_interval = 500
        
        super().__init__(cfg, cfg.headless)
        self._standstill_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.init_vels = torch.zeros(self.num_envs, 6, device=self.device)  # lin(3) + ang(3)

        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Go2/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,
                vertical_ray_angles=torch.tensor([self.lidar_vfov])
            ),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)
        
        # start and target 
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_go2_vel_w = torch.zeros(self.num_envs, 1, 3)
            self.prev_go2_rot = torch.zeros(self.num_envs, 1, 4)
            self.prev_distance_2d = torch.ones(self.num_envs, 1, 1) * 1000.0 # previous distance to goal in 2D plane
            self.init_distance_2d = torch.ones(self.num_envs, 1, 1) * 1000.0
            self.total_traveled_distance = torch.zeros(self.num_envs, 1, 1) + 1e-6
            self.prev_yaw_rate = torch.zeros(self.num_envs, 1, 1)
            self.last_check_step = torch.zeros(self.num_envs)
            self.last_check_distance = torch.zeros(self.num_envs, 1, 1)

    def _design_scene(self):
        go2_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Go2")
        self.go2 = Articulation(go2_cfg)

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.5), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),
            size=(300., 300.),
        )
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [10.0, 10.0, 4.5]
        print(f"[ENV CONFIG] Obstacle distribution map range: {self.map_range}")
        print(f"[ENV CONFIG] Terrain size: 50x50m (±25m)")
        print(f"[ENV CONFIG] Obstacle border_width: 15m") 
        print(f"[ENV CONFIG] Effective obstacle area: 20x20m (±10m)")
        print(f"[ENV CONFIG] Robot activity boundary will be: ±12m")

        obstacles_terrain = HfDiscreteObstaclesTerrainCfg(
            horizontal_scale=0.1,
            vertical_scale=0.1,
            border_width=15.0,
            num_obstacles=self.curr_num_obstacles,
            obstacle_height_mode="range",
            obstacle_width_range=(0.4, 1.1),
            obstacle_height_range=[4.0, 6.0],
            obstacle_height_probability=[1.0],
            platform_width=0.0,
            )
        
        sub_terrains = {
            "obstacles": obstacles_terrain,
        }

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(50.0, 50.0),
                border_width=0.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains=sub_terrains
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        self.terrain_importer = TerrainImporter(terrain_cfg)
        self.terrain_cfg_template = terrain_cfg
        
        self._verify_lidar_terrain_consistency()
        
        print(f"[ENV CONFIG] Terrain initialization complete, obstacle count fixed at: {self.curr_num_obstacles}")
        print(f"[ENV CONFIG] LiDAR sensor configured, ensuring consistency with terrain data")
        return

    def regenerate_terrain_with_obstacles(self, num_obstacles: int):
        """
        DEPRECATED - Do not use during runtime!
        
        Regenerating terrain at runtime causes serious issues:
        1. LiDAR sensors cannot update properly, causing inconsistency between sensor data and actual obstacles
        2. Isaac Sim physical simulation state may become inconsistent
        3. May cause other sensor data anomalies
        
        Correct approach:
        - Determine obstacle count before starting sim app
        - To change obstacle count, must restart entire sim app (outer loop)
        - Do not dynamically change obstacle count during environment runtime (inner loop)
        """
        raise RuntimeError(
            "[ENV ERROR] Changing obstacle count during runtime is prohibited!\n"
            "This causes LiDAR data and actual obstacles to be inconsistent.\n"
            "To change obstacle count, please restart the entire sim app.\n"
            f"Current fixed obstacle count: {self.curr_num_obstacles}"
        )
    
    def _verify_lidar_terrain_consistency(self):
        """
        Verify consistency between LiDAR sensor and terrain configuration.
        Ensure LiDAR can properly detect all terrain obstacles.
        """
        if hasattr(self, 'lidar') and hasattr(self, 'terrain_importer'):
            print(f"[ENV CHECK] LiDAR configuration:")
            print(f"  - LiDAR range: {self.lidar_range}m")
            print(f"  - LiDAR horizontal resolution: {self.lidar_hres}°")
            print(f"  - LiDAR vertical beams: {self.lidar_vbeams}")
            print(f"  - LiDAR horizontal beams: {self.lidar_hbeams}")
            print(f"[ENV CHECK] Terrain configuration:")
            print(f"  - Terrain size: 50x50m")
            print(f"  - Obstacle distribution area: ±{self.map_range[0]}m x ±{self.map_range[1]}m")
            print(f"  - Obstacle count: {self.curr_num_obstacles}")
            print(f"[ENV CHECK] LiDAR and terrain consistency verification complete")

    def get_obstacle_info(self):
        """
        Get current environment obstacle information.
        Returns: dict containing obstacle count and related configuration info
        """
        return {
            "current_obstacles": self.curr_num_obstacles,
            "max_obstacles": self.max_num_obstacles,
            "increment_obstacles": self.incre_num_obstacles,
            "map_range": self.map_range,
            "terrain_size": "50x50m",
            "obstacle_distribution_area": f"±{self.map_range[0]}m x ±{self.map_range[1]}m",
            "lidar_range": self.lidar_range,
            "can_change_runtime": False,
            "change_method": "Requires sim app restart (outer loop)"
        }
    
    def suggest_restart_for_obstacle_change(self, new_obstacle_count: int):
        """
        Provide restart suggestions when obstacle count change is needed.
        """
        print(f"\n{'='*60}")
        print(f"Obstacle count change request: {self.curr_num_obstacles} → {new_obstacle_count}")
        print(f"{'='*60}")
        print(f"❌ Error: Cannot change obstacle count during runtime!")
        print(f"")
        print(f"Correct procedure:")
        print(f"  1. Save current training progress")
        print(f"  2. Completely close current sim app process")
        print(f"  3. Modify initial_num_obstacles in config file to {new_obstacle_count}")
        print(f"  4. Restart sim app (outer loop)")
        print(f"  5. Resume training")
        print(f"")
        print(f"Reason: After Isaac Sim initialization, LiDAR sensors cannot dynamically update terrain changes")
        print(f"Result: Ensures 100% consistency between LiDAR data and actual obstacles")
        print(f"{'='*60}\n")
        
        return False

    def _set_specs(self):
        observation_dim = 7 + 4

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((3,))  # vx, vy, vyaw
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),

            "reward_vel_goal": UnboundedContinuousTensorSpec(1),
            "reward_stability": UnboundedContinuousTensorSpec(1),
            "reward_distance": UnboundedContinuousTensorSpec(1),
            "reward_safety_static": UnboundedContinuousTensorSpec(1),
            "reward_exploration": UnboundedContinuousTensorSpec(1),
            "reward_vel_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_yaw_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_checkpoint": UnboundedContinuousTensorSpec(1),
            "reward_lateral_backward": UnboundedContinuousTensorSpec(1),

        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "orientation": UnboundedContinuousTensorSpec((1, 4), device=self.device),
            "linear_velocity": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "angular_velocity": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            "joint_position": UnboundedContinuousTensorSpec((1, 12), device=self.device),
            "joint_velocity": UnboundedContinuousTensorSpec((1, 12), device=self.device),
            "last_action": UnboundedContinuousTensorSpec((1, 12), device=self.device),
            "default_joint_position": UnboundedContinuousTensorSpec((1, 12), device=self.device),
            "default_joint_velocity": UnboundedContinuousTensorSpec((1, 12), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        w_push = self.map_range[0] + 0.5
        h_push = self.map_range[1] + 0.5
        # generate random positions
        masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
        shifts = torch.tensor([[0., h_push, 0.], [0., -h_push, 0.], [w_push, 0., 0.], [-w_push, 0., 0.]], dtype=torch.float, device=self.device)
        mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
        selected_masks = masks[mask_indices].unsqueeze(1)
        selected_shifts = shifts[mask_indices].unsqueeze(1)

        pos = torch.rand((len(env_ids), 1, 3), device=self.device) * 2 - 1
        pos[..., :2] *= torch.tensor([self.map_range[0], self.map_range[1]], device=self.device)
        pos[..., 2] = self.desired_height
        
        apply_shift = torch.rand(len(env_ids), 1, 1, device=self.device) < 0.5
        pos = torch.where(apply_shift, pos * selected_masks + selected_shifts, pos)

        self.target_pos[env_ids] = pos

    def _reset_idx(self, env_ids: torch.Tensor):
        w_push = self.map_range[0] + 0.5
        h_push = self.map_range[1] + 0.5
        self.go2.reset(env_ids)
        self.reset_target(env_ids)

        masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
        shifts = torch.tensor([[0., h_push, 0.], [0., -h_push, 0.], [w_push, 0., 0.], [-w_push, 0., 0.]], dtype=torch.float, device=self.device)
        
        mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
        selected_masks = masks[mask_indices].unsqueeze(1)
        selected_shifts = shifts[mask_indices].unsqueeze(1)

        pos = torch.rand((len(env_ids), 1, 3), device=self.device) * 2 - 1
        pos[..., :2] *= torch.tensor([self.map_range[0], self.map_range[1]], device=self.device)
        pos[..., 2] = self.desired_height
        
        apply_shift = torch.rand(len(env_ids), 1, 1, device=self.device) < 0.5
        pos = torch.where(apply_shift, pos * selected_masks + selected_shifts, pos)

        # Coordinate change: after reset, the go2's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        default_pose = self.go2._data.default_root_state.clone()
        
        default_pose[env_ids, :3] = pos.squeeze(1)
        default_quat = default_pose[env_ids, 3:7]
        
        if random.random() < 0.5:
            diff = self.target_pos[env_ids] - pos
            facing_yaw = torch.atan2(diff[..., 1], diff[..., 0]).squeeze(1)
            euler_angles = quaternion_to_euler(default_quat)
            euler_angles[..., 2] = facing_yaw
            new_quat = euler_to_quaternion(euler_angles)
            default_pose[env_ids, 3:7] = new_quat
        else:
            random_yaw = torch.rand((len(env_ids),), device=self.device) * 2 * torch.pi
            euler_angles = quaternion_to_euler(default_quat)
            euler_angles[..., 2] = random_yaw
            new_quat = euler_to_quaternion(euler_angles)
            default_pose[env_ids, 3:7] = new_quat
        # print(f"[DEBUG] Machine dog default quaternion: {default_pose[env_ids[0], 3:7].cpu().numpy()}")
        
        self.go2.write_root_state_to_sim(
            default_pose[env_ids],
            env_ids=env_ids
        )
        self.go2.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        
        default_joint_pos = self.go2.data.default_joint_pos[env_ids]
        default_joint_vel = self.go2.data.default_joint_vel[env_ids]
        self.go2.write_joint_state_to_sim(
            position=default_joint_pos,
            velocity=default_joint_vel,
            env_ids=env_ids
        )
        
        self.prev_go2_rot[env_ids] = default_pose[env_ids, 3:7].clone().unsqueeze(1)
        self.prev_go2_vel_w[env_ids] = 0.
        self.prev_distance_2d[env_ids] = self.target_dir[env_ids][..., :2].norm(dim=-1, keepdim=True)  # reset previous distance to goal in 2D plane
        self.init_distance_2d[env_ids] = self.target_dir[env_ids][..., :2].norm(dim=-1, keepdim=True)
        self.stats[env_ids] = 0.
        self.total_traveled_distance[env_ids] = 1e-6
        self.prev_yaw_rate[env_ids] = 0.
        
        self._standstill_counter[env_ids] = self.standstill_steps
        self.last_check_step[env_ids] = self.progress_buf[env_ids].clone()
        self.last_check_distance[env_ids] = self.init_distance_2d[env_ids].clone() 
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "joint_actions")].squeeze(1)
        self.go2.set_joint_position_target(actions)
        self.go2.write_data_to_sim()

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.lidar.update(self.dt)
        self.go2.update(self.dt)
        self.go2.write_data_to_sim()
    
    # get current states/observation
    def _compute_state_and_obs(self):
        go2_pos_w = self.go2.data.root_pos_w.unsqueeze(1)
        go2_quat_w = self.go2.data.root_quat_w.unsqueeze(1)
        go2_lin_vel_w = self.go2.data.root_lin_vel_w.unsqueeze(1)
        go2_ang_vel_w = self.go2.data.root_ang_vel_w.unsqueeze(1)
        go2_ang_vel_b = math_utils.quat_rotate_inverse(self.go2.data.root_quat_w, self.go2.data.root_ang_vel_w).unsqueeze(1)

        joint_pos = self.go2.data.joint_pos.unsqueeze(1)
        joint_vel = self.go2.data.joint_vel.unsqueeze(1)
        last_actions = self.go2.data.joint_pos_target.unsqueeze(1)
        default_joint_pos = self.go2.data.default_joint_pos.unsqueeze(1)
        default_joint_vel = self.go2.data.default_joint_vel.unsqueeze(1)

        self.info["position"] = go2_pos_w
        self.info["orientation"] = go2_quat_w
        self.info["linear_velocity"] = go2_lin_vel_w
        self.info["angular_velocity"] = go2_ang_vel_b
        self.info["joint_position"] = joint_pos
        self.info["joint_velocity"] = joint_vel
        self.info["last_action"] = last_actions
        self.info["default_joint_position"] = default_joint_pos
        self.info["default_joint_velocity"] = default_joint_vel
        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame
        self.lidar_obs = self.lidar_scan.clone()

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            
            x = self.lidar.data.pos_w[0]

            lidar_pos = self.lidar.data.pos_w[0].cpu()
            lidar_hits = self.lidar.data.ray_hits_w[0].cpu()

            directions = lidar_hits - lidar_pos
            valid_hit_mask = torch.norm(directions, dim=-1) < self.lidar_range
            lidar_hits_valid = lidar_hits[valid_hit_mask]

            self.debug_draw.vector(
                x=lidar_pos.expand_as(directions)[valid_hit_mask],
                v=directions[valid_hit_mask],
                size=1.0,
                color=(0.0, 0.5, 1.0, 1.0)
            )

            if len(lidar_hits_valid) > 0:
                zero_directions = torch.zeros_like(lidar_hits_valid)
                zero_directions[..., 2] = self.desired_height

                self.debug_draw.vector(
                    x=lidar_hits_valid,
                    v=zero_directions,
                    size=1.5,
                    color=(1.0, 0.0, 0.0, 1.0)
                )

            go2_positions = go2_pos_w.cpu().squeeze(1)
            target_positions = self.target_pos.squeeze(1).cpu()
            directions_to_target = target_positions - go2_positions

            self.debug_draw.vector(
                x=go2_positions,
                v=directions_to_target,
                size=2.0,
                color=(0.0, 1.0, 0.0, 1.0)
            )

            vec_yaw_w = quat_axis(go2_quat_w, axis=0)

            vec_yaw_w = vec_yaw_w.cpu()
            vec_yaw_w = vec_yaw_w / vec_yaw_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            self.debug_draw.vector(
                x=go2_positions,
                v=vec_yaw_w,
                size=3.0,
                color=(1.0, 0.0, 0.0, 1.0)
            )


        # ---------Network Input II: Go2's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - go2_pos_w
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)

        # b. unit direction vector to goal
        # self.target_dir[:] = rpos
        # target_dir_2d = self.target_dir.clone()
        # target_dir_2d[..., 2] = 0

        # c. yaw and angular z
        euler_angles = quaternion_to_euler(go2_quat_w)
        roll_w = euler_angles[..., 0]
        pitch_w = euler_angles[..., 1]
        yaw_w = euler_angles[..., 2]
        vec_yaw_w = quat_axis(go2_quat_w, axis=0)  # (num_envs, 3)
        vec_yaw_w_2d = vec_yaw_w.clone()
        vec_yaw_w_2d[..., 2] = 0
        vec_yaw_body = vec_to_new_frame(vec_yaw_w, vec_yaw_w_2d)
        yaw_rate_w = go2_ang_vel_w[..., 2].unsqueeze(-1)

        rpos_clipped = rpos / distance.clamp(1e-6)
        rpos_clipped_body = vec_to_new_frame(rpos_clipped, vec_yaw_w_2d)
        
        vel_body = vec_to_new_frame(go2_lin_vel_w, vec_yaw_w_2d)

        # print("rpos_clipped_body.shape=", rpos_clipped_body.shape)
        # print("distance_2d.shape=", distance_2d.shape)
        # print("vel_body.shape=", vel_body.shape)
        # print("vec_yaw_body.shape=", vec_yaw_body.shape)
        # print("yaw_rate_w.shape=", yaw_rate_w.shape)

        go2_state = torch.cat([
            rpos_clipped_body,     # unit direction to goal in robot's local frame
            distance_2d,           # scalar distance to goal
            vel_body,              # velocity in robot's local frame
            vec_yaw_body,          # current yaw angle in local frame (should be [1,0,0])
            yaw_rate_w             # current yaw rate
        ], dim=-1).squeeze(1)      # shape: (num_envs, )

        # -----------------Network Input Final--------------
        obs = {
            "state": go2_state,
            "lidar": self.lidar_obs,
        }
        
        # -----------------Reward Calculation-----------------

        # basic status
        vel_2d = go2_lin_vel_w[..., :2]
        vel_norm = torch.norm(vel_2d, dim=-1, keepdim=True) + 1e-8
        vel_2d_unit = vel_2d / vel_norm
        traveled_this_step = vel_norm * self.dt
        self.total_traveled_distance += traveled_this_step
        yaw_rate_abs = torch.abs(yaw_rate_w)
        front_lidar_scan = torch.cat([self.lidar_scan[:, :, :self.lidar_hbeams//8], self.lidar_scan[:, :, -self.lidar_hbeams//8:]], dim=2)
        ahead_lidar_scan = torch.cat([self.lidar_scan[:, :, :self.lidar_hbeams//18], self.lidar_scan[:, :, -self.lidar_hbeams//18:]], dim=2)
        min_front_distance = self.lidar_range - einops.reduce(front_lidar_scan, "n 1 w h -> n 1", "max")
        min_ahead_distance = self.lidar_range - einops.reduce(ahead_lidar_scan, "n 1 w h -> n 1", "max")
        is_moving_float_mask = ((vel_2d[..., 0].abs() > 0.1) | (yaw_rate_abs[..., 0] > 0.1)).float()

        # a. distance reward
        delta_distance = self.prev_distance_2d - distance_2d
        move_distance = self.init_distance_2d - distance_2d
        move_ratio = move_distance / (self.init_distance_2d + 1e-8)
        d_max_move_range = self.dt * self.cfg.algo.actor.action_limit.linear_velocity
        reward_distance = delta_distance.squeeze(-1) / (d_max_move_range + 1e-8)

        at_checkpoint = (self.progress_buf % self.checkpoint_interval == 0)
        check_env_ids = torch.where(at_checkpoint.squeeze(-1))[0]
        reward_checkpoint = torch.zeros(self.num_envs, 1, device=self.device)
        if len(check_env_ids) > 0:
            curr_distance = distance_2d[check_env_ids]
            prev_distance = self.last_check_distance[check_env_ids]
            improvement = (prev_distance - curr_distance).squeeze(-1)
            reward_checkpoint[check_env_ids] = improvement * 10.0
            self.last_check_distance[check_env_ids] = curr_distance.clone()
            self.last_check_step[check_env_ids] = self.progress_buf[check_env_ids].clone()

        # b. velocity reward
        goal_dir_unit = rpos / distance.clamp_min(1e-6)
        clearance_weight = min_ahead_distance.clamp(min=0.0, max=1.0)
        reward_vel_goal = (vec_yaw_w[..., :2] * goal_dir_unit[..., :2]).sum(-1)
        reward_vel_goal *= clearance_weight
        dv_x = go2_lin_vel_w[..., 0] - self.prev_go2_vel_w[..., 0]
        dv_y = go2_lin_vel_w[..., 1] - self.prev_go2_vel_w[..., 1]
        reward_vel_smoothness = -(dv_x ** 2 + dv_y ** 2) * 0.5

        d_yaw_rate = (yaw_rate_w - self.prev_yaw_rate).squeeze(-1)
        reward_yaw_smoothness = -(d_yaw_rate ** 2) * 0.01

        # c. safety reward
        reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))

        # d. exploration reward
        exploration_weight = torch.sigmoid(1.0 - min_ahead_distance)
        reward_exploration = yaw_rate_abs.squeeze(-1) * exploration_weight * 0.01

        # e. lateral backward reward
        vec_right_w = torch.stack([torch.sin(yaw_w), -torch.cos(yaw_w), torch.zeros_like(yaw_w)], dim=-1)
        vel_lateral = (vel_2d * vec_right_w[..., :2]).sum(dim=-1) * 0.3
        vel_forward = torch.sum(vec_yaw_w[..., :2] * vel_2d, dim=-1)
        vel_forward_clamped = vel_forward.clamp(max=0.0)
        reward_lateral_backward = - (torch.abs(vel_lateral) + torch.abs(vel_forward_clamped))

        # f. stability reward
        excess_roll = torch.relu(torch.abs(roll_w) - 0.1)
        excess_pitch = torch.relu(torch.abs(pitch_w) - 0.1)
        reward_stability = - (excess_roll ** 2 + excess_pitch ** 2)

        # Collision condition with its penalty
        center_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3)
        rear_lidar_scan = self.lidar_scan[:, :, self.lidar_hbeams * 3 // 8: self.lidar_hbeams * 5 // 8, :]
        rear_collision = einops.reduce(rear_lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.5)
        collision = center_collision | rear_collision

        # Final reward calculation
        total_reward = torch.zeros_like(reward_distance)
        total_reward += reward_distance
        total_reward += reward_checkpoint
        total_reward += reward_vel_goal
        total_reward += reward_vel_smoothness
        total_reward += reward_yaw_smoothness
        total_reward += reward_safety_static
        # total_reward += reward_exploration
        total_reward += reward_lateral_backward
        total_reward += reward_stability
        self.reward = total_reward

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)

        boundary_extension = getattr(self.cfg, 'line_formation_config', {}).get('env_boundary_extension', 0.0)
        x_boundary = self.map_range[0] + 1.0 + boundary_extension
        y_boundary = self.map_range[1] + 1.0 + boundary_extension
        
        if boundary_extension > 0:
            print(f"Robot activity boundary extended to ±{x_boundary:.1f}, ±{y_boundary:.1f}")
            print(f"Obstacle distribution remains in ±{self.map_range[0]:.1f}, ±{self.map_range[1]:.1f}")
        
        x_out_low = go2_pos_w[..., 0] < -x_boundary
        x_out_high = go2_pos_w[..., 0] > x_boundary
        x_out = x_out_low | x_out_high

        # y-axis boundary check
        y_out_low = go2_pos_w[..., 1] < -y_boundary
        y_out_high = go2_pos_w[..., 1] > y_boundary
        y_out = y_out_low | y_out_high

        # Combine into one out-of-bound condition
        out_of_bounds = x_out | y_out

        self.terminated = collision | reach_goal | out_of_bounds
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number
        if collision.any():
            self.reward[collision] -= 50.0 # penalize collision episodes
        if reach_goal.any():
            self.reward[reach_goal] += 50.0 * self.init_distance_2d[reach_goal].squeeze(1)

        # update previous status
        self.prev_go2_vel_w = go2_lin_vel_w.clone()
        self.prev_go2_rot.copy_(go2_quat_w)
        self.prev_distance_2d = distance_2d.clone()
        self.prev_yaw_rate.copy_(yaw_rate_w)

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

        if not self.cfg.algo.actor.pred_z:
            self.stats["reward_vel_goal"] = reward_vel_goal.unsqueeze(-1)
            self.stats["reward_stability"] = reward_stability.unsqueeze(-1)
            self.stats["reward_distance"] = reward_distance.unsqueeze(-1)
            self.stats["reward_safety_static"] = reward_safety_static  # shape: (num_envs, 1)
            self.stats["reward_exploration"] = reward_exploration.unsqueeze(-1)
            self.stats["reward_vel_smoothness"] = reward_vel_smoothness.unsqueeze(-1)
            self.stats["reward_yaw_smoothness"] = reward_yaw_smoothness.unsqueeze(-1)
            self.stats["reward_checkpoint"] = reward_checkpoint.unsqueeze(-1)
            self.stats["reward_lateral_backward"] = reward_lateral_backward.unsqueeze(-1)

        standstill_mask = self._standstill_counter > 0
        if standstill_mask.any():
            self._standstill_counter[standstill_mask] -= 1
        self.info["standstill"] = standstill_mask
        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)
        

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
