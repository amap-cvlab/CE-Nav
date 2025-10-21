# from omni.isaac.lab.scene import InteractiveSceneCfg
# from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG
# from omni.isaac.lab.sensors import RayCasterCfg, patterns, ContactSensorCfg
# from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
# import omni.isaac.lab.sim as sim_utils
# import omni.isaac.lab.envs.mdp as mdp
# from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
# from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
# from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
# from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.utils.noise import UniformNoiseCfg
# from omni.isaac.core.utils.viewports import set_camera_view
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import go2.go2_ctrl as go2_ctrl

# @configclass
# class Go2SimCfg(InteractiveSceneCfg):
#     # ground plane
#     ground = AssetBaseCfg(prim_path="/World/ground", 
#                           spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.)),
#                           init_state=AssetBaseCfg.InitialStateCfg(
#                               pos=(0, 0, 1e-4)
#                           ))
    
#     # Lights
#     light = AssetBaseCfg(
#         prim_path="/World/light",
#         spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
#     )
#     sky_light = AssetBaseCfg(
#         prim_path="/World/skyLight",
#         spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
#     )

#     # Go2 Robot
#     unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")

#     # Go2 foot contact sensor
#     contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Go2/.*_foot", history_length=3, track_air_time=True)

#     # Go2 height scanner
#     height_scanner = RayCasterCfg(
#         prim_path="{ENV_REGEX_NS}/Go2/base",
#         offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)), 
#         attach_yaw_only=True,
#         pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), 
#         debug_vis=False,
#         mesh_prim_paths=["/World/ground"],
#     )

# @configclass
# class ActionsCfg:
#     """Action specifications for the environment."""
#     joint_pos = mdp.JointPositionActionCfg(asset_name="unitree_go2", joint_names=[".*"])

# @configclass
# class ObservationsCfg:
#     """Observation specifications for the environment."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         # observation terms (order preserved)
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
#                                params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
#                                params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
#         projected_gravity = ObsTerm(func=mdp.projected_gravity,
#                                     params={"asset_cfg": SceneEntityCfg(name="unitree_go2")},
#                                     noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
#         # velocity command
#         base_vel_cmd = ObsTerm(func=go2_ctrl.base_vel_cmd)

#         joint_pos = ObsTerm(func=mdp.joint_pos_rel,
#                             params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel,
#                             params={"asset_cfg": SceneEntityCfg(name="unitree_go2")})
#         actions = ObsTerm(func=mdp.last_action)
        
#         # Height scan
#         height_scan = ObsTerm(func=mdp.height_scan,
#                               params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#                               clip=(-1.0, 1.0))

#         def __post_init__(self) -> None:
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()

# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""
#     base_vel_cmd = mdp.UniformVelocityCommandCfg(
#         asset_name="unitree_go2",
#         resampling_time_range=(0.0, 0.0),
#         debug_vis=True,
#         ranges=mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
#         ),
#     )

# @configclass
# class EventCfg:
#     """Configuration for events."""
#     pass

# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""
#     pass

# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""
#     pass

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""
#     pass

# @configclass
# class Go2RSLEnvCfg(ManagerBasedRLEnvCfg):
#     """Configuration for the Go2 environment."""
#     # scene settings
#     scene = Go2SimCfg(num_envs=2, env_spacing=2.0)

#     # basic settings
#     observations = ObservationsCfg()
#     actions = ActionsCfg()
    
#     # dummy settings
#     commands = CommandsCfg()
#     rewards = RewardsCfg()
#     terminations = TerminationsCfg()
#     events = EventCfg()
#     curriculum = CurriculumCfg()

#     def __post_init__(self):
#         # viewer settings
#         self.viewer.eye = [-4.0, 0.0, 5.0]
#         self.viewer.lookat = [0.0, 0.0, 0.0]

#         # step settings
#         self.decimation = 8  # step

#         # simulation settings
#         self.sim.dt = 0.005  # sim step every 
#         self.sim.render_interval = self.decimation  
#         self.sim.disable_contact_processing = True
#         self.sim.antialiasing_mode = None
#         # self.sim.physics_material = self.scene.terrain.physics_material

#         # settings for rsl env control
#         self.episode_length_s = 20.0 # can be ignored
#         self.is_finite_horizon = False
#         self.actions.joint_pos.scale = 0.25

#         if self.scene.height_scanner is not None:
#             self.scene.height_scanner.update_period = self.decimation * self.sim.dt

# def camera_follow(env):
#     if (env.unwrapped.scene.num_envs == 1):
#         robot_position = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, :3].cpu().numpy()
#         robot_orientation = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, 3:7].cpu().numpy()
#         rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
#                                 robot_orientation[3], robot_orientation[0]])
#         yaw = rotation.as_euler('zyx')[0]
#         yaw_rotation = R.from_euler('z', yaw).as_matrix()
#         set_camera_view(
#             yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
#             robot_position
#         )