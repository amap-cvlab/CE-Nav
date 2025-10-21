import os
import torch
import carb
import gymnasium as gym
# from omni.isaac.lab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_him_rough_cfg
# from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.orbit_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

base_vel_cmd_input = None

# # Initialize base_vel_cmd_input as a tensor when created
# def init_base_vel_cmd(num_envs):
#     global base_vel_cmd_input
#     base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)

# # Modify base_vel_cmd to use the tensor directly
# def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
#     global base_vel_cmd_input
#     return base_vel_cmd_input.clone().to(env.device)

# # Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
# def sub_keyboard_event(event) -> bool:
#     global base_vel_cmd_input
#     lin_vel = 1.5
#     ang_vel = 1.5
    
#     if base_vel_cmd_input is not None:
#         if event.type == carb.input.KeyboardEventType.KEY_PRESS:
#             # Update tensor values for environment 0
#             if event.input.name == 'W':
#                 base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
#             elif event.input.name == 'S':
#                 base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
#             elif event.input.name == 'A':
#                 base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
#             elif event.input.name == 'D':
#                 base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
#             elif event.input.name == 'Z':
#                 base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
#             elif event.input.name == 'C':
#                 base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
            
#             # If there are multiple environments, handle inputs for env 1
#             if base_vel_cmd_input.shape[0] > 1:
#                 if event.input.name == 'I':
#                     base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
#                 elif event.input.name == 'K':
#                     base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
#                 elif event.input.name == 'J':
#                     base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
#                 elif event.input.name == 'L':
#                     base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
#                 elif event.input.name == 'M':
#                     base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
#                 elif event.input.name == '>':
#                     base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        
#         # Reset commands to zero on key release
#         elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
#             base_vel_cmd_input.zero_()
#     return True

# def get_rsl_flat_policy(cfg):
#     cfg.observations.policy.height_scan = None
#     env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
#     env = RslRlVecEnvWrapper(env)

#     # Low level control: rsl control policy
#     agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
#     ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
#                                     run_dir=agent_cfg["load_run"], 
#                                     checkpoint=agent_cfg["load_checkpoint"])
#     ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
#     ppo_runner.load(ckpt_path)
#     policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
#     return env, policy

def get_rsl_him_rough_policy():
    """
    Load RSL-RL policy only, without creating a new Gym environment.
    Suitable for nested control scenarios (e.g., integration with NavigationEnv).
    Supports both PyTorch checkpoint and JIT model formats.
    """
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_him_rough_cfg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "../ckpts/unitree_go2/model.pt")
    policy = torch.jit.load(ckpt_path, map_location='cuda:0')
    policy = torch.jit.freeze(policy.eval())
    return policy

# def get_rsl_rough_policy(cfg):
#     env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
#     env = RslRlVecEnvWrapper(env)

#     # Low level control: rsl control policy
#     agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
#     ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
#                                     run_dir=agent_cfg["load_run"], 
#                                     checkpoint=agent_cfg["load_checkpoint"])
#     ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
#     ppo_runner.load(ckpt_path)
#     policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
#     return env, policy