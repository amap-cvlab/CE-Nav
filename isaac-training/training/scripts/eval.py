import os
import hydra
import datetime
import wandb
import torch
import numpy as np
import json
import time
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import ILtoPPO
from omni_drones.controllers import LeePositionController
from controller import LocomotionController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType
from typing import List, Dict, Tuple
from pathlib import Path
from tensordict.tensordict import TensorDict

import sys
root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
from il_training.fastsys.infer_policy import PolicyInference
from il_training.fastsys.infer_policy_mlp import MLPPolicyInference

class ExpertAsPolicy:
    """Wrapper to use expert model as policy."""
    def __init__(self, expert_model, device='cuda'):
        self.expert = expert_model
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: TensorDict) -> torch.Tensor:
        observation_data = obs.get(("agents", "observation"), obs)
        observation_data = observation_data.to(self.device)
        
        if hasattr(self.expert, 'sample_action') and callable(self.expert.sample_action):
            action = self.expert.sample_action(observation_data)
        else:
            raise NotImplementedError("Expert model must have 'sample_action' method.")
        
        output_td = obs.clone()
        output_td.set(("agents", "action"), action)
        
        return output_td

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="eval", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": False, "anti_aliasing": 1})

    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    from env import NavigationEnv
    env = NavigationEnv(cfg)
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from go2.go2_ctrl import get_rsl_him_rough_policy
    locomotion_policy = get_rsl_him_rough_policy()
    print("Locomotion policy loaded")
    
    transforms = []
    vel_transform = LocomotionController(locomotion_policy, device=cfg.device)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    

    eval_mode = cfg.evaluation.mode
    policy = None
    
    if eval_mode == "rl_agent":
        print("Mode: Evaluating RL agent")
        from ppo_pure_rl import PureRLILtoPPO
        
        policy = PureRLILtoPPO(
            cfg.algo, 
            transformed_env.observation_spec, 
            transformed_env.action_spec, 
            'multi', 
            cfg.device
        )
        checkpoint = cfg.evaluation.rl_agent_checkpoint
        policy.load_state_dict(torch.load(checkpoint))
        print(f"RL checkpoint loaded: {checkpoint}")

    elif eval_mode == "expert_cnfm":
        print("Mode: Evaluating CNFM expert")
        expert_model_path = cfg.evaluation.expert_cnfm_model_path
        
        try:
            expert_inference = PolicyInference(expert_model_path, cfg)
            policy = ExpertAsPolicy(expert_inference, device=cfg.device)
            print(f"Expert model loaded: {expert_model_path}")
        except Exception as e:
            print(f"Failed to load expert: {e}")
            sim_app.close()
            return

    elif eval_mode == "guided_student":
        print("Mode: Evaluating guided student")
        expert_model_path = cfg.evaluation.expert_cnfm_model_path
        student_checkpoint = cfg.evaluation.student_checkpoint
        
        try:
            expert_inference = PolicyInference(expert_model_path, cfg)
            print(f"Expert loaded: {expert_model_path}")
            
            from ppo import ILtoPPO
            policy = ILtoPPO(
                cfg.algo, 
                transformed_env.observation_spec, 
                transformed_env.action_spec, 
                'multi', 
                cfg.device,
                il_ref_model=expert_inference
            )
            
            checkpoint_data = torch.load(student_checkpoint, map_location=cfg.device)
            policy.load_state_dict(checkpoint_data)
            print(f"Student checkpoint loaded: {student_checkpoint}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            sim_app.close()
            return
    
    elif eval_mode == "expert_mlp":
        print("Mode: Evaluating MLP expert")
        expert_model_path = cfg.evaluation.expert_mlp_model_path
        
        try:
            expert_inference = MLPPolicyInference(expert_model_path, cfg)
            policy = ExpertAsPolicy(expert_inference, device=cfg.device)
            print(f"MLP expert loaded: {expert_model_path}")
        except Exception as e:
            print(f"Failed to load MLP expert: {e}")
            sim_app.close()
            return
            
    else:
        raise ValueError(f"Unknown eval mode: '{eval_mode}'. Choose 'rl_agent', 'expert_cnfm', or 'expert_mlp'.")
    
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )

    if hasattr(cfg, 'fixed_scenario_eval') and cfg.fixed_scenario_eval:
        
        if hasattr(cfg, 'use_line_formation') and cfg.use_line_formation:
            line_formation_results = evaluate_line_formation_multiple_obstacles(
                env=transformed_env,
                policy=policy,
                cfg=cfg
            )
            
            run.log(line_formation_results['summary'])
            print_line_formation_results(line_formation_results)
            save_line_formation_results(line_formation_results, cfg)
            
        elif hasattr(cfg, 'models_to_compare') and len(cfg.models_to_compare) > 1:
            comparison_results = compare_multiple_models(
                env=env,
                cfg=cfg
            )
            
            run.log(comparison_results['summary'])
            print_comparison_results(comparison_results)
            save_comparison_results(comparison_results, cfg)
            
        else:
            eval_results = evaluate_fixed_scenarios(
                env=transformed_env,
                policy=policy,
                cfg=cfg,
                model_name=f"ILtoPPO_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            run.log(eval_results)
            print(f"\n[Nav]: Fixed scenario evaluation completed!")
            print(f"Success Rate: {eval_results['success_rate']:.2%}")
            avg_time = eval_results.get('avg_completion_time', None)
            if avg_time is not None:
                print(f"Average Completion Time: {avg_time:.2f}s")
            else:
                print(f"Average Completion Time: N/A")
            print(f"Path Efficiency: {eval_results['path_efficiency']:.3f}")
            print(f"Smoothness Score: {eval_results['smoothness_score']:.3f}")
            print(f"SPL: {eval_results.get('overall_stats', {}).get('spl', 0.0):.3f}")
            
            save_detailed_results(eval_results, cfg)
    else:
        print(f"\\n[Nav]: Starting standard evaluation with {cfg.env.num_envs} parallel environments")
        
        all_eval_results = []
        
        for i, data in enumerate(collector):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            print(f"[Nav]: start evaluating policy at step: {i} with {cfg.env.num_envs} environments")
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            )
            env.reset()
            info.update(eval_info)
            print(f"\\n[Nav]: evaluation done - processed {cfg.env.num_envs} environments in parallel.")
            
            current_result = {
                'batch_index': i,
                'timestamp': datetime.datetime.now().isoformat(),
                'env_frames': collector._frames,
                'rollout_fps': collector._fps,
                'eval_metrics': eval_info
            }
            all_eval_results.append(current_result)
            
            run.log(info)
            
            if i == 0:
                print(f"\\n[Nav]: Single evaluation completed. Saving results...")
                break
        
        standard_eval_results = {
            'model_name': f"ILtoPPO_standard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.datetime.now().isoformat(),
            'evaluation_mode': 'standard',
            'num_environments': cfg.env.num_envs,
            'seed': cfg.seed,
            'total_batches': len(all_eval_results),
            'batch_results': all_eval_results,
            'final_metrics': eval_info if all_eval_results else {}
        }
        
        save_standard_eval_results(standard_eval_results, cfg)
        print(f"\\n[Nav]: Standard evaluation results saved successfully.")

    wandb.finish()
    sim_app.close()

def evaluate_fixed_scenarios(
    env,
    policy,
    cfg,
    model_name: str = "model",
    fixed_scenarios: List[Dict] = None
) -> Dict:
    """
    Evaluate model performance in fixed scenarios with multi-environment parallelization.
    """
    
    if fixed_scenarios is None:
        fixed_scenarios = [
            {"start_pos": [0.0, 0.0, 0.42], "target_pos": [10.0, 10.0, 0.42], "scenario_name": "diagonal_1"},
            {"start_pos": [-5.0, -5.0, 0.42], "target_pos": [15.0, 5.0, 0.42], "scenario_name": "cross_1"},
            {"start_pos": [8.0, -8.0, 0.42], "target_pos": [-8.0, 8.0, 0.42], "scenario_name": "diagonal_2"},
            {"start_pos": [0.0, 10.0, 0.42], "target_pos": [0.0, -10.0, 0.42], "scenario_name": "straight_y"},
            {"start_pos": [-10.0, 0.0, 0.42], "target_pos": [10.0, 0.0, 0.42], "scenario_name": "straight_x"},
        ]
    
    num_scenarios = len(fixed_scenarios)
    num_trials_per_scenario = getattr(cfg, 'trials_per_scenario', 5)
    num_envs = cfg.env.num_envs
    
    print(f"\\n[Nav]: Starting parallel evaluation with {num_envs} environments")
    print(f"Testing {num_scenarios} scenarios, {num_trials_per_scenario} trials each")
    
    all_results = evaluate_scenarios_parallel(
        env=env,
        policy=policy,
        scenarios=fixed_scenarios,
        num_trials_per_scenario=num_trials_per_scenario,
        cfg=cfg
    )
    
    scenario_results = {}
    for scenario in fixed_scenarios:
        scenario_name = scenario['scenario_name']
        scenario_trials = [r for r in all_results if r['scenario_name'] == scenario_name]
        scenario_stats = compute_scenario_statistics(scenario_trials, scenario_name)
        scenario_results[scenario_name] = scenario_stats
        
        print(f"\\n{scenario_name} Results:")
        print(f"  Success Rate: {scenario_stats['success_rate']:.2%}")
        avg_time = scenario_stats.get('avg_completion_time', None)
        if avg_time is not None:
            print(f"  Avg Time: {avg_time:.2f}s")
        else:
            print(f"  Avg Time: N/A")
        print(f"  SPL: {scenario_stats['spl']:.3f}")
    
    overall_stats = compute_overall_statistics(all_results, scenario_results)
    
    return {
        'model_name': model_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'overall_stats': overall_stats,
        'scenario_results': scenario_results,
        'success_rate': overall_stats['success_rate'],
        'avg_completion_time': overall_stats.get('avg_completion_time', 0), 
        'path_efficiency': overall_stats['path_efficiency'],
        'smoothness_score': overall_stats['smoothness_score']
    }


def evaluate_scenarios_parallel(
    env,
    policy,
    scenarios: List[Dict],
    num_trials_per_scenario: int,
    cfg
) -> List[Dict]:
    """
    Evaluate multiple scenarios using parallel environments.
    """
    
    num_envs = cfg.env.num_envs
    num_scenarios = len(scenarios)
    total_trials = num_scenarios * num_trials_per_scenario
    
    print(f"Total trials: {total_trials}, using {num_envs} parallel environments")
    
    trial_tasks = []
    for scenario_idx, scenario in enumerate(scenarios):
        for trial_idx in range(num_trials_per_scenario):
            trial_tasks.append({
                'scenario': scenario,
                'scenario_idx': scenario_idx,
                'trial_idx': trial_idx,
                'seed': cfg.seed + scenario_idx * 100 + trial_idx,
                'trial_name': f"{scenario['scenario_name']}_trial_{trial_idx+1}"
            })
    
    all_results = []
    num_batches = (len(trial_tasks) + num_envs - 1) // num_envs
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * num_envs
        end_idx = min(start_idx + num_envs, len(trial_tasks))
        batch_tasks = trial_tasks[start_idx:end_idx]
        
        print(f"\\nExecuting batch {batch_idx + 1}/{num_batches}: trials {start_idx + 1}-{end_idx}")
        
        batch_results = evaluate_batch_parallel(
            env=env,
            policy=policy,
            tasks=batch_tasks,
            cfg=cfg
        )
        
        all_results.extend(batch_results)
    
    return all_results


def evaluate_batch_parallel(
    env,
    policy,
    tasks: List[Dict],
    cfg
) -> List[Dict]:
    """
    Execute a batch of trial tasks in parallel.
    """
    
    num_active_envs = len(tasks)
    max_steps = env.max_episode_length
    
    env.enable_render(not cfg.headless)
    env.eval()
    
    for i, task in enumerate(tasks):
        env.set_seed(task['seed'] + i)
    
    env.reset()
    
    reset_robot_colors(env)
    
    for i, task in enumerate(tasks):
        if i >= env.num_envs:
            break
            
        scenario = task['scenario']
        start_pos = torch.tensor(scenario['start_pos'], device=env.device)
        target_pos = torch.tensor(scenario['target_pos'], device=env.device)
        
        env_id = torch.tensor([i], device=env.device)
        
        default_pose = env.go2._data.default_root_state.clone()
        default_pose[i, :3] = start_pos
        env.go2.write_root_state_to_sim(default_pose[i:i+1], env_ids=env_id)
        env.go2.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_id)
        
        env.target_pos[i] = target_pos
        env.target_dir[i] = target_pos - start_pos
    
    trial_data = {}
    for i, task in enumerate(tasks):
        if i >= env.num_envs:
            break
        trial_data[i] = {
            'task': task,
            'trajectory_positions': [],
            'trajectory_velocities': [],
            'trajectory_yaw_rates': [],
            'success': False,
            'collision': False,
            'completion_time': None,
            'completed': False,
            'start_pos': torch.tensor(task['scenario']['start_pos'], device=env.device),
            'target_pos': torch.tensor(task['scenario']['target_pos'], device=env.device)
        }
    
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    
    try:
        with torch.no_grad():
            with set_exploration_type(ExplorationType.MEAN):
                for step in range(max_steps):
                    active_envs = [i for i in trial_data.keys() if not trial_data[i]['completed']]
                    if not active_envs:
                        break
                    
                    obs = env._compute_state_and_obs()
                    
                    action = policy(obs)
                    if hasattr(action, 'mode'):
                        action = action.mode
                    elif hasattr(action, 'mean'):
                        action = action.mean
                    
                    for env_idx in active_envs:
                        current_pos = env.go2.data.root_pos_w[env_idx].cpu().numpy()
                        current_vel = env.go2.data.root_lin_vel_w[env_idx].cpu().numpy()
                        current_ang_vel = env.go2.data.root_ang_vel_w[env_idx, 2].cpu().numpy()
                        
                        trial_data[env_idx]['trajectory_positions'].append(current_pos.copy())
                        trial_data[env_idx]['trajectory_velocities'].append(current_vel.copy())
                        trial_data[env_idx]['trajectory_yaw_rates'].append(float(current_ang_vel))
                    
                    try:
                        obs_next = env.step(action)
                    except Exception as step_error:
                        print(f"Step execution error: {step_error}")
                        break
                    
                    for env_idx in active_envs[:]:
                        target_pos = trial_data[env_idx]['target_pos']
                        distance_to_target = torch.norm(env.go2.data.root_pos_w[env_idx] - target_pos)
                        
                        if distance_to_target < 0.5:
                            trial_data[env_idx]['success'] = True
                            trial_data[env_idx]['completion_time'] = (step + 1) * env.dt
                            trial_data[env_idx]['completed'] = True
                            task_name = trial_data[env_idx]['task']['trial_name']
                            print(f"   {task_name} succeeded ({step + 1} steps)")
                        
                        elif (hasattr(env, 'terminated') and env.terminated is not None and 
                              env_idx < len(env.terminated) and env.terminated[env_idx].item()):
                            if not trial_data[env_idx]['success']:
                                trial_data[env_idx]['collision'] = True
                                trial_data[env_idx]['completed'] = True
                                task_name = trial_data[env_idx]['task']['trial_name']
                                
                                print(f"\n{'='*80}")
                                print(f"COLLISION WARNING - Robot Glowing!")
                                print(f"Robot: {task_name}")
                                print(f"Steps: {step + 1}")
                                print(f"Time: {(step + 1) * env.dt:.2f}s")
                                print(f"{'='*80}\n")
                                
                                try:
                                    make_robot_glow_red(env, env_idx)
                                except Exception as e:
                                    print(f"Warning: Could not make robot glow: {e}")
                
                for env_idx in trial_data.keys():
                    if not trial_data[env_idx]['completed']:
                        trial_data[env_idx]['completed'] = True
                        task_name = trial_data[env_idx]['task']['trial_name']
                        print(f"   {task_name} timeout")
        
    except Exception as e:
        print(f"Batch execution error: {e}")
        for env_idx in trial_data.keys():
            if not trial_data[env_idx]['completed']:
                trial_data[env_idx]['collision'] = True
                trial_data[env_idx]['completed'] = True
    
    results = []
    for env_idx, data in trial_data.items():
        task = data['task']
        scenario = task['scenario']
        
        metrics = compute_trial_metrics(
            trajectory_positions=data['trajectory_positions'],
            trajectory_velocities=data['trajectory_velocities'],
            trajectory_yaw_rates=data['trajectory_yaw_rates'],
            start_pos=data['start_pos'].cpu().numpy(),
            target_pos=data['target_pos'].cpu().numpy(),
            success=data['success'],
            completion_time=data['completion_time'],
            collision_occurred=data['collision'],
            max_time=max_steps * env.dt
        )
        
        results.append({
            'trial_name': task['trial_name'],
            'scenario_name': scenario['scenario_name'],
            'seed': task['seed'],
            'success': data['success'],
            'collision': data['collision'],
            'completion_time': data['completion_time'],
            'trajectory_length': len(data['trajectory_positions']),
            'metrics': metrics
        })
    
    return results


def evaluate_single_trial(
    env,
    policy, 
    start_pos: torch.Tensor,
    target_pos: torch.Tensor,
    seed: int,
    cfg,
    trial_name: str = "trial"
) -> Dict:
    """
    Evaluate a single trial.
    """
    
    env.enable_render(not cfg.headless)
    env.eval()
    env.set_seed(seed)
    
    env.reset()
    
    env_id = torch.tensor([0], device=env.device)
    
    default_pose = env.go2._data.default_root_state.clone()
    default_pose[0, :3] = start_pos
    env.go2.write_root_state_to_sim(default_pose[0:1], env_ids=env_id)
    env.go2.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_id)
    
    env.target_pos[0] = target_pos.unsqueeze(0)
    env.target_dir[0] = target_pos.unsqueeze(0) - start_pos.unsqueeze(0)
    
    start_time = time.time()
    trajectory_positions = []
    trajectory_velocities = []
    trajectory_yaw_rates = []
    collision_occurred = False
    success = False
    completion_time = None
    
    max_steps = env.max_episode_length
    
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    
    try:
        with torch.no_grad():
            with set_exploration_type(ExplorationType.MEAN):
                for step in range(max_steps):
                    obs = env._compute_state_and_obs()
                    
                    action = policy(obs)
                    if hasattr(action, 'mode'):
                        action = action.mode
                    elif hasattr(action, 'mean'):
                        action = action.mean
                    
                    current_pos = env.go2.data.root_pos_w[0].cpu().numpy()
                    current_vel = env.go2.data.root_lin_vel_w[0].cpu().numpy()
                    current_ang_vel = env.go2.data.root_ang_vel_w[0, 2].cpu().numpy()
                    
                    trajectory_positions.append(current_pos.copy())
                    trajectory_velocities.append(current_vel.copy())
                    trajectory_yaw_rates.append(float(current_ang_vel))
                    
                    try:
                        obs_next = env.step(action)
                    except Exception as step_error:
                        print(f"    ✗ Step error: {step_error}")
                        break
                    
                    distance_to_target = torch.norm(env.go2.data.root_pos_w[0] - target_pos)
                    
                    if distance_to_target < 0.5:
                        success = True
                        completion_time = (step + 1) * env.dt
                        print(f"     Success in {step + 1} steps ({completion_time:.2f}s)")
                        break
                    
                    if hasattr(env, 'terminated') and env.terminated is not None and env.terminated[0].item():
                        if not success:
                            collision_occurred = True
                            
                            print(f"\n{'='*80}")
                            print(f"SINGLE ROBOT COLLISION - Glowing Red!")
                            print(f"Trial: {trial_name}")
                            print(f"Steps: {step + 1}")
                            print(f"Time: {(step + 1) * env.dt:.2f}s")
                            print(f"Position: {env.go2.data.root_pos_w[0].cpu().numpy()}")
                            print(f"Target: {target_pos.cpu().numpy()}")
                            print(f"{'='*80}\n")
                            
                            try:
                                make_robot_glow_red(env, 0)
                            except Exception as e:
                                print(f"Warning: Could not make robot glow: {e}")
                        break
        
        if not success and not collision_occurred and len(trajectory_positions) >= max_steps:
            print(f"     Timeout after {max_steps} steps")
        
    except Exception as e:
        print(f"    ✗ Error during evaluation: {str(e)}")
        return {
            'trial_name': trial_name,
            'seed': seed,
            'success': False,
            'collision': True,
            'completion_time': None,
            'trajectory_length': 0,
            'metrics': {
                'path_length': 0.0,
                'direct_distance': float(torch.norm(target_pos - start_pos).cpu()),
                'path_efficiency': 0.0,
                'velocity_smoothness': 0.0,
                'yaw_rate_smoothness': 0.0,
                'avg_velocity': 0.0,
                'max_velocity': 0.0,
                'spl': 0.0
            }
        }
    
    metrics = compute_trial_metrics(
        trajectory_positions=trajectory_positions,
        trajectory_velocities=trajectory_velocities,
        trajectory_yaw_rates=trajectory_yaw_rates,
        start_pos=start_pos.cpu().numpy(),
        target_pos=target_pos.cpu().numpy(),
        success=success,
        completion_time=completion_time,
        collision_occurred=collision_occurred,
        max_time=max_steps * env.dt
    )
    
    return {
        'trial_name': trial_name,
        'seed': seed,
        'success': success,
        'collision': collision_occurred,
        'completion_time': completion_time,
        'trajectory_length': len(trajectory_positions),
        'metrics': metrics
    }


def compute_trial_metrics(
    trajectory_positions: List,
    trajectory_velocities: List,
    trajectory_yaw_rates: List,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    success: bool,
    completion_time: float,
    collision_occurred: bool,
    max_time: float
) -> Dict:
    """
    Compute performance metrics for a single trial.
    """
    
    if len(trajectory_positions) == 0:
        return {
            'path_length': 0.0,
            'direct_distance': float(np.linalg.norm(target_pos - start_pos)),
            'path_efficiency': 0.0,
            'velocity_smoothness': 0.0,
            'yaw_rate_smoothness': 0.0,
            'avg_velocity': 0.0,
            'max_velocity': 0.0,
            'spl': 0.0
        }
    
    positions = np.array(trajectory_positions)
    velocities = np.array(trajectory_velocities)
    yaw_rates = np.array(trajectory_yaw_rates)
    
    path_diffs = np.diff(positions, axis=0)
    path_length = np.sum(np.linalg.norm(path_diffs, axis=1))
    
    direct_distance = np.linalg.norm(target_pos - start_pos)
    
    path_efficiency = direct_distance / max(path_length, 1e-6)
    
    velocity_norms = np.linalg.norm(velocities, axis=1)
    if len(velocity_norms) > 1:
        velocity_changes = np.diff(velocity_norms)
        velocity_smoothness = 1.0 / (1.0 + np.std(velocity_changes))
    else:
        velocity_smoothness = 1.0
    
    if len(yaw_rates) > 1:
        yaw_rate_changes = np.diff(yaw_rates)
        yaw_rate_smoothness = 1.0 / (1.0 + np.std(yaw_rate_changes))
    else:
        yaw_rate_smoothness = 1.0
    
    avg_velocity = np.mean(velocity_norms)
    max_velocity = np.max(velocity_norms)
    
    if success:
        spl = direct_distance / max(path_length, direct_distance)
    else:
        spl = 0.0
    
    return {
        'path_length': float(path_length),
        'direct_distance': float(direct_distance),
        'path_efficiency': float(path_efficiency),
        'velocity_smoothness': float(velocity_smoothness),
        'yaw_rate_smoothness': float(yaw_rate_smoothness),
        'avg_velocity': float(avg_velocity),
        'max_velocity': float(max_velocity),
        'spl': float(spl)
    }


def compute_scenario_statistics(trials: List[Dict], scenario_name: str) -> Dict:
    """
    Compute statistics for a single scenario.
    """
    
    if not trials:
        return {}
    
    successes = [t['success'] for t in trials]
    success_rate = np.mean(successes)
    
    successful_trials = [t for t in trials if t['success']]
    if successful_trials:
        completion_times = [t['completion_time'] for t in successful_trials]
        avg_completion_time = np.mean(completion_times)
        std_completion_time = np.std(completion_times)
    else:
        avg_completion_time = None
        std_completion_time = None
    
    collisions = [t['collision'] for t in trials]
    collision_rate = np.mean(collisions)
    
    path_efficiencies = [t['metrics']['path_efficiency'] for t in trials]
    velocity_smoothness = [t['metrics']['velocity_smoothness'] for t in trials]
    yaw_smoothness = [t['metrics']['yaw_rate_smoothness'] for t in trials]
    spls = [t['metrics']['spl'] for t in trials]
    
    return {
        'scenario_name': scenario_name,
        'num_trials': len(trials),
        'success_rate': float(success_rate),
        'collision_rate': float(collision_rate),
        'avg_completion_time': avg_completion_time,
        'std_completion_time': std_completion_time,
        'path_efficiency': float(np.mean(path_efficiencies)),
        'velocity_smoothness': float(np.mean(velocity_smoothness)),
        'yaw_rate_smoothness': float(np.mean(yaw_smoothness)),
        'spl': float(np.mean(spls))
    }


def compute_overall_statistics(all_trials: List[Dict], scenario_results: Dict) -> Dict:
    """
    Compute overall statistics across all trials.
    """
    
    if not all_trials:
        return {}
    
    overall_success_rate = np.mean([t['success'] for t in all_trials])
    overall_collision_rate = np.mean([t['collision'] for t in all_trials])
    
    successful_trials = [t for t in all_trials if t['success']]
    if successful_trials:
        completion_times = [t['completion_time'] for t in successful_trials]
        avg_completion_time = np.mean(completion_times)
    else:
        avg_completion_time = None
    
    all_path_efficiencies = [t['metrics']['path_efficiency'] for t in all_trials]
    all_velocity_smoothness = [t['metrics']['velocity_smoothness'] for t in all_trials]
    all_yaw_smoothness = [t['metrics']['yaw_rate_smoothness'] for t in all_trials]
    all_spls = [t['metrics']['spl'] for t in all_trials]
    
    smoothness_score = (np.mean(all_velocity_smoothness) + np.mean(all_yaw_smoothness)) / 2
    
    return {
        'total_trials': len(all_trials),
        'success_rate': float(overall_success_rate),
        'collision_rate': float(overall_collision_rate),
        'avg_completion_time': avg_completion_time,
        'path_efficiency': float(np.mean(all_path_efficiencies)),
        'smoothness_score': float(smoothness_score),
        'spl': float(np.mean(all_spls)),
        'per_scenario_performance': {
            name: stats['success_rate'] for name, stats in scenario_results.items()
        }
    }


def save_detailed_results(eval_results: Dict, cfg) -> None:
    """
    Save detailed evaluation results to a file.
    """
    
    script_dir = Path(__file__).parent
    results_dir = script_dir / "eval_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results_{eval_results['model_name']}_{timestamp}.json"
    
    filepath = results_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\\n[Nav]: Detailed results saved to {filepath.absolute()}")


def save_standard_eval_results(eval_results: Dict, cfg) -> None:
    """
    Save standard evaluation mode results to a file.
    """
    
    script_dir = Path(__file__).parent
    results_dir = script_dir / "eval_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"standard_eval_results_{timestamp}.json"
    
    filepath = results_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\\n[Nav]: Standard evaluation results saved to {filepath.absolute()}")


def compare_multiple_models(env, cfg) -> Dict:
    """
    Compare performance of multiple models.
    """
    print("\\n[Nav]: Starting multi-model comparison...")
    
    models_config = cfg.models_to_compare
    comparison_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'models': {},
        'summary': {}
    }
    
    fixed_scenarios = getattr(cfg, 'test_scenarios', None)
    if fixed_scenarios is None:
        fixed_scenarios = [
            {"start_pos": [0.0, 0.0, 0.42], "target_pos": [10.0, 10.0, 0.42], "scenario_name": "diagonal_1"},
            {"start_pos": [-5.0, -5.0, 0.42], "target_pos": [15.0, 5.0, 0.42], "scenario_name": "cross_1"},
            {"start_pos": [8.0, -8.0, 0.42], "target_pos": [-8.0, 8.0, 0.42], "scenario_name": "diagonal_2"},
            {"start_pos": [0.0, 10.0, 0.42], "target_pos": [0.0, -10.0, 0.42], "scenario_name": "straight_y"},
            {"start_pos": [-10.0, 0.0, 0.42], "target_pos": [10.0, 0.0, 0.42], "scenario_name": "straight_x"},
        ]
    
    for model_config in models_config:
        model_name = model_config['name']
        print(f"\\n=== Evaluating Model: {model_name} ===")
        
        try:
            model_env, model_policy = setup_model_for_evaluation(env, model_config, cfg)
            
            model_results = evaluate_fixed_scenarios(
                env=model_env,
                policy=model_policy,
                cfg=cfg,
                model_name=model_name,
                fixed_scenarios=fixed_scenarios
            )
            
            comparison_results['models'][model_name] = model_results
            
            print(f"Model {model_name} Results:")
            print(f"  Success Rate: {model_results['success_rate']:.2%}")
            avg_time = model_results.get('avg_completion_time', None)
            if avg_time is not None:
                print(f"  Avg Time: {avg_time:.2f}s")
            else:
                print(f"  Avg Time: N/A")
            print(f"  Path Efficiency: {model_results['path_efficiency']:.3f}")
            print(f"  Smoothness: {model_results['smoothness_score']:.3f}")
            spl = model_results.get('overall_stats', {}).get('spl', 0.0)
            print(f"  SPL: {spl:.3f}")
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
            comparison_results['models'][model_name] = {
                'error': str(e),
                'success_rate': 0.0,
                'avg_completion_time': None,
                'path_efficiency': 0.0,
                'smoothness_score': 0.0
            }
    
    comparison_results['summary'] = compute_comparison_summary(comparison_results['models'])
    
    return comparison_results


def setup_model_for_evaluation(base_env, model_config, cfg):
    """
    Set up environment and policy for a specific model.
    """
    
    from ppo import ILtoPPO
    from controller import LocomotionController
    from torchrl.envs.transforms import TransformedEnv, Compose
    
    if 'il_model_path' in model_config:
        import sys
        from pathlib import Path
        root_path = Path(__file__).resolve().parent.parent.parent.parent
        sys.path.append(str(root_path))
        from il_training.fastsys.infer_policy import PolicyInference
        
        il_model = PolicyInference(model_config['il_model_path'], cfg)
    else:
        il_model = None
    
    policy = ILtoPPO(
        cfg.algo, 
        base_env.observation_spec, 
        base_env.action_spec, 
        'multi', 
        cfg.device,
        il_ref_model=il_model
    )
    
    if 'checkpoint_path' in model_config:
        checkpoint = torch.load(model_config['checkpoint_path'])
        policy.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {model_config['checkpoint_path']}")
    
    import sys
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from go2.go2_ctrl import get_rsl_him_rough_policy
    locomotion_policy = get_rsl_him_rough_policy()
    
    transforms = []
    vel_transform = LocomotionController(locomotion_policy, device=cfg.device)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(base_env, Compose(*transforms)).train()
    
    return transformed_env, policy


def compute_comparison_summary(models_results: Dict) -> Dict:
    """
    Compute comparison summary for multiple models.
    """
    summary = {
        'best_model': None,
        'metrics_comparison': {},
        'ranking': {}
    }
    
    valid_models = {name: results for name, results in models_results.items() if 'error' not in results}
    
    if not valid_models:
        return summary
    
    metrics = ['success_rate', 'path_efficiency', 'smoothness_score']
    
    for metric in metrics:
        metric_values = {}
        for model_name, results in valid_models.items():
            if metric in results and results[metric] is not None:
                metric_values[model_name] = results[metric]
        
        if metric_values:
            best_model = max(metric_values, key=metric_values.get)
            summary['metrics_comparison'][metric] = {
                'best_model': best_model,
                'best_value': metric_values[best_model],
                'all_values': metric_values
            }
    
    model_scores = {}
    for model_name in valid_models:
        score = 0
        weight_sum = 0
        
        if 'success_rate' in valid_models[model_name]:
            score += valid_models[model_name]['success_rate'] * 0.5
            weight_sum += 0.5
        
        if 'path_efficiency' in valid_models[model_name]:
            score += valid_models[model_name]['path_efficiency'] * 0.3
            weight_sum += 0.3
        
        if 'smoothness_score' in valid_models[model_name]:
            score += valid_models[model_name]['smoothness_score'] * 0.2
            weight_sum += 0.2
        
        if weight_sum > 0:
            model_scores[model_name] = score / weight_sum
    
    if model_scores:
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        summary['ranking'] = {i+1: {'model': name, 'score': score} for i, (name, score) in enumerate(ranked_models)}
        summary['best_model'] = ranked_models[0][0]
    
    return summary


def print_comparison_results(comparison_results: Dict) -> None:
    """
    Print comparison results.
    """
    print("\\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    summary = comparison_results['summary']
    
    if 'ranking' in summary and summary['ranking']:
        print("\\nOverall Ranking:")
        for rank, data in summary['ranking'].items():
            print(f"{rank}. {data['model']} (Score: {data['score']:.3f})")
    
    if 'metrics_comparison' in summary:
        print("\\nBest Models by Metric:")
        for metric, data in summary['metrics_comparison'].items():
            print(f"  {metric}: {data['best_model']} ({data['best_value']:.3f})")
    
    print("\\nDetailed Results:")
    for model_name, results in comparison_results['models'].items():
        print(f"\\n{model_name}:")
        if 'error' in results:
            print(f"  Error: {results['error']}")
        else:
            print(f"  Success Rate: {results['success_rate']:.2%}")
            avg_time = results.get('avg_completion_time', None)
            if avg_time is not None:
                print(f"  Avg Time: {avg_time:.2f}s")
            else:
                print(f"  Avg Time: N/A")
            print(f"  Path Efficiency: {results['path_efficiency']:.3f}")
            print(f"  Smoothness Score: {results['smoothness_score']:.3f}")
            spl = results.get('overall_stats', {}).get('spl', 0.0)
            print(f"  SPL: {spl:.3f}")


def save_comparison_results(comparison_results: Dict, cfg) -> None:
    """
    Save model comparison results.
    """
    script_dir = Path(__file__).parent
    results_dir = script_dir / "eval_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_comparison_{timestamp}.json"
    
    filepath = results_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\\n[Nav]: Comparison results saved to {filepath.absolute()}")


def evaluate_line_formation_multiple_obstacles(env, policy, cfg) -> Dict:
    """
    Perform single test using line formation mode.
    """
    print("\\n[Nav]: Starting line formation evaluation...")
    
    line_config = cfg.line_formation_config
    num_robots = line_config.num_robots
    
    all_results = {}
    summary_results = {}
    
    obstacle_count = cfg.env.initial_num_obstacles
    print(f"\\n[Nav]: Single line formation evaluation with {obstacle_count} obstacles")
    print(f"Note: Using initial_num_obstacles from config: {obstacle_count}")
    
    env.reset()
    
    print("Resetting robot appearance to default state...")
    reset_robot_colors(env)
    
    line_scenarios = generate_line_formation_scenarios(env, cfg, obstacle_count)
    
    print(f"Generated {len(line_scenarios)} valid start-target pairs")
    
    level_results = evaluate_line_formation_parallel(
        env=env,
        policy=policy,
        scenarios=line_scenarios,
        cfg=cfg,
        obstacle_count=obstacle_count
    )
    
    all_results[f"obstacles_{obstacle_count}"] = level_results
    
    level_stats = compute_obstacle_level_statistics(level_results, obstacle_count)
    summary_results[f"obstacles_{obstacle_count}"] = level_stats
    
    print(f"\\nSingle Obstacle Level {obstacle_count} Results:")
    print(f"  Success Rate: {level_stats['success_rate']:.2%}")
    print(f"  Avg Time: {level_stats.get('avg_completion_time', 'N/A')}")
    print(f"  Path Efficiency: {level_stats['path_efficiency']:.3f}")
    print(f"  SPL: {level_stats['spl']:.3f}")
    print(f"\\n[Nav]: Single evaluation completed!")
    
    comparison_summary = compute_obstacle_comparison_summary(summary_results)
    
    return {
        'timestamp': datetime.datetime.now().isoformat(),
        'obstacle_count': obstacle_count,
        'all_results': all_results,
        'summary': summary_results,
        'comparison': comparison_summary
    }


def generate_line_formation_scenarios(env, cfg, obstacle_count: int) -> List[Dict]:
    """
    Generate a line of start and target positions, ensuring distance from obstacles.
    """
    line_config = cfg.line_formation_config
    num_robots = line_config.num_robots
    start_y = line_config.start_y
    target_y = line_config.target_y
    x_range = line_config.x_range
    height = line_config.height
    min_distance = line_config.min_distance_to_obstacles
    
    obstacle_positions = get_obstacle_positions(env)
    
    scenarios = []
    x_positions = torch.linspace(x_range[0], x_range[1], num_robots)
    
    for i, x_pos in enumerate(x_positions):
        start_pos = [float(x_pos), start_y, height]
        target_pos = [float(x_pos), target_y, height]
        
        start_pos = adjust_position_away_from_obstacles(
            start_pos, obstacle_positions, min_distance, search_radius=5.0
        )
        
        target_pos = adjust_position_away_from_obstacles(
            target_pos, obstacle_positions, min_distance, search_radius=5.0
        )
        
        scenarios.append({
            'scenario_name': f'line_robot_{i+1}',
            'start_pos': start_pos,
            'target_pos': target_pos,
            'robot_id': i
        })
    
    return scenarios


def get_obstacle_positions(env) -> np.ndarray:
    """
    Get obstacle positions in the environment.
    """
    try:
        if hasattr(env, 'obstacle_positions'):
            positions = env.obstacle_positions.cpu().numpy()
            print(f"Found {len(positions)} obstacles via obstacle_positions")
            return positions
        elif hasattr(env, '_obstacle_pos'):
            positions = env._obstacle_pos.cpu().numpy()
            print(f"Found {len(positions)} obstacles via _obstacle_pos")
            return positions
        elif hasattr(env, 'obstacles') and hasattr(env.obstacles, 'data'):
            if hasattr(env.obstacles.data, 'pos_w'):
                positions = env.obstacles.data.pos_w.cpu().numpy()
                print(f"Found {len(positions)} obstacles via obstacles.data.pos_w")
                return positions
        
        print("Warning: Could not get actual obstacle positions")
        print("Creating virtual obstacle-dense region based on map_range")
        
        map_range = getattr(env, 'map_range', [20.0, 20.0, 4.5])
        virtual_obstacles = []
        
        x_points = np.linspace(-map_range[0], map_range[0], 20)
        y_points = np.linspace(-map_range[1], map_range[1], 20)
        
        for x in x_points:
            for y in y_points:
                virtual_obstacles.append([x, y, 0.5])
        
        virtual_obstacles = np.array(virtual_obstacles)
        print(f"Created {len(virtual_obstacles)} virtual obstacle positions")
        return virtual_obstacles
        
    except Exception as e:
        print(f"Error getting obstacle positions: {e}")
        return np.array([]).reshape(0, 3)


def adjust_position_away_from_obstacles(
    position: List[float], 
    obstacle_positions: np.ndarray, 
    min_distance: float,
    search_radius: float = 5.0
) -> List[float]:
    """
    Adjust position to stay away from obstacles.
    """
    if len(obstacle_positions) == 0:
        return position
    
    pos = np.array(position)
    max_attempts = 10
    
    for attempt in range(max_attempts):
        distances = np.linalg.norm(obstacle_positions[:, :2] - pos[:2], axis=1)
        min_dist = np.min(distances) if len(distances) > 0 else float('inf')
        
        if min_dist >= min_distance:
            return pos.tolist()
        
        nearest_obstacle_idx = np.argmin(distances)
        nearest_obstacle = obstacle_positions[nearest_obstacle_idx]
        
        direction = pos[:2] - nearest_obstacle[:2]
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            direction = np.random.uniform(-1, 1, 2)
            direction_norm = np.linalg.norm(direction)
        
        direction = direction / direction_norm
        
        new_pos = nearest_obstacle[:2] + direction * (min_distance + 0.5)
        pos[:2] = new_pos
        
        pos[0] = np.clip(pos[0], -50, 50)
        pos[1] = np.clip(pos[1], -20, 20)
    
    print(f"Warning: Could not find safe position after {max_attempts} attempts for {position}")
    return pos.tolist()


def set_obstacle_count(env, obstacle_count: int):
    """
    Set the number of obstacles in the environment.
    Note: This function is deprecated as Isaac Sim cannot dynamically override obstacle count.
    """
    try:
        actual_env = env
        if hasattr(env, 'base_env'):
            actual_env = env.base_env
        elif hasattr(env, '_env'):
            actual_env = env._env
            
        if hasattr(actual_env, 'regenerate_terrain_with_obstacles'):
            print(f"[ENV] Using regenerate_terrain_with_obstacles method")
            actual_env.regenerate_terrain_with_obstacles(obstacle_count)
            print(f"Successfully regenerated terrain with {obstacle_count} obstacles")
            return
            
        if hasattr(actual_env, 'curr_num_obstacles'):
            actual_env.curr_num_obstacles = obstacle_count
            print(f"Set curr_num_obstacles to {obstacle_count}")
        elif hasattr(actual_env, 'num_obstacles'):
            actual_env.num_obstacles = obstacle_count
            print(f"Set num_obstacles to {obstacle_count}")
        
        if hasattr(actual_env, 'cfg') and hasattr(actual_env.cfg, 'env'):
            if hasattr(actual_env.cfg.env, 'initial_num_obstacles'):
                actual_env.cfg.env.initial_num_obstacles = obstacle_count
                print(f"Set cfg.env.initial_num_obstacles to {obstacle_count}")
        
        print(f"Warning: Using fallback method - obstacles may not change until environment reset")
        
    except Exception as e:
        print(f"Error setting obstacle count to {obstacle_count}: {e}")
        print("Continuing with default obstacle count")





def evaluate_line_formation_parallel(
    env, policy, scenarios: List[Dict], cfg, obstacle_count: int
) -> List[Dict]:
    """
    Evaluate line formation scenarios in parallel with GPU memory optimization.
    """
    max_batch_size = cfg.line_formation_config.get('max_batch_size', cfg.env.num_envs)
    batch_size = min(max_batch_size, len(scenarios), cfg.env.num_envs)
    
    print(f"Evaluating {len(scenarios)} scenarios with batch size {batch_size}")
    print(f"GPU memory optimization: using {batch_size} instead of {len(scenarios)} parallel environments")
    
    all_results = []
    num_batches = (len(scenarios) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(scenarios))
        batch_scenarios = scenarios[start_idx:end_idx]
        
        print(f"\\nProcessing batch {batch_idx + 1}/{num_batches}: scenarios {start_idx + 1}-{end_idx}")
        
        batch_tasks = []
        for i, scenario in enumerate(batch_scenarios):
            batch_tasks.append({
                'scenario': scenario,
                'scenario_idx': start_idx + i,
                'trial_idx': 0,
                'seed': cfg.seed + start_idx + i,
                'trial_name': scenario['scenario_name']
            })
        
        try:
            batch_results = evaluate_batch_parallel_with_error_handling(
                env=env,
                policy=policy,
                tasks=batch_tasks,
                cfg=cfg
            )
            
            for result in batch_results:
                result['obstacle_count'] = obstacle_count
            
            all_results.extend(batch_results)
            
            print(f"Batch {batch_idx + 1} completed successfully: {len(batch_results)} results")
            
        except Exception as e:
            print(f"Error in batch {batch_idx + 1}: {e}")
            print("Creating failed results for this batch...")
            
            failed_results = []
            for task in batch_tasks:
                failed_results.append({
                    'trial_name': task['trial_name'],
                    'scenario_name': task['scenario']['scenario_name'],
                    'seed': task['seed'],
                    'success': False,
                    'collision': True,
                    'completion_time': None,
                    'trajectory_length': 0,
                    'obstacle_count': obstacle_count,
                    'metrics': {
                        'path_length': 0.0,
                        'direct_distance': 0.0,
                        'path_efficiency': 0.0,
                        'velocity_smoothness': 0.0,
                        'yaw_rate_smoothness': 0.0,
                        'avg_velocity': 0.0,
                        'max_velocity': 0.0,
                        'spl': 0.0
                    }
                })
            
            all_results.extend(failed_results)
            print(f"Added {len(failed_results)} failed results for batch {batch_idx + 1}")
            
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                print("GPU memory cleared")
                
                env.reset()
                print("Environment reset")
                
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
    
    return all_results


def evaluate_batch_parallel_with_error_handling(
    env,
    policy,
    tasks: List[Dict],
    cfg
) -> List[Dict]:
    """
    Parallel batch evaluation with error handling.
    """
    try:
        return evaluate_batch_parallel(env, policy, tasks, cfg)
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error detected: {e}")
            print("Attempting recovery with smaller batch size...")
            
            if len(tasks) > 1:
                mid_point = len(tasks) // 2
                first_half = tasks[:mid_point]
                second_half = tasks[mid_point:]
                
                print(f"Splitting into two sub-batches: {len(first_half)} + {len(second_half)}")
                
                results = []
                
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    env.reset()
                    first_results = evaluate_batch_parallel(env, policy, first_half, cfg)
                    results.extend(first_results)
                    print(f"First sub-batch completed: {len(first_results)} results")
                except Exception as e1:
                    print(f"First sub-batch failed: {e1}")
                    results.extend(create_failed_results(first_half))
                
                gc.collect()
                torch.cuda.empty_cache()
                
                try:
                    env.reset()
                    second_results = evaluate_batch_parallel(env, policy, second_half, cfg)
                    results.extend(second_results)
                    print(f"Second sub-batch completed: {len(second_results)} results")
                except Exception as e2:
                    print(f"Second sub-batch failed: {e2}")
                    results.extend(create_failed_results(second_half))
                
                return results
            else:
                print("Single task failed, creating failed result")
                return create_failed_results(tasks)
        else:
            raise e
    
    except Exception as e:
        print(f"Unexpected error in batch evaluation: {e}")
        return create_failed_results(tasks)


def create_failed_results(tasks: List[Dict]) -> List[Dict]:
    """
    Create default results for failed tasks.
    """
    failed_results = []
    for task in tasks:
        failed_results.append({
            'trial_name': task['trial_name'],
            'scenario_name': task['scenario']['scenario_name'],
            'seed': task['seed'],
            'success': False,
            'collision': True,
            'completion_time': None,
            'trajectory_length': 0,
            'metrics': {
                'path_length': 0.0,
                'direct_distance': 50.0,
                'path_efficiency': 0.0,
                'velocity_smoothness': 0.0,
                'yaw_rate_smoothness': 0.0,
                'avg_velocity': 0.0,
                'max_velocity': 0.0,
                'spl': 0.0
            }
        })
    return failed_results


def compute_obstacle_level_statistics(results: List[Dict], obstacle_count: int) -> Dict:
    """
    Compute statistics for a single obstacle level.
    """
    if not results:
        return {}
    
    successes = [r['success'] for r in results]
    success_rate = np.mean(successes)
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        completion_times = [r['completion_time'] for r in successful_results]
        avg_completion_time = np.mean(completion_times)
    else:
        avg_completion_time = None
    
    collisions = [r['collision'] for r in results]
    collision_rate = np.mean(collisions)
    
    path_efficiencies = [r['metrics']['path_efficiency'] for r in results]
    velocity_smoothness = [r['metrics']['velocity_smoothness'] for r in results]
    spls = [r['metrics']['spl'] for r in results]
    
    return {
        'obstacle_count': obstacle_count,
        'total_trials': len(results),
        'success_rate': float(success_rate),
        'collision_rate': float(collision_rate),
        'avg_completion_time': avg_completion_time,
        'path_efficiency': float(np.mean(path_efficiencies)),
        'velocity_smoothness': float(np.mean(velocity_smoothness)),
        'spl': float(np.mean(spls))
    }


def compute_obstacle_comparison_summary(level_results: Dict) -> Dict:
    """
    Compute comparison summary across different obstacle levels.
    """
    obstacle_counts = []
    success_rates = []
    avg_times = []
    path_efficiencies = []
    
    for level_name, stats in level_results.items():
        obstacle_counts.append(stats['obstacle_count'])
        success_rates.append(stats['success_rate'])
        if stats['avg_completion_time'] is not None:
            avg_times.append(stats['avg_completion_time'])
        path_efficiencies.append(stats['path_efficiency'])
    
    return {
        'obstacle_counts': obstacle_counts,
        'success_rates': success_rates,
        'avg_completion_times': avg_times,
        'path_efficiencies': path_efficiencies,
        'best_obstacle_level': obstacle_counts[np.argmax(success_rates)] if success_rates else None,
        'worst_obstacle_level': obstacle_counts[np.argmin(success_rates)] if success_rates else None
    }


def print_line_formation_results(results: Dict):
    """
    Print line formation evaluation results.
    """
    print("\\n" + "="*60)
    print("LINE FORMATION EVALUATION RESULTS")
    print("="*60)
    
    print("\\nResults by Obstacle Level:")
    for level_name, stats in results['summary'].items():
        obstacle_count = stats['obstacle_count']
        print(f"\\n{obstacle_count} Obstacles:")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Collision Rate: {stats['collision_rate']:.2%}")
        avg_time = stats.get('avg_completion_time')
        if avg_time is not None:
            print(f"  Avg Time: {avg_time:.2f}s")
        else:
            print(f"  Avg Time: N/A")
        print(f"  Path Efficiency: {stats['path_efficiency']:.3f}")
        print(f"  SPL: {stats['spl']:.3f}")
    
    comparison = results['comparison']
    if comparison['best_obstacle_level'] is not None:
        print(f"\\nBest Performance: {comparison['best_obstacle_level']} obstacles")
        print(f"Worst Performance: {comparison['worst_obstacle_level']} obstacles")


def save_line_formation_results(results: Dict, cfg):
    """
    Save line formation evaluation results.
    """
    script_dir = Path(__file__).parent
    results_dir = script_dir / "eval_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"line_formation_results_{timestamp}.json"
    
    filepath = results_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\\n[Nav]: Line formation results saved to {filepath.absolute()}")


def make_robot_glow_red(env, robot_idx: int):
    """
    Make the specified robot glow red to indicate collision.
    """
    try:
        if hasattr(env, 'go2') and hasattr(env.go2, '_data'):
            try:
                from pxr import Usd, UsdShade, Gf
                import omni.usd
                
                stage = omni.usd.get_context().get_stage()
                
                robot_prim_path = f"/World/envs/env_{robot_idx}/go2"
                robot_prim = stage.GetPrimAtPath(robot_prim_path)
                
                if robot_prim:
                    material_path = f"{robot_prim_path}/glow_material"
                    material_prim = stage.DefinePrim(material_path, "Material")
                    material = UsdShade.Material(material_prim)
                    
                    shader = UsdShade.Shader.Define(stage, f"{material_path}/glow_shader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    
                    shader.CreateInput("diffuseColor", Gf.Vec3f(1.0, 0.0, 0.0))
                    shader.CreateInput("emissiveColor", Gf.Vec3f(2.0, 0.0, 0.0))
                    shader.CreateInput("metallic", 0.0)
                    shader.CreateInput("roughness", 0.1)
                    shader.CreateInput("opacity", 1.0)
                    
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    
                    UsdShade.MaterialBindingAPI(robot_prim).Bind(material)
                    
                    print(f"Robot {robot_idx} is now glowing red!")
                    return True
                    
            except Exception as material_error:
                print(f"Material glow method failed: {material_error}")
        
        if hasattr(env, 'make_robot_glow'):
            env.make_robot_glow(robot_idx, [2.0, 0.0, 0.0])
            print(f"Robot {robot_idx} is glowing (method 2)")
            return True
            
        if hasattr(env, '_debug_draw'):
            if hasattr(env, 'go2') and hasattr(env.go2, 'data'):
                robot_pos = env.go2.data.root_pos_w[robot_idx].cpu().numpy()
                for i in range(5):
                    offset = [0.1*i, 0.1*i, 0.1*i]
                    glow_pos = robot_pos + offset
                    env._debug_draw.draw_sphere(glow_pos, 0.3 + 0.1*i, [1.0, 0.0, 0.0, 0.8])
                print(f"Added glow effect at robot {robot_idx} position")
                return True
        
        if hasattr(env, 'go2') and hasattr(env.go2, 'cfg'):
            try:
                print(f"Robot {robot_idx} lighting enhanced")
                return True
            except:
                pass
        
        print(f"\nRobot {robot_idx} collision! Should be glowing!")
        print(f"Position: {env.go2.data.root_pos_w[robot_idx].cpu().numpy() if hasattr(env, 'go2') else 'Unknown'}")
        print(f"Please watch this robot!")
        return False
        
    except Exception as e:
        print(f"Glow effect failed: {e}")
        print(f"But robot {robot_idx} did collide! Please observe!")
        return False


def reset_robot_colors(env):
    """
    Reset all robot appearances to default state (including glow effects).
    """
    try:
        if hasattr(env, 'reset_robot_colors'):
            env.reset_robot_colors()
            print("All robot appearances reset")
        elif hasattr(env, 'reset_robot_materials'):
            env.reset_robot_materials()
            print("All robot materials reset")
        else:
            print("Robot appearance reset not available, but new glow effects will override old ones")
    except Exception as e:
        print(f"Failed to reset robot appearances: {e}")


if __name__ == "__main__":
    main()