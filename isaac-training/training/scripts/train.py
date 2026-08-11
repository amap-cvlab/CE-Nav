import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from isaacsim import SimulationApp
from ppo import ILtoPPO
from omni_drones.controllers import LeePositionController
# from omni_drones.utils.torchrl.transforms import VelController, CustomVelYawRateController, LocomotionController, ravel_composite
from controller import LocomotionController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType
import hydra.compose, hydra.initialize

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
from il_training.ppo import ILtoPPO as IL_PPO
from il_training.fastsys.infer_policy import PolicyInference

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({
        "headless": cfg.headless,
        "anti_aliasing": 1,
    })

    # Use Wandb to monitor training
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

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from go2.go2_ctrl import get_rsl_him_rough_policy
    locomotion_policy = get_rsl_him_rough_policy()
    print("locomotion policy loaded!!!")
    # Transformed Environment
    transforms = []
    vel_transform = LocomotionController(locomotion_policy, device=cfg.device)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    
    # IL expert reference model, used to compute il_loss during RL training.
    # The expert uses a fixed architecture independent of the environment spec,
    # so that it always matches the shipped checkpoint.
    expert_checkpoint_path = str(root_path / "il_training" / "fastsys" / "checkpoints" / "dynfji91" / "best_model.pt")
    il_model = PolicyInference(expert_checkpoint_path, cfg)

    # PPO Policy
    policy = ILtoPPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, 'multi', cfg.device, il_ref_model=il_model)

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    total_frames = cfg.max_frame_num
    frames_per_batch = cfg.env.num_envs * cfg.algo.training_frame_num
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=frames_per_batch, 
        total_frames=total_frames,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )
    num_batches = total_frames // frames_per_batch
    # Training Loop
    for i, data in enumerate(collector):
        train_process = i * 1.0 / num_batches
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        print('i: {}'.format(i))

        # Train Policy
        train_loss_stats = policy.train_rl(data)
        info.update(train_loss_stats)
        
        # Calculate and log training episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info
        if i > 0 and i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)

        # Update wand info
        run.log(info)

        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: {} at {}".format(i, ckpt_path))

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
