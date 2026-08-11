import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import hydra
import torch
from torch.utils.data import DataLoader
# from dataset import ExpertDataset
from ppo import ILtoPPO
import yaml
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torch.utils.data._utils.collate import default_collate_fn_map
from functools import partial
from data import create_data_loaders, get_state
from tqdm import tqdm
import wandb
import os
import torch.multiprocessing as mp
from eval import DWASimulation
import numpy as np

# def collate_batch(batch):
#     obs_list = [item[0] for item in batch]
#     action_list = [item[1] for item in batch]

#     obs_batch = {}
#     for key in obs_list[0]:
#         obs_batch[key] = torch.stack([obs[key] for obs in obs_list], dim=0)

#     action_batch = torch.stack(action_list, dim=0)
#     return obs_batch, action_batch

class Trainer:
    def __init__(self, cfg, cfg_navrl, train_loader, test_loader):
        self.cfg = cfg
        self.cfg_navrl = cfg_navrl
        self.cfg_navrl.algo.feature_extractor.learning_rate = self.cfg['training']['lr']  # 设置学习率
        self.cfg_navrl.algo.actor.learning_rate = self.cfg['training']['lr']
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('self.device: {}'.format(self.device))
        
        # 构建 observation_spec（必须与实际数据一致）
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    # state: [1, 11]
                    "state": UnboundedContinuousTensorSpec(shape=(1, 11), device=self.device),
                    # lidar: [1, 1, 144, 1]
                    "lidar": UnboundedContinuousTensorSpec(shape=(1, 1, 144, 1), device=self.device),
                }, shape=(1,), device=self.device)
            }, shape=(1,), device=self.device)
        }, shape=(1,), device=self.device)  # 可根据需要设为 (n_envs,) 或 ()


        # 动作空间：[1, 3] -> vx, vy, vyaw
        action_spec = CompositeSpec({
                "agents": CompositeSpec({
                    "action": UnboundedContinuousTensorSpec(shape=(1, 3), device=self.device)
            }, shape=(1,), device=self.device)
        }, shape=(1,), device=self.device)

        # 初始化模型
        self.model = ILtoPPO(self.cfg_navrl.algo, observation_spec, action_spec, 'single', device=self.device)
        # state_dict = torch.load(pretrained_path, map_location=self.device)
        # model.load_state_dict(state_dict, strict=False)  # 可设为 False 避免不匹配层报错
        # model.train()
        
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['training']['lr'])
        self.criterion = torch.nn.MSELoss()
        # self.best_test_loss = float('inf')
        self.best_succ_num = 0
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.75)
        self.simulator_easy = DWASimulation('easy')
        self.simulator_hard = DWASimulation('hard')
        
        if cfg['wandb']['use_wandb']:
            wandb.init(project=cfg['wandb']['project'],
                        entity=cfg['wandb']['entity'],
                        mode=cfg['wandb']['mode'])
            wandb.config.update(cfg)
            os.makedirs(os.path.join('checkpoints',wandb.run.id), exist_ok=True)
        
        print('train init finish')
        
    
        
    def train(self):
        total_steps = 0
        for epoch in range(self.cfg['training']['epochs']):
            epoch_loss = 0.0
             
            # Save checkpoint到wandb files
            if (epoch) % self.cfg['training']['save_interval'] == 0:
                ckpt_path = f"model_epoch_{epoch}.pt"
                ckpt_abs_path = os.path.join('checkpoints', wandb.run.id, ckpt_path)
                torch.save(self.model.state_dict(), ckpt_abs_path)
                # if self.cfg['wandb']['use_wandb']:
                #     wandb.save(os.path.join('checkpoints',wandb.run.id,ckpt_path))
                cur_easy_eval_path = os.path.join('./eval_res/', wandb.run.id, 'epoch_' + str(epoch), 'easy')
                cur_hard_eval_path = os.path.join('./eval_res/', wandb.run.id, 'epoch_' + str(epoch), 'hard')
                os.makedirs(cur_easy_eval_path, exist_ok=True)
                os.makedirs(cur_hard_eval_path, exist_ok=True)
                # self.simulator.run_simulation_without_multi_thread(ckpt_abs_path, cur_eval_path, num_scenarios=100)
                cur_success_num_easy = self.simulator_easy.run_simulation(ckpt_abs_path, cur_easy_eval_path, 
                                                                self.cfg,
                                                                self.cfg_navrl,
                                                                num_scenarios=100, max_workers=16, mode='navrl_il', eval_scene='easy')
                cur_success_num_hard = self.simulator_hard.run_simulation(ckpt_abs_path, cur_hard_eval_path, 
                                                                self.cfg,
                                                                self.cfg_navrl,
                                                                num_scenarios=100, max_workers=16, mode='navrl_il', eval_scene='hard')
                print('cur_success_num_easy: {}'.format(cur_success_num_easy))
                print('cur_success_num_hard: {}'.format(cur_success_num_hard))
                if self.cfg['wandb']['use_wandb']:
                    wandb.log({"test/cur_success_num_easy": cur_success_num_easy}, step=total_steps)
                    wandb.log({"test/cur_success_num_hard": cur_success_num_hard}, step=total_steps)
                if cur_success_num_hard >= self.best_succ_num:
                    self.best_succ_num = cur_success_num_hard
                    if self.cfg['wandb']['use_wandb']:
                        # 保存到wandb的files目录
                        best_ckpt_path = "best_model.pt"
                        torch.save(self.model.state_dict(), os.path.join('checkpoints',wandb.run.id,best_ckpt_path))
                        wandb.save(os.path.join('checkpoints',wandb.run.id,best_ckpt_path))
            
            for batch_idx, (state, target, obstacle, action) in enumerate(tqdm(self.train_loader)):
                state = state.to(self.device)
                target = target.to(self.device)
                obstacle = obstacle.to(self.device)
                action = action.to(self.device)
                state_navrl, ray_cast_navrl, action_navrl = get_state(state, target, obstacle, action, self.device)
                              
                input = {
                    "state": state_navrl,
                    "lidar": ray_cast_navrl,
                    "expert_action": action_navrl
                }
                total_steps += 1
                
                loss_dict = self.model.train_il_step(input)
                epoch_loss += loss_dict["nll_loss"]
                
                # Log training
                if total_steps % self.cfg['training']['log_interval'] == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"Epoch: {epoch+1}, Step: {total_steps}, Loss: {avg_loss:.6f}")
                    if self.cfg['wandb']['use_wandb']:
                        wandb.log({"train/loss": avg_loss, "lr": loss_dict['lr']}, step=total_steps)
                        # wandb.log({"train/loss": avg_loss}, step=total_steps)
                
                # Test evaluation
                if total_steps % self.cfg['training']['test_interval'] == 0:
                    test_loss = self.evaluate()
                    print(f"Test Loss @ Step {total_steps}: {test_loss:.6f}")
                    if self.cfg['wandb']['use_wandb']:
                        wandb.log({"test/loss": test_loss}, step=total_steps)
                        # wandb.log({"test/loss": test_loss, "lr": self.optimizer.param_groups[0]['lr']}, step=total_steps)
            
                
                
                
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.cfg['training']['epochs']}], Loss: {avg_loss:.4f}")
            ckpt_path = f"model_epoch_{epoch}.pt"
            ckpt_abs_path = os.path.join('checkpoints', wandb.run.id, ckpt_path)
            torch.save(self.model.state_dict(), ckpt_abs_path)
                
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for state, target, obstacle, action in self.test_loader:
            state = state.to(self.device)
            target = target.to(self.device)
            obstacle = obstacle.to(self.device)
            action = action.to(self.device)
            state_navrl, ray_cast_navrl, action_navrl = get_state(state, target, obstacle, action, self.device)
            input = {
                "state": state_navrl,
                "lidar": ray_cast_navrl,
                "expert_action": action_navrl
            }
            
            loss_dict = self.model.train_il_step(input, only_eval=True)
            total_loss += loss_dict["nll_loss"]
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches

@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'isaac-training/training/cfg'), config_name="ppo", version_base=None)


def main(cfg_navrl):
    mp.set_start_method('spawn', force=True)
    
    # Load config
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_navrl_il.yaml')) as f:
        cfg = yaml.safe_load(f)
    
    # Prepare data
    train_loader, test_loader = create_data_loaders(cfg, output_norm=False)
    
    # Create trainer
    trainer = Trainer(cfg, cfg_navrl, train_loader, test_loader)
    trainer.train()
    


if __name__ == "__main__":
    main()
