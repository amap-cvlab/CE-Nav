import torch
import time
import numpy as np
import yaml
from tensordict.tensordict import TensorDict

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
from il_training.ppo import construct_policy
from il_training.fastsys.data import get_state
from collections import OrderedDict
import re

class PolicyInference:
    def __init__(self, model_path, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = construct_policy(cfg, self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()

        for old_key, value in state_dict.items():
            pattern = r'feature_extractor_cnn\.(\d+)\.'
            replacement = r'feature_extractor_cnn.module.0.module.\1.'
            new_key = re.sub(pattern, replacement, old_key)
            new_state_dict[new_key] = value

        self.model.load_state_dict(new_state_dict)
        print("Policy loaded successfully")
        self.model.eval()

    def predict(self, state, target, obstacle, batch_mode=False):
        if not batch_mode:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
            obstacle = torch.tensor(obstacle, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            target = torch.tensor(target, dtype=torch.float32).to(self.device)
            obstacle = torch.tensor(obstacle, dtype=torch.float32).unsqueeze(1).to(self.device)
            
        state_processed, ray_cast, _ = get_state(state, target, obstacle, None, self.device)
        
        input_batch = {
            "state": state_processed,
            "lidar": ray_cast,
        }
        
        action_3d = self.model.sample_action(input_batch)
        output = action_3d
        if not batch_mode:
            output = output.squeeze(0)
            
        return output.cpu().numpy()

    def sample_raw_action(self, batch: dict) -> torch.Tensor:
        return self.model.sample_raw_action(batch)
    
    def sample_action(self, batch: dict) -> torch.Tensor:
        return self.model.sample_action(batch)

