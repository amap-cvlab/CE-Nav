import torch
import time
import numpy as np
import yaml
from model import PolicyNetwork
from data import vectorized_raycast

class PolicyInference:
    def __init__(self, model_path, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNetwork(cfg).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, state, target, obstacle):
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
        obstacle = torch.tensor(obstacle, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        ray_cast, raycast_occ = vectorized_raycast(obstacle.squeeze(0).squeeze(0), state.squeeze(0)[0:2], target.squeeze(0))
        
        
        # Predict normalized action
        action_norm = self.model(state, target, obstacle, ray_cast.unsqueeze(0), raycast_occ.unsqueeze(0).unsqueeze(0))
        
        # Denormalize
        action = self.model.denormalize(action_norm)

        return action.squeeze(0).cpu().numpy()

if __name__ == "__main__":



    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    checkpoint = "checkpoints/<run_id>/best_model.pt"  # set to your trained ResNet policy checkpoint

    infer = PolicyInference(checkpoint, cfg)

    x= np.load("data/fastsys/test_data/scene_1_obj_5/scene_1_1.npz")  # set to a local .npz sample
    state = x['state']
    target = x['target']
    obstacle = x['obstacle']

    for _ in range(10):
        action = infer.predict(state, target, obstacle)

    # 测试推理速度
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        action = infer.predict(state, target, obstacle, ray_cast)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs

    print(f"网络预测动作: {action}")
    print(f"gt: {x['action']}")
    print(f"平均单次推理耗时: {avg_time*1000:.2f} ms")

    
