import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from il_training.utils import ValueNorm, make_mlp, GAE, make_batch, IndependentBeta, BetaActor

class MLPPolicyInference:
    def __init__(self, model_path, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self._construct_mlp_model(cfg)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print("MLP policy loaded successfully")
        except Exception as e:
            print(f"Error loading MLP state_dict: {e}")
        
        self.model.eval()

    def _construct_mlp_model(self, cfg):
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec(shape=(1, 11), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec(shape=(1, 1, 144, 1), device=self.device),
                }, shape=(1,), device=self.device)
            }, shape=(1,), device=self.device)
        }, shape=(1,), device=self.device)

        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec(shape=(1, 3), device=self.device)
            }, shape=(1,), device=self.device)
        }, shape=(1,), device=self.device)
        
        model = MLPModel(cfg.algo, observation_spec, action_spec, device=self.device)
        return model

    @torch.no_grad()
    def sample_action(self, obs: TensorDict) -> torch.Tensor:
        input_td = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": obs["state"].to(self.device),
                    "lidar": obs["lidar"].to(self.device),
                }, batch_size=obs["state"].shape[0])
            }, batch_size=obs["state"].shape[0])
        }, batch_size=obs["state"].shape[0], device=self.device)
        
        output = self.model(input_td)
        action = output["agents", "action"]
        return action


class MLPModel(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)

        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        self.action_dim = 3
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)
        self.gae = GAE(0.99, 0.95)

        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        linear_limit = self.cfg.actor.action_limit.linear_velocity
        angular_limit = self.cfg.actor.action_limit.angular_velocity

        actions_normalized = tensordict["agents", "action_normalized"]
        actions = torch.zeros_like(actions_normalized)
        actions[..., :2] = 2 * actions_normalized[..., :2] * linear_limit - linear_limit
        actions[..., 2] = 2 * actions_normalized[..., 2] * angular_limit - angular_limit
        tensordict["agents", "action"] = actions
        return tensordict

