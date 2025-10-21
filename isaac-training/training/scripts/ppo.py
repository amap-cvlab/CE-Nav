import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world
import torch.distributions as D
from copy import deepcopy

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
# from il_training.ppo import ILtoPPO as IL_PPO
from il_training.fastsys.infer_policy import PolicyInference


class SquashedNormal(D.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        base_dist = D.Normal(loc, scale)
        transforms = [D.transforms.TanhTransform()]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        return torch.tanh(self.base_dist.mean)

    @propertyi
    def mode(self):
        return torch.tanh(self.base_dist.mean)

    def log_prob(self, value):
        value = torch.clamp(value, -0.999, 0.999)
        return super().log_prob(value).sum(-1)

    def entropy(self):
        return self.base_dist.entropy()


class GuidedActor(nn.Module):
    def __init__(self, feature_dim, expert_action_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = feature_dim + expert_action_dim
        
        self.actor_net = make_mlp(
            [input_dim, hidden_dim, hidden_dim],
        )
        self.alpha_net = nn.Linear(hidden_dim, action_dim)
        self.beta_net = nn.Linear(hidden_dim, action_dim)

    def forward(self, feature, expert_action):
        x = torch.cat([feature, expert_action], dim=-1)
        
        x = self.actor_net(x)
        alpha = F.softplus(self.alpha_net(x)) + 1.0
        beta = F.softplus(self.beta_net(x)) + 1.0
        return alpha, beta



class ILtoPPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, mode='single', device="cpu", il_ref_model=None, expert_config=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        if mode == 'multi':
            self.n_agents, self.action_dim = action_spec.shape
        else:
            self.n_agents = 1
            self.action_dim = 3

        feature_dim = 256
        expert_action_dim = 3

        guided_actor_module = GuidedActor(feature_dim, expert_action_dim, self.action_dim)
        
        self.actor = ProbabilisticActor(
            module=TensorDictModule(
                guided_actor_module, 
                in_keys=["_feature", "expert_action_normalized"],
                out_keys=["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # self.actor = ProbabilisticActor(
        #     TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
        #     in_keys=["loc", "scale"],
        #     out_keys=[("agents", "action_normalized")],
        #     distribution_class=IndependentNormal,
        #     return_log_prob=True
        # ).to(self.device)

        # actor_module = TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"])
        # self.actor = ProbabilisticActor(
        #     module=actor_module,
        #     in_keys=["loc", "scale"],
        #     out_keys=[("agents", "action_normalized")],
        #     distribution_class=SquashedNormal,
        #     distribution_kwargs={},
        #     return_log_prob=True
        # ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        if expert_config is not None:
            if expert_config.type == "cnfm":
                from omegaconf import OmegaConf
                minimal_cfg = OmegaConf.create({
                    'algo': {
                        'actor': {
                            'blocks': cfg.actor.blocks,
                            'channels': cfg.actor.channels,
                            'rep_dim': cfg.actor.rep_dim,
                            'action_limit': cfg.actor.action_limit,
                            'learning_rate': cfg.actor.learning_rate
                        }
                    }
                })
                self.il_ref_model = PolicyInference(expert_config.cnfm.model_path, minimal_cfg)
                print(f"Expert model loaded: {expert_config.cnfm.model_path}")
            else:
                raise ValueError(f"Currently only support cnfm expert model, got {expert_config.type}")
        elif il_ref_model is not None:
            self.il_ref_model = il_ref_model
            print("Using provided expert model")
        else:
            raise ValueError("GuidedActor requires either expert_config or il_ref_model.")
        self.current_step = 0

        dummy_input = observation_spec.zero()
        dummy_input.set("expert_action_normalized", torch.zeros(dummy_input.shape[0], expert_action_dim, device=self.device))
        dummy_input.set("expert_action_physical", torch.zeros(dummy_input.shape[0], expert_action_dim, device=self.device))


        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)

        with torch.no_grad():
            ref_model_input = {
                "lidar": tensordict.get(("agents", "observation", "lidar")),
                "state": tensordict.get(("agents", "observation", "state"))
            }
            expert_actions_3d_physical = self.il_ref_model.sample_raw_action(ref_model_input)
            
            linear_limit = self.cfg.actor.action_limit.linear_velocity
            angular_limit = self.cfg.actor.action_limit.angular_velocity
            
            expert_actions_normalized_for_input = torch.zeros_like(expert_actions_3d_physical)
            expert_actions_normalized_for_input[..., :2] = (expert_actions_3d_physical[..., :2] + linear_limit) / (2 * linear_limit)
            expert_actions_normalized_for_input[..., 2] = (expert_actions_3d_physical[..., 2] + angular_limit) / (2 * angular_limit)
            
            tensordict.set("expert_action_physical", expert_actions_3d_physical)
            tensordict.set("expert_action_normalized", torch.clamp(expert_actions_normalized_for_input, 0.0, 1.0))

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

    def train_rl(self, tensordict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"]
        dones = tensordict["next", "terminated"]

        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update_rl(minibatch))
        infos = torch.stack(infos).to_tensordict()
        self.current_step += 1
        return infos.apply(torch.mean, batch_size=[]).to_dict()

    def _update_rl(self, tensordict):
        self.feature_extractor(tensordict)
        
        action_dist = self.actor.get_dist(tensordict)
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])

        action_entropy = action_dist.entropy()
        entropy_weights = torch.ones_like(action_entropy)
        entropy_weights[..., 2] = 3.0
        entropy_loss = -self.cfg.entropy_loss_coefficient * (action_entropy * entropy_weights).mean()


        # Actor Loss
        advantage = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"]
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio)
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss

        il_loss = torch.tensor(0.0, device=self.device)
        il_loss_with_coeff = 0.0
        if self.il_ref_model is not None:
            start_decay_step = 1000
            end_decay_step = 5000
            start_coeff = 0.5
            end_coeff = 0.05

            if self.current_step <= start_decay_step:
                self.il_coeff = start_coeff
            elif self.current_step <= end_decay_step:
                total_decay_steps = end_decay_step - start_decay_step
                gamma = (end_coeff / start_coeff) ** (1 / total_decay_steps)
                
                current_decay_step = self.current_step - start_decay_step
                self.il_coeff = start_coeff * (gamma ** current_decay_step)
            else:
                self.il_coeff = 0.05
            # # elif self.current_step < 10000: self.il_coeff = 0.75 ** (self.current_step // 200)
            # elif self.current_step < 3000: self.il_coeff = 0.1
            # else: self.il_coeff = 0.05
            # # self.il_coeff = 0.0
            
            if self.il_coeff > 0:
                student_mean_action_normalized = action_dist.mean
                linear_limit = self.cfg.actor.action_limit.linear_velocity
                angular_limit = self.cfg.actor.action_limit.angular_velocity
                student_mean_action_physical = torch.zeros_like(student_mean_action_normalized)
                student_mean_action_physical[..., :2] = 2 * student_mean_action_normalized[..., :2] * linear_limit - linear_limit
                student_mean_action_physical[..., 2] = 2 * student_mean_action_normalized[..., 2] * angular_limit - angular_limit
                
                expert_actions_3d_physical = tensordict.get("expert_action_physical")
                action_indices_for_il = torch.tensor([0, 2], device=self.device)
                
                student_actions_for_il = student_mean_action_physical.index_select(-1, action_indices_for_il)
                expert_actions_for_il = expert_actions_3d_physical.index_select(-1, action_indices_for_il)
                
                il_loss = F.mse_loss(student_actions_for_il, expert_actions_for_il)
                il_loss_with_coeff = il_loss * self.il_coeff
                loss = loss + il_loss_with_coeff

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()

        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "loss": loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "il_loss": il_loss,
            "il_loss_with_coeff": il_loss_with_coeff,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

    # def freeze_feature_extractor(self):
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = False

    # def unfreeze_feature_extractor(self):
    #     for param in self.feature_extractor.parameters():
    #         param.requires_grad = True

    # def freeze_critic(self):
    #     for param in self.critic.parameters():
    #         param.requires_grad = False

    # def unfreeze_critic(self):
    #     for param in self.critic.parameters():
    #         param.requires_grad = True
