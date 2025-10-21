import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np
import math

from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from utils import ValueNorm, make_mlp, GAE, make_batch, IndependentBeta, BetaActor
from torchrl.envs.transforms import CatTensors

def construct_policy(cfg, device, observation_spec=None, action_spec=None):
    if observation_spec is None:
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec(shape=(1, 11), device=device),
                    "lidar": UnboundedContinuousTensorSpec(shape=(1, 1, 144, 1), device=device),
                }, shape=(1,), device=device)
            }, shape=(1,), device=device)
        }, shape=(1,), device=device)

    if action_spec is None:
        action_spec = CompositeSpec({
                "agents": CompositeSpec({
                    "action": UnboundedContinuousTensorSpec(shape=(1, 3), device=device)
            }, shape=(1,), device=device)
        }, shape=(1,), device=device)
    model = ILtoPPO(cfg.algo, observation_spec, action_spec, 'single', device=device)
    return model

class InvertiblePLU(nn.Module):
    """Invertible PLU decomposition linear layer for mixing dimensions in flow models."""
    def __init__(self, features: int):
        super().__init__()
        self.features = features
        w_shape = (self.features, self.features)
        w = torch.empty(w_shape)
        nn.init.orthogonal_(w)
        P, L, U = torch.linalg.lu(w)

        self.s = nn.Parameter(torch.diag(U))
        self.U = nn.Parameter(U - torch.diag(self.s))
        self.L = nn.Parameter(L)
        
        self.P = nn.Parameter(P, requires_grad=False)
        self.P_inv = nn.Parameter(torch.linalg.inv(P), requires_grad=False)

    def forward(self, x):
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.features, device=x.device)
        U = torch.triu(self.U, diagonal=1)
        s = self.s
        W = self.P @ L @ (U + torch.diag(s))
        z = x @ W
        logdet = torch.sum(torch.log(torch.abs(s)), dim=0, keepdim=True)
        return z, logdet

    def reverse(self, x):
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.features, device=x.device)
        U = torch.triu(self.U, diagonal=1)
        s = self.s
        eye = torch.eye(self.features, device=x.device, dtype=U.dtype)
        U_inv = torch.linalg.solve_triangular(U + torch.diag(s), eye, upper=True)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False, unitriangular=True)
        W_inv = U_inv @ L_inv @ self.P_inv
        z = x @ W_inv
        logdet = torch.sum(torch.log(torch.abs(s)), dim=0, keepdim=True)
        return z, -logdet

class MetaBlock(torch.nn.Module):
    """Core transformation block for RealNVP, a conditional affine coupling layer."""
    def __init__(self, in_channels: int, channels: int, cond_channels: int):
        super().__init__()
        final_cond_channels = int(np.ceil(in_channels / 2)) + cond_channels
        self.l = InvertiblePLU(features=in_channels)
        self.t_net = nn.Sequential(
            nn.Linear(final_cond_channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, in_channels // 2)
        )
        nn.init.zeros_(self.t_net[-1].weight)
        if self.t_net[-1].bias is not None:
            nn.init.zeros_(self.t_net[-1].bias)

        self.s_net = nn.Sequential(
            nn.Linear(final_cond_channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, in_channels // 2)
        )
        nn.init.zeros_(self.s_net[-1].weight)
        if self.s_net[-1].bias is not None:
            nn.init.zeros_(self.s_net[-1].bias)
            
    def forward(self, x, y):
        x, log_det_l = self.l.forward(x)
        x_cond, x_trans = torch.tensor_split(x, 2, dim=1)
        condition = torch.cat([x_cond, y], dim=-1)
        s = self.s_net(condition)
        t = self.t_net(condition)
        x_trans = (x_trans - t) * torch.exp(-s)
        x = torch.cat((x_cond, x_trans), dim=1)
        log_det_s = -s.sum(dim=1)
        return x, log_det_l + log_det_s

    def reverse(self, z, y):
        z_cond, z_trans = torch.tensor_split(z, 2, dim=1)
        condition = torch.cat([z_cond, y], dim=-1)
        s = self.s_net(condition)
        t = self.t_net(condition)
        z_trans = z_trans * torch.exp(s) + t
        z = torch.cat((z_cond, z_trans), dim=1)
        z, _ = self.l.reverse(z)
        return z

class RealNVP(nn.Module):
    """Complete RealNVP model, stacking multiple MetaBlocks."""
    def __init__(self, in_channels, channels, cond_channels, n_layers, prior):
        super().__init__()
        self.blocks = nn.ModuleList([
            MetaBlock(in_channels, channels, cond_channels) for _ in range(n_layers)
        ])
        self.prior = prior

    def forward(self, x, y):
        log_dets = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            x, log_det = block(x, y)
            log_dets = log_dets + log_det
        return x, log_dets

    def reverse(self, x, y):
        for block in reversed(self.blocks):
            x = block.reverse(x, y)
        return x

class GEncoder(nn.Module):
    """State encoder that encodes high-dimensional state s into condition vector y."""
    def __init__(self, input_size, rep_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, rep_size)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.network(s)


class ILtoPPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, mode='single', device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Feature extractor using TensorDictSequential and Lazy modules
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        self.feature_extractor_cnn = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
        ).to(self.device)

        feature_dim = 139
        self.encoder = GEncoder(input_size=feature_dim, rep_size=cfg.actor.rep_dim).to(self.device)
        self.action_dim_il = 2 
        prior = distributions.MultivariateNormal(
            torch.zeros(self.action_dim_il).to(device), torch.eye(self.action_dim_il).to(device)
        )
        self.actor = RealNVP(
            in_channels=self.action_dim_il, channels=cfg.actor.channels,
            cond_channels=cfg.actor.rep_dim, n_layers=cfg.actor.blocks, prior=prior
        ).to(self.device)

        il_params = list(self.feature_extractor_cnn.parameters()) + list(self.encoder.parameters()) + list(self.actor.parameters())
        self.il_optimizer = torch.optim.Adam(il_params, lr=cfg.actor.learning_rate)

        if observation_spec is not None:
            self.feature_extractor_cnn(observation_spec.zero())

    def _get_condition_from_batch(self, batch):
        """Helper function to convert batch dict to condition vector."""
        input_td = TensorDict({
            ("agents", "observation", "lidar"): batch["lidar"].to(self.device),
            ("agents", "observation", "state"): batch["state"].to(self.device),
        }, batch_size=batch["state"].shape[0], device=self.device)
        
        self.feature_extractor_cnn(input_td)
        feature = input_td["_feature"]
        condition = self.encoder(feature)
        return condition

    def train_il_step(self, batch, only_eval=False):
        if not only_eval: self.train(); self.il_optimizer.zero_grad()
        else: self.eval()

        expert_actions_3d = batch["expert_action"].to(self.device).float()
        expert_actions_2d = expert_actions_3d[:, [0, 2]]

        with torch.set_grad_enabled(not only_eval):
            condition = self._get_condition_from_batch(batch)

            noise = torch.randn_like(expert_actions_2d) * self.cfg.actor.noise_std
            noisy_expert_actions = expert_actions_2d + noise
            
            z, logdets = self.actor(noisy_expert_actions, condition)
            log_prob = self.actor.prior.log_prob(z) + logdets
            loss = -log_prob.mean()

        if not only_eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.il_optimizer.step()

        return {"nll_loss": loss.item(), 'lr': self.il_optimizer.param_groups[0]['lr']}

    def sample_action(self, batch):
        self.eval()
        with torch.no_grad():
            condition = self._get_condition_from_batch(batch)
            batch_size = condition.shape[0]
            prior_sample = self.actor.prior.sample((batch_size,))
            action_2d_raw = self.actor.reverse(prior_sample, condition)
        
        action_2d_raw.requires_grad_()
        with torch.enable_grad():
            z, logdets = self.actor(action_2d_raw, condition)
            log_prob = self.actor.prior.log_prob(z) + logdets
            grad = torch.autograd.grad(log_prob.sum(), [action_2d_raw])[0]
            # grad.clamp_(-5.0, 5.0)
        
        with torch.no_grad():
            denoised_action = action_2d_raw.detach() + self.cfg.actor.noise_std**2 * grad

        action_3d = torch.zeros(batch_size, 3, device=self.device)
        action_3d[:, 0] = denoised_action[:, 0]
        action_3d[:, 2] = denoised_action[:, 1]
        return action_3d

    @torch.no_grad()
    def sample_raw_action(self, batch):
        self.eval()
        condition = self._get_condition_from_batch(batch)
        batch_size = condition.shape[0]
        prior_sample = self.actor.prior.sample((batch_size,))
        action_2d_raw = self.actor.reverse(prior_sample, condition)
        
        action_3d = torch.zeros(batch_size, 3, device=self.device)
        action_3d[:, 0] = action_2d_raw[:, 0]
        action_3d[:, 2] = action_2d_raw[:, 1]
        return action_3d