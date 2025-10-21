from typing import Any, Dict, Optional, Sequence, Union, Tuple

import torch
from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import (
 TransformedEnv,
 Transform,
 Compose,
 FlattenObservation,
 CatTensors
)
from torchrl.data import (
 TensorSpec,
 BoundedTensorSpec,
 UnboundedContinuousTensorSpec,
 DiscreteTensorSpec,
 MultiDiscreteTensorSpec,
 CompositeSpec,
)
# from .env import AgentSpec
from omni_drones.utils.torchrl.env import AgentSpec
from dataclasses import replace
from omni_drones.utils.torch import quaternion_to_euler
import omni.isaac.orbit.utils.math as math_utils
import math
from collections import deque

def _transform_agent_spec(self: Transform, agent_spec: AgentSpec) -> AgentSpec:
    return agent_spec
Transform.transform_agent_spec = _transform_agent_spec

def _transform_agent_spec(self: Compose, agent_spec: AgentSpec) -> AgentSpec:
    for transform in self.transforms:
        agent_spec = transform.transform_agent_spec(agent_spec)
    return agent_spec
Compose.transform_agent_spec = _transform_agent_spec

def _agent_spec(self: TransformedEnv) -> AgentSpec:
    agent_spec = self.transform.transform_agent_spec(self.base_env.agent_spec)
    return {name: replace(spec, _env=self) for name, spec in agent_spec.items()}
TransformedEnv.agent_spec = property(_agent_spec)

class LocomotionController(Transform):
    """

    padding(zero//)
    last actions:target actions
    scale factor
    """
    def __init__(
        self,
        locomotion_policy,
        action_key: str = ("agents", "action"),
        joint_action_key: str = ("agents", "joint_actions"),
        device="cuda"
    ):
        in_keys = [action_key, ("agents", "observation", "state")]
        super().__init__(in_keys=in_keys, out_keys=[joint_action_key])
        self.locomotion_policy = locomotion_policy
        self.action_key = action_key
        self.joint_action_key = joint_action_key
        self.device = device

        self.queue_len = 11 # T
        self.obs_dim = 45 # D
        self.obs_buffer = deque(maxlen=self.queue_len)
        self.default_obs = None
        self.last_actions = None
        self.joint_index_mapping = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
        self.joint_inverse_mapping = [self.joint_index_mapping.index(i) for i in range(12)]
        self.angle_scale = torch.tensor([0.25, 0.25, 0.25], device=self.device)

        self.command_scale = torch.tensor(
            [2.0, 2.0, 0.25],
            device=self.device
        )
        self.joint_vel_scale = torch.tensor(
            [0.05, 0.05, 0.05,
            0.05, 0.05, 0.05,
            0.05, 0.05, 0.05,
            0.05, 0.05, 0.05],
            device=self.device
        )
        self.action_scale = torch.tensor(
            [0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25],
            device=self.device
        )

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        return input_spec

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        standstill_flag = tensordict.get(("info", "standstill"), None)
        if standstill_flag is not None and standstill_flag[0]:
            tensordict.set(self.joint_action_key, tensordict[("info", "default_joint_position")])
            return tensordict

        first_frame_debug = False
        if self.action_key not in tensordict.keys(True, True):
            print(f"Warning: '{self.action_key}' not found in tensordict")
            return tensordict

        base_quat = tensordict[("info", "orientation")] # shape: (num_envs, 1, 4)
        base_ang_vel = tensordict[("info", "angular_velocity")] # shape: (num_envs, 1, 3)
        # velocity_commands = torch.tensor([[0.0, 0.0, 0.5]], device=self.device).expand(tensordict.batch_size[0], 1, 3)
        velocity_commands = tensordict[self.action_key]
        gravity_world = torch.tensor([0., 0., -1.], device=self.device)
        projected_gravity = math_utils.quat_rotate_inverse(
            base_quat.squeeze(1), gravity_world.unsqueeze(0).expand(base_quat.shape[0], 3)
        )
        joint_pos = tensordict[("info", "joint_position")][..., self.joint_index_mapping]
        default_joint_pos = tensordict[("info", "default_joint_position")][..., self.joint_index_mapping]
        last_actions = self.last_actions if self.last_actions is not None else torch.zeros_like(joint_pos)
        joint_pos_rel = joint_pos - default_joint_pos
        joint_vel_rel = tensordict[("info", "joint_velocity")][..., self.joint_index_mapping]

        if first_frame_debug:
            print("------- observation -------")
            print("default_joint_pos: ", default_joint_pos.detach().cpu().numpy())
            print("base_ang_vel: ", base_ang_vel.detach().cpu().numpy())
            print("projected_gravity: ", projected_gravity.detach().cpu().numpy())
            print("velocity_commands: ", velocity_commands.detach().cpu().numpy())
            print("joint_pos_rel: ", joint_pos_rel.detach().cpu().numpy())
            print("joint_vel_rel: ", joint_vel_rel.detach().cpu().numpy())
            print("last_actions: ", last_actions.detach().cpu().numpy())

        obs_tensor = torch.cat([
            # base_ang_vel.squeeze(1) * self.angle_scale.unsqueeze(0),
            torch.zeros_like(base_ang_vel.squeeze(1)),
            projected_gravity,
            velocity_commands.squeeze(1) * self.command_scale.unsqueeze(0),
            joint_pos_rel.squeeze(1),
            joint_vel_rel.squeeze(1) * self.joint_vel_scale.unsqueeze(0),
            last_actions.squeeze(1)
        ], dim=-1).to(self.device)
        obs_tensor = torch.clamp(obs_tensor, -100.0, 100.0)
        padding_tensor = torch.zeros_like(obs_tensor)
        if len(self.obs_buffer) < self.queue_len:
            for i in range(self.queue_len - len(self.obs_buffer)):
                self.obs_buffer.append(padding_tensor.clone())
        self.obs_buffer.append(obs_tensor.clone())
        hist_buffer = list(self.obs_buffer)[:-1]
        hist_tensor = torch.stack(hist_buffer, dim=0)
        hist_tensor = hist_tensor.permute(1, 0, 2).contiguous() # (N, T, D)
        hist_tensor = hist_tensor.view(hist_tensor.shape[0], -1) # (N, T*D) -> (N, 10*45)
        hist_tensor = torch.clamp(hist_tensor, -100.0, 100.0)
        input_tensor = torch.cat([hist_tensor, obs_tensor], dim=-1)
        # input_tensor = torch.tensor([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.000, -0.034, -0.000, -0.999, 2.000, 0.000, -0.000, -0.025, -0.002, -0.098, 0.025, -0.002, -0.098, -0.111, 0.007, -0.069, 0.111, 0.007, -0.069, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]).to(hist_tensor.device)
        # input_tensor = input_tensor.view(hist_tensor.shape[0], -1)

        if first_frame_debug:
            print("input_tensor: ", input_tensor.detach().cpu().numpy())
        with torch.no_grad():
            joint_outputs = self.locomotion_policy(input_tensor)
        if first_frame_debug:
            print("model_outputs: ", joint_outputs.detach().cpu().numpy())

        # BE CAREFUL
        self.last_actions = joint_outputs
        target_joint_pos = joint_outputs * self.action_scale.unsqueeze(0)
        target_joint_pos = target_joint_pos + default_joint_pos.squeeze(1)
        if first_frame_debug:
            print("------- predicted joint actions -------")
            print("target_joint_pos: ", target_joint_pos.detach().cpu().numpy())
            exit()
        target_joint_pos_sim = target_joint_pos[:, self.joint_inverse_mapping]
        tensordict.set(self.joint_action_key, target_joint_pos_sim)
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict)

    def _step(self, tensordict, next_tensordict):

        return next_tensordict
