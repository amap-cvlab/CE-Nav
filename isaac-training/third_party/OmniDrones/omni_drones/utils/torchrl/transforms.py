# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
from .env import AgentSpec
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


class FromDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = None,
    ):
        if nbins is None:
            nbins = 2
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            self.minimum = action_spec.space.minimum.unsqueeze(-2)
            self.maximum = action_spec.space.maximum.unsqueeze(-2)
            self.mapping = torch.cartesian_prod(
                *[torch.linspace(0, 1, dim_nbins) for dim_nbins in nbins]
            ).to(action_spec.device)  # [prod(nbins), len(nbins)]
            n = self.mapping.shape[0]
            spec = DiscreteTensorSpec(
                n, shape=[*action_spec.shape[:-1], 1], device=action_spec.device
            )
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping * (self.maximum - self.minimum) + self.minimum
        action = action.unsqueeze(-1)
        action = torch.take_along_dim(mapping, action, dim=-2).squeeze(-2)
        return action


class FromMultiDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = 2,
    ):
        if action_key is None:
            action_key = "action"
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            spec = MultiDiscreteTensorSpec(
                nbins, shape=action_spec.shape, device=action_spec.device
            )
            self.nvec = spec.nvec.to(action_spec.device)
            self.minimum = action_spec.space.minimum
            self.maximum = action_spec.space.maximum
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action = action / (self.nvec - 1) * (self.maximum - self.minimum) + self.minimum
        return action

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._inv_call(tensordict)


class DepthImageNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        min_range: float,
        max_range: float,
        inverse: bool=False
    ):
        super().__init__(in_keys=in_keys)
        self.max_range = max_range
        self.min_range = min_range
        self.inverse = inverse

    def _apply_transform(self, obs: torch.Tensor) -> None:
        obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
        obs = obs.clip(self.min_range, self.max_range)
        if self.inverse:
            obs = (obs - self.min_range) / (self.max_range - self.min_range)
        else:
            obs = (self.max_range - obs) / (self.max_range - self.min_range)
        return obs


def ravel_composite(
    spec: CompositeSpec, key: str, start_dim: int=-2, end_dim: int=-1
):
    r"""
    
    Examples:
    >>> obs_spec = CompositeSpec({
    ...     "obs_self": UnboundedContinuousTensorSpec((1, 19)),
    ...     "obs_others": UnboundedContinuousTensorSpec((3, 13)),
    ... })
    >>> spec = CompositeSpec({
            "agents": {
                "observation": obs_spec
            }
    ... })
    >>> t = ravel_composite(spec, ("agents", "observation"))

    """
    composite_spec = spec[key]
    if not isinstance(key, tuple):
        key = (key,)
    if isinstance(composite_spec, CompositeSpec):
        in_keys = [k for k in spec.keys(True, True) if k[:len(key)] == key]

        # print("my print: ", in_keys)
        # ('agents', 'intrinsics', 'mass'), ('agents', 'intrinsics', 'inertia'), ('agents', 'intrinsics', 'com'), ('agents', 'intrinsics', 'KF'), ('agents', 'intrinsics', 'KM'), ('agents', 'intrinsics', 'tau_up'), ('agents', 'intrinsics', 'tau_down'), ('agents', 'intrinsics', 'drag_coef')]
        return Compose(
            FlattenObservation(start_dim, end_dim, in_keys),
            CatTensors(in_keys, out_key=key, del_keys=False)
        )
    else:
        raise TypeError


class VelController(Transform):
    def __init__(
        self,
        controller,
        yaw_control: bool = True,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.yaw_control = yaw_control
        self.action_key = action_key
    

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if (self.yaw_control):
            spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        else:
            spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(3,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # print("tensordict size: ", tensordict.shape)
        # print("tensor dict: ", tensordict)
        drone_state = tensordict[("info", "drone_state")][..., :13]
        # print("drone state shape: ", drone_state.shape)

        action = tensordict[self.action_key]
        if (self.yaw_control):
            target_vel, target_yaw = action.split([3, 1], -1)
            target_vel = target_vel.unsqueeze(1)
            target_yaw = target_yaw.unsqueeze(1)
            target_yaw = target_yaw * torch.pi
        else:
            target_vel = action.unsqueeze(1)
            # print("target vel: ", target_vel)
            # target_yaw = torch.zeros(action.shape[:-1] + (1,), device=action.device)
            target_yaw = None

        # print("drone vel shape: ", target_vel.shape)
        # print("target vel: ", target_vel)
        cmds = self.controller(
            drone_state, 
            target_vel=target_vel, 
            target_yaw=target_yaw
        )

        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class RateController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.) * self.max_thrust
        cmds = self.controller(
            drone_state, 
            target_rate=target_rate * torch.pi, 
            target_thrust=target_thrust
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class AttitudeController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_thrust, target_yaw_rate, target_roll, target_pitch = action.split(1, dim=-1)
        cmds = self.controller(
            drone_state,
            target_thrust=((target_thrust+1)/2).clip(0.) * self.max_thrust,
            target_yaw_rate=target_yaw_rate * torch.pi,
            target_roll=target_roll * torch.pi,
            target_pitch=target_pitch * torch.pi
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class History(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        out_keys: Sequence[str]=None,
        steps: int = 32,
    ):
        if out_keys is None:
            out_keys = [
                f"{key}_h" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_h",)
                for key in in_keys
            ]
        if any(key in in_keys for key in out_keys):
            raise ValueError
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.steps = steps
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(-1).expand(*spec.shape, self.steps)
                observation_spec[out_key] = spec
        return observation_spec

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key)
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key).clone()
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
            tensordict.set(("next", out_key), item_history)
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(tensordict.batch_size, dtype=bool, device=tensordict.device)
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if out_key not in tensordict.keys(True, True):
                item = tensordict.get(in_key)
                item_history = (
                    item.unsqueeze(-1)
                    .expand(*item.shape, self.steps)
                    .clone()
                    .zero_()
                )
                tensordict.set(out_key, item_history)
            else:
                item_history = tensordict.get(out_key)
                item_history[_reset] = 0.
        return tensordict


class CustomVelYawRateController(Transform):
    def __init__(
        self,
        controller,
        sim_dt,
        debug = False,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.sim_dt = sim_dt
        self.action_key = action_key
        self.debug = debug

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1] + (3,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]

        vx, vy, vyaw = torch.split(action, 1, dim=-1)

        # 构造目标速度：z方向不动（可改为高度保持）
        target_vel = torch.cat([vx, vy, torch.zeros_like(vx)], dim=-1)

        # 构造目标偏航角（通过积分偏航率得到）
        current_yaw = quaternion_to_euler(drone_state[..., 3:7])[..., -1].unsqueeze(1)
        # real_control_dt = tensordict[("info", "real_control_dt")]
        target_yaw = current_yaw + vyaw * self.sim_dt
        target_yaw = torch.fmod(target_yaw + torch.pi, 2 * torch.pi) - torch.pi
        
        if self.debug:
            print("===== debug info in _inv_call =====")
            print("real_control_dt=", real_control_dt)
            print("vx: ", vx)
            print("vy: ", vy)
            print("vyaw: ", vyaw)
            print("current_yaw: ", current_yaw)
            print("vyaw * self.sim_dt = ", vyaw * self.sim_dt)
            print("target_yaw: ", target_yaw)
        
        cmds = self.controller(drone_state, target_vel=target_vel, target_yaw=target_yaw)
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    Input shape: (..., 4)
    Output shape: (..., 3, 3)
    """
    w, x, y, z = quat.unbind(dim=-1)
    N = quat.shape[-1]
    if not (N == 4):
        raise ValueError(f"Input quaternions should be of shape (..., 4), but are {quat.shape}")

    device = quat.device
    mat = torch.stack(
        (
            1 - 2 * (y**2 + z**2),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x**2 + y**2),
        ),
        dim=-1,
    ).view(*quat.shape[:-1], 3, 3)

    return mat


class LocomotionController(Transform):
    """
    队列长度和观测维度
    队列padding逻辑(zero/初始观测/当前观测)
    last actions:是上次模型输出的值还是上次的target actions
    输入输出scale factor问题
    """
    def __init__(
        self,
        locomotion_policy,
        action_key: str = ("agents", "action"),
        joint_action_key: str = ("agents", "joint_actions"),
        device="cuda"
    ):
        in_keys = [action_key, ("agents", "observation", "state")]  # 假设 state 包含 base lin/ang vel 等信息
        super().__init__(in_keys=in_keys, out_keys=[joint_action_key])
        self.locomotion_policy = locomotion_policy
        self.action_key = action_key
        self.joint_action_key = joint_action_key
        self.device = device

        # 根据模型配置设置
        self.queue_len = 11   # T
        self.obs_dim = 45       # D
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

        # 提取基础状态
        base_quat = tensordict[("info", "orientation")]     # shape: (num_envs, 1, 4)
        base_ang_vel = tensordict[("info", "angular_velocity")]  # shape: (num_envs, 1, 3)
        # velocity_commands = torch.tensor([[0.0, 0.0, 0.5]], device=self.device).expand(tensordict.batch_size[0], 1, 3)
        velocity_commands = tensordict[self.action_key]
        gravity_world = torch.tensor([0., 0., -1.], device=self.device)
        projected_gravity = math_utils.quat_rotate_inverse(
            base_quat.squeeze(1), gravity_world.unsqueeze(0).expand(base_quat.shape[0], 3)
        )
        # 读取进来的joints顺序:['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        # 需要转换成:[FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        joint_pos = tensordict[("info", "joint_position")][..., self.joint_index_mapping]
        default_joint_pos = tensordict[("info", "default_joint_position")][..., self.joint_index_mapping]
        last_actions = self.last_actions if self.last_actions is not None else torch.zeros_like(joint_pos)
        joint_pos_rel = joint_pos - default_joint_pos
        joint_vel_rel = tensordict[("info", "joint_velocity")][..., self.joint_index_mapping]

        # 打印所有缩放前的观测量
        if first_frame_debug:
            print("------- observation -------")
            print("default_joint_pos: ", default_joint_pos.detach().cpu().numpy())
            print("base_ang_vel: ", base_ang_vel.detach().cpu().numpy())
            print("projected_gravity: ", projected_gravity.detach().cpu().numpy())
            print("velocity_commands: ", velocity_commands.detach().cpu().numpy())
            print("joint_pos_rel: ", joint_pos_rel.detach().cpu().numpy())
            print("joint_vel_rel: ", joint_vel_rel.detach().cpu().numpy())
            print("last_actions: ", last_actions.detach().cpu().numpy())

        # 构造当前帧观测（必须严格匹配训练时的顺序和维度）并添加到buffer
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
        hist_tensor = hist_tensor.permute(1, 0, 2).contiguous()  # (N, T, D)
        hist_tensor = hist_tensor.view(hist_tensor.shape[0], -1)  # (N, T*D) -> (N, 10*45)
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
        # ❌ 显式禁用在 step 后的 transform 上运行
        return next_tensordict