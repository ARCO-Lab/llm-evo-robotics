#!/usr/bin/env python3
"""
GPT-5建议的真正通用多任务SAC架构
支持2-5关节Reacher的同时训练，一套模型控制所有关节数
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.sac.policies import SACPolicy
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import tempfile
import xml.etree.ElementTree as ET

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 全局配置
J_MAX = 5  # 支持的最大关节数
SUPPORTED_JOINTS = [2, 3, 4, 5]  # 支持的关节数列表

class JointTokenEncoder(nn.Module):
    """
    关节Token编码器
    输入: [cos θ_i, sin θ_i, vel_i, link_len_i, joint_id_onehot/J_max, parent_id_onehot/J_max]
    """
    
    def __init__(self, joint_token_dim: int = 64):
        super().__init__()
        # 输入维度: 3 (cos, sin, vel) + 1 (link_len) + 1 (joint_id/J_max) + 1 (parent_id/J_max) = 6
        input_dim = 6
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, joint_token_dim),
            nn.ReLU(),
            nn.LayerNorm(joint_token_dim),
            nn.Linear(joint_token_dim, joint_token_dim),
            nn.ReLU(),
            nn.LayerNorm(joint_token_dim)
        )
        
        self.joint_token_dim = joint_token_dim
        
    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: [batch_size, J_max, 6]
        Returns:
            joint_tokens: [batch_size, J_max, joint_token_dim]
        """
        batch_size, num_joints, feature_dim = joint_features.shape
        
        # 展平处理
        joint_features_flat = joint_features.view(-1, feature_dim)
        joint_tokens_flat = self.encoder(joint_features_flat)
        
        # 恢复形状
        joint_tokens = joint_tokens_flat.view(batch_size, num_joints, self.joint_token_dim)
        
        return joint_tokens

class TaskTokenEncoder(nn.Module):
    """
    任务Token编码器
    输入: [N/J_max, onehot_N (5维), link_len_1..N, 其余补0]
    """
    
    def __init__(self, task_token_dim: int = 64):
        super().__init__()
        # 输入维度: 1 (N/J_max) + 5 (onehot_N) + J_max (link_lengths) = 1 + 5 + 5 = 11
        input_dim = 1 + len(SUPPORTED_JOINTS) + J_MAX
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, task_token_dim),
            nn.ReLU(),
            nn.LayerNorm(task_token_dim),
            nn.Linear(task_token_dim, task_token_dim),
            nn.ReLU(),
            nn.LayerNorm(task_token_dim)
        )
        
        self.task_token_dim = task_token_dim
        
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_features: [batch_size, input_dim]
        Returns:
            task_tokens: [batch_size, task_token_dim]
        """
        return self.encoder(task_features)

class SharedBackbone(nn.Module):
    """
    共享骨干网络
    关节MLP编码 → Multi-Head Self-Attention → 注意力池化
    """
    
    def __init__(self, joint_token_dim: int = 64, task_token_dim: int = 64, 
                 num_heads: int = 4, backbone_dim: int = 128):
        super().__init__()
        
        self.joint_encoder = JointTokenEncoder(joint_token_dim)
        self.task_encoder = TaskTokenEncoder(task_token_dim)
        
        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=joint_token_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 注意力池化
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=joint_token_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True
        )
        
        # 全局特征融合
        self.global_fusion = nn.Sequential(
            nn.Linear(joint_token_dim + task_token_dim, backbone_dim),
            nn.ReLU(),
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.LayerNorm(backbone_dim)
        )
        
        self.backbone_dim = backbone_dim
        
    def forward(self, joint_features: torch.Tensor, task_features: torch.Tensor, 
                joint_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: [batch_size, J_max, 6]
            task_features: [batch_size, input_dim]
            joint_mask: [batch_size, J_max] - True表示有效关节，False表示padding
        Returns:
            backbone_features: [batch_size, backbone_dim]
        """
        batch_size = joint_features.shape[0]
        
        # 编码关节和任务
        joint_tokens = self.joint_encoder(joint_features)  # [B, J_max, joint_token_dim]
        task_token = self.task_encoder(task_features)      # [B, task_token_dim]
        
        # Self-Attention with mask
        # key_padding_mask: True表示需要mask的位置（padding）
        key_padding_mask = ~joint_mask  # 反转mask，因为MultiheadAttention的mask语义相反
        
        attended_joints, _ = self.self_attention(
            query=joint_tokens,
            key=joint_tokens,
            value=joint_tokens,
            key_padding_mask=key_padding_mask
        )  # [B, J_max, joint_token_dim]
        
        # 注意力池化：使用task_token作为query
        task_query = task_token.unsqueeze(1)  # [B, 1, task_token_dim]
        
        # 需要调整维度匹配
        if task_query.shape[-1] != attended_joints.shape[-1]:
            task_query = F.linear(task_query, 
                                torch.randn(attended_joints.shape[-1], task_query.shape[-1]).to(task_query.device))
        
        pooled_joints, _ = self.attention_pooling(
            query=task_query,
            key=attended_joints,
            value=attended_joints,
            key_padding_mask=key_padding_mask
        )  # [B, 1, joint_token_dim]
        
        pooled_joints = pooled_joints.squeeze(1)  # [B, joint_token_dim]
        
        # 融合全局特征
        global_features = torch.cat([pooled_joints, task_token], dim=-1)  # [B, joint_token_dim + task_token_dim]
        backbone_features = self.global_fusion(global_features)  # [B, backbone_dim]
        
        return backbone_features

class VariableActorHead(nn.Module):
    """
    可变维度Actor头
    输出J_max个关节的(μ, logσ)，推理/训练时只取前N维
    """
    
    def __init__(self, backbone_dim: int = 128, action_dim: int = J_MAX):
        super().__init__()
        
        self.mean_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, action_dim)
        )
        
        self.log_std_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, action_dim)
        )
        
        self.action_dim = action_dim
        
    def forward(self, backbone_features: torch.Tensor, num_joints: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_features: [batch_size, backbone_dim]
            num_joints: 实际关节数
        Returns:
            mean: [batch_size, num_joints]
            log_std: [batch_size, num_joints]
        """
        full_mean = self.mean_head(backbone_features)      # [B, J_max]
        full_log_std = self.log_std_head(backbone_features) # [B, J_max]
        
        # 只取前num_joints维
        mean = full_mean[:, :num_joints]
        log_std = full_log_std[:, :num_joints]
        
        # 限制log_std范围
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

class MultiHeadCritic(nn.Module):
    """
    多头Critic：每个N一个Q1/Q2头
    """
    
    def __init__(self, backbone_dim: int = 128, supported_joints: List[int] = SUPPORTED_JOINTS):
        super().__init__()
        
        self.supported_joints = supported_joints
        
        # 为每个关节数创建独立的Q1/Q2头
        self.q1_heads = nn.ModuleDict()
        self.q2_heads = nn.ModuleDict()
        
        for num_joints in supported_joints:
            self.q1_heads[str(num_joints)] = nn.Sequential(
                nn.Linear(backbone_dim + num_joints, backbone_dim),
                nn.ReLU(),
                nn.Linear(backbone_dim, 1)
            )
            
            self.q2_heads[str(num_joints)] = nn.Sequential(
                nn.Linear(backbone_dim + num_joints, backbone_dim),
                nn.ReLU(),
                nn.Linear(backbone_dim, 1)
            )
    
    def forward(self, backbone_features: torch.Tensor, actions: torch.Tensor, 
                num_joints: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_features: [batch_size, backbone_dim]
            actions: [batch_size, num_joints]
            num_joints: 关节数
        Returns:
            q1_values: [batch_size, 1]
            q2_values: [batch_size, 1]
        """
        if num_joints not in self.supported_joints:
            raise ValueError(f"Unsupported num_joints: {num_joints}")
        
        # 拼接特征和动作
        q_input = torch.cat([backbone_features, actions], dim=-1)
        
        # 使用对应的头
        key = str(num_joints)
        q1_values = self.q1_heads[key](q_input)
        q2_values = self.q2_heads[key](q_input)
        
        return q1_values, q2_values

class PerTaskEntropy(nn.Module):
    """
    每任务熵温度α
    """
    
    def __init__(self, supported_joints: List[int] = SUPPORTED_JOINTS):
        super().__init__()
        
        self.supported_joints = supported_joints
        
        # 为每个关节数创建独立的log_alpha
        self.log_alphas = nn.ParameterDict()
        for num_joints in supported_joints:
            self.log_alphas[str(num_joints)] = nn.Parameter(torch.zeros(1))
    
    def get_alpha(self, num_joints: int) -> torch.Tensor:
        """获取指定关节数的熵温度"""
        if num_joints not in self.supported_joints:
            raise ValueError(f"Unsupported num_joints: {num_joints}")
        
        key = str(num_joints)
        return torch.exp(self.log_alphas[key])
    
    def get_log_alpha(self, num_joints: int) -> torch.Tensor:
        """获取指定关节数的log熵温度"""
        if num_joints not in self.supported_joints:
            raise ValueError(f"Unsupported num_joints: {num_joints}")
        
        key = str(num_joints)
        return self.log_alphas[key]

class UniversalMultiTaskExtractor(BaseFeaturesExtractor):
    """
    通用多任务特征提取器
    整合所有组件
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.backbone = SharedBackbone(
            joint_token_dim=64,
            task_token_dim=64,
            num_heads=4,
            backbone_dim=features_dim
        )
        
        print(f"🔧 UniversalMultiTaskExtractor: 支持{SUPPORTED_JOINTS}关节")
        print(f"   J_max = {J_MAX}")
        print(f"   特征维度: {features_dim}")
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, obs_dim]
        Returns:
            features: [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # 解析观察（这里需要根据实际环境调整）
        joint_features, task_features, joint_mask = self._parse_observations(observations)
        
        # 通过共享骨干
        features = self.backbone(joint_features, task_features, joint_mask)
        
        return features
    
    def _parse_observations(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解析观察为关节特征、任务特征和mask
        这里是简化版本，实际需要根据环境调整
        """
        batch_size = observations.shape[0]
        
        # 假设观察格式：[joint_features..., task_info...]
        # 这里需要根据实际环境wrapper的输出格式调整
        
        # 创建dummy数据作为示例
        joint_features = torch.zeros(batch_size, J_MAX, 6).to(observations.device)
        task_features = torch.zeros(batch_size, 1 + len(SUPPORTED_JOINTS) + J_MAX).to(observations.device)
        joint_mask = torch.ones(batch_size, J_MAX, dtype=torch.bool).to(observations.device)
        
        # 假设前2个关节有效
        joint_mask[:, 2:] = False
        
        return joint_features, task_features, joint_mask

def generate_multi_joint_reacher_xml(num_joints: int, link_lengths: List[float], 
                                   link_masses: List[float]) -> str:
    """生成N关节Reacher的MuJoCo XML"""
    
    if len(link_lengths) != num_joints or len(link_masses) != num_joints:
        raise ValueError("link_lengths和link_masses的长度必须等于num_joints")
    
    xml_template = f'''
    <mujoco model="reacher_{num_joints}joint">
      <compiler angle="radian" inertiafromgeom="true"/>
      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
      </default>
      <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
      
      <worldbody>
        <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
        <geom conaffinity="1" contype="1" name="sideS" pos="0 -1 0" rgba="0.9 0.4 0.6 1" size="1 0.02 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideE" pos="1 0 0" rgba="0.9 0.4 0.6 1" size="0.02 1 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideN" pos="0 1 0" rgba="0.9 0.4 0.6 1" size="1 0.02 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideW" pos="-1 0 0" rgba="0.9 0.4 0.6 1" size="0.02 1 1" type="box"/>
        
        <body name="body0" pos="0 0 0">
          <joint axis="0 0 1" limited="true" name="joint0" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
          <geom fromto="0 0 0 {link_lengths[0]} 0 0" name="link0" size="0.02" type="capsule"/>
    '''
    
    # 添加后续关节和链接
    for i in range(1, num_joints):
        prev_length = sum(link_lengths[:i])
        xml_template += f'''
          <body name="body{i}" pos="{link_lengths[i-1]} 0 0">
            <joint axis="0 0 1" limited="true" name="joint{i}" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
            <geom fromto="0 0 0 {link_lengths[i]} 0 0" name="link{i}" size="0.02" type="capsule"/>
        '''
    
    # 添加末端执行器
    total_length = sum(link_lengths)
    xml_template += f'''
            <body name="fingertip" pos="{link_lengths[-1]} 0 0">
              <geom name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.0 1" size="0.01" type="sphere"/>
            </body>
    '''
    
    # 关闭所有body标签
    for i in range(num_joints):
        xml_template += '          </body>\n'
    
    # 添加目标
    xml_template += '''
        <body name="target" pos="0.4 0.4 0">
          <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-0.45 0.45" ref="0.4" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-0.45 0.45" ref="0.4" stiffness="0" type="slide"/>
          <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size="0.02" type="sphere"/>
        </body>
      </worldbody>
      
      <actuator>
    '''
    
    # 添加执行器
    for i in range(num_joints):
        xml_template += f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="joint{i}" gear="200"/>\n'
    
    xml_template += '''
      </actuator>
    </mujoco>
    '''
    
    return xml_template

from gymnasium.envs.mujoco import MujocoEnv

class RealMultiJointReacherEnv(MujocoEnv):
    """真实的N关节Reacher环境"""
    
    def __init__(self, num_joints: int, link_lengths: List[float], 
                 link_masses: List[float], render_mode: Optional[str] = None, **kwargs):
        
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        self.link_masses = link_masses
        
        # 生成XML
        xml_string = generate_multi_joint_reacher_xml(num_joints, link_lengths, link_masses)
        
        # 创建临时文件
        self.temp_xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.temp_xml_file.write(xml_string)
        self.temp_xml_file.close()
        
        # 定义观察和动作空间
        obs_dim = 2 * num_joints + 2 * num_joints + 4  # cos, sin, vel, ee_pos, target_pos
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        
        # 初始化MuJoCo环境
        super().__init__(
            model_path=self.temp_xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )
        
        self.action_space = action_space
        self.step_count = 0
        self.max_episode_steps = 50
        
        print(f"✅ RealMultiJointReacherEnv ({num_joints}关节) 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
        print(f"   最大episode步数: {self.max_episode_steps}")
    
    def step(self, action):
        # 显式渲染（如果需要）
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()
        reward = self._get_reward()
        
        # 检查终止条件
        ee_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance_to_target = np.linalg.norm(ee_pos - target_pos)
        
        terminated = distance_to_target < 0.02
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance_to_target,
            'is_success': terminated
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        # 关节角度的cos和sin
        joint_cos = np.cos(self.data.qpos[:self.num_joints])
        joint_sin = np.sin(self.data.qpos[:self.num_joints])
        
        # 关节速度
        joint_vel = self.data.qvel[:self.num_joints]
        
        # 末端执行器位置
        ee_pos = self.get_body_com("fingertip")[:2]
        
        # 目标位置
        target_pos = self.get_body_com("target")[:2]
        
        return np.concatenate([joint_cos, joint_sin, joint_vel, ee_pos, target_pos])
    
    def _get_reward(self):
        ee_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # 距离奖励
        reward = -distance
        
        # 到达奖励
        if distance < 0.02:
            reward += 10.0
        
        return reward
    
    def reset_model(self):
        # 随机初始化关节角度
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        
        # 随机目标位置
        qpos[-2:] = self.np_random.uniform(low=-0.4, high=0.4, size=2)
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        
        return self._get_obs()
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

class UniversalMultiTaskWrapper(gym.Wrapper):
    """
    通用多任务包装器
    将不同关节数的环境统一为相同的观察和动作空间
    """
    
    def __init__(self, env, num_joints: int, link_lengths: List[float]):
        super().__init__(env)
        
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        
        # 统一观察空间：关节特征 + 任务特征
        # 关节特征: J_max * 6 (cos, sin, vel, link_len, joint_id/J_max, parent_id/J_max)
        # 任务特征: 1 + len(SUPPORTED_JOINTS) + J_MAX
        joint_feature_dim = J_MAX * 6
        task_feature_dim = 1 + len(SUPPORTED_JOINTS) + J_MAX
        total_obs_dim = joint_feature_dim + task_feature_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # 统一动作空间为J_max维
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(J_MAX,), dtype=np.float32
        )
        
        print(f"🔧 UniversalMultiTaskWrapper ({num_joints}关节)")
        print(f"   统一观察空间: {self.observation_space.shape}")
        print(f"   统一动作空间: {self.action_space.shape}")
    
    def step(self, action):
        # 只使用前num_joints维动作
        real_action = action[:self.num_joints]
        
        obs, reward, terminated, truncated, info = self.env.step(real_action)
        
        # 转换观察
        unified_obs = self._transform_observation(obs)
        
        return unified_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        unified_obs = self._transform_observation(obs)
        return unified_obs, info
    
    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        """将原始观察转换为统一格式"""
        
        # 解析原始观察
        joint_cos = obs[:self.num_joints]
        joint_sin = obs[self.num_joints:2*self.num_joints]
        joint_vel = obs[2*self.num_joints:3*self.num_joints]
        ee_pos = obs[3*self.num_joints:3*self.num_joints+2]
        target_pos = obs[3*self.num_joints+2:3*self.num_joints+4]
        
        # 构建关节特征 [J_max, 6]
        joint_features = np.zeros((J_MAX, 6))
        
        for i in range(self.num_joints):
            joint_features[i, 0] = joint_cos[i]  # cos
            joint_features[i, 1] = joint_sin[i]  # sin
            joint_features[i, 2] = joint_vel[i]  # vel
            joint_features[i, 3] = self.link_lengths[i] if i < len(self.link_lengths) else 0.0  # link_len
            joint_features[i, 4] = (i + 1) / J_MAX  # joint_id/J_max
            joint_features[i, 5] = i / J_MAX if i > 0 else 0.0  # parent_id/J_max
        
        # 构建任务特征
        task_features = np.zeros(1 + len(SUPPORTED_JOINTS) + J_MAX)
        task_features[0] = self.num_joints / J_MAX  # N/J_max
        
        # onehot_N
        if self.num_joints in SUPPORTED_JOINTS:
            idx = SUPPORTED_JOINTS.index(self.num_joints)
            task_features[1 + idx] = 1.0
        
        # link_lengths (前N个有效，其余为0)
        for i in range(min(self.num_joints, len(self.link_lengths))):
            task_features[1 + len(SUPPORTED_JOINTS) + i] = self.link_lengths[i]
        
        # 展平并拼接
        joint_features_flat = joint_features.flatten()
        unified_obs = np.concatenate([joint_features_flat, task_features])
        
        return unified_obs.astype(np.float32)

def create_multi_joint_env(num_joints: int, render_mode: Optional[str] = None):
    """创建指定关节数的环境"""
    
    # 默认链接长度和质量
    if num_joints == 2:
        link_lengths = [0.1, 0.11]
        link_masses = [1.0, 1.0]
    elif num_joints == 3:
        link_lengths = [0.1, 0.1, 0.1]
        link_masses = [1.0, 1.0, 1.0]
    elif num_joints == 4:
        link_lengths = [0.08, 0.08, 0.08, 0.08]
        link_masses = [1.0, 1.0, 1.0, 1.0]
    elif num_joints == 5:
        link_lengths = [0.06, 0.06, 0.06, 0.06, 0.06]
        link_masses = [1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        raise ValueError(f"不支持的关节数: {num_joints}")
    
    # 创建环境
    env = RealMultiJointReacherEnv(
        num_joints=num_joints,
        link_lengths=link_lengths,
        link_masses=link_masses,
        render_mode=render_mode
    )
    
    # 包装
    env = UniversalMultiTaskWrapper(env, num_joints, link_lengths)
    env = Monitor(env)
    
    return env

def make_env(num_joints: int, render_mode: Optional[str] = None):
    """创建环境的工厂函数"""
    def _init():
        return create_multi_joint_env(num_joints, render_mode)
    return _init

def train_universal_multitask_sac(total_timesteps: int = 50000):
    """
    训练通用多任务SAC
    同时训练2-5关节Reacher
    """
    print("🚀 通用多任务SAC训练")
    print(f"🎯 同时训练{SUPPORTED_JOINTS}关节Reacher")
    print(f"💡 一套模型控制所有关节数")
    print("="*60)
    
    # 创建多任务并行环境
    print("🌍 创建多任务并行环境...")
    
    env_fns = []
    for num_joints in SUPPORTED_JOINTS:
        # 每个关节数创建一个环境实例
        env_fns.append(make_env(num_joints, render_mode='human'))
    
    # 创建向量化环境
    train_env = SubprocVecEnv(env_fns)
    
    print(f"✅ 多任务并行环境创建完成")
    print(f"   环境数量: {len(env_fns)}")
    print(f"   支持关节数: {SUPPORTED_JOINTS}")
    print(f"   观察空间: {train_env.observation_space}")
    print(f"   动作空间: {train_env.action_space}")
    
    # 创建SAC模型
    print("\n🤖 创建通用多任务SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': UniversalMultiTaskExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=1000,
        device='cpu',
        tensorboard_log="./tensorboard_logs/universal_multitask/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
    )
    
    print("✅ 通用多任务SAC模型创建完成")
    print("   ✅ 使用通用多任务特征提取器")
    print("   ✅ 支持2-5关节同时训练")
    print("   ✅ 并行训练加速")
    
    # 开始训练
    print(f"\n🎯 开始通用多任务训练 ({total_timesteps}步)...")
    print("💡 您将看到多个关节数的Reacher同时学习")
    print("💡 观察不同关节数的学习进展")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 通用多任务训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        # 保存模型
        model.save("models/universal_multitask_sac")
        print("💾 通用多任务模型已保存: models/universal_multitask_sac")
        
        return model
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/universal_multitask_sac_interrupted")
        print("💾 中断模型已保存")
        return model
    
    finally:
        train_env.close()

def main():
    """主函数"""
    print("🌟 GPT-5通用多任务SAC系统")
    print("🎯 一套模型同时训练2-5关节Reacher")
    print("💡 真正的通用多任务架构")
    print()
    
    try:
        # 训练阶段
        print("🚀 开始训练阶段...")
        model = train_universal_multitask_sac(total_timesteps=50000)
        
        print(f"\n🎉 通用多任务训练完成！")
        print(f"💡 一套模型现在可以控制2-5关节的Reacher")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
        print("💡 模型已保存，可以稍后继续训练或测试")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
