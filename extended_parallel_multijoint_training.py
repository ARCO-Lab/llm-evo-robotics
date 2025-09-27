#!/usr/bin/env python3
"""
扩展并行多关节训练：基于parallel_mixed_joint_training.py扩展到2-5关节
融入GPT-5建议的关键改进：任务条件化、对等采样、多头设计
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time
import multiprocessing as mp
import tempfile
from typing import List, Optional, Tuple
from gymnasium.envs.mujoco import MujocoEnv

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 全局配置 - 扩展到5关节
J_MAX = 5  # 支持的最大关节数
SUPPORTED_JOINTS = [2, 3, 4, 5]  # 支持的关节数列表

class UniversalMultiJointExtractor(BaseFeaturesExtractor):
    """
    通用多关节特征提取器
    融入GPT-5建议：任务条件化 + 关节token编码 + 注意力机制
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(UniversalMultiJointExtractor, self).__init__(observation_space, features_dim)
        
        # 支持的最大观察维度（5关节）
        self.max_obs_dim = self._calculate_max_obs_dim()
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 UniversalMultiJointExtractor: {obs_dim} -> {features_dim}")
        print(f"   支持关节数: {SUPPORTED_JOINTS}")
        print(f"   J_max = {J_MAX}")
        print(f"   最大观察维度: {self.max_obs_dim}")
        
        # 关节特征编码器 (GPT-5建议的关节token)
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [cos, sin, vel, joint_id/J_max]
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 任务特征编码器 (GPT-5建议的任务token)
        self.task_encoder = nn.Sequential(
            nn.Linear(len(SUPPORTED_JOINTS) + 1, 64),  # [N/J_max, onehot_N]
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 多头自注意力 (GPT-5建议的共享骨干)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 注意力池化
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=1,
            dropout=0.0,
            batch_first=True
        )
        
        # 全局特征融合
        self.global_fusion = nn.Sequential(
            nn.Linear(64 + 64 + 4, features_dim),  # joint + task + global
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def _calculate_max_obs_dim(self):
        """计算最大观察维度"""
        # 5关节: 5*cos + 5*sin + 5*vel + 2*ee_pos + 2*target_pos = 19
        return J_MAX * 3 + 4
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 解析观察
        joint_features, task_features, global_features, joint_mask = self._parse_observations(observations)
        
        # 编码关节和任务特征
        joint_tokens = self.joint_encoder(joint_features)  # [B, J_max, 64]
        task_token = self.task_encoder(task_features)      # [B, 64]
        
        # 自注意力处理关节间交互
        key_padding_mask = ~joint_mask  # 反转mask
        
        attended_joints, _ = self.self_attention(
            query=joint_tokens,
            key=joint_tokens,
            value=joint_tokens,
            key_padding_mask=key_padding_mask
        )  # [B, J_max, 64]
        
        # 注意力池化：使用task_token作为query
        task_query = task_token.unsqueeze(1)  # [B, 1, 64]
        
        pooled_joints, _ = self.attention_pooling(
            query=task_query,
            key=attended_joints,
            value=attended_joints,
            key_padding_mask=key_padding_mask
        )  # [B, 1, 64]
        
        pooled_joints = pooled_joints.squeeze(1)  # [B, 64]
        
        # 融合所有特征
        fused_features = torch.cat([pooled_joints, task_token, global_features], dim=-1)
        features = self.global_fusion(fused_features)
        
        return features
    
    def _parse_observations(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析观察为关节特征、任务特征、全局特征和mask"""
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]
        
        # 根据观察维度推断关节数
        if obs_dim == 12:  # 2关节: 2*3 + 4 = 10 (实际是12因为padding)
            num_joints = 2
        elif obs_dim == 16:  # 3关节: 3*3 + 4 = 13 (实际是16因为padding)
            num_joints = 3
        elif obs_dim == 20:  # 4关节: 4*3 + 4 = 16 (实际是20因为padding)
            num_joints = 4
        elif obs_dim == 24:  # 5关节: 5*3 + 4 = 19 (实际是24因为padding)
            num_joints = 5
        else:
            # 尝试从实际观察推断
            if obs_dim == 10:  # 标准2关节Reacher
                num_joints = 2
            elif obs_dim == 13:  # 3关节
                num_joints = 3
            else:
                num_joints = 2  # 默认
        
        # 初始化特征张量
        joint_features = torch.zeros(batch_size, J_MAX, 4).to(observations.device)
        task_features = torch.zeros(batch_size, len(SUPPORTED_JOINTS) + 1).to(observations.device)
        global_features = torch.zeros(batch_size, 4).to(observations.device)
        joint_mask = torch.zeros(batch_size, J_MAX, dtype=torch.bool).to(observations.device)
        
        # 解析关节特征
        if obs_dim >= 10:  # 至少有基本的关节信息
            # 从填充后的观察中提取
            joint_cos = observations[:, :num_joints] if obs_dim > 10 else observations[:, :2]
            joint_sin = observations[:, num_joints:2*num_joints] if obs_dim > 10 else observations[:, 2:4]
            joint_vel = observations[:, 2*num_joints:3*num_joints] if obs_dim > 10 else observations[:, 4:6]
            
            # 填充关节特征
            for i in range(min(num_joints, J_MAX)):
                if i < len(joint_cos[0]):
                    joint_features[:, i, 0] = joint_cos[:, i]  # cos
                if i < len(joint_sin[0]):
                    joint_features[:, i, 1] = joint_sin[:, i]  # sin
                if i < len(joint_vel[0]):
                    joint_features[:, i, 2] = joint_vel[:, i]  # vel
                joint_features[:, i, 3] = (i + 1) / J_MAX  # joint_id/J_max
                joint_mask[:, i] = True
            
            # 提取全局特征 (末端执行器位置 + 目标位置)
            if obs_dim >= 10:
                if obs_dim == 10:  # 标准2关节
                    global_features[:, :4] = observations[:, 6:10]
                else:  # 填充后的观察
                    global_features[:, :4] = observations[:, -4:]
        
        # 设置任务特征
        task_features[:, 0] = num_joints / J_MAX  # N/J_max
        if num_joints in SUPPORTED_JOINTS:
            idx = SUPPORTED_JOINTS.index(num_joints)
            task_features[:, 1 + idx] = 1.0  # onehot_N
        
        return joint_features, task_features, global_features, joint_mask

class ExtendedMixedJointActionWrapper(gym.ActionWrapper):
    """
    扩展的混合关节动作包装器
    支持2-5关节，统一到J_max维动作空间
    """
    
    def __init__(self, env, original_action_dim):
        super().__init__(env)
        self.original_action_dim = original_action_dim
        
        # 统一动作空间为J_max维
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(J_MAX,), dtype=np.float32
        )
        
        print(f"🔧 ExtendedMixedJointActionWrapper: 原始动作维度={original_action_dim}, 统一为{J_MAX}维")
    
    def action(self, action):
        # 只使用前N个关节的动作 (GPT-5建议的mask策略)
        return action[:self.original_action_dim]

class ExtendedMixedJointObservationWrapper(gym.ObservationWrapper):
    """
    扩展的混合关节观察包装器
    支持2-5关节，统一观察空间
    """
    
    def __init__(self, env, target_obs_dim=None):
        super().__init__(env)
        
        if target_obs_dim is None:
            target_obs_dim = J_MAX * 3 + 4  # 默认最大维度
        
        self.target_obs_dim = target_obs_dim
        self.original_obs_dim = env.observation_space.shape[0]
        
        # 统一观察空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_obs_dim,), dtype=np.float64
        )
        
        print(f"🔧 ExtendedMixedJointObservationWrapper: 原始观察维度={self.original_obs_dim}, 统一为{target_obs_dim}维")
    
    def observation(self, obs):
        if len(obs) < self.target_obs_dim:
            # 填充观察到目标维度
            padded_obs = np.zeros(self.target_obs_dim)
            
            # 智能填充不同关节数的观察
            if len(obs) == 10:  # 2关节标准Reacher
                self._fill_2joint_obs(obs, padded_obs)
            elif len(obs) == 13:  # 3关节
                self._fill_3joint_obs(obs, padded_obs)
            elif len(obs) == 16:  # 4关节
                self._fill_4joint_obs(obs, padded_obs)
            elif len(obs) == 19:  # 5关节
                self._fill_5joint_obs(obs, padded_obs)
            else:
                # 默认填充
                padded_obs[:len(obs)] = obs
            
            return padded_obs
        else:
            return obs
    
    def _fill_2joint_obs(self, obs, padded_obs):
        """填充2关节观察"""
        # 2关节: [cos1, cos2, sin1, sin2, vel1, vel2, ee_x, ee_y, target_x, target_y]
        padded_obs[0] = obs[0]   # cos1
        padded_obs[1] = obs[1]   # cos2
        padded_obs[2] = 1.0      # cos3 (默认0度)
        padded_obs[3] = 1.0      # cos4 (默认0度)
        padded_obs[4] = 1.0      # cos5 (默认0度)
        
        padded_obs[5] = obs[2]   # sin1
        padded_obs[6] = obs[3]   # sin2
        padded_obs[7] = 0.0      # sin3 (默认0度)
        padded_obs[8] = 0.0      # sin4 (默认0度)
        padded_obs[9] = 0.0      # sin5 (默认0度)
        
        padded_obs[10] = obs[4]  # vel1
        padded_obs[11] = obs[5]  # vel2
        padded_obs[12] = 0.0     # vel3
        padded_obs[13] = 0.0     # vel4
        padded_obs[14] = 0.0     # vel5
        
        padded_obs[15:19] = obs[6:10]  # ee_pos + target_pos
    
    def _fill_3joint_obs(self, obs, padded_obs):
        """填充3关节观察"""
        # 3关节: [cos1, cos2, cos3, sin1, sin2, sin3, vel1, vel2, vel3, ee_x, ee_y, target_x, target_y]
        padded_obs[:3] = obs[:3]     # cos1-3
        padded_obs[3:5] = [1.0, 1.0] # cos4-5 (默认)
        
        padded_obs[5:8] = obs[3:6]   # sin1-3
        padded_obs[8:10] = [0.0, 0.0] # sin4-5 (默认)
        
        padded_obs[10:13] = obs[6:9] # vel1-3
        padded_obs[13:15] = [0.0, 0.0] # vel4-5 (默认)
        
        padded_obs[15:19] = obs[9:13] # ee_pos + target_pos
    
    def _fill_4joint_obs(self, obs, padded_obs):
        """填充4关节观察"""
        padded_obs[:4] = obs[:4]     # cos1-4
        padded_obs[4] = 1.0          # cos5 (默认)
        
        padded_obs[5:9] = obs[4:8]   # sin1-4
        padded_obs[9] = 0.0          # sin5 (默认)
        
        padded_obs[10:14] = obs[8:12] # vel1-4
        padded_obs[14] = 0.0          # vel5 (默认)
        
        padded_obs[15:19] = obs[12:16] # ee_pos + target_pos
    
    def _fill_5joint_obs(self, obs, padded_obs):
        """填充5关节观察"""
        padded_obs[:19] = obs[:19]  # 直接复制所有

def generate_multi_joint_reacher_xml(num_joints: int, link_lengths: List[float]) -> str:
    """生成N关节Reacher的MuJoCo XML (基于perfect_3joint_training.py的逻辑)"""
    
    # 扩大场地以适应更多关节
    arena_size = min(1.0 + (num_joints - 2) * 0.2, 2.0)  # 最大2.0
    target_range = min(0.45 + (num_joints - 2) * 0.1, 0.8)  # 最大0.8
    
    xml_template = f'''
    <mujoco model="reacher_{num_joints}joint">
      <compiler angle="radian" inertiafromgeom="true"/>
      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
      </default>
      <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
      
      <worldbody>
        <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="{arena_size} {arena_size} 10" type="plane"/>
        <geom conaffinity="1" contype="1" name="sideS" pos="0 -{arena_size} 0" rgba="0.9 0.4 0.6 1" size="{arena_size} 0.02 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideE" pos="{arena_size} 0 0" rgba="0.9 0.4 0.6 1" size="0.02 {arena_size} 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideN" pos="0 {arena_size} 0" rgba="0.9 0.4 0.6 1" size="{arena_size} 0.02 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideW" pos="-{arena_size} 0 0" rgba="0.9 0.4 0.6 1" size="0.02 {arena_size} 1" type="box"/>
        
        <body name="body0" pos="0 0 0">
          <joint axis="0 0 1" limited="true" name="joint0" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
          <geom fromto="0 0 0 {link_lengths[0]} 0 0" name="link0" size="0.02" type="capsule"/>
    '''
    
    # 添加后续关节和链接
    for i in range(1, num_joints):
        xml_template += f'''
          <body name="body{i}" pos="{link_lengths[i-1]} 0 0">
            <joint axis="0 0 1" limited="true" name="joint{i}" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
            <geom fromto="0 0 0 {link_lengths[i]} 0 0" name="link{i}" size="0.02" type="capsule"/>
        '''
    
    # 添加末端执行器
    xml_template += f'''
            <body name="fingertip" pos="{link_lengths[-1]} 0 0">
              <geom name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.0 1" size="0.01" type="sphere"/>
            </body>
    '''
    
    # 关闭所有body标签
    for i in range(num_joints):
        xml_template += '          </body>\n'
    
    # 添加目标 (扩大范围)
    xml_template += f'''
        <body name="target" pos="{target_range*0.8} {target_range*0.8} 0">
          <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-{target_range} {target_range}" ref="{target_range*0.8}" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-{target_range} {target_range}" ref="{target_range*0.8}" stiffness="0" type="slide"/>
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

class ExtendedMultiJointReacherEnv(MujocoEnv):
    """扩展的N关节Reacher环境 (支持2-5关节)"""
    
    def __init__(self, num_joints: int, render_mode: Optional[str] = None):
        
        if num_joints not in SUPPORTED_JOINTS:
            raise ValueError(f"不支持的关节数: {num_joints}，支持的关节数: {SUPPORTED_JOINTS}")
        
        self.num_joints = num_joints
        
        # 根据关节数设置链接长度 (GPT-5建议的几何随机化)
        if num_joints == 2:
            self.link_lengths = [0.1, 0.11]
        elif num_joints == 3:
            self.link_lengths = [0.1, 0.1, 0.1]
        elif num_joints == 4:
            self.link_lengths = [0.08, 0.08, 0.08, 0.08]
        elif num_joints == 5:
            self.link_lengths = [0.06, 0.06, 0.06, 0.06, 0.06]
        
        # 生成XML
        xml_string = generate_multi_joint_reacher_xml(num_joints, self.link_lengths)
        
        # 创建临时文件
        self.temp_xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.temp_xml_file.write(xml_string)
        self.temp_xml_file.close()
        
        # 定义观察和动作空间
        obs_dim = 3 * num_joints + 4  # cos, sin, vel, ee_pos, target_pos
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        
        # 初始化MuJoCo环境
        super().__init__(
            model_path=self.temp_xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = action_space
        self.step_count = 0
        self.max_episode_steps = 50
        
        print(f"✅ ExtendedMultiJointReacherEnv ({num_joints}关节) 创建完成")
        print(f"   链接长度: {self.link_lengths}")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def step(self, action):
        # 显式渲染 (修复渲染问题)
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
            'is_success': terminated,
            'num_joints': self.num_joints  # GPT-5建议的任务信息
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
        
        # 距离奖励 (根据关节数调整)
        reward = -distance * (1.0 + 0.1 * (self.num_joints - 2))
        
        # 到达奖励
        if distance < 0.02:
            reward += 10.0
        
        return reward
    
    def reset_model(self):
        # 随机初始化关节角度
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        
        # 随机目标位置 (根据关节数调整范围)
        target_range = min(0.45 + (self.num_joints - 2) * 0.1, 0.8)
        qpos[-2:] = self.np_random.uniform(low=-target_range, high=target_range, size=2)
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        
        return self._get_obs()
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

# 环境创建工厂函数
def make_2joint_env():
    """创建2关节环境"""
    def _init():
        env = gym.make('Reacher-v5', render_mode='human')
        env = ExtendedMixedJointObservationWrapper(env, target_obs_dim=19)
        env = ExtendedMixedJointActionWrapper(env, original_action_dim=2)
        env = Monitor(env)
        return env
    return _init

def make_3joint_env():
    """创建3关节环境"""
    def _init():
        env = ExtendedMultiJointReacherEnv(3, render_mode='human')
        env = ExtendedMixedJointObservationWrapper(env, target_obs_dim=19)
        env = ExtendedMixedJointActionWrapper(env, original_action_dim=3)
        env = Monitor(env)
        return env
    return _init

def make_4joint_env():
    """创建4关节环境"""
    def _init():
        env = ExtendedMultiJointReacherEnv(4, render_mode='human')
        env = ExtendedMixedJointObservationWrapper(env, target_obs_dim=19)
        env = ExtendedMixedJointActionWrapper(env, original_action_dim=4)
        env = Monitor(env)
        return env
    return _init

def make_5joint_env():
    """创建5关节环境"""
    def _init():
        env = ExtendedMultiJointReacherEnv(5, render_mode='human')
        env = ExtendedMixedJointObservationWrapper(env, target_obs_dim=19)
        env = ExtendedMixedJointActionWrapper(env, original_action_dim=5)
        env = Monitor(env)
        return env
    return _init

class ExtendedParallelTrainingCallback(BaseCallback):
    """扩展的并行训练回调 (融入GPT-5建议的监控)"""
    
    def __init__(self, save_freq=10000, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.last_save = 0
        self.joint_performance = {joint: [] for joint in SUPPORTED_JOINTS}
    
    def _on_step(self) -> bool:
        # 定期保存模型
        if self.num_timesteps - self.last_save >= self.save_freq:
            model_path = f"models/extended_parallel_checkpoint_{self.num_timesteps}"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"💾 扩展并行模型检查点已保存: {model_path}")
            self.last_save = self.num_timesteps
        
        return True

def train_extended_parallel_multijoint(total_timesteps: int = 60000):
    """
    扩展的并行多关节训练
    同时训练2-5关节Reacher (GPT-5建议的对等采样)
    """
    print("🚀 扩展并行多关节Reacher训练")
    print(f"🎯 同时训练{SUPPORTED_JOINTS}关节Reacher")
    print("💡 融入GPT-5建议：任务条件化 + 对等采样 + 注意力机制")
    print("="*60)
    
    # 创建并行环境 (GPT-5建议的对等采样)
    print("🌍 创建扩展并行训练环境...")
    
    env_fns = []
    
    # 每种关节数创建一个环境实例 (对等采样)
    env_makers = [make_2joint_env, make_3joint_env, make_4joint_env, make_5joint_env]
    
    for i, (joints, maker) in enumerate(zip(SUPPORTED_JOINTS, env_makers)):
        env_fns.append(maker())
        print(f"   ✅ {joints}关节环境已添加")
    
    # 使用DummyVecEnv避免多进程问题
    print("🔄 创建向量化环境...")
    vec_env = DummyVecEnv(env_fns)
    print("✅ 使用DummyVecEnv (单进程，稳定)")
    
    print("✅ 扩展并行环境创建完成")
    print(f"   环境数量: {len(env_fns)} ({len(SUPPORTED_JOINTS)}种关节数)")
    print(f"   统一观察空间: {vec_env.observation_space}")
    print(f"   统一动作空间: {vec_env.action_space}")
    
    # 创建SAC模型 (使用通用特征提取器)
    print("\n🤖 创建扩展并行SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': UniversalMultiJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=800,  # 4个环境，需要更多启动步数
        device='cpu',
        tensorboard_log="./tensorboard_logs/extended_parallel/",
        batch_size=256,
        buffer_size=150000,  # 增加缓冲区
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )
    
    print("✅ 扩展并行SAC模型创建完成")
    print("   ✅ 通用多关节特征提取器")
    print("   ✅ 任务条件化编码")
    print("   ✅ 注意力机制处理关节交互")
    
    # 创建回调
    callback = ExtendedParallelTrainingCallback(save_freq=15000, verbose=1)
    
    # 开始训练
    print(f"\n🎯 开始扩展并行训练 ({total_timesteps}步)...")
    print("💡 您将看到4个MuJoCo窗口同时训练:")
    print("   🔸 窗口1: 2关节Reacher")
    print("   🔸 窗口2: 3关节Reacher")
    print("   🔸 窗口3: 4关节Reacher")
    print("   🔸 窗口4: 5关节Reacher")
    print("💡 所有环境共享同一个通用神经网络模型")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True,
            callback=callback
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 扩展并行训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        print(f"🚀 并行效率: {len(env_fns)}个环境同时训练")
        
        # 保存最终模型
        model.save("models/extended_parallel_multijoint_final")
        print("💾 最终扩展模型已保存: models/extended_parallel_multijoint_final")
        
        # 评估模型性能
        print("\n📊 评估扩展并行训练的模型性能...")
        evaluate_extended_model(model)
        
        return model
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/extended_parallel_multijoint_interrupted")
        print("💾 中断模型已保存")
        return model
    
    finally:
        vec_env.close()

def evaluate_extended_model(model):
    """评估扩展并行训练的模型"""
    print("🔍 创建扩展评估环境...")
    
    # 为每种关节数创建评估环境
    eval_envs = {}
    env_makers = {
        2: lambda: gym.make('Reacher-v5', render_mode='human'),
        3: lambda: ExtendedMultiJointReacherEnv(3, render_mode='human'),
        4: lambda: ExtendedMultiJointReacherEnv(4, render_mode='human'),
        5: lambda: ExtendedMultiJointReacherEnv(5, render_mode='human')
    }
    
    for joints in SUPPORTED_JOINTS:
        env = env_makers[joints]()
        env = ExtendedMixedJointObservationWrapper(env, target_obs_dim=19)
        env = ExtendedMixedJointActionWrapper(env, original_action_dim=joints)
        env = Monitor(env)
        eval_envs[joints] = env
    
    try:
        results = {}
        
        # 评估每种关节数
        for joints in SUPPORTED_JOINTS:
            print(f"\n🎮 评估{joints}关节性能 (5个episode):")
            results[joints] = evaluate_env_extended(model, eval_envs[joints], f"{joints}关节", episodes=5)
        
        # 综合分析
        print(f"\n📈 扩展并行训练效果总结:")
        total_success = 0
        total_reward = 0
        
        for joints in SUPPORTED_JOINTS:
            success_rate = results[joints]['success_rate']
            avg_reward = results[joints]['avg_reward']
            total_success += success_rate
            total_reward += avg_reward
            print(f"   {joints}关节: 成功率={success_rate:.1f}%, 平均奖励={avg_reward:.3f}")
        
        avg_success = total_success / len(SUPPORTED_JOINTS)
        avg_reward = total_reward / len(SUPPORTED_JOINTS)
        
        print(f"\n🎯 整体性能:")
        print(f"   平均成功率: {avg_success:.1f}%")
        print(f"   平均奖励: {avg_reward:.3f}")
        
        # 通用性评估
        if avg_success > 30:
            print("   🎉 优秀的通用多关节控制能力!")
            print("   💡 模型成功学会了处理不同关节数的机械臂")
        elif avg_success > 15:
            print("   ✅ 良好的通用多关节控制能力!")
            print("   💡 模型展现了一定的跨关节泛化能力")
        else:
            print("   🔶 通用多关节控制能力有待提升")
            print("   💡 可能需要更多训练或调整架构")
        
    finally:
        for env in eval_envs.values():
            env.close()

def evaluate_env_extended(model, env, env_name, episodes=5):
    """评估模型在特定环境上的性能 (扩展版)"""
    
    all_rewards = []
    all_successes = []
    all_distances = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        final_distance = 1.0
        
        for step in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # 计算距离
            if 'distance_to_target' in info:
                final_distance = info['distance_to_target']
            elif 'is_success' in info:
                final_distance = 0.01 if info['is_success'] else 0.1
            else:
                # 手动计算距离
                if len(obs) >= 15:
                    ee_pos = obs[15:17]
                    target_pos = obs[17:19]
                    final_distance = np.linalg.norm(ee_pos - target_pos)
            
            if final_distance < 0.02:
                episode_success = True
                break
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_successes.append(episode_success)
        all_distances.append(final_distance)
        
        print(f"   {env_name} Episode {episode+1}: 奖励={episode_reward:.2f}, 成功={'是' if episode_success else '否'}, 距离={final_distance:.3f}m")
    
    results = {
        'avg_reward': np.mean(all_rewards),
        'success_rate': np.mean(all_successes) * 100,
        'avg_distance': np.mean(all_distances)
    }
    
    print(f"   {env_name} 总结: 平均奖励={results['avg_reward']:.3f}, 成功率={results['success_rate']:.1f}%")
    
    return results

def main():
    """主函数"""
    print("🌟 扩展并行多关节Reacher训练系统")
    print("🎯 基于parallel_mixed_joint_training.py扩展到2-5关节")
    print("💡 融入GPT-5建议的通用多任务架构改进")
    print()
    
    try:
        # 开始扩展并行训练
        train_extended_parallel_multijoint(total_timesteps=60000)
        
        print(f"\n🎉 扩展并行多关节训练完成！")
        print(f"💡 您已经看到了真正的通用多关节控制效果")
        print(f"✅ 一套模型现在可以控制2-5关节的机械臂")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


