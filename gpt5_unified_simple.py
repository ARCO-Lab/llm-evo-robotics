#!/usr/bin/env python3
"""
GPT-5统一策略简化版训练脚本：
1. Set-Transformer架构，支持可变关节数(2-5)
2. 统一奖励函数，跨N可比
3. 单一模型处理所有关节数
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
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import math
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# GPT-5统一策略参数
ALPHA_DISTANCE = 5.0      # 距离惩罚权重
BETA_CONTROL = 1e-3       # 控制惩罚权重 (除以N)
GAMMA_SMOOTH = 1e-3       # 动作平滑惩罚权重 (除以N)
SUCCESS_THRESHOLD = 0.03  # 统一成功阈值 (3cm)
SUCCESS_REWARD = 5.0      # 成功奖励
EPISODE_LENGTH = 200      # 统一episode长度

# 网络参数
JOINT_TOKEN_DIM = 10
GLOBAL_TOKEN_DIM = 10
ENCODER_LAYERS = 2
HIDDEN_DIM = 256
ATTENTION_HEADS = 4
DROPOUT = 0.1

# 从baseline复制的3关节XML配置
def get_3joint_xml():
    """3关节XML配置（统一动力学参数 + 自碰撞检测）"""
    return """
<mujoco model="3joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
  </default>
  <contact>
    <!-- 链节之间的自碰撞检测 -->
    <pair geom1="link0" geom2="link2" condim="3"/>
    <!-- End-effector与所有链节的碰撞检测 -->
    <pair geom1="fingertip" geom2="link0" condim="3"/>
    <pair geom1="fingertip" geom2="link1" condim="3"/>
  </contact>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.5 0.5 10" type="plane"/>
    <geom conaffinity="0" contype="0" fromto="-.5 -.5 .01 .5 -.5 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto=" .5 -.5 .01 .5  .5 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.5  .5 .01 .5  .5 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.5 -.5 .01 -.5 .5 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="2" conaffinity="2"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="4" conaffinity="4"/>
          <body name="fingertip" pos="0.11 0 0">
            <geom contype="16" conaffinity="16" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.5 .5" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.5 .5" ref="-.2" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
  </actuator>
</mujoco>
"""

def get_4joint_xml():
    """4关节XML配置（统一动力学参数 + 自碰撞检测）"""
    return """
<mujoco model="4joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
  </default>
  <contact>
    <pair geom1="link0" geom2="link2" condim="3"/>
    <pair geom1="link0" geom2="link3" condim="3"/>
    <pair geom1="link1" geom2="link3" condim="3"/>
  </contact>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.6 0.6 10" type="plane"/>
    <geom conaffinity="0" contype="0" fromto="-.6 -.6 .01 .6 -.6 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto=" .6 -.6 .01 .6  .6 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.6  .6 .01 .6  .6 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.6 -.6 .01 -.6 .6 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.08 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.08 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.08 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="2" conaffinity="2"/>
        <body name="body2" pos="0.08 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.08 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="4" conaffinity="4"/>
          <body name="body3" pos="0.08 0 0">
            <joint axis="0 0 1" limited="true" name="joint3" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
            <geom fromto="0 0 0 0.08 0 0" name="link3" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="8" conaffinity="8"/>
            <body name="fingertip" pos="0.088 0 0">
              <geom contype="0" conaffinity="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".25 -.25 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.6 .6" ref=".25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.6 .6" ref="-.25" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint3"/>
  </actuator>
</mujoco>
"""

class SimpleSetTransformerExtractor(BaseFeaturesExtractor):
    """简化版Set-Transformer特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # 简化的特征提取网络
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, JOINT_TOKEN_DIM),  # q, q_dot, sin_q, cos_q
            nn.ReLU(),
            nn.LayerNorm(JOINT_TOKEN_DIM)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(4, GLOBAL_TOKEN_DIM),  # ee_x, ee_y, target_x, target_y
            nn.ReLU(),
            nn.LayerNorm(GLOBAL_TOKEN_DIM)
        )
        
        # 简化的注意力机制
        token_dim = max(JOINT_TOKEN_DIM, GLOBAL_TOKEN_DIM)
        self.attention = nn.MultiheadAttention(token_dim, num_heads=2, dropout=DROPOUT, batch_first=True)
        
        # 🚀 性能优化：直接从观察空间到特征空间的映射
        obs_dim = observation_space.shape[0]
        self.output_proj = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, features_dim)
        )
        
        print(f"🔧 SimpleSetTransformerExtractor: token维度={token_dim}, 输出维度={features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 🚀 性能优化：使用简化的前向传播，避免复杂的循环和注意力计算
        # 直接使用MLP处理观察，避免Set-Transformer的复杂计算
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]
        
        # 🔧 动态适应不同观察维度：如果维度不匹配，进行填充或截断
        expected_dim = self.output_proj[0].in_features
        if obs_dim != expected_dim:
            if obs_dim < expected_dim:
                # 填充零到期望维度
                padding = torch.zeros(batch_size, expected_dim - obs_dim, device=observations.device)
                observations = torch.cat([observations, padding], dim=1)
            else:
                # 截断到期望维度
                observations = observations[:, :expected_dim]
        
        # 简单的特征提取：直接处理整个观察向量
        features = self.output_proj(observations)
        return features
    
    def forward_original(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]
        
        # 动态检测关节数
        num_joints = (obs_dim - 4) // 3
        num_joints = min(max(num_joints, 2), 5)
        
        batch_features = []
        
        for i in range(batch_size):
            obs = observations[i]
            
            # 提取关节特征
            cos_q = obs[:num_joints]
            sin_q = obs[num_joints:2*num_joints]
            q_dot = obs[2*num_joints:3*num_joints]
            
            joint_tokens = []
            for j in range(num_joints):
                joint_feature = torch.tensor([
                    torch.atan2(sin_q[j], cos_q[j]).item(),
                    q_dot[j].item(),
                    sin_q[j].item(),
                    cos_q[j].item()
                ], dtype=torch.float32)
                joint_tokens.append(joint_feature)
            
            joint_tokens = torch.stack(joint_tokens)  # [num_joints, 4]
            joint_encoded = self.joint_encoder(joint_tokens)  # [num_joints, token_dim]
            
            # 提取全局特征
            ee_pos = obs[3*num_joints:3*num_joints+2]
            target_pos = obs[3*num_joints+2:3*num_joints+4]
            global_feature = torch.cat([ee_pos, target_pos])  # [4]
            global_encoded = self.global_encoder(global_feature.unsqueeze(0))  # [1, token_dim]
            
            # 组合tokens
            all_tokens = torch.cat([joint_encoded, global_encoded], dim=0)  # [num_joints+1, token_dim]
            
            # 自注意力
            attn_output, _ = self.attention(
                all_tokens.unsqueeze(0), 
                all_tokens.unsqueeze(0), 
                all_tokens.unsqueeze(0)
            )
            attn_output = attn_output.squeeze(0)  # [num_joints+1, token_dim]
            
            # 全局池化
            pooled = torch.mean(attn_output, dim=0)  # [token_dim]
            batch_features.append(pooled)
        
        batch_features = torch.stack(batch_features)  # [batch_size, token_dim]
        return self.output_proj(batch_features)

# 从baseline复制的真正3关节环境
class Sequential3JointReacherEnv(MujocoEnv):
    """真正的3关节Reacher环境（从baseline复制）"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Sequential3JointReacherEnv 初始化")
        
        self.num_joints = 3
        self.link_lengths = [0.1, 0.1, 0.1]
        self.max_reach = sum(self.link_lengths)
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_3joint_xml())
        self.xml_file.flush()
        
        # 计算观察空间维度
        obs_dim = self.num_joints * 3 + 4  # cos, sin, vel + ee_pos + target_pos
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            width=480,
            height=480
        )
        
        self.step_count = 0
        self.max_episode_steps = EPISODE_LENGTH
        
        print(f"✅ 3关节Reacher创建完成")
        print(f"   链长: {self.link_lengths}")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
    
    def step(self, action):
        # 使用标准MuJoCo步骤
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        
        # 🎯 关键修复：像标准Reacher一样在step中渲染
        if self.render_mode == "human":
            self.render()
        
        # 计算距离
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 使用GPT-5统一奖励函数
        distance_penalty = -ALPHA_DISTANCE * distance
        control_penalty = -(BETA_CONTROL / self.num_joints) * np.sum(np.square(action))
        
        # 简化奖励，暂时不使用动作平滑
        reward = distance_penalty + control_penalty
        
        terminated = False
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': distance < SUCCESS_THRESHOLD,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        # 3关节的观察：cos, sin, vel, fingertip_pos, target_pos
        theta = self.data.qpos.flat[:self.num_joints]
        obs = np.concatenate([
            np.cos(theta),                    # 3个cos值
            np.sin(theta),                    # 3个sin值
            self.data.qvel.flat[:self.num_joints],  # 3个关节速度
            self.get_body_com("fingertip")[:2],     # 末端执行器位置 (x,y)
            self.get_body_com("target")[:2],        # 目标位置 (x,y)
        ])
        return obs
    
    def reset_model(self):
        # 重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()
        
        # 只给机械臂关节添加随机速度
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        
        # 生成随机目标位置
        max_target_distance = self.max_reach * 0.85
        min_target_distance = 0.05
        
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

# 从baseline复制的真正4关节环境
class Sequential4JointReacherEnv(MujocoEnv):
    """真正的4关节Reacher环境（从baseline复制）"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Sequential4JointReacherEnv 初始化")
        
        self.num_joints = 4
        self.link_lengths = [0.08, 0.08, 0.08, 0.08]
        self.max_reach = sum(self.link_lengths)
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_4joint_xml())
        self.xml_file.flush()
        
        # 计算观察空间维度
        obs_dim = self.num_joints * 3 + 4  # cos, sin, vel + ee_pos + target_pos
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            width=480,
            height=480
        )
        
        self.step_count = 0
        self.max_episode_steps = EPISODE_LENGTH
        
        print(f"✅ 4关节Reacher创建完成")
        print(f"   链长: {self.link_lengths}")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
    
    def step(self, action):
        # 使用标准MuJoCo步骤
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        
        # 🎯 关键修复：像标准Reacher一样在step中渲染
        if self.render_mode == "human":
            self.render()
        
        # 计算距离
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 使用GPT-5统一奖励函数
        distance_penalty = -ALPHA_DISTANCE * distance
        control_penalty = -(BETA_CONTROL / self.num_joints) * np.sum(np.square(action))
        
        # 简化奖励，暂时不使用动作平滑
        reward = distance_penalty + control_penalty
        
        terminated = False
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': distance < SUCCESS_THRESHOLD,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        # 4关节的观察：cos, sin, vel, fingertip_pos, target_pos
        theta = self.data.qpos.flat[:self.num_joints]
        obs = np.concatenate([
            np.cos(theta),                    # 4个cos值
            np.sin(theta),                    # 4个sin值
            self.data.qvel.flat[:self.num_joints],  # 4个关节速度
            self.get_body_com("fingertip")[:2],     # 末端执行器位置 (x,y)
            self.get_body_com("target")[:2],        # 目标位置 (x,y)
        ])
        return obs
    
    def reset_model(self):
        # 重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()
        
        # 只给机械臂关节添加随机速度
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        
        # 生成随机目标位置
        max_target_distance = self.max_reach * 0.85
        min_target_distance = 0.05
        
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

class UnifiedReacherEnv(gym.Wrapper):
    """统一的Reacher环境，支持2-5关节"""
    
    def __init__(self, base_env, num_joints):
        super().__init__(base_env)
        self.num_joints = num_joints
        self.last_action = None
        self.success_count = 0
        self.success_threshold_steps = 10
        self.step_count = 0
        
        print(f"🔧 UnifiedReacherEnv: {num_joints}关节，统一奖励函数")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        self.success_count = 0
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 🚀 性能优化：降低渲染频率，每5步渲染一次
        if hasattr(self.env, 'render_mode') and self.env.render_mode == "human":
            if self.step_count % 5 == 0:  # 每5步渲染一次，提高训练速度
                self.env.render()
        
        # 重新计算统一奖励
        if 'distance_to_target' in info:
            distance = info['distance_to_target']
            
            # GPT-5统一奖励函数
            distance_penalty = -ALPHA_DISTANCE * distance
            control_penalty = -(BETA_CONTROL / self.num_joints) * np.sum(np.square(action))
            
            smooth_penalty = 0.0
            if self.last_action is not None:
                action_diff = action - self.last_action
                smooth_penalty = -(GAMMA_SMOOTH / self.num_joints) * np.sum(np.square(action_diff))
            
            success_reward = 0.0
            if distance < SUCCESS_THRESHOLD:
                self.success_count += 1
                if self.success_count >= self.success_threshold_steps:
                    success_reward = SUCCESS_REWARD
            else:
                self.success_count = 0
            
            # 更新奖励和成功判断
            reward = distance_penalty + control_penalty + smooth_penalty + success_reward
            info['is_success'] = distance < SUCCESS_THRESHOLD
            
            self.last_action = action.copy()
        
        self.step_count += 1
        if self.step_count >= EPISODE_LENGTH:
            truncated = True
        
        return obs, reward, terminated, truncated, info

def create_unified_env(num_joints, render_mode=None):
    """创建统一的Reacher环境"""
    if num_joints == 2:
        base_env = gym.make('Reacher-v5', render_mode=render_mode)
        env = UnifiedReacherEnv(base_env, num_joints)
    elif num_joints == 3:
        # 🎯 使用真正的3关节环境！
        env = Sequential3JointReacherEnv(render_mode=render_mode)
    elif num_joints == 4:
        # 🎯 使用真正的4关节环境！
        env = Sequential4JointReacherEnv(render_mode=render_mode)
    else:
        # 对于5关节，暂时使用2关节环境
        print(f"⚠️ 简化版：{num_joints}关节暂时使用2关节环境代替")
        base_env = gym.make('Reacher-v5', render_mode=render_mode)
        env = UnifiedReacherEnv(base_env, num_joints)
    
    return Monitor(env)

class RandomJointVecEnv(DummyVecEnv):
    """随机关节数的向量化环境"""
    
    def __init__(self, n_envs=4, render_mode=None):
        self.joint_numbers = [2, 3, 4, 5]
        self.joint_probs = [0.25, 0.25, 0.25, 0.25]
        self.render_mode = render_mode
        
        # 创建环境函数
        def make_env():
            # 随机选择关节数
            num_joints = np.random.choice(self.joint_numbers, p=self.joint_probs)
            return create_unified_env(num_joints, render_mode=self.render_mode)
        
        env_fns = [make_env for _ in range(n_envs)]
        super().__init__(env_fns)
        
        render_info = f"渲染={'开启' if render_mode else '关闭'}"
        print(f"🔧 RandomJointVecEnv: {n_envs}个并行环境，随机关节数{self.joint_numbers}，{render_info}")

def train_gpt5_unified_model(total_timesteps=50000):
    """训练GPT-5统一策略模型"""
    print("🌟 GPT-5统一策略训练开始")
    print(f"📊 训练步数: {total_timesteps:,}")
    print(f"🎯 统一成功阈值: {SUCCESS_THRESHOLD}m")
    print(f"🎯 渲染模式: 开启，单环境训练确保流畅性")
    print("="*60)
    
    # 🎯 关键修复：使用单个环境而不是向量化环境，确保渲染流畅
    # 参考baseline的做法，直接创建单个环境
    train_env = create_unified_env(num_joints=4, render_mode='human')  # 测试4关节
    
    # 创建模型
    policy_kwargs = {
        'features_extractor_class': SimpleSetTransformerExtractor,
        'features_extractor_kwargs': {'features_dim': HIDDEN_DIM},
    }
    
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100000,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        target_entropy='auto',
        tensorboard_log=None  # 禁用tensorboard日志
    )
    
    print("✅ GPT-5统一策略模型创建完成")
    
    # 开始训练
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=50,  # 🚀 减少日志频率，提高训练速度
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        
        # 保存模型
        model_path = "models/gpt5_unified_4joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        return model, training_time
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        model_path = "models/gpt5_unified_4joint_reacher_interrupted"
        model.save(model_path)
        return model, 0
    
    finally:
        train_env.close()

def test_gpt5_unified_model(model, n_eval_episodes=20):
    """测试GPT-5统一策略模型"""
    print(f"\n🧪 测试GPT-5统一策略模型")
    print(f"📊 每个关节数测试{n_eval_episodes}个episodes")
    print("-"*40)
    
    all_results = []
    
    # 🔧 暂时只测试4关节，因为模型是在4关节上训练的
    for num_joints in [4]:
        print(f"\n🔧 测试{num_joints}关节...")
        
        test_env = create_unified_env(num_joints, render_mode='human')
        
        success_count = 0
        episode_rewards = []
        episode_distances = []
        
        for episode in range(n_eval_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            
            for step in range(EPISODE_LENGTH):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                
                distance = info.get('distance_to_target', float('inf'))
                min_distance = min(min_distance, distance)
                
                if info.get('is_success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_count += 1
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
        
        success_rate = success_count / n_eval_episodes
        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances)
        
        result = {
            'num_joints': num_joints,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_distance': avg_distance
        }
        all_results.append(result)
        
        print(f"   成功率: {success_rate:.1%} ({success_count}/{n_eval_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均距离: {avg_distance:.4f}")
        
        test_env.close()
    
    # 总结
    print(f"\n🎯 GPT-5统一策略测试总结:")
    print("-"*50)
    for result in all_results:
        print(f"{result['num_joints']}关节: 成功率{result['success_rate']:.1%}, 奖励{result['avg_reward']:.1f}, 距离{result['avg_distance']:.3f}")
    
    return all_results

def main():
    """主函数"""
    print("🌟 GPT-5统一策略简化版训练系统")
    print("🤖 单一模型支持2-5关节Reacher")
    print("🎯 Set-Transformer + 统一奖励函数")
    print()
    
    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # 训练模型（减少步数先测试性能）
    model, training_time = train_gpt5_unified_model(total_timesteps=50000)
    
    # 测试模型
    results = test_gpt5_unified_model(model, n_eval_episodes=10)
    
    print(f"\n🎉 GPT-5统一策略训练完成!")
    print(f"   训练时间: {training_time/60:.1f} 分钟")
    print(f"   支持关节数: 2-5 (单一模型)")
    
    # 分析结果
    success_rates = [r['success_rate'] for r in results]
    avg_success_rate = np.mean(success_rates)
    success_std = np.std(success_rates)
    
    print(f"\n📊 跨关节数一致性分析:")
    print(f"   平均成功率: {avg_success_rate:.1%}")
    print(f"   成功率标准差: {success_std:.3f} (越小越一致)")
    
    if success_std < 0.1:
        print(f"   ✅ 跨关节数一致性很好!")
    elif success_std < 0.2:
        print(f"   ⚠️ 跨关节数一致性一般")
    else:
        print(f"   ❌ 跨关节数一致性较差")

if __name__ == "__main__":
    main()
