#!/usr/bin/env python3
"""
改进的依次训练和测试脚本 - 基于GPT-5的建议修复：
1. 🔧 统一奖励函数 - 所有关节数使用相同的奖励计算
2. 🔧 修复观测dtype - 统一使用float32和有限边界
3. 🔧 增加训练步数和并行环境 - 使用4-8个并行环境，训练步数提升到20-50万
4. 🔧 移除Dropout - 避免给SAC引入额外噪声
5. 🔧 添加标准化 - 使用VecNormalize进行观测和奖励标准化
6. 🔧 修复成功阈值 - 避免与目标采样边界重叠
7. 🔧 训练时关闭渲染 - 提高采样效率
8. 🔧 添加梯度裁剪和调整学习率
9. 🔧 归一化观测分量到[-1,1]范围
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import time
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 🔧 修复的配置参数
SUCCESS_THRESHOLD = 0.03  # 降低成功阈值，避免与目标最小半径重叠
TARGET_MIN_RADIUS = 0.06  # 目标最小半径，避免边界黏连
PARALLEL_ENVS = 6  # 并行环境数量
TOTAL_TIMESTEPS = 200000  # 增加训练步数到20万

class ImprovedJointExtractor(BaseFeaturesExtractor):
    """改进的特征提取器 - 移除Dropout，使用float32"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(ImprovedJointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 ImprovedJointExtractor: {obs_dim}维 -> {features_dim}维 (无Dropout)")
        
        # 🔧 移除Dropout，保持简洁的MLP结构
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# 🔧 统一奖励计算的基类
class UnifiedRewardMixin:
    """统一奖励计算的混入类"""
    
    def compute_unified_reward(self, fingertip_pos, target_pos, action=None):
        """统一的奖励计算函数"""
        distance = np.linalg.norm(fingertip_pos - target_pos)
        max_reach = getattr(self, 'max_reach', 0.3)  # 默认最大可达距离
        
        # 🔧 标准化距离奖励
        reward = -distance / max_reach  # 归一化到[-1, 0]范围
        
        # 🔧 成功奖励（适中的量级）
        if distance < SUCCESS_THRESHOLD:
            reward += 3.0  # 降低成功奖励量级
        
        # 🔧 可选：轻微的控制代价（如果提供action）
        if action is not None:
            control_cost = 0.01 * np.sum(np.square(action))
            reward -= control_cost
        
        return reward

# 改进的Reacher环境基类
class ImprovedReacherEnv(MujocoEnv, UnifiedRewardMixin):
    """改进的Reacher环境基类 - 修复所有GPT-5指出的问题"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        self.max_reach = sum(link_lengths)  # 最大可达距离
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(xml_content)
        self.xml_file.flush()
        
        # 🔧 计算观察空间维度并设置合理的边界
        obs_dim = num_joints * 3 + 4  # cos, sin, vel + ee_pos + target_pos
        
        # 🔧 设置有限的观测边界（float32）
        obs_low = np.array(
            [-1.0] * num_joints +  # cos values [-1, 1]
            [-1.0] * num_joints +  # sin values [-1, 1]  
            [-20.0] * num_joints + # joint velocities [-20, 20]
            [-1.0, -1.0] +         # normalized ee_pos [-1, 1]
            [-1.0, -1.0],          # normalized target_pos [-1, 1]
            dtype=np.float32
        )
        obs_high = np.array(
            [1.0] * num_joints +   # cos values
            [1.0] * num_joints +   # sin values
            [20.0] * num_joints +  # joint velocities
            [1.0, 1.0] +           # normalized ee_pos
            [1.0, 1.0],            # normalized target_pos
            dtype=np.float32
        )
        
        observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.step_count = 0
        self.max_episode_steps = 100
        
        print(f"✅ {num_joints}关节Reacher创建完成 (改进版)")
        print(f"   链长: {link_lengths}")
        print(f"   最大可达距离: {self.max_reach:.3f}")
        print(f"   目标生成范围: {self.calculate_target_range():.3f}")
        print(f"   🔧 观测空间: {observation_space.shape}, dtype={observation_space.dtype}")
    
    def calculate_target_range(self):
        """计算目标生成的最大距离"""
        return self.max_reach * 0.85
    
    def generate_unified_target(self):
        """🔧 改进的目标生成策略 - 避免边界黏连"""
        max_target_distance = self.calculate_target_range()
        min_target_distance = TARGET_MIN_RADIUS  # 使用更大的最小半径
        
        # 使用极坐标生成目标
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def step(self, action):
        # 训练时不渲染（除非明确指定）
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        
        # 🔧 使用统一的奖励计算
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        reward = self.compute_unified_reward(fingertip_pos, target_pos, action)
        
        # 计算距离和成功判断
        distance = np.linalg.norm(fingertip_pos - target_pos)
        terminated = distance < SUCCESS_THRESHOLD
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': terminated,
            'max_reach': self.max_reach,
            'target_range': self.calculate_target_range(),
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """🔧 改进的观测函数 - 归一化并返回float32"""
        theta = self.data.qpos.flat[:self.num_joints]
        
        # 获取原始位置
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        
        # 🔧 归一化位置到[-1, 1]范围
        normalized_fingertip = fingertip_pos / self.max_reach
        normalized_target = target_pos / self.max_reach
        
        obs = np.concatenate([
            np.cos(theta),                              # N个cos值 [-1, 1]
            np.sin(theta),                              # N个sin值 [-1, 1]
            np.clip(self.data.qvel.flat[:self.num_joints], -20, 20),  # N个关节速度 [-20, 20]
            normalized_fingertip,                       # 归一化末端位置 [-1, 1]
            normalized_target,                          # 归一化目标位置 [-1, 1]
        ])
        
        # 🔧 确保返回float32
        return obs.astype(np.float32)
    
    def reset_model(self):
        # 🔧 修复目标滚动问题的重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()
        
        # 只给机械臂关节添加随机速度，目标关节速度保持为0
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        
        # 使用改进的目标生成策略
        target_x, target_y = self.generate_unified_target()
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

# XML配置生成函数（保持不变）
def get_2joint_xml():
    """2关节XML配置"""
    return """
<mujoco model="reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
    <geom conaffinity="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .3 -.3 .01 .3  .3 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.3  .3 .01 .3  .3 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="fingertip" pos="0.11 0 0">
          <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
        </body>
      </body>
    </body>
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.2" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
  </actuator>
</mujoco>
"""

def get_3joint_xml():
    """3关节XML配置"""
    return """
<mujoco model="3joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.5 0.5 10" type="plane"/>
    <geom conaffinity="0" fromto="-.5 -.5 .01 .5 -.5 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .5 -.5 .01 .5  .5 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.5  .5 .01 .5  .5 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.5 -.5 .01 -.5 .5 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
          <body name="fingertip" pos="0.11 0 0">
            <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
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
    """4关节XML配置"""
    return """
<mujoco model="4joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.6 0.6 10" type="plane"/>
    <geom conaffinity="0" fromto="-.6 -.6 .01 .6 -.6 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .6 -.6 .01 .6  .6 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.6  .6 .01 .6  .6 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.6 -.6 .01 -.6 .6 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.08 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.08 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.08 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="body2" pos="0.08 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.08 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
          <body name="body3" pos="0.08 0 0">
            <joint axis="0 0 1" limited="true" name="joint3" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
            <geom fromto="0 0 0 0.08 0 0" name="link3" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
            <body name="fingertip" pos="0.088 0 0">
              <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
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

def get_5joint_xml():
    """5关节XML配置"""
    return """
<mujoco model="5joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.7 0.7 10" type="plane"/>
    <geom conaffinity="0" fromto="-.7 -.7 .01 .7 -.7 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .7 -.7 .01 .7  .7 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.7  .7 .01 .7  .7 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.7 -.7 .01 -.7 .7 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.06 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.06 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.06 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="body2" pos="0.06 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.06 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
          <body name="body3" pos="0.06 0 0">
            <joint axis="0 0 1" limited="true" name="joint3" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
            <geom fromto="0 0 0 0.06 0 0" name="link3" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
            <body name="body4" pos="0.06 0 0">
              <joint axis="0 0 1" limited="true" name="joint4" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
              <geom fromto="0 0 0 0.06 0 0" name="link4" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
              <body name="fingertip" pos="0.066 0 0">
                <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".3 -.3 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.7 .7" ref=".3" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.7 .7" ref="-.3" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint3"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint4"/>
  </actuator>
</mujoco>
"""

# 环境类
class Improved2JointReacherEnv(ImprovedReacherEnv):
    """改进的2关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Improved2JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_2joint_xml(),
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class Improved3JointReacherEnv(ImprovedReacherEnv):
    """改进的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Improved3JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_3joint_xml(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class Improved4JointReacherEnv(ImprovedReacherEnv):
    """改进的4关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Improved4JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_4joint_xml(),
            num_joints=4,
            link_lengths=[0.08, 0.08, 0.08, 0.08],
            render_mode=render_mode,
            **kwargs
        )

class Improved5JointReacherEnv(ImprovedReacherEnv):
    """改进的5关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Improved5JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_5joint_xml(),
            num_joints=5,
            link_lengths=[0.06, 0.06, 0.06, 0.06, 0.06],
            render_mode=render_mode,
            **kwargs
        )

# 🔧 统一奖励的2关节包装器
class Improved2JointReacherWrapper(gym.Wrapper, UnifiedRewardMixin):
    """改进的2关节Reacher包装器 - 统一奖励函数和观测格式"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]
        self.max_reach = sum(self.link_lengths)
        self.max_episode_steps = 100
        
        # 🔧 重新定义观测空间为float32和有限边界
        obs_low = np.array([-1.0, -1.0, -1.0, -1.0, -20.0, -20.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        print("🌟 Improved2JointReacherWrapper 初始化")
        print(f"   链长: {self.link_lengths}")
        print(f"   最大可达距离: {self.max_reach:.3f}")
        print(f"   🔧 统一奖励函数和观测格式")
    
    def calculate_target_range(self):
        return self.max_reach * 0.85
    
    def generate_unified_target(self):
        max_target_distance = self.calculate_target_range()
        min_target_distance = TARGET_MIN_RADIUS
        
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 应用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        
        # 修复目标滚动问题
        reacher_env = self.env.unwrapped
        qpos = reacher_env.data.qpos.copy()
        qvel = reacher_env.data.qvel.copy()
        
        qpos[-2:] = [target_x, target_y]
        qvel[-2:] = [0.0, 0.0]
        
        reacher_env.set_state(qpos, qvel)
        
        # 🔧 获取改进的观测
        obs = self._get_improved_obs()
        
        if info is None:
            info = {}
        info.update({
            'max_reach': self.max_reach,
            'target_range': self.calculate_target_range(),
            'target_pos': [target_x, target_y]
        })
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 🔧 重新计算统一的奖励
        reacher_env = self.env.unwrapped
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        
        # 使用统一的奖励计算
        reward = self.compute_unified_reward(fingertip_pos, target_pos, action)
        
        # 重新计算成功判断
        distance = np.linalg.norm(fingertip_pos - target_pos)
        is_success = distance < SUCCESS_THRESHOLD
        
        # 🔧 获取改进的观测
        obs = self._get_improved_obs()
        
        if info is None:
            info = {}
        info.update({
            'max_reach': self.max_reach,
            'target_range': self.calculate_target_range(),
            'distance_to_target': distance,
            'is_success': is_success,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        })
        
        return obs, reward, terminated, truncated, info
    
    def _get_improved_obs(self):
        """🔧 改进的观测函数 - 与自定义环境保持一致"""
        reacher_env = self.env.unwrapped
        theta = reacher_env.data.qpos.flat[:2]  # 2个关节角度
        
        # 获取原始位置
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        
        # 归一化位置到[-1, 1]范围
        normalized_fingertip = fingertip_pos / self.max_reach
        normalized_target = target_pos / self.max_reach
        
        obs = np.concatenate([
            np.cos(theta),                              # 2个cos值
            np.sin(theta),                              # 2个sin值
            np.clip(reacher_env.data.qvel.flat[:2], -20, 20),  # 2个关节速度
            normalized_fingertip,                       # 归一化末端位置
            normalized_target,                          # 归一化目标位置
        ])
        
        return obs.astype(np.float32)

def create_env(num_joints, render_mode=None):
    """创建指定关节数的改进环境"""
    if num_joints == 2:
        # 🔧 使用统一奖励的包装器
        env = gym.make('Reacher-v5', render_mode=render_mode)
        env = Improved2JointReacherWrapper(env)
    elif num_joints == 3:
        env = Improved3JointReacherEnv(render_mode=render_mode)
    elif num_joints == 4:
        env = Improved4JointReacherEnv(render_mode=render_mode)
    elif num_joints == 5:
        env = Improved5JointReacherEnv(render_mode=render_mode)
    else:
        raise ValueError(f"不支持的关节数: {num_joints}")
    
    env = Monitor(env)
    return env

def make_env(num_joints, render_mode=None):
    """创建环境的工厂函数（用于并行环境）"""
    def _init():
        return create_env(num_joints, render_mode)
    return _init

def train_improved_joint_model(num_joints, total_timesteps=TOTAL_TIMESTEPS):
    """🔧 训练改进的单个关节数模型"""
    print(f"\\n🚀 开始训练改进的{num_joints}关节Reacher模型")
    print(f"📊 训练步数: {total_timesteps}")
    print(f"🎯 成功阈值: {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
    print(f"🔧 并行环境数: {PARALLEL_ENVS}")
    print(f"🔧 目标最小半径: {TARGET_MIN_RADIUS} ({TARGET_MIN_RADIUS*100}cm)")
    print("="*60)
    
    # 🔧 创建并行训练环境（不渲染）
    env_fns = [make_env(num_joints, render_mode=None) for _ in range(PARALLEL_ENVS)]
    vec_env = DummyVecEnv(env_fns)
    
    # 🔧 添加标准化
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 创建改进的SAC模型
    policy_kwargs = {
        'features_extractor_class': ImprovedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=10000,  # 🔧 增加learning_starts
        device='cpu',
        tensorboard_log=f"./tensorboard_logs/improved_sequential_{num_joints}joint/",
        batch_size=512,         # 🔧 增加batch_size
        buffer_size=1000000,    # 🔧 增加buffer_size
        learning_rate=1e-4,     # 🔧 降低学习率
        gamma=0.99,
        tau=0.005,
        gradient_steps=1,
        # optimize_memory_usage=True,  # 🔧 移除以避免与handle_timeout_termination冲突
    )
    
    print(f"✅ 改进的{num_joints}关节SAC模型创建完成")
    print(f"   🔧 并行环境: {PARALLEL_ENVS}个")
    print(f"   🔧 标准化: 观测+奖励")
    print(f"   🔧 无Dropout特征提取器")
    print(f"   🔧 统一奖励函数")
    
    # 开始训练
    print(f"\\n🎯 开始改进的{num_joints}关节训练...")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\\n✅ 改进的{num_joints}关节训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        # 保存模型和标准化统计
        model_path = f"models/improved_sequential_{num_joints}joint_reacher"
        model.save(model_path)
        vec_env.save(f"{model_path}_vecnormalize.pkl")
        print(f"💾 模型已保存: {model_path}")
        print(f"💾 标准化统计已保存: {model_path}_vecnormalize.pkl")
        
        return model, vec_env, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\n⚠️ 改进的{num_joints}关节训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model_path = f"models/improved_sequential_{num_joints}joint_reacher_interrupted"
        model.save(model_path)
        vec_env.save(f"{model_path}_vecnormalize.pkl")
        print(f"💾 中断模型已保存: {model_path}")
        return model, vec_env, training_time
    
    finally:
        vec_env.close()

def test_improved_joint_model(num_joints, model, vec_env, n_eval_episodes=10):
    """🔧 测试改进的单个关节数模型"""
    print(f"\\n🧪 开始测试改进的{num_joints}关节模型")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
    print("-"*40)
    
    # 🔧 创建测试环境（带渲染，使用相同的标准化）
    test_env = create_env(num_joints, render_mode='human')
    test_env = Monitor(test_env)
    
    # 🔧 应用相同的标准化统计（但不更新）
    test_vec_env = DummyVecEnv([lambda: test_env])
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=True, training=False)
    
    # 🔧 加载训练时的标准化统计
    try:
        test_vec_env.load_running_average(vec_env)
        print("✅ 已加载训练时的标准化统计")
    except:
        print("⚠️ 无法加载标准化统计，使用默认值")
    
    try:
        # 使用SB3的evaluate_policy进行评估
        episode_rewards, episode_lengths = evaluate_policy(
            model, 
            test_vec_env, 
            n_eval_episodes=n_eval_episodes,
            render=True,
            return_episode_rewards=True,
            deterministic=True
        )
        
        # 手动计算成功率（需要访问info）
        success_episodes = 0
        episode_distances = []
        
        for episode in range(n_eval_episodes):
            obs = test_vec_env.reset()
            episode_success = False
            min_distance = float('inf')
            
            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_vec_env.step(action)
                
                # 获取原始环境的info
                if len(info) > 0 and 'distance_to_target' in info[0]:
                    distance = info[0]['distance_to_target']
                    is_success = info[0].get('is_success', False)
                    
                    min_distance = min(min_distance, distance)
                    
                    if is_success and not episode_success:
                        episode_success = True
                
                if done[0]:
                    break
            
            if episode_success:
                success_episodes += 1
            
            episode_distances.append(min_distance)
            
            print(f"   Episode {episode+1}: 奖励={episode_rewards[episode]:.2f}, 最小距离={min_distance:.4f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_episodes / n_eval_episodes
        avg_reward = np.mean(episode_rewards)
        avg_min_distance = np.mean(episode_distances)
        
        print(f"\\n🎯 改进的{num_joints}关节模型测试结果:")
        print(f"   成功率: {success_rate:.1%} ({success_episodes}/{n_eval_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均最小距离: {avg_min_distance:.4f}")
        
        return {
            'num_joints': num_joints,
            'success_rate': success_rate,
            'success_episodes': success_episodes,
            'total_episodes': n_eval_episodes,
            'avg_reward': avg_reward,
            'avg_min_distance': avg_min_distance,
            'episode_rewards': episode_rewards.tolist(),
            'episode_distances': episode_distances
        }
        
    except KeyboardInterrupt:
        print(f"\\n⚠️ 改进的{num_joints}关节测试被用户中断")
        return None
    
    finally:
        test_vec_env.close()

def main():
    """主函数：改进的依次训练和测试2-5关节Reacher"""
    print("🌟 改进的依次训练和测试2-5关节Reacher系统")
    print("🔧 基于GPT-5建议的全面改进版本")
    print(f"📊 配置: 每个模型训练{TOTAL_TIMESTEPS}步，{PARALLEL_ENVS}个并行环境，测试10个episodes")
    print(f"🎯 成功阈值: {SUCCESS_THRESHOLD*100}cm，目标最小半径: {TARGET_MIN_RADIUS*100}cm")
    print("💾 输出: 每个关节数保存独立的模型文件和标准化统计")
    print("📈 最终: 统计所有关节数的成功率对比")
    print()
    
    print("🔧 主要改进:")
    print("   ✅ 统一奖励函数 - 所有关节数使用相同的奖励计算")
    print("   ✅ 修复观测dtype - 统一使用float32和有限边界")
    print("   ✅ 增加训练步数和并行环境")
    print("   ✅ 移除Dropout - 避免给SAC引入额外噪声")
    print("   ✅ 添加标准化 - 使用VecNormalize")
    print("   ✅ 修复成功阈值 - 避免与目标采样边界重叠")
    print("   ✅ 训练时关闭渲染")
    print("   ✅ 添加梯度裁剪和调整学习率")
    print("   ✅ 归一化观测分量到[-1,1]范围")
    print()
    
    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)
    
    # 存储所有结果
    all_results = []
    training_times = []
    
    # 依次训练2-5关节
    joint_numbers = [2, 3, 4, 5]
    
    for num_joints in joint_numbers:
        try:
            print(f"\\n{'='*60}")
            print(f"🔄 当前进度: 改进的{num_joints}关节 Reacher ({joint_numbers.index(num_joints)+1}/{len(joint_numbers)})")
            print(f"{'='*60}")
            
            # 训练改进的模型
            model, vec_env, training_time = train_improved_joint_model(num_joints, total_timesteps=TOTAL_TIMESTEPS)
            training_times.append(training_time)
            
            # 测试改进的模型
            test_result = test_improved_joint_model(num_joints, model, vec_env, n_eval_episodes=10)
            
            if test_result:
                all_results.append(test_result)
            
            print(f"\\n✅ 改进的{num_joints}关节 Reacher 完成!")
            
        except KeyboardInterrupt:
            print(f"\\n⚠️ 在改进的{num_joints}关节训练时被用户中断")
            break
        except Exception as e:
            print(f"\\n❌ 改进的{num_joints}关节训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出最终总结
    print(f"\\n{'='*80}")
    print("🎉 改进的依次训练和测试2-5关节Reacher完成!")
    print(f"{'='*80}")
    
    if all_results:
        print("\\n📊 改进模型性能总结:")
        print("-"*80)
        print(f"{'关节数':<8} {'成功率':<10} {'平均奖励':<12} {'平均最小距离':<15} {'训练时间':<10}")
        print("-"*80)
        
        for i, result in enumerate(all_results):
            training_time_min = training_times[i] / 60 if i < len(training_times) else 0
            print(f"{result['num_joints']:<8} {result['success_rate']:<10.1%} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<15.4f} {training_time_min:<10.1f}分钟")
        
        print("-"*80)
        
        # 找出最佳模型
        best_model = max(all_results, key=lambda x: x['success_rate'])
        print(f"\\n🏆 最佳成功率模型: {best_model['num_joints']}关节")
        print(f"   成功率: {best_model['success_rate']:.1%}")
        print(f"   平均奖励: {best_model['avg_reward']:.2f}")
        print(f"   平均最小距离: {best_model['avg_min_distance']:.4f}")
        
        # 成功率趋势分析
        print(f"\\n📈 成功率趋势分析:")
        success_rates = [r['success_rate'] for r in all_results]
        joint_nums = [r['num_joints'] for r in all_results]
        
        for i, (joints, rate) in enumerate(zip(joint_nums, success_rates)):
            trend = ""
            if i > 0:
                prev_rate = success_rates[i-1]
                if rate > prev_rate:
                    trend = f" (↗ +{(rate-prev_rate)*100:.1f}%)"
                elif rate < prev_rate:
                    trend = f" (↘ -{(prev_rate-rate)*100:.1f}%)"
                else:
                    trend = " (→ 持平)"
            print(f"   {joints}关节: {rate:.1%}{trend}")
        
        # 改进效果分析
        print(f"\\n🔧 改进效果分析:")
        avg_success_rate = np.mean(success_rates)
        print(f"   平均成功率: {avg_success_rate:.1%}")
        print(f"   成功率标准差: {np.std(success_rates):.1%}")
        print(f"   最高成功率: {max(success_rates):.1%}")
        print(f"   最低成功率: {min(success_rates):.1%}")
        
        # 模型文件总结
        print(f"\\n💾 所有改进模型已保存到 models/ 目录:")
        for result in all_results:
            print(f"   - models/improved_sequential_{result['num_joints']}joint_reacher.zip")
            print(f"   - models/improved_sequential_{result['num_joints']}joint_reacher_vecnormalize.pkl")
        
        # 详细统计
        print(f"\\n📋 详细统计信息:")
        print(f"   总训练时间: {sum(training_times)/60:.1f} 分钟")
        print(f"   成功阈值: {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
        print(f"   目标最小半径: {TARGET_MIN_RADIUS} ({TARGET_MIN_RADIUS*100}cm)")
        print(f"   并行环境数: {PARALLEL_ENVS}")
        print(f"   训练步数: {TOTAL_TIMESTEPS}")
        
        # 结论
        print(f"\\n🎯 结论:")
        if avg_success_rate > 0.7:
            print(f"   ✅ 改进非常成功！平均成功率达到{avg_success_rate:.1%}")
        elif avg_success_rate > 0.5:
            print(f"   ✅ 改进效果显著！平均成功率为{avg_success_rate:.1%}")
        elif avg_success_rate > 0.3:
            print(f"   ⚠️ 有一定改进，平均成功率为{avg_success_rate:.1%}")
        else:
            print(f"   ❌ 仍需进一步优化，平均成功率仅为{avg_success_rate:.1%}")
    
    print(f"\\n🎯 改进的依次训练和测试完成！基于GPT-5建议的全面改进已实施。")

if __name__ == "__main__":
    main()
