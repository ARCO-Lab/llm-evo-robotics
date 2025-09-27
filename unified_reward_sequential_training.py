#!/usr/bin/env python3
"""
统一奖励规范的完整依次训练脚本：
🎯 基于GPT-5建议，实现所有关节数的奖励可比性
🔧 核心改进：
   1. 距离归一化：用可达半径R归一化距离
   2. 成功阈值归一化：SUCCESS_THRESHOLD = k * R，统一k=0.25
   3. 成功奖励与距离项量级匹配：success_bonus=+2.0
   4. 控制代价按关节数均值化：避免多关节平白多惩罚
   5. 目标分布统一按R取比例：保证可到达性一致
   6. 动力学时间尺度一致：统一gear、质量、阻尼
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import time
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 🎯 统一奖励规范参数
SUCCESS_THRESHOLD_RATIO = 0.25  # k = 0.25，成功阈值与可达半径的比例
DISTANCE_WEIGHT = 1.0          # w_d = 1.0，距离项权重
SUCCESS_BONUS = 2.0            # 成功奖励，与归一化距离项量级匹配
CONTROL_WEIGHT = 0.01          # λ_u = 0.01，控制代价权重
TARGET_MIN_RATIO = 0.15        # 目标最小距离比例
TARGET_MAX_RATIO = 0.85        # 目标最大距离比例

class UnifiedRewardExtractor(BaseFeaturesExtractor):
    """统一奖励规范的特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(UnifiedRewardExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 UnifiedRewardExtractor: {obs_dim}维 -> {features_dim}维")
        
        # 简化网络结构，移除Dropout
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

class UnifiedRewardReacherEnv(MujocoEnv):
    """统一奖励规范的Reacher环境基类"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        
        # 🎯 计算可达半径R和统一的成功阈值
        self.max_reach = sum(link_lengths)
        self.success_threshold = SUCCESS_THRESHOLD_RATIO * self.max_reach
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(xml_content)
        self.xml_file.flush()
        
        # 计算观察空间维度
        obs_dim = num_joints * 3 + 4  # cos, sin, vel + ee_pos + target_pos
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.step_count = 0
        self.max_episode_steps = 100
        
        print(f"✅ {num_joints}关节Reacher创建完成 (统一奖励规范)")
        print(f"   链长: {link_lengths}")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
        print(f"   🎯 成功阈值: {self.success_threshold:.3f} ({SUCCESS_THRESHOLD_RATIO:.1%} * R)")
        print(f"   🎯 目标生成范围: {self.calculate_target_min_distance():.3f} ~ {self.calculate_target_max_distance():.3f}")
    
    def calculate_target_min_distance(self):
        """计算目标生成的最小距离"""
        return TARGET_MIN_RATIO * self.max_reach
    
    def calculate_target_max_distance(self):
        """计算目标生成的最大距离"""
        return TARGET_MAX_RATIO * self.max_reach
    
    def generate_unified_target(self):
        """🎯 统一的目标生成策略 - 按R取比例"""
        min_distance = self.calculate_target_min_distance()
        max_distance = self.calculate_target_max_distance()
        
        # 使用极坐标生成目标
        target_distance = self.np_random.uniform(min_distance, max_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def step(self, action):
        # 显式渲染
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward = self.unified_reward(action)
        
        # 计算归一化距离
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        normalized_distance = distance / self.max_reach
        
        # 🎯 统一的成功判断：距离小于统一的成功阈值
        terminated = distance < self.success_threshold
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'normalized_distance': normalized_distance,
            'is_success': terminated,
            'max_reach': self.max_reach,
            'success_threshold': self.success_threshold,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def unified_reward(self, action):
        """🎯 统一奖励规范的奖励函数"""
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 1. 🎯 距离归一化：用可达半径R归一化
        normalized_distance = distance / self.max_reach
        distance_reward = -DISTANCE_WEIGHT * normalized_distance
        
        # 2. 🎯 成功奖励：与距离项量级匹配
        success_reward = SUCCESS_BONUS if distance < self.success_threshold else 0.0
        
        # 3. 🎯 控制代价按关节数均值化：避免多关节平白多惩罚
        control_cost = -CONTROL_WEIGHT * np.mean(np.square(action))
        
        total_reward = distance_reward + success_reward + control_cost
        
        return total_reward
    
    def _get_obs(self):
        # N关节的观察：cos, sin, vel, fingertip_pos, target_pos
        theta = self.data.qpos.flat[:self.num_joints]
        obs = np.concatenate([
            np.cos(theta),                    # N个cos值
            np.sin(theta),                    # N个sin值
            self.data.qvel.flat[:self.num_joints],  # N个关节速度
            self.get_body_com("fingertip")[:2],     # 末端执行器位置 (x,y)
            self.get_body_com("target")[:2],        # 目标位置 (x,y)
        ])
        return obs
    
    def reset_model(self):
        # 重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()
        
        # 只给机械臂关节添加随机速度，目标关节速度保持为0
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        
        # 🎯 使用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

# 🎯 统一动力学时间尺度的XML配置
def get_unified_2joint_xml():
    """2关节XML配置（统一动力学参数）"""
    return """
<mujoco model="unified_2joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
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

def get_unified_3joint_xml():
    """3关节XML配置（统一动力学参数）"""
    return """
<mujoco model="unified_3joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
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

def get_unified_4joint_xml():
    """4关节XML配置（统一动力学参数）"""
    return """
<mujoco model="unified_4joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
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

def get_unified_5joint_xml():
    """5关节XML配置（统一动力学参数）"""
    return """
<mujoco model="unified_5joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
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
class Unified2JointReacherEnv(UnifiedRewardReacherEnv):
    """统一奖励规范的2关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified2JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_unified_2joint_xml(),
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class Unified3JointReacherEnv(UnifiedRewardReacherEnv):
    """统一奖励规范的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified3JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_unified_3joint_xml(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class Unified4JointReacherEnv(UnifiedRewardReacherEnv):
    """统一奖励规范的4关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified4JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_unified_4joint_xml(),
            num_joints=4,
            link_lengths=[0.08, 0.08, 0.08, 0.08],
            render_mode=render_mode,
            **kwargs
        )

class Unified5JointReacherEnv(UnifiedRewardReacherEnv):
    """统一奖励规范的5关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified5JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_unified_5joint_xml(),
            num_joints=5,
            link_lengths=[0.06, 0.06, 0.06, 0.06, 0.06],
            render_mode=render_mode,
            **kwargs
        )

# 🎯 统一奖励规范的2关节包装器（用于标准Reacher-v5）
class Unified2JointReacherWrapper(gym.Wrapper):
    """统一奖励规范的2关节Reacher包装器 - 使用标准Reacher-v5"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]
        self.max_episode_steps = 100
        
        # 🎯 计算可达半径R和统一的成功阈值
        self.max_reach = sum(self.link_lengths)
        self.success_threshold = SUCCESS_THRESHOLD_RATIO * self.max_reach
        
        print("🌟 Unified2JointReacherWrapper 初始化")
        print(f"   链长: {self.link_lengths}")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
        print(f"   🎯 成功阈值: {self.success_threshold:.3f} ({SUCCESS_THRESHOLD_RATIO:.1%} * R)")
        print(f"   🎯 目标生成范围: {self.calculate_target_min_distance():.3f} ~ {self.calculate_target_max_distance():.3f}")
    
    def calculate_target_min_distance(self):
        return TARGET_MIN_RATIO * self.max_reach
    
    def calculate_target_max_distance(self):
        return TARGET_MAX_RATIO * self.max_reach
    
    def generate_unified_target(self):
        min_distance = self.calculate_target_min_distance()
        max_distance = self.calculate_target_max_distance()
        
        target_distance = self.np_random.uniform(min_distance, max_distance)
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
        
        # 获取新的观察
        obs = reacher_env._get_obs()
        
        if info is None:
            info = {}
        info.update({
            'max_reach': self.max_reach,
            'success_threshold': self.success_threshold,
            'target_pos': [target_x, target_y]
        })
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 🎯 重新计算统一奖励
        reacher_env = self.env.unwrapped
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 统一奖励规范
        normalized_distance = distance / self.max_reach
        distance_reward = -DISTANCE_WEIGHT * normalized_distance
        success_reward = SUCCESS_BONUS if distance < self.success_threshold else 0.0
        control_cost = -CONTROL_WEIGHT * np.mean(np.square(action))
        
        unified_reward = distance_reward + success_reward + control_cost
        
        # 🎯 统一的成功判断
        is_success = distance < self.success_threshold
        
        if info is None:
            info = {}
        info.update({
            'max_reach': self.max_reach,
            'success_threshold': self.success_threshold,
            'distance_to_target': distance,
            'normalized_distance': normalized_distance,
            'is_success': is_success,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        })
        
        return obs, unified_reward, terminated, truncated, info

def create_unified_env(num_joints, render_mode=None):
    """创建指定关节数的统一奖励环境"""
    if num_joints == 2:
        # 使用标准Reacher-v5 + 统一奖励包装器
        env = gym.make('Reacher-v5', render_mode=render_mode)
        env = Unified2JointReacherWrapper(env)
    elif num_joints == 3:
        env = Unified3JointReacherEnv(render_mode=render_mode)
    elif num_joints == 4:
        env = Unified4JointReacherEnv(render_mode=render_mode)
    elif num_joints == 5:
        env = Unified5JointReacherEnv(render_mode=render_mode)
    else:
        raise ValueError(f"不支持的关节数: {num_joints}")
    
    env = Monitor(env)
    return env

def train_unified_model(num_joints, total_timesteps=30000):
    """训练统一奖励规范的单个关节数模型"""
    print(f"\\n🚀 开始训练统一奖励规范的{num_joints}关节Reacher模型")
    print(f"📊 训练步数: {total_timesteps}")
    print(f"🎯 统一奖励规范:")
    print(f"   - 成功阈值比例: {SUCCESS_THRESHOLD_RATIO:.1%}")
    print(f"   - 距离权重: {DISTANCE_WEIGHT}")
    print(f"   - 成功奖励: {SUCCESS_BONUS}")
    print(f"   - 控制权重: {CONTROL_WEIGHT}")
    print(f"   - 目标范围: {TARGET_MIN_RATIO:.1%} ~ {TARGET_MAX_RATIO:.1%} * R")
    print("="*60)
    
    # 创建训练环境（开启渲染）
    train_env = create_unified_env(num_joints, render_mode='human')
    
    # 创建SAC模型
    policy_kwargs = {
        'features_extractor_class': UnifiedRewardExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=1000,
        device='cpu',
        tensorboard_log=f"./tensorboard_logs/unified_reward_{num_joints}joint/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )
    
    print(f"✅ 统一奖励规范的{num_joints}关节SAC模型创建完成")
    
    # 开始训练
    print(f"\\n🎯 开始{num_joints}关节训练...")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\\n✅ {num_joints}关节训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        # 保存模型
        model_path = f"models/unified_reward_{num_joints}joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        return model, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\n⚠️ {num_joints}关节训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model_path = f"models/unified_reward_{num_joints}joint_reacher_interrupted"
        model.save(model_path)
        print(f"💾 中断模型已保存: {model_path}")
        return model, training_time
    
    finally:
        train_env.close()

def test_unified_model(num_joints, model, n_eval_episodes=10):
    """测试统一奖励规范的单个关节数模型"""
    print(f"\\n🧪 开始测试统一奖励规范的{num_joints}关节模型")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    print("-"*40)
    
    # 创建测试环境（带渲染）
    test_env = create_unified_env(num_joints, render_mode='human')
    
    try:
        # 手动运行episodes来计算成功率
        success_episodes = 0
        total_episodes = n_eval_episodes
        episode_rewards = []
        episode_distances = []
        episode_normalized_distances = []
        
        for episode in range(n_eval_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            min_normalized_distance = float('inf')
            
            # 获取环境信息
            max_reach = info.get('max_reach', 1.0)
            success_threshold = info.get('success_threshold', 0.05)
            
            for step in range(100):  # 每个episode 100步
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                
                # 获取距离和成功信息
                distance = info.get('distance_to_target', float('inf'))
                normalized_distance = info.get('normalized_distance', float('inf'))
                is_success = info.get('is_success', False)
                
                min_distance = min(min_distance, distance)
                min_normalized_distance = min(min_normalized_distance, normalized_distance)
                
                if is_success and not episode_success:
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_episodes += 1
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
            episode_normalized_distances.append(min_normalized_distance)
            
            print(f"   Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, 归一化距离={min_normalized_distance:.3f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_episodes / total_episodes
        avg_reward = np.mean(episode_rewards)
        avg_min_distance = np.mean(episode_distances)
        avg_normalized_distance = np.mean(episode_normalized_distances)
        
        print(f"\\n🎯 统一奖励规范的{num_joints}关节模型测试结果:")
        print(f"   成功率: {success_rate:.1%} ({success_episodes}/{total_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均最小距离: {avg_min_distance:.4f}")
        print(f"   平均归一化距离: {avg_normalized_distance:.3f}")
        print(f"   🎯 可达半径R: {max_reach:.3f}")
        print(f"   🎯 成功阈值: {success_threshold:.3f} ({SUCCESS_THRESHOLD_RATIO:.1%} * R)")
        
        return {
            'num_joints': num_joints,
            'success_rate': success_rate,
            'success_episodes': success_episodes,
            'total_episodes': total_episodes,
            'avg_reward': avg_reward,
            'avg_min_distance': avg_min_distance,
            'avg_normalized_distance': avg_normalized_distance,
            'max_reach': max_reach,
            'success_threshold': success_threshold,
            'episode_rewards': episode_rewards,
            'episode_distances': episode_distances,
            'episode_normalized_distances': episode_normalized_distances
        }
        
    except KeyboardInterrupt:
        print(f"\\n⚠️ {num_joints}关节测试被用户中断")
        return None
    
    finally:
        test_env.close()

def main():
    """主函数：统一奖励规范的依次训练和测试2-5关节Reacher"""
    print("🌟 统一奖励规范的完整依次训练和测试2-5关节Reacher系统")
    print("🎯 基于GPT-5建议，实现所有关节数的奖励可比性")
    print(f"📊 配置: 每个模型训练30000步，测试10个episodes")
    print("🔧 统一奖励规范:")
    print(f"   1. 距离归一化: 用可达半径R归一化距离")
    print(f"   2. 成功阈值归一化: SUCCESS_THRESHOLD = {SUCCESS_THRESHOLD_RATIO:.1%} * R")
    print(f"   3. 成功奖励与距离项量级匹配: success_bonus = {SUCCESS_BONUS}")
    print(f"   4. 控制代价按关节数均值化: 避免多关节平白多惩罚")
    print(f"   5. 目标分布统一按R取比例: {TARGET_MIN_RATIO:.1%} ~ {TARGET_MAX_RATIO:.1%} * R")
    print(f"   6. 动力学时间尺度一致: 统一gear=200、density=1000、damping=1")
    print("💾 输出: 每个关节数保存独立的模型文件")
    print("📈 最终: 统计所有关节数的成功率和归一化距离对比")
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
            print(f"🔄 当前进度: {num_joints}关节 Reacher ({joint_numbers.index(num_joints)+1}/{len(joint_numbers)})")
            print(f"{'='*60}")
            
            # 训练模型
            model, training_time = train_unified_model(num_joints, total_timesteps=30000)
            training_times.append(training_time)
            
            # 测试模型
            test_result = test_unified_model(num_joints, model, n_eval_episodes=10)
            
            if test_result:
                all_results.append(test_result)
            
            print(f"\\n✅ {num_joints}关节 Reacher 完成!")
            
        except KeyboardInterrupt:
            print(f"\\n⚠️ 在{num_joints}关节训练时被用户中断")
            break
        except Exception as e:
            print(f"\\n❌ {num_joints}关节训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出最终总结
    print(f"\\n{'='*80}")
    print("🎉 统一奖励规范的完整依次训练和测试2-5关节Reacher完成!")
    print(f"{'='*80}")
    
    if all_results:
        print("\\n📊 所有模型性能总结:")
        print("-"*100)
        print(f"{'关节数':<8} {'成功率':<10} {'平均奖励':<12} {'平均距离':<12} {'归一化距离':<12} {'可达半径R':<12} {'训练时间':<10}")
        print("-"*100)
        
        for i, result in enumerate(all_results):
            training_time_min = training_times[i] / 60 if i < len(training_times) else 0
            print(f"{result['num_joints']:<8} {result['success_rate']:<10.1%} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<12.4f} {result['avg_normalized_distance']:<12.3f} {result['max_reach']:<12.3f} {training_time_min:<10.1f}分钟")
        
        print("-"*100)
        
        # 找出最佳模型
        best_model = max(all_results, key=lambda x: x['success_rate'])
        print(f"\\n🏆 最佳成功率模型: {best_model['num_joints']}关节")
        print(f"   成功率: {best_model['success_rate']:.1%}")
        print(f"   平均奖励: {best_model['avg_reward']:.2f}")
        print(f"   平均归一化距离: {best_model['avg_normalized_distance']:.3f}")
        print(f"   可达半径R: {best_model['max_reach']:.3f}")
        
        # 🎯 统一奖励规范效果分析
        print(f"\\n🎯 统一奖励规范效果分析:")
        success_rates = [r['success_rate'] for r in all_results]
        normalized_distances = [r['avg_normalized_distance'] for r in all_results]
        rewards = [r['avg_reward'] for r in all_results]
        
        print(f"   成功率一致性: 标准差 {np.std(success_rates):.3f} (越小越一致)")
        print(f"   归一化距离一致性: 标准差 {np.std(normalized_distances):.3f} (越小越一致)")
        print(f"   奖励一致性: 标准差 {np.std(rewards):.3f} (越小越一致)")
        
        # 成功率趋势分析
        print(f"\\n📈 成功率趋势分析:")
        joint_nums = [r['num_joints'] for r in all_results]
        
        for i, (joints, rate, norm_dist) in enumerate(zip(joint_nums, success_rates, normalized_distances)):
            trend = ""
            if i > 0:
                prev_rate = success_rates[i-1]
                if rate > prev_rate:
                    trend = f" (↗ +{(rate-prev_rate)*100:.1f}%)"
                elif rate < prev_rate:
                    trend = f" (↘ -{(prev_rate-rate)*100:.1f}%)"
                else:
                    trend = " (→ 持平)"
            print(f"   {joints}关节: 成功率{rate:.1%}, 归一化距离{norm_dist:.3f}{trend}")
        
        # 模型文件总结
        print(f"\\n💾 所有模型已保存到 models/ 目录:")
        for result in all_results:
            print(f"   - models/unified_reward_{result['num_joints']}joint_reacher.zip")
        
        # 详细统计
        print(f"\\n📋 统一奖励规范统计:")
        print(f"   总训练时间: {sum(training_times)/60:.1f} 分钟")
        print(f"   平均成功率: {np.mean(success_rates):.1%}")
        print(f"   成功率标准差: {np.std(success_rates):.1%}")
        print(f"   平均归一化距离: {np.mean(normalized_distances):.3f}")
        print(f"   归一化距离标准差: {np.std(normalized_distances):.3f}")
        print(f"   🎯 成功阈值比例: {SUCCESS_THRESHOLD_RATIO:.1%}")
        print(f"   🎯 距离权重: {DISTANCE_WEIGHT}")
        print(f"   🎯 成功奖励: {SUCCESS_BONUS}")
        print(f"   🎯 控制权重: {CONTROL_WEIGHT}")
        
        # 结论
        print(f"\\n🎯 结论:")
        if np.std(success_rates) < 0.1:  # 成功率标准差小于10%
            print(f"   ✅ 统一奖励规范非常成功！各关节数成功率一致性很好")
        elif np.std(success_rates) < 0.2:  # 成功率标准差小于20%
            print(f"   ✅ 统一奖励规范效果良好！各关节数成功率相对一致")
        else:
            print(f"   ⚠️ 统一奖励规范有一定效果，但仍有改进空间")
        
        if best_model['success_rate'] > 0.5:
            print(f"   🏆 整体训练成功！{best_model['num_joints']}关节模型表现最佳")
        elif max(success_rates) > 0.3:
            print(f"   ⚠️ 部分成功，最佳模型成功率为{max(success_rates):.1%}")
        else:
            print(f"   ❌ 整体表现较差，可能需要进一步调整参数")
    
    print(f"\\n🎯 统一奖励规范的完整训练完成！实现了所有关节数的奖励可比性和公平性。")

if __name__ == "__main__":
    main()


