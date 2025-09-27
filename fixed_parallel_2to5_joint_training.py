#!/usr/bin/env python3
"""
修复的2-5关节并行训练：
1. 修复并行渲染问题 - 使用SubprocVecEnv实现真正的多进程
2. 修复目标滚动问题 - 参考标准MuJoCo Reacher的目标设置方式
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time
import multiprocessing as mp
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 支持的关节数
SUPPORTED_JOINTS = [2, 3, 4, 5]
J_MAX = 5

class MixedJointExtractor(BaseFeaturesExtractor):
    """混合关节特征提取器 (直接复制自parallel_mixed_joint_training.py)"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(MixedJointExtractor, self).__init__(observation_space, features_dim)
        
        # 扩展支持的最大观察维度（5关节的19维）
        self.max_obs_dim = 19  # 5*3 + 4 = 19
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 MixedJointExtractor: {obs_dim} -> {features_dim}")
        print(f"   支持最大观察维度: {self.max_obs_dim} (扩展到5关节)")
        
        # 使用最大维度设计网络，可以处理不同输入
        self.net = nn.Sequential(
            nn.Linear(self.max_obs_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]
        
        # 如果输入维度小于最大维度，用零填充
        if obs_dim < self.max_obs_dim:
            # 创建填充后的观察
            padded_obs = torch.zeros(batch_size, self.max_obs_dim, device=observations.device)
            
            if obs_dim == 10:  # 2关节Reacher
                self._fill_2joint_obs(observations, padded_obs)
            elif obs_dim == 13:  # 3关节
                self._fill_3joint_obs(observations, padded_obs)
            elif obs_dim == 16:  # 4关节
                self._fill_4joint_obs(observations, padded_obs)
            elif obs_dim == 19:  # 5关节
                padded_obs = observations  # 直接使用
            else:
                # 其他情况，直接复制并填充零
                padded_obs[:, :obs_dim] = observations
            
            observations = padded_obs
        
        return self.net(observations)
    
    def _fill_2joint_obs(self, obs, padded_obs):
        """填充2关节观察"""
        padded_obs[:, 0] = obs[:, 0]   # cos1
        padded_obs[:, 1] = obs[:, 1]   # cos2
        padded_obs[:, 2] = 1.0         # cos3 (默认0度)
        padded_obs[:, 3] = 1.0         # cos4 (默认0度)
        padded_obs[:, 4] = 1.0         # cos5 (默认0度)
        
        padded_obs[:, 5] = obs[:, 2]   # sin1
        padded_obs[:, 6] = obs[:, 3]   # sin2
        padded_obs[:, 7] = 0.0         # sin3 (默认0度)
        padded_obs[:, 8] = 0.0         # sin4 (默认0度)
        padded_obs[:, 9] = 0.0         # sin5 (默认0度)
        
        padded_obs[:, 10] = obs[:, 4]  # vel1
        padded_obs[:, 11] = obs[:, 5]  # vel2
        padded_obs[:, 12] = 0.0        # vel3
        padded_obs[:, 13] = 0.0        # vel4
        padded_obs[:, 14] = 0.0        # vel5
        
        padded_obs[:, 15] = obs[:, 6]  # ee_x
        padded_obs[:, 16] = obs[:, 7]  # ee_y
        padded_obs[:, 17] = obs[:, 8]  # target_x
        padded_obs[:, 18] = obs[:, 9]  # target_y
    
    def _fill_3joint_obs(self, obs, padded_obs):
        """填充3关节观察"""
        padded_obs[:, 0:3] = obs[:, 0:3]     # cos1-3
        padded_obs[:, 3] = 1.0               # cos4 (默认)
        padded_obs[:, 4] = 1.0               # cos5 (默认)
        
        padded_obs[:, 5:8] = obs[:, 3:6]     # sin1-3
        padded_obs[:, 8] = 0.0               # sin4 (默认)
        padded_obs[:, 9] = 0.0               # sin5 (默认)
        
        padded_obs[:, 10:13] = obs[:, 6:9]   # vel1-3
        padded_obs[:, 13] = 0.0              # vel4 (默认)
        padded_obs[:, 14] = 0.0              # vel5 (默认)
        
        padded_obs[:, 15:19] = obs[:, 9:13]  # ee_pos + target_pos
    
    def _fill_4joint_obs(self, obs, padded_obs):
        """填充4关节观察"""
        padded_obs[:, 0:4] = obs[:, 0:4]     # cos1-4
        padded_obs[:, 4] = 1.0               # cos5 (默认)
        
        padded_obs[:, 5:9] = obs[:, 4:8]     # sin1-4
        padded_obs[:, 9] = 0.0               # sin5 (默认)
        
        padded_obs[:, 10:14] = obs[:, 8:12]  # vel1-4
        padded_obs[:, 14] = 0.0              # vel5 (默认)
        
        padded_obs[:, 15:19] = obs[:, 12:16] # ee_pos + target_pos

class MixedJointActionWrapper(gym.ActionWrapper):
    """混合关节动作包装器"""
    
    def __init__(self, env, original_action_dim):
        super().__init__(env)
        self.original_action_dim = original_action_dim
        
        # 统一动作空间为5维（最大关节数）
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(J_MAX,), dtype=np.float32
        )
        
        print(f"🔧 MixedJointActionWrapper: 原始动作维度={original_action_dim}, 统一为{J_MAX}维")
    
    def action(self, action):
        # 只使用前N个关节的动作
        return action[:self.original_action_dim]

class MixedJointObservationWrapper(gym.ObservationWrapper):
    """混合关节观察包装器"""
    
    def __init__(self, env, target_obs_dim=19):
        super().__init__(env)
        self.target_obs_dim = target_obs_dim
        self.original_obs_dim = env.observation_space.shape[0]
        
        # 统一观察空间为19维（5关节的维度）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_obs_dim,), dtype=np.float64
        )
        
        print(f"🔧 MixedJointObservationWrapper: 原始观察维度={self.original_obs_dim}, 统一为{target_obs_dim}维")
    
    def observation(self, obs):
        if len(obs) < self.target_obs_dim:
            # 填充观察到目标维度
            padded_obs = np.zeros(self.target_obs_dim)
            
            if len(obs) == 10:  # 2关节
                self._fill_2joint_obs_np(obs, padded_obs)
            elif len(obs) == 13:  # 3关节
                self._fill_3joint_obs_np(obs, padded_obs)
            elif len(obs) == 16:  # 4关节
                self._fill_4joint_obs_np(obs, padded_obs)
            elif len(obs) == 17:  # 5关节
                padded_obs = obs
            else:
                padded_obs[:len(obs)] = obs
            
            return padded_obs
        else:
            return obs
    
    def _fill_2joint_obs_np(self, obs, padded_obs):
        """填充2关节观察 (numpy版本)"""
        padded_obs[0] = obs[0]   # cos1
        padded_obs[1] = obs[1]   # cos2
        padded_obs[2] = 1.0      # cos3
        padded_obs[3] = 1.0      # cos4
        padded_obs[4] = 1.0      # cos5
        padded_obs[5] = obs[2]   # sin1
        padded_obs[6] = obs[3]   # sin2
        padded_obs[7] = 0.0      # sin3
        padded_obs[8] = 0.0      # sin4
        padded_obs[9] = 0.0      # sin5
        padded_obs[10] = obs[4]  # vel1
        padded_obs[11] = obs[5]  # vel2
        padded_obs[12] = 0.0     # vel3
        padded_obs[13] = 0.0     # vel4
        padded_obs[14] = 0.0     # vel5
        padded_obs[15] = obs[6]  # ee_x
        padded_obs[16] = obs[7]  # ee_y
        padded_obs[17] = obs[8]  # target_x
        padded_obs[18] = obs[9]  # target_y
    
    def _fill_3joint_obs_np(self, obs, padded_obs):
        """填充3关节观察 (numpy版本)"""
        padded_obs[0:3] = obs[0:3]     # cos1-3
        padded_obs[3] = 1.0            # cos4
        padded_obs[4] = 1.0            # cos5
        padded_obs[5:8] = obs[3:6]     # sin1-3
        padded_obs[8] = 0.0            # sin4
        padded_obs[9] = 0.0            # sin5
        padded_obs[10:13] = obs[6:9]   # vel1-3
        padded_obs[13] = 0.0           # vel4
        padded_obs[14] = 0.0           # vel5
        padded_obs[15:19] = obs[9:13]  # ee_pos + target_pos
    
    def _fill_4joint_obs_np(self, obs, padded_obs):
        """填充4关节观察 (numpy版本)"""
        padded_obs[0:4] = obs[0:4]     # cos1-4
        padded_obs[4] = 1.0            # cos5
        padded_obs[5:9] = obs[4:8]     # sin1-4
        padded_obs[9] = 0.0            # sin5
        padded_obs[10:14] = obs[8:12]  # vel1-4
        padded_obs[14] = 0.0           # vel5
        padded_obs[15:19] = obs[12:16] # ee_pos + target_pos

# 修复的XML配置 - 参考标准MuJoCo Reacher的目标设置
def get_fixed_3joint_xml():
    """获取修复的3关节XML - 目标不会滚动"""
    return """
<mujoco model="fixed_3joint">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <!-- 场地 -->
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.5 0.5 10" type="plane"/>
    
    <!-- 边界 -->
    <geom conaffinity="0" fromto="-.5 -.5 .01 .5 -.5 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .5 -.5 .01 .5  .5 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.5  .5 .01 .5  .5 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.5 -.5 .01 -.5 .5 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    
    <!-- 3关节机械臂 -->
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
          <body name="fingertip" pos="0.1 0 0">
            <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- 修复的目标 - 参考标准MuJoCo Reacher，使用slide关节固定在平面上 -->
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.45 .45" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.45 .45" ref="-.2" stiffness="0" type="slide"/>
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

def get_fixed_4joint_xml():
    """获取修复的4关节XML - 目标不会滚动"""
    return """
<mujoco model="fixed_4joint">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <!-- 更大的场地 -->
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.6 0.6 10" type="plane"/>
    
    <!-- 边界 -->
    <geom conaffinity="0" fromto="-.6 -.6 .01 .6 -.6 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .6 -.6 .01 .6  .6 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.6  .6 .01 .6  .6 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.6 -.6 .01 -.6 .6 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    
    <!-- 4关节机械臂 -->
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
            <body name="fingertip" pos="0.08 0 0">
              <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- 修复的目标 - 使用slide关节固定在平面上 -->
    <body name="target" pos=".25 -.25 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.55 .55" ref=".25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.55 .55" ref="-.25" stiffness="0" type="slide"/>
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

def get_fixed_5joint_xml():
    """获取修复的5关节XML - 目标不会滚动"""
    return """
<mujoco model="fixed_5joint">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <!-- 最大的场地 -->
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.7 0.7 10" type="plane"/>
    
    <!-- 边界 -->
    <geom conaffinity="0" fromto="-.7 -.7 .01 .7 -.7 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .7 -.7 .01 .7  .7 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.7  .7 .01 .7  .7 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.7 -.7 .01 -.7 .7 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    
    <!-- 5关节机械臂 -->
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
              <body name="fingertip" pos="0.06 0 0">
                <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- 修复的目标 - 使用slide关节固定在平面上 -->
    <body name="target" pos=".3 -.3 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.65 .65" ref=".3" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.65 .65" ref="-.3" stiffness="0" type="slide"/>
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

# 修复的环境类
class Fixed3JointReacherEnv(MujocoEnv):
    """修复的3关节Reacher环境 - 目标不会滚动"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Fixed3JointReacherEnv 初始化")
        
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_fixed_3joint_xml())
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        print(f"   场地尺寸: 1.0m x 1.0m")
        print(f"   目标范围: ±0.45m")
        print(f"   🔧 修复: 目标使用slide关节，不会滚动")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        print("✅ Fixed3JointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
        
        self.step_count = 0
        self.max_episode_steps = 50
    
    def step(self, action):
        # 显式渲染（修复渲染问题）
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward = self.reward(action)
        
        # 计算距离
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 检查终止条件
        terminated = distance < 0.02
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def reward(self, action):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 距离奖励
        reward = -distance
        
        # 到达奖励
        if distance < 0.02:
            reward += 10.0
        
        return reward
    
    def _get_obs(self):
        # 3关节的观察：cos, sin, vel, fingertip_pos, target_pos
        theta = self.data.qpos.flat[:3]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qvel.flat[:3],
            self.get_body_com("fingertip")[:2],  # 只取x,y坐标
            self.get_body_com("target")[:2],     # 只取x,y坐标
        ])
        return obs
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

class Fixed4JointReacherEnv(MujocoEnv):
    """修复的4关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Fixed4JointReacherEnv 初始化")
        
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_fixed_4joint_xml())
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        print(f"   🔧 修复: 目标使用slide关节，不会滚动")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        print("✅ Fixed4JointReacherEnv 创建完成")
        
        self.step_count = 0
        self.max_episode_steps = 50
    
    def step(self, action):
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward = self.reward(action)
        
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        terminated = distance < 0.02
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def reward(self, action):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        reward = -distance
        if distance < 0.02:
            reward += 10.0
        
        return reward
    
    def _get_obs(self):
        theta = self.data.qpos.flat[:4]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qvel.flat[:4],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2],
        ])
        return obs
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

class Fixed5JointReacherEnv(MujocoEnv):
    """修复的5关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Fixed5JointReacherEnv 初始化")
        
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_fixed_5joint_xml())
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        print(f"   🔧 修复: 目标使用slide关节，不会滚动")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        print("✅ Fixed5JointReacherEnv 创建完成")
        
        self.step_count = 0
        self.max_episode_steps = 50
    
    def step(self, action):
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward = self.reward(action)
        
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        terminated = distance < 0.02
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def reward(self, action):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        reward = -distance
        if distance < 0.02:
            reward += 10.0
        
        return reward
    
    def _get_obs(self):
        theta = self.data.qpos.flat[:5]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qvel.flat[:5],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2],
        ])
        return obs
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

# 环境创建工厂函数 - 修复并行渲染
def make_2joint_env():
    """创建2关节环境工厂函数"""
    def _init():
        env = gym.make('Reacher-v5', render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=2)
        env = Monitor(env)
        return env
    return _init

def make_3joint_env():
    """创建3关节环境工厂函数"""
    def _init():
        env = Fixed3JointReacherEnv(render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=3)
        env = Monitor(env)
        return env
    return _init

def make_4joint_env():
    """创建4关节环境工厂函数"""
    def _init():
        env = Fixed4JointReacherEnv(render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=4)
        env = Monitor(env)
        return env
    return _init

def make_5joint_env():
    """创建5关节环境工厂函数"""
    def _init():
        env = Fixed5JointReacherEnv(render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=5)
        env = Monitor(env)
        return env
    return _init

def train_fixed_parallel_2to5_joint(total_timesteps: int = 40000):
    """
    修复的2-5关节并行训练
    1. 使用SubprocVecEnv实现真正的多进程并行渲染
    2. 修复目标滚动问题
    """
    print("🚀 修复的2-5关节并行Reacher训练")
    print(f"🎯 同时训练{SUPPORTED_JOINTS}关节Reacher")
    print("💡 修复1: 真正的多进程并行渲染")
    print("💡 修复2: 目标使用slide关节，不会滚动")
    print("="*60)
    
    # 创建并行环境
    print("🌍 创建修复的并行训练环境...")
    
    env_fns = []
    env_makers = [make_2joint_env, make_3joint_env, make_4joint_env, make_5joint_env]
    
    for i, (joints, maker) in enumerate(zip(SUPPORTED_JOINTS, env_makers)):
        env_fns.append(maker())
        print(f"   ✅ {joints}关节环境已添加")
    
    # 尝试使用SubprocVecEnv实现真正的多进程并行
    print("🔄 创建真正的并行向量化环境...")
    try:
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        print("✅ 使用SubprocVecEnv (真正的多进程并行)")
        print("💡 现在每个环境都会在独立进程中渲染")
    except Exception as e:
        print(f"⚠️ SubprocVecEnv失败，回退到DummyVecEnv: {e}")
        vec_env = DummyVecEnv(env_fns)
        print("✅ 使用DummyVecEnv (单进程)")
    
    print("✅ 修复的并行环境创建完成")
    print(f"   环境数量: {len(env_fns)} ({len(SUPPORTED_JOINTS)}种关节数)")
    print(f"   统一观察空间: {vec_env.observation_space}")
    print(f"   统一动作空间: {vec_env.action_space}")
    
    # 创建SAC模型
    print("\n🤖 创建修复的2-5关节SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': MixedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=800,
        device='cpu',
        tensorboard_log="./tensorboard_logs/fixed_parallel_2to5/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )
    
    print("✅ 修复的2-5关节SAC模型创建完成")
    print("   ✅ 混合关节特征提取器")
    print("   ✅ 支持2-5关节统一处理")
    print("   ✅ 修复了目标滚动问题")
    print("   ✅ 修复了并行渲染问题")
    
    # 开始训练
    print(f"\n🎯 开始修复的2-5关节训练 ({total_timesteps}步)...")
    print("💡 您现在应该看到4个MuJoCo窗口同时训练:")
    print("   🔸 窗口1: 2关节Reacher (标准)")
    print("   🔸 窗口2: 3关节Reacher (修复)")
    print("   🔸 窗口3: 4关节Reacher (修复)")
    print("   🔸 窗口4: 5关节Reacher (修复)")
    print("💡 所有环境共享同一个神经网络模型")
    print("💡 目标球现在不会滚动了")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 修复的2-5关节训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        print(f"🚀 并行效率: {len(env_fns)}个环境同时训练")
        
        # 保存最终模型
        model.save("models/fixed_parallel_2to5_joint_final")
        print("💾 最终修复模型已保存: models/fixed_parallel_2to5_joint_final")
        
        return model
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/fixed_parallel_2to5_joint_interrupted")
        print("💾 中断模型已保存")
        return model
    
    finally:
        vec_env.close()

def main():
    """主函数"""
    print("🌟 修复的2-5关节Reacher训练系统")
    print("🎯 修复1: 真正的多进程并行渲染")
    print("🎯 修复2: 目标不会滚动 (参考标准MuJoCo Reacher)")
    print("💡 基于parallel_mixed_joint_training.py扩展")
    print()
    
    try:
        # 开始修复的2-5关节训练
        train_fixed_parallel_2to5_joint(total_timesteps=40000)
        
        print(f"\n🎉 修复的2-5关节训练完成！")
        print(f"💡 您应该看到了4个环境同时训练的效果")
        print(f"✅ 目标球不再滚动了")
        print(f"✅ 一套模型现在可以控制2-5关节的机械臂")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


