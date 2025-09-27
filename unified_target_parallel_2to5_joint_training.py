#!/usr/bin/env python3
"""
统一目标生成策略的2-5关节并行训练：
1. 所有环境使用相同的基于可达范围的智能目标生成策略
2. 确保各关节数的成功率和训练体验相似
3. 使用极坐标方式在合理可达范围内生成目标
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
    """混合关节特征提取器"""
    
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

# 统一的目标生成策略基类
class UnifiedTargetReacherEnv(MujocoEnv):
    """统一目标生成策略的基类"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        
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
        self.max_episode_steps = 50
        
        print(f"✅ {num_joints}关节Reacher创建完成")
        print(f"   链长: {link_lengths}")
        print(f"   最大可达距离: {self.calculate_max_reach():.3f}")
        print(f"   目标生成范围: {self.calculate_target_range():.3f}")
    
    def calculate_max_reach(self):
        """计算理论最大可达距离"""
        return sum(self.link_lengths)
    
    def calculate_target_range(self):
        """计算目标生成的最大距离"""
        max_reach = self.calculate_max_reach()
        return max_reach * 0.85  # 85%的可达范围，留15%挑战性
    
    def generate_unified_target(self):
        """统一的目标生成策略 - 基于可达范围的智能生成"""
        max_target_distance = self.calculate_target_range()
        min_target_distance = 0.05  # 最小距离，避免太容易
        
        # 使用极坐标生成目标
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
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
            'is_success': terminated,
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range()
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
        # 统一的重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        
        # 🎯 使用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

# XML配置生成函数
def get_unified_3joint_xml():
    """统一的3关节XML配置"""
    return """
<mujoco model="unified_3joint">
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
          <body name="fingertip" pos="0.11 0 0">
            <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- 目标 - 使用slide关节 -->
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
    """统一的4关节XML配置"""
    return """
<mujoco model="unified_4joint">
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
            <body name="fingertip" pos="0.088 0 0">
              <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- 目标 - 使用slide关节 -->
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
    """统一的5关节XML配置"""
    return """
<mujoco model="unified_5joint">
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
              <body name="fingertip" pos="0.066 0 0">
                <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- 目标 - 使用slide关节 -->
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

# 统一的环境类
class Unified3JointReacherEnv(UnifiedTargetReacherEnv):
    """统一目标生成的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified3JointReacherEnv 初始化")
        print("   🎯 使用统一的基于可达范围的目标生成策略")
        
        super().__init__(
            xml_content=get_unified_3joint_xml(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],  # 3个0.1长度的链节
            render_mode=render_mode,
            **kwargs
        )

class Unified4JointReacherEnv(UnifiedTargetReacherEnv):
    """统一目标生成的4关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified4JointReacherEnv 初始化")
        print("   🎯 使用统一的基于可达范围的目标生成策略")
        
        super().__init__(
            xml_content=get_unified_4joint_xml(),
            num_joints=4,
            link_lengths=[0.08, 0.08, 0.08, 0.08],  # 4个0.08长度的链节
            render_mode=render_mode,
            **kwargs
        )

class Unified5JointReacherEnv(UnifiedTargetReacherEnv):
    """统一目标生成的5关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Unified5JointReacherEnv 初始化")
        print("   🎯 使用统一的基于可达范围的目标生成策略")
        
        super().__init__(
            xml_content=get_unified_5joint_xml(),
            num_joints=5,
            link_lengths=[0.06, 0.06, 0.06, 0.06, 0.06],  # 5个0.06长度的链节
            render_mode=render_mode,
            **kwargs
        )

# 统一的2关节环境包装器
class Unified2JointReacherWrapper(gym.Wrapper):
    """统一的2关节Reacher包装器 - 使用相同的目标生成策略"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]  # 标准2关节的链长
        print("🌟 Unified2JointReacherWrapper 初始化")
        print("   🎯 为标准2关节Reacher应用统一的目标生成策略")
        print(f"   链长: {self.link_lengths}")
        print(f"   最大可达距离: {self.calculate_max_reach():.3f}")
        print(f"   目标生成范围: {self.calculate_target_range():.3f}")
    
    def calculate_max_reach(self):
        """计算理论最大可达距离"""
        return sum(self.link_lengths)
    
    def calculate_target_range(self):
        """计算目标生成的最大距离"""
        max_reach = self.calculate_max_reach()
        return max_reach * 0.85  # 85%的可达范围
    
    def generate_unified_target(self):
        """统一的目标生成策略"""
        max_target_distance = self.calculate_target_range()
        min_target_distance = 0.05
        
        # 使用极坐标生成目标
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def reset(self, **kwargs):
        # 调用原始环境的reset
        obs, info = self.env.reset(**kwargs)
        
        # 应用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        
        # 直接设置目标位置
        reacher_env = self.env.unwrapped
        qpos = reacher_env.data.qpos.copy()
        qpos[-2:] = [target_x, target_y]
        qvel = reacher_env.data.qvel.copy()
        reacher_env.set_state(qpos, qvel)
        
        # 获取新的观察
        obs = reacher_env._get_obs()
        
        # 更新info
        if info is None:
            info = {}
        info.update({
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range(),
            'target_pos': [target_x, target_y]
        })
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 添加统一的信息
        if info is None:
            info = {}
        info.update({
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range()
        })
        
        return obs, reward, terminated, truncated, info

# 环境创建工厂函数 - 统一目标生成版本
def make_unified_2joint_env():
    """创建统一目标生成的2关节环境工厂函数"""
    def _init():
        env = gym.make('Reacher-v5', render_mode='human')
        env = Unified2JointReacherWrapper(env)  # 应用统一目标生成策略
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=2)
        env = Monitor(env)
        return env
    return _init

def make_unified_3joint_env():
    """创建统一目标生成的3关节环境工厂函数"""
    def _init():
        env = Unified3JointReacherEnv(render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=3)
        env = Monitor(env)
        return env
    return _init

def make_unified_4joint_env():
    """创建统一目标生成的4关节环境工厂函数"""
    def _init():
        env = Unified4JointReacherEnv(render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=4)
        env = Monitor(env)
        return env
    return _init

def make_unified_5joint_env():
    """创建统一目标生成的5关节环境工厂函数"""
    def _init():
        env = Unified5JointReacherEnv(render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=19)
        env = MixedJointActionWrapper(env, original_action_dim=5)
        env = Monitor(env)
        return env
    return _init

def train_unified_parallel_2to5_joint(total_timesteps: int = 40000):
    """
    统一目标生成策略的2-5关节并行训练
    所有环境使用相同的基于可达范围的智能目标生成策略
    """
    print("🚀 统一目标生成策略的2-5关节并行Reacher训练")
    print(f"🎯 同时训练{SUPPORTED_JOINTS}关节Reacher")
    print("💡 创新1: 所有环境使用统一的基于可达范围的目标生成策略")
    print("💡 创新2: 极坐标方式在合理可达范围内生成目标")
    print("💡 创新3: 确保各关节数的成功率和训练体验相似")
    print("="*60)
    
    # 创建并行环境
    print("🌍 创建统一目标生成的并行训练环境...")
    
    env_fns = []
    env_makers = [make_unified_2joint_env, make_unified_3joint_env, make_unified_4joint_env, make_unified_5joint_env]
    
    for i, (joints, maker) in enumerate(zip(SUPPORTED_JOINTS, env_makers)):
        env_fns.append(maker())
        print(f"   ✅ {joints}关节环境已添加 (统一目标生成策略)")
    
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
    
    print("✅ 统一目标生成的并行环境创建完成")
    print(f"   环境数量: {len(env_fns)} ({len(SUPPORTED_JOINTS)}种关节数)")
    print(f"   统一观察空间: {vec_env.observation_space}")
    print(f"   统一动作空间: {vec_env.action_space}")
    print(f"   🎯 所有环境现在都使用相同的智能目标生成策略")
    
    # 创建SAC模型
    print("\\n🤖 创建统一目标生成的2-5关节SAC模型...")
    
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
        tensorboard_log="./tensorboard_logs/unified_parallel_2to5/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )
    
    print("✅ 统一目标生成的2-5关节SAC模型创建完成")
    print("   ✅ 混合关节特征提取器")
    print("   ✅ 支持2-5关节统一处理")
    print("   ✅ 统一的智能目标生成策略")
    print("   ✅ 基于可达范围的极坐标目标分布")
    print("   ✅ 各关节数训练体验一致")
    
    # 开始训练
    print(f"\\n🎯 开始统一目标生成的2-5关节训练 ({total_timesteps}步)...")
    print("💡 您现在应该看到4个MuJoCo窗口同时训练:")
    print("   🔸 窗口1: 2关节Reacher (统一目标生成)")
    print("   🔸 窗口2: 3关节Reacher (统一目标生成)")
    print("   🔸 窗口3: 4关节Reacher (统一目标生成)")
    print("   🔸 窗口4: 5关节Reacher (统一目标生成)")
    print("💡 所有环境共享同一个神经网络模型")
    print("💡 所有环境使用相同的智能目标生成策略")
    print("💡 目标生成范围: 85%可达范围，确保合理挑战性")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\\n✅ 统一目标生成的2-5关节训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        print(f"🚀 并行效率: {len(env_fns)}个环境同时训练")
        
        # 保存最终模型
        model.save("models/unified_parallel_2to5_joint_final")
        print("💾 统一目标生成模型已保存: models/unified_parallel_2to5_joint_final")
        
        return model
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/unified_parallel_2to5_joint_interrupted")
        print("💾 中断模型已保存")
        return model
    
    finally:
        vec_env.close()

def main():
    """主函数"""
    print("🌟 统一目标生成策略的2-5关节Reacher训练系统")
    print("🎯 创新1: 所有环境使用统一的基于可达范围的目标生成策略")
    print("🎯 创新2: 极坐标方式确保目标在合理可达范围内")
    print("🎯 创新3: 各关节数的训练体验和成功率相似")
    print("💡 解决了不同关节数目标reset行为不一致的问题")
    print()
    
    try:
        # 开始统一目标生成的2-5关节训练
        train_unified_parallel_2to5_joint(total_timesteps=40000)
        
        print(f"\\n🎉 统一目标生成的2-5关节训练完成！")
        print(f"💡 您应该看到了4个环境同时训练的效果")
        print(f"✅ 所有环境现在都使用相同的智能目标生成策略")
        print(f"✅ 目标分布更合理，各关节数成功率应该更相近")
        print(f"✅ 一套模型现在可以更好地控制2-5关节的机械臂")
        
    except KeyboardInterrupt:
        print(f"\\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


