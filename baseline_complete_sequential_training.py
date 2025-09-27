#!/usr/bin/env python3
"""
完整的依次训练和测试脚本：
1. 依次训练2-5关节Reacher，每个训练30000步
2. 每次训练完立即测试10个episodes
3. 最后统计所有关节数的成功率对比
4. 修复了成功判断逻辑，使用合理的成功阈值
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

# 🎯 统一为标准Reacher-v5奖励参数
SUCCESS_THRESHOLD_2JOINT = 0.05  # 2关节保持原有阈值 5cm
SUCCESS_THRESHOLD_RATIO = 0.25   # 3+关节：k = 0.25，成功阈值与可达半径的比例
# 标准Reacher-v5奖励参数
REWARD_NEAR_WEIGHT = 1.0         # 距离奖励权重（标准Reacher-v5）
REWARD_CONTROL_WEIGHT = 0.1      # 控制惩罚权重（标准Reacher-v5）
TARGET_MIN_RATIO = 0.15          # 目标最小距离比例（3+关节）
TARGET_MAX_RATIO = 0.85          # 目标最大距离比例（3+关节）

# 移除自定义特征提取器，使用纯baseline SAC
# class SpecializedJointExtractor(BaseFeaturesExtractor):
#     """专门针对特定关节数的特征提取器（移除Dropout）"""
#     
#     def __init__(self, observation_space: gym.Space, features_dim: int = 128):
#         super(SpecializedJointExtractor, self).__init__(observation_space, features_dim)
#         
#         obs_dim = observation_space.shape[0]
#         
#         print(f"🔧 SpecializedJointExtractor: {obs_dim}维 -> {features_dim}维")
#         
#         # 简化网络结构，移除Dropout
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.ReLU(),
#             nn.LayerNorm(256),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.LayerNorm(256),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.LayerNorm(128),
#             nn.Linear(128, features_dim),
#             nn.ReLU()
#         )
#     
#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         return self.net(observations)

# 修复目标滚动问题的基类
class SequentialReacherEnv(MujocoEnv):
    """依次训练用的Reacher环境基类（3+关节应用统一奖励规范）"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, show_position_info=False, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        self.show_position_info = show_position_info  # 🎯 控制是否显示实时位置信息
        
        # 🎯 计算可达半径R和统一的成功阈值（仅3+关节）
        self.max_reach = sum(link_lengths)
        if num_joints >= 3:
            self.success_threshold = SUCCESS_THRESHOLD_RATIO * self.max_reach
            self.use_unified_reward = True
        else:
            self.success_threshold = SUCCESS_THRESHOLD_2JOINT
            self.use_unified_reward = False
        
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
            render_mode=render_mode,
            width=480,  # 与标准Reacher一致
            height=480  # 与标准Reacher一致
        )
        
        self.step_count = 0
        self.max_episode_steps = 100  # 测试时每个episode 100步
        
        # 确保使用标准MuJoCo渲染机制（V-Sync会自动处理FPS）
        # 移除自定义FPS控制，依赖MuJoCo的内置机制
        
        reward_type = "统一奖励规范" if self.use_unified_reward else "默认奖励"
        position_info_status = "开启" if self.show_position_info else "关闭"
        print(f"✅ {num_joints}关节Reacher创建完成 ({reward_type}, 位置信息显示: {position_info_status})")
        print(f"   链长: {link_lengths}")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
        print(f"   🎯 成功阈值: {self.success_threshold:.3f}")
        if self.use_unified_reward:
            print(f"   🎯 成功阈值比例: {SUCCESS_THRESHOLD_RATIO:.1%} * R")
            print(f"   🎯 目标生成范围: {self.calculate_unified_target_min():.3f} ~ {self.calculate_unified_target_max():.3f}")
        else:
            print(f"   🎯 目标生成范围: {self.calculate_target_range():.3f}")
        
        if self.show_position_info:
            print(f"   📍 实时位置信息: 每10步显示一次end-effector位置")
    
    def calculate_max_reach(self):
        """计算理论最大可达距离"""
        return sum(self.link_lengths)
    
    def calculate_target_range(self):
        """计算目标生成的最大距离（2关节用）"""
        max_reach = self.calculate_max_reach()
        return max_reach * 0.85  # 85%的可达范围，留15%挑战性
    
    def calculate_unified_target_min(self):
        """计算统一目标生成的最小距离（3+关节用）"""
        return TARGET_MIN_RATIO * self.max_reach
    
    def calculate_unified_target_max(self):
        """计算统一目标生成的最大距离（3+关节用）"""
        return TARGET_MAX_RATIO * self.max_reach
    
    def generate_unified_target(self):
        """🎯 统一的目标生成策略 - 基于可达范围的智能生成"""
        if self.use_unified_reward:
            # 3+关节：使用统一的目标生成策略
            min_distance = self.calculate_unified_target_min()
            max_distance = self.calculate_unified_target_max()
        else:
            # 2关节：保持原有策略
            max_distance = self.calculate_target_range()
            min_distance = 0.05  # 最小距离，避免太容易
        
        # 使用极坐标生成目标
        target_distance = self.np_random.uniform(min_distance, max_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def step(self, action):
        # 使用标准MuJoCo步骤，让内置的V-Sync处理FPS
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward = self.reward(action)
        
        # 🎯 关键修复：像标准Reacher一样在step中渲染
        if self.render_mode == "human":
            self.render()
        
        # 计算距离
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 移除terminate选项：不再因为到达目标而提前结束
        terminated = False
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        # 计算归一化距离（仅3+关节）
        normalized_distance = distance / self.max_reach if self.use_unified_reward else None
        
        # 🎯 实时显示end-effector位置信息（每10步显示一次，避免刷屏）
        # if hasattr(self, 'show_position_info') and self.show_position_info and self.step_count % 10 == 0:
        #     joint_angles = self.data.qpos[:self.num_joints]
        #     print(f"📍 Step {self.step_count}: End-effector=({fingertip_pos[0]:.4f}, {fingertip_pos[1]:.4f}), Target=({target_pos[0]:.4f}, {target_pos[1]:.4f}), 距离={distance:.4f}, 奖励={reward:.3f}")
        #     if self.num_joints >= 3:
        #         print(f"   关节角度: [{', '.join([f'{angle:.3f}' for angle in joint_angles])}], 归一化距离={normalized_distance:.3f}")
        
        info = {
            'distance_to_target': distance,
            'normalized_distance': normalized_distance,
            'is_success': distance < self.success_threshold,  # 🔧 关键：统一的成功判断（仅用于统计）
            'max_reach': self.max_reach,
            'success_threshold': self.success_threshold,
            'use_unified_reward': self.use_unified_reward,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    # def reward(self, action):
    #     fingertip_pos = self.get_body_com("fingertip")[:2]
    #     target_pos = self.get_body_com("target")[:2]
    #     distance = np.linalg.norm(fingertip_pos - target_pos)
        
     
    #         # 2关节：保持默认奖励
    #         # 距离奖励
    #     reward = -distance
            
        
         
    #     total_reward = reward
        
    #     return total_reward
    
    # def reward(self, action):
    #     fingertip_pos = self.get_body_com("fingertip")[:2]
    #     target_pos = self.get_body_com("target")[:2]
    #     distance = np.linalg.norm(fingertip_pos - target_pos)
        
    #     # 🎯 所有关节统一使用标准Reacher-v5奖励函数
    #     # 1. 距离奖励：-1.0 * distance_to_target
    #     distance_reward = -REWARD_NEAR_WEIGHT * distance
        
    #     # 2. 控制惩罚：-0.1 * sum(action²)
    #     control_penalty = -REWARD_CONTROL_WEIGHT * np.sum(np.square(action))
        
    #     # 标准Reacher-v5总奖励
    #     total_reward = distance_reward + control_penalty
        
    #     return total_reward

    def reward(self, action):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 统一奖励尺度：3+关节使用归一化距离，2关节保持标准Reacher-v5
        if self.use_unified_reward:  # 3+关节使用归一化距离
            normalized_distance = distance / self.max_reach
            distance_reward = -REWARD_NEAR_WEIGHT * normalized_distance
            # 调试信息：显示归一化距离的使用
            # if hasattr(self, 'show_position_info') and self.show_position_info and self.step_count % 20 == 0:
            #     print(f"   🎯 使用归一化距离奖励: 实际距离={distance:.3f}, 归一化={normalized_distance:.3f}, 距离奖励={distance_reward:.3f}")
        else:  # 2关节保持标准Reacher-v5奖励
            distance_reward = -REWARD_NEAR_WEIGHT * distance
        
        # 控制惩罚保持不变
        control_penalty = -REWARD_CONTROL_WEIGHT * np.sum(np.square(action))
        
        total_reward = distance_reward + control_penalty
        
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
        # 🔧 修复目标滚动问题的重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()  # 从初始速度开始
        
        # 🎯 只给机械臂关节添加随机速度，目标关节速度保持为0
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        # qvel[-2:] 保持为0，这样目标就不会滚动了！
        
        # 🎯 使用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

# XML配置生成函数
def get_2joint_xml():
    """2关节XML配置（使用标准Reacher-v5的结构）"""
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

def get_5joint_xml():
    """5关节XML配置（统一动力学参数 + 自碰撞检测）"""
    return """
<mujoco model="5joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
  </default>
  <contact>
    <pair geom1="link0" geom2="link2" condim="3"/>
    <pair geom1="link0" geom2="link3" condim="3"/>
    <pair geom1="link0" geom2="link4" condim="3"/>
    <pair geom1="link1" geom2="link3" condim="3"/>
    <pair geom1="link1" geom2="link4" condim="3"/>
    <pair geom1="link2" geom2="link4" condim="3"/>
  </contact>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.7 0.7 10" type="plane"/>
    <geom conaffinity="0" contype="0" fromto="-.7 -.7 .01 .7 -.7 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto=" .7 -.7 .01 .7  .7 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.7  .7 .01 .7  .7 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.7 -.7 .01 -.7 .7 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.06 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.06 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.06 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="2" conaffinity="2"/>
        <body name="body2" pos="0.06 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.06 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="4" conaffinity="4"/>
          <body name="body3" pos="0.06 0 0">
            <joint axis="0 0 1" limited="true" name="joint3" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
            <geom fromto="0 0 0 0.06 0 0" name="link3" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="8" conaffinity="8"/>
            <body name="body4" pos="0.06 0 0">
              <joint axis="0 0 1" limited="true" name="joint4" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
              <geom fromto="0 0 0 0.06 0 0" name="link4" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="16" conaffinity="16"/>
              <body name="fingertip" pos="0.066 0 0">
                <geom contype="0" conaffinity="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
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
class Sequential2JointReacherEnv(SequentialReacherEnv):
    """依次训练的2关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Sequential2JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_2joint_xml(),
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class Sequential3JointReacherEnv(SequentialReacherEnv):
    """依次训练的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, show_position_info=False, **kwargs):
        print("🌟 Sequential3JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_3joint_xml(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            show_position_info=show_position_info,
            **kwargs
        )

class Sequential4JointReacherEnv(SequentialReacherEnv):
    """依次训练的4关节Reacher环境"""
    
    def __init__(self, render_mode=None, show_position_info=False, **kwargs):
        print("🌟 Sequential4JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_4joint_xml(),
            num_joints=4,
            link_lengths=[0.08, 0.08, 0.08, 0.08],
            render_mode=render_mode,
            show_position_info=show_position_info,
            **kwargs
        )

class Sequential5JointReacherEnv(SequentialReacherEnv):
    """依次训练的5关节Reacher环境"""
    
    def __init__(self, render_mode=None, show_position_info=False, **kwargs):
        print("🌟 Sequential5JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_5joint_xml(),
            num_joints=5,
            link_lengths=[0.06, 0.06, 0.06, 0.06, 0.06],
            render_mode=render_mode,
            show_position_info=show_position_info,
            **kwargs
        )

# 修复2关节环境包装器（用于标准Reacher-v5）
class Sequential2JointReacherWrapper(gym.Wrapper):
    """依次训练的2关节Reacher包装器 - 使用标准Reacher-v5（保持默认奖励）"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]
        self.max_episode_steps = 100  # 测试时每个episode 100步
        
        # 🎯 2关节保持默认设置
        self.max_reach = sum(self.link_lengths)
        self.success_threshold = SUCCESS_THRESHOLD_2JOINT
        self.use_unified_reward = False
        
        print("🌟 Sequential2JointReacherWrapper 初始化 (默认奖励)")
        print(f"   链长: {self.link_lengths}")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
        print(f"   🎯 成功阈值: {self.success_threshold:.3f}")
        print(f"   🎯 目标生成范围: {self.calculate_target_range():.3f}")
    
    def calculate_max_reach(self):
        return sum(self.link_lengths)
    
    def calculate_target_range(self):
        max_reach = self.calculate_max_reach()
        return max_reach * 0.85
    
    def generate_unified_target(self):
        max_target_distance = self.calculate_target_range()
        min_target_distance = 0.05
        
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 应用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        
        # 🔧 修复目标滚动问题：确保目标速度为0
        reacher_env = self.env.unwrapped
        qpos = reacher_env.data.qpos.copy()
        qvel = reacher_env.data.qvel.copy()
        
        # 设置目标位置
        qpos[-2:] = [target_x, target_y]
        # 🔧 关键修复：确保目标速度为0
        qvel[-2:] = [0.0, 0.0]
        
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
        
        # 🔧 重新计算成功判断 - 这是关键修复！
        reacher_env = self.env.unwrapped
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 2关节：使用默认成功阈值
        is_success = distance < self.success_threshold
        
        # 🎯 实时显示end-effector位置信息（2关节环境）
        if hasattr(self, 'show_position_info') and self.show_position_info:
            if not hasattr(self, 'step_count'):
                self.step_count = 0
            self.step_count += 1
            
            # if self.step_count % 10 == 0:
            #     joint_angles = reacher_env.data.qpos[:2]  # 2关节
            #     print(f"📍 Step {self.step_count}: End-effector=({fingertip_pos[0]:.4f}, {fingertip_pos[1]:.4f}), Target=({target_pos[0]:.4f}, {target_pos[1]:.4f}), 距离={distance:.4f}, 奖励={reward:.3f}")
            #     print(f"   关节角度: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}], 成功={'✅' if is_success else '❌'}")
        
        # 添加统一的信息
        if info is None:
            info = {}
        info.update({
            'distance_to_target': distance,
            'normalized_distance': None,  # 2关节不使用归一化距离
            'is_success': is_success,  # 🔧 关键修复：添加正确的成功判断
            'max_reach': self.max_reach,
            'success_threshold': self.success_threshold,
            'use_unified_reward': self.use_unified_reward,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        })
        
        return obs, reward, terminated, truncated, info

def create_env(num_joints, render_mode=None, show_position_info=False):
    """创建指定关节数的环境"""
    if num_joints == 2:
        # 使用标准Reacher-v5 + 包装器
        env = gym.make('Reacher-v5', render_mode=render_mode)
        env = Sequential2JointReacherWrapper(env)
        # 为2关节环境添加位置信息显示功能
        env.show_position_info = show_position_info
    elif num_joints == 3:
        env = Sequential3JointReacherEnv(render_mode=render_mode, show_position_info=show_position_info)
    elif num_joints == 4:
        env = Sequential4JointReacherEnv(render_mode=render_mode, show_position_info=show_position_info)
    elif num_joints == 5:
        env = Sequential5JointReacherEnv(render_mode=render_mode, show_position_info=show_position_info)
    else:
        raise ValueError(f"不支持的关节数: {num_joints}")
    
    env = Monitor(env)
    return env

def train_single_joint_model(num_joints, total_timesteps=50000):
    """训练单个关节数的模型"""
    print(f"\\n🚀 开始训练{num_joints}关节Reacher模型")
    print(f"📊 训练步数: {total_timesteps}")
    if num_joints >= 3:
        print(f"🎯 成功阈值: {SUCCESS_THRESHOLD_RATIO:.1%} * R (统一奖励规范)")
    else:
        print(f"🎯 成功阈值: {SUCCESS_THRESHOLD_2JOINT} ({SUCCESS_THRESHOLD_2JOINT*100}cm) (默认奖励)")
    print("="*60)
    
    # 创建训练环境（开启渲染和位置信息显示）
    train_env = create_env(num_joints, render_mode='human', show_position_info=True)
    
    # 创建纯baseline SAC模型（移除自定义特征提取器）
    # policy_kwargs = {
    #     'features_extractor_class': SpecializedJointExtractor,
    #     'features_extractor_kwargs': {'features_dim': 128},
    # }
    
    model = SAC(
        'MlpPolicy',  # 使用标准MlpPolicy，不传入任何policy_kwargs
        train_env,
        # policy_kwargs=policy_kwargs,  # 移除自定义policy_kwargs
        verbose=2,
        learning_starts=1000,
        device='cpu',
        tensorboard_log=f"./tensorboard_logs/baseline_sequential_{num_joints}joint/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,  # 使用标准学习率
        gamma=0.99,
        tau=0.005,
    )
    
    print(f"✅ {num_joints}关节纯baseline SAC模型创建完成（无自定义特征提取器）")
    
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
        model_path = f"models/baseline_sequential_{num_joints}joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        return model, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\n⚠️ {num_joints}关节训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model_path = f"models/baseline_sequential_{num_joints}joint_reacher_interrupted"
        model.save(model_path)
        print(f"💾 中断模型已保存: {model_path}")
        return model, training_time
    
    finally:
        train_env.close()

def test_single_joint_model(num_joints, model, n_eval_episodes=10):
    """测试单个关节数的模型"""
    print(f"\\n🧪 开始测试{num_joints}关节模型")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    if num_joints >= 3:
        print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD_RATIO:.1%} * R (统一奖励规范)")
    else:
        print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD_2JOINT} ({SUCCESS_THRESHOLD_2JOINT*100}cm) (默认奖励)")
    print("-"*40)
    
    # 创建测试环境（带渲染和位置信息显示）
    test_env = create_env(num_joints, render_mode='human', show_position_info=True)
    
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
            use_unified_reward = info.get('use_unified_reward', False)
            
            for step in range(100):  # 每个episode 100步
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                
                # 获取距离和成功信息
                distance = info.get('distance_to_target', float('inf'))
                normalized_distance = info.get('normalized_distance', None)
                is_success = info.get('is_success', False)
                
                min_distance = min(min_distance, distance)
                if normalized_distance is not None:
                    min_normalized_distance = min(min_normalized_distance, normalized_distance)
                
                if is_success and not episode_success:
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_episodes += 1
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
            episode_normalized_distances.append(min_normalized_distance if min_normalized_distance != float('inf') else None)
            
            normalized_dist_str = f", 归一化距离={min_normalized_distance:.3f}" if min_normalized_distance != float('inf') else ""
            print(f"   Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}{normalized_dist_str}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_episodes / total_episodes
        avg_reward = np.mean(episode_rewards)
        avg_min_distance = np.mean(episode_distances)
        avg_normalized_distance = np.mean([d for d in episode_normalized_distances if d is not None]) if any(d is not None for d in episode_normalized_distances) else None
        
        reward_type = "默认奖励" if num_joints == 2 else "统一奖励规范"
        print(f"\\n🎯 {num_joints}关节模型测试结果 ({reward_type}):")
        print(f"   成功率: {success_rate:.1%} ({success_episodes}/{total_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均最小距离: {avg_min_distance:.4f}")
        if avg_normalized_distance is not None:
            print(f"   平均归一化距离: {avg_normalized_distance:.3f}")
        print(f"   🎯 可达半径R: {max_reach:.3f}")
        print(f"   🎯 成功阈值: {success_threshold:.3f}")
        
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
            'use_unified_reward': use_unified_reward,
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
    """主函数：依次训练和测试3-5关节Reacher（统一奖励规范 + 自碰撞检测）"""
    print("🌟 Baseline版本：完整的依次训练和测试2-5关节Reacher系统")
    print("🎯 策略: 每个关节数单独训练，使用纯baseline SAC（无自定义特征提取器）")
    print("🔧 纯baseline SAC配置:")
    print(f"   1. 标准MlpPolicy（默认网络结构）")
    print(f"   2. 标准学习率: 3e-4")
    print(f"   3. 统一标准Reacher-v5奖励: -1.0*distance - 0.1*sum(action²)")
    print(f"   4. 成功阈值: 2关节={SUCCESS_THRESHOLD_2JOINT}m, 3+关节={SUCCESS_THRESHOLD_RATIO:.1%}*R")
    print(f"   5. 目标分布统一按R取比例: {TARGET_MIN_RATIO:.1%} ~ {TARGET_MAX_RATIO:.1%} * R")
    print(f"🛡️ 自碰撞检测: 防止机械臂穿透自己，提高物理真实性")
    print(f"📊 配置: 每个模型训练30000步，测试10个episodes")
    print(f"   - 3+关节成功阈值: {SUCCESS_THRESHOLD_RATIO:.1%} * R (统一)")
    print("💾 输出: 每个关节数保存独立的模型文件")
    print("📈 最终: 统计所有关节数的成功率和奖励一致性对比")
    print()
    
    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)
    
    # 存储所有结果
    all_results = []
    training_times = []
    
    # 从3关节开始训练（跳过2关节）
    joint_numbers = [3]
    
    for num_joints in joint_numbers:
        try:
            print(f"\\n{'='*60}")
            print(f"🔄 当前进度: {num_joints}关节 Reacher ({joint_numbers.index(num_joints)+1}/{len(joint_numbers)})")
            print(f"{'='*60}")
            
            # 训练模型
            model, training_time = train_single_joint_model(num_joints, total_timesteps=50000)
            training_times.append(training_time)
            
            # 测试模型
            test_result = test_single_joint_model(num_joints, model, n_eval_episodes=10)
            
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
    print("🎉 完整的依次训练和测试3-5关节Reacher完成!")
    print(f"{'='*80}")
    
    if all_results:
        print("\\n📊 所有模型性能总结:")
        print("-"*100)
        print(f"{'关节数':<8} {'奖励类型':<12} {'成功率':<10} {'平均奖励':<12} {'平均距离':<12} {'归一化距离':<12} {'训练时间':<10}")
        print("-"*100)
        
        for i, result in enumerate(all_results):
            training_time_min = training_times[i] / 60 if i < len(training_times) else 0
            reward_type = "统一奖励"  # 3+关节都使用统一奖励
            normalized_dist = result.get('avg_normalized_distance', 'N/A')
            normalized_dist_str = f"{normalized_dist:.3f}" if normalized_dist != 'N/A' and normalized_dist is not None else 'N/A'
            print(f"{result['num_joints']:<8} {reward_type:<12} {result['success_rate']:<10.1%} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<12.4f} {normalized_dist_str:<12} {training_time_min:<10.1f}分钟")
        
        print("-"*100)
        
        # 找出最佳模型
        best_model = max(all_results, key=lambda x: x['success_rate'])
        print(f"\\n🏆 最佳成功率模型: {best_model['num_joints']}关节")
        print(f"   成功率: {best_model['success_rate']:.1%}")
        print(f"   平均奖励: {best_model['avg_reward']:.2f}")
        print(f"   平均最小距离: {best_model['avg_min_distance']:.4f}")
        
        # 🎯 奖励一致性分析
        print(f"\\n🎯 奖励一致性分析:")
        success_rates = [r['success_rate'] for r in all_results]
        rewards = [r['avg_reward'] for r in all_results]
        normalized_distances = [r.get('avg_normalized_distance') for r in all_results if r.get('avg_normalized_distance') is not None]
        
        print(f"   成功率一致性: 标准差 {np.std(success_rates):.3f} (越小越一致)")
        print(f"   奖励一致性: 标准差 {np.std(rewards):.3f} (越小越一致)")
        if len(normalized_distances) > 1:
            print(f"   归一化距离一致性: 标准差 {np.std(normalized_distances):.3f} (越小越一致)")
        
        # 成功率趋势分析
        print(f"\\n📈 成功率趋势分析:")
        joint_nums = [r['num_joints'] for r in all_results]
        
        for i, (joints, rate) in enumerate(zip(joint_nums, success_rates)):
            reward_type = "统一"  # 3+关节都使用统一奖励
            trend = ""
            if i > 0:
                prev_rate = success_rates[i-1]
                if rate > prev_rate:
                    trend = f" (↗ +{(rate-prev_rate)*100:.1f}%)"
                elif rate < prev_rate:
                    trend = f" (↘ -{(prev_rate-rate)*100:.1f}%)"
                else:
                    trend = " (→ 持平)"
            print(f"   {joints}关节({reward_type}): {rate:.1%}{trend}")
        
        # 模型文件总结
        print(f"\\n💾 所有模型已保存到 models/ 目录:")
        for result in all_results:
            print(f"   - models/baseline_sequential_{result['num_joints']}joint_reacher.zip")
        
        # 详细统计
        print(f"\\n📋 详细统计信息:")
        print(f"   总训练时间: {sum(training_times)/60:.1f} 分钟")
        print(f"   平均成功率: {np.mean(success_rates):.1%}")
        print(f"   成功率标准差: {np.std(success_rates):.1%}")
        print(f"   平均奖励标准差: {np.std(rewards):.3f}")
        print(f"   🎯 3+关节成功阈值比例: {SUCCESS_THRESHOLD_RATIO:.1%} * R")
        
        # 结论
        print(f"\\n🎯 结论:")
        
        # 统一奖励规范效果评估
        if len(normalized_distances) > 1:
            normalized_std = np.std(normalized_distances)
            if normalized_std < 0.1:
                print(f"   ✅ 统一奖励规范非常成功！3+关节归一化距离一致性很好 (标准差: {normalized_std:.3f})")
            elif normalized_std < 0.2:
                print(f"   ✅ 统一奖励规范效果良好！3+关节归一化距离相对一致 (标准差: {normalized_std:.3f})")
            else:
                print(f"   ⚠️ 统一奖励规范有一定效果，但仍有改进空间 (标准差: {normalized_std:.3f})")
        
        # 整体训练效果评估
        if best_model['success_rate'] > 0.5:
            print(f"   🏆 整体训练成功！{best_model['num_joints']}关节模型表现最佳")
        elif max(success_rates) > 0.3:
            print(f"   ⚠️ 部分成功，最佳模型成功率为{max(success_rates):.1%}")
        else:
            print(f"   ❌ 整体表现较差，可能需要进一步调整参数")
        
        # 奖励一致性评估
        reward_std = np.std(rewards)
        if reward_std < 5.0:
            print(f"   ✅ 奖励一致性良好 (标准差: {reward_std:.3f})")
        elif reward_std < 10.0:
            print(f"   ⚠️ 奖励一致性一般 (标准差: {reward_std:.3f})")
        else:
            print(f"   ❌ 奖励一致性较差 (标准差: {reward_std:.3f})，建议进一步优化统一奖励规范")
    
    print(f"\\n🎯 完整的依次训练和测试完成！")
    print(f"   - 3+关节：应用GPT-5统一奖励规范，实现奖励可比性")
    print(f"   - 自碰撞检测：防止机械臂穿透自己，提高物理真实性")
    print(f"   - 每个关节数都有了专门优化的模型和详细的性能统计")

if __name__ == "__main__":
    main()
