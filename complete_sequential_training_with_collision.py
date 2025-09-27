#!/usr/bin/env python3
"""
带碰撞检测的完整依次训练脚本：
🔧 为3+关节环境添加自碰撞检测
🎯 防止机械臂不现实的姿态和穿透
💡 保持训练稳定性的同时增加物理真实性
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

# 成功判断阈值
SUCCESS_THRESHOLD = 0.05  # 5cm，更合理的阈值

class SpecializedJointExtractor(BaseFeaturesExtractor):
    """专门针对特定关节数的特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(SpecializedJointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 SpecializedJointExtractor: {obs_dim}维 -> {features_dim}维")
        
        # 针对具体观察维度设计网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
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
        return self.net(observations)

# 修复目标滚动问题的基类
class CollisionAwareReacherEnv(MujocoEnv):
    """带碰撞检测的Reacher环境基类"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(xml_content)
        self.xml_file.flush()
        
        # 计算观察空间维度 - 增加碰撞信息
        obs_dim = num_joints * 3 + 4 + 1  # cos, sin, vel + ee_pos + target_pos + collision_penalty
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.step_count = 0
        self.max_episode_steps = 100  # 测试时每个episode 100步
        self.collision_penalty = 0.0  # 碰撞惩罚
        
        print(f"✅ {num_joints}关节Reacher创建完成 (带碰撞检测)")
        print(f"   链长: {link_lengths}")
        print(f"   最大可达距离: {self.calculate_max_reach():.3f}")
        print(f"   目标生成范围: {self.calculate_target_range():.3f}")
        print(f"   🔧 碰撞检测: {'启用' if num_joints > 2 else '禁用'}")
    
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
    
    def check_self_collision(self):
        """检查自碰撞"""
        if self.num_joints <= 2:
            return False, 0.0
        
        # 获取所有接触信息
        collision_detected = False
        collision_penalty = 0.0
        
        # 检查MuJoCo的接触数据
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            
            # 检查是否是机械臂内部的碰撞（排除与地面和边界的碰撞）
            if (geom1_name.startswith('link') and geom2_name.startswith('link') and 
                geom1_name != geom2_name):
                # 检查是否是相邻链段（相邻链段的轻微接触是正常的）
                link1_id = int(geom1_name.replace('link', '')) if geom1_name.replace('link', '').isdigit() else -1
                link2_id = int(geom2_name.replace('link', '')) if geom2_name.replace('link', '').isdigit() else -1
                
                if abs(link1_id - link2_id) > 1:  # 非相邻链段碰撞
                    collision_detected = True
                    collision_penalty += 1.0  # 每次碰撞惩罚1.0
        
        return collision_detected, collision_penalty
    
    def step(self, action):
        # 显式渲染（修复渲染问题）
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        # 检查碰撞
        collision_detected, collision_penalty = self.check_self_collision()
        self.collision_penalty = collision_penalty
        
        observation = self._get_obs()
        reward = self.reward(action, collision_detected, collision_penalty)
        
        # 计算距离
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 统一的成功判断：距离小于SUCCESS_THRESHOLD且无碰撞
        terminated = distance < SUCCESS_THRESHOLD and not collision_detected
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance,
            'is_success': terminated,  # 🔧 关键：统一的成功判断
            'collision_detected': collision_detected,
            'collision_penalty': collision_penalty,
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range(),
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def reward(self, action, collision_detected=False, collision_penalty=0.0):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 距离奖励
        reward = -distance
        
        # 到达奖励
        if distance < SUCCESS_THRESHOLD:
            reward += 10.0
        
        # 🔧 碰撞惩罚
        if collision_detected:
            reward -= collision_penalty * 5.0  # 碰撞严重惩罚
        
        # 控制代价
        control_cost = 0.01 * np.sum(np.square(action))
        reward -= control_cost
        
        return reward
    
    def _get_obs(self):
        # N关节的观察：cos, sin, vel, fingertip_pos, target_pos, collision_penalty
        theta = self.data.qpos.flat[:self.num_joints]
        obs = np.concatenate([
            np.cos(theta),                    # N个cos值
            np.sin(theta),                    # N个sin值
            self.data.qvel.flat[:self.num_joints],  # N个关节速度
            self.get_body_com("fingertip")[:2],     # 末端执行器位置 (x,y)
            self.get_body_com("target")[:2],        # 目标位置 (x,y)
            [self.collision_penalty]                # 碰撞惩罚信息
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
        self.collision_penalty = 0.0
        return self._get_obs()

# 🔧 带碰撞检测的XML配置生成函数
def get_2joint_xml_with_collision():
    """2关节XML配置（保持原有设置，无碰撞检测）"""
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
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="0"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="0"/>
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

def get_3joint_xml_with_collision():
    """3关节XML配置（启用碰撞检测）"""
    return """
<mujoco model="3joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
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
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
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

def get_4joint_xml_with_collision():
    """4关节XML配置（启用碰撞检测）"""
    return """
<mujoco model="4joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
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
      <geom fromto="0 0 0 0.08 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.08 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.08 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
        <body name="body2" pos="0.08 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.08 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
          <body name="body3" pos="0.08 0 0">
            <joint axis="0 0 1" limited="true" name="joint3" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
            <geom fromto="0 0 0 0.08 0 0" name="link3" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
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

def get_5joint_xml_with_collision():
    """5关节XML配置（启用碰撞检测）"""
    return """
<mujoco model="5joint_reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
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
      <geom fromto="0 0 0 0.06 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.06 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.06 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
        <body name="body2" pos="0.06 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.06 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
          <body name="body3" pos="0.06 0 0">
            <joint axis="0 0 1" limited="true" name="joint3" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
            <geom fromto="0 0 0 0.06 0 0" name="link3" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
            <body name="body4" pos="0.06 0 0">
              <joint axis="0 0 1" limited="true" name="joint4" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
              <geom fromto="0 0 0 0.06 0 0" name="link4" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
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
class CollisionAware2JointReacherEnv(CollisionAwareReacherEnv):
    """带碰撞检测的2关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 CollisionAware2JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_2joint_xml_with_collision(),
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class CollisionAware3JointReacherEnv(CollisionAwareReacherEnv):
    """带碰撞检测的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 CollisionAware3JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_3joint_xml_with_collision(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class CollisionAware4JointReacherEnv(CollisionAwareReacherEnv):
    """带碰撞检测的4关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 CollisionAware4JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_4joint_xml_with_collision(),
            num_joints=4,
            link_lengths=[0.08, 0.08, 0.08, 0.08],
            render_mode=render_mode,
            **kwargs
        )

class CollisionAware5JointReacherEnv(CollisionAwareReacherEnv):
    """带碰撞检测的5关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 CollisionAware5JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_5joint_xml_with_collision(),
            num_joints=5,
            link_lengths=[0.06, 0.06, 0.06, 0.06, 0.06],
            render_mode=render_mode,
            **kwargs
        )

# 修复2关节环境包装器（用于标准Reacher-v5）
class CollisionAware2JointReacherWrapper(gym.Wrapper):
    """带碰撞检测的2关节Reacher包装器 - 使用标准Reacher-v5"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]
        self.max_episode_steps = 100  # 测试时每个episode 100步
        self.collision_penalty = 0.0
        print("🌟 CollisionAware2JointReacherWrapper 初始化")
        print(f"   链长: {self.link_lengths}")
        print(f"   最大可达距离: {self.calculate_max_reach():.3f}")
        print(f"   目标生成范围: {self.calculate_target_range():.3f}")
        print(f"   🔧 碰撞检测: 禁用 (2关节无需)")
        
        # 修改观察空间以包含碰撞信息
        original_obs_space = env.observation_space
        new_obs_dim = original_obs_space.shape[0] + 1  # 增加碰撞惩罚维度
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(new_obs_dim,), 
            dtype=original_obs_space.dtype
        )
    
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
        
        # 获取新的观察并添加碰撞信息
        obs = reacher_env._get_obs()
        self.collision_penalty = 0.0
        obs_with_collision = np.append(obs, [self.collision_penalty])
        
        # 更新info
        if info is None:
            info = {}
        info.update({
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range(),
            'target_pos': [target_x, target_y],
            'collision_detected': False,
            'collision_penalty': self.collision_penalty
        })
        
        return obs_with_collision, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2关节无碰撞检测
        collision_detected = False
        self.collision_penalty = 0.0
        
        # 🔧 重新计算成功判断 - 这是关键修复！
        reacher_env = self.env.unwrapped
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 统一的成功判断：距离小于SUCCESS_THRESHOLD
        is_success = distance < SUCCESS_THRESHOLD
        
        # 添加碰撞信息到观察
        obs_with_collision = np.append(obs, [self.collision_penalty])
        
        # 添加统一的信息
        if info is None:
            info = {}
        info.update({
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range(),
            'distance_to_target': distance,
            'is_success': is_success,  # 🔧 关键修复：添加正确的成功判断
            'collision_detected': collision_detected,
            'collision_penalty': self.collision_penalty,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        })
        
        return obs_with_collision, reward, terminated, truncated, info

def create_collision_aware_env(num_joints, render_mode=None):
    """创建指定关节数的带碰撞检测环境"""
    if num_joints == 2:
        # 使用标准Reacher-v5 + 包装器
        env = gym.make('Reacher-v5', render_mode=render_mode)
        env = CollisionAware2JointReacherWrapper(env)
    elif num_joints == 3:
        env = CollisionAware3JointReacherEnv(render_mode=render_mode)
    elif num_joints == 4:
        env = CollisionAware4JointReacherEnv(render_mode=render_mode)
    elif num_joints == 5:
        env = CollisionAware5JointReacherEnv(render_mode=render_mode)
    else:
        raise ValueError(f"不支持的关节数: {num_joints}")
    
    env = Monitor(env)
    return env

def train_collision_aware_model(num_joints, total_timesteps=30000):
    """训练带碰撞检测的单个关节数模型"""
    print(f"\\n🚀 开始训练带碰撞检测的{num_joints}关节Reacher模型")
    print(f"📊 训练步数: {total_timesteps}")
    print(f"🎯 成功阈值: {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
    print(f"🔧 碰撞检测: {'启用' if num_joints > 2 else '禁用'}")
    print("="*60)
    
    # 创建训练环境（开启渲染）
    train_env = create_collision_aware_env(num_joints, render_mode='human')
    
    # 创建SAC模型
    policy_kwargs = {
        'features_extractor_class': SpecializedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=1000,
        device='cpu',
        tensorboard_log=f"./tensorboard_logs/collision_aware_{num_joints}joint/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )
    
    print(f"✅ 带碰撞检测的{num_joints}关节SAC模型创建完成")
    
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
        model_path = f"models/collision_aware_{num_joints}joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        return model, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\n⚠️ {num_joints}关节训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model_path = f"models/collision_aware_{num_joints}joint_reacher_interrupted"
        model.save(model_path)
        print(f"💾 中断模型已保存: {model_path}")
        return model, training_time
    
    finally:
        train_env.close()

def test_collision_aware_model(num_joints, model, n_eval_episodes=10):
    """测试带碰撞检测的单个关节数模型"""
    print(f"\\n🧪 开始测试带碰撞检测的{num_joints}关节模型")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm) 且无碰撞")
    print("-"*40)
    
    # 创建测试环境（带渲染）
    test_env = create_collision_aware_env(num_joints, render_mode='human')
    
    try:
        # 手动运行episodes来计算成功率
        success_episodes = 0
        collision_episodes = 0
        total_episodes = n_eval_episodes
        episode_rewards = []
        episode_distances = []
        episode_collisions = []
        
        for episode in range(n_eval_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_success = False
            episode_collision = False
            min_distance = float('inf')
            max_collision_penalty = 0.0
            
            for step in range(100):  # 每个episode 100步
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                
                # 获取距离和成功信息
                distance = info.get('distance_to_target', float('inf'))
                is_success = info.get('is_success', False)
                collision_detected = info.get('collision_detected', False)
                collision_penalty = info.get('collision_penalty', 0.0)
                
                min_distance = min(min_distance, distance)
                max_collision_penalty = max(max_collision_penalty, collision_penalty)
                
                if collision_detected:
                    episode_collision = True
                
                if is_success and not episode_success:
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_episodes += 1
            if episode_collision:
                collision_episodes += 1
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
            episode_collisions.append(max_collision_penalty)
            
            collision_status = f"碰撞={max_collision_penalty:.1f}" if episode_collision else "无碰撞"
            print(f"   Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, {collision_status}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_episodes / total_episodes
        collision_rate = collision_episodes / total_episodes
        avg_reward = np.mean(episode_rewards)
        avg_min_distance = np.mean(episode_distances)
        avg_collision_penalty = np.mean(episode_collisions)
        
        print(f"\\n🎯 带碰撞检测的{num_joints}关节模型测试结果:")
        print(f"   成功率: {success_rate:.1%} ({success_episodes}/{total_episodes})")
        print(f"   碰撞率: {collision_rate:.1%} ({collision_episodes}/{total_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均最小距离: {avg_min_distance:.4f}")
        print(f"   平均碰撞惩罚: {avg_collision_penalty:.2f}")
        
        return {
            'num_joints': num_joints,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'success_episodes': success_episodes,
            'collision_episodes': collision_episodes,
            'total_episodes': total_episodes,
            'avg_reward': avg_reward,
            'avg_min_distance': avg_min_distance,
            'avg_collision_penalty': avg_collision_penalty,
            'episode_rewards': episode_rewards,
            'episode_distances': episode_distances,
            'episode_collisions': episode_collisions
        }
        
    except KeyboardInterrupt:
        print(f"\\n⚠️ {num_joints}关节测试被用户中断")
        return None
    
    finally:
        test_env.close()

def main():
    """主函数：带碰撞检测的依次训练和测试2-5关节Reacher"""
    print("🌟 带碰撞检测的完整依次训练和测试2-5关节Reacher系统")
    print("🎯 策略: 每个关节数单独训练，3+关节启用碰撞检测")
    print(f"📊 配置: 每个模型训练30000步，测试10个episodes，成功阈值{SUCCESS_THRESHOLD*100}cm")
    print("🔧 碰撞检测: 2关节禁用，3+关节启用自碰撞检测")
    print("💾 输出: 每个关节数保存独立的模型文件")
    print("📈 最终: 统计所有关节数的成功率和碰撞率对比")
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
            model, training_time = train_collision_aware_model(num_joints, total_timesteps=30000)
            training_times.append(training_time)
            
            # 测试模型
            test_result = test_collision_aware_model(num_joints, model, n_eval_episodes=10)
            
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
    print("🎉 带碰撞检测的完整依次训练和测试2-5关节Reacher完成!")
    print(f"{'='*80}")
    
    if all_results:
        print("\\n📊 所有模型性能总结:")
        print("-"*90)
        print(f"{'关节数':<8} {'成功率':<10} {'碰撞率':<10} {'平均奖励':<12} {'平均最小距离':<15} {'训练时间':<10}")
        print("-"*90)
        
        for i, result in enumerate(all_results):
            training_time_min = training_times[i] / 60 if i < len(training_times) else 0
            collision_rate = result.get('collision_rate', 0.0)
            print(f"{result['num_joints']:<8} {result['success_rate']:<10.1%} {collision_rate:<10.1%} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<15.4f} {training_time_min:<10.1f}分钟")
        
        print("-"*90)
        
        # 找出最佳模型
        best_model = max(all_results, key=lambda x: x['success_rate'])
        print(f"\\n🏆 最佳成功率模型: {best_model['num_joints']}关节")
        print(f"   成功率: {best_model['success_rate']:.1%}")
        print(f"   碰撞率: {best_model.get('collision_rate', 0.0):.1%}")
        print(f"   平均奖励: {best_model['avg_reward']:.2f}")
        print(f"   平均最小距离: {best_model['avg_min_distance']:.4f}")
        
        # 碰撞检测效果分析
        print(f"\\n🔧 碰撞检测效果分析:")
        for result in all_results:
            joints = result['num_joints']
            collision_rate = result.get('collision_rate', 0.0)
            if joints > 2:
                print(f"   {joints}关节: 碰撞率 {collision_rate:.1%} (碰撞检测启用)")
            else:
                print(f"   {joints}关节: 无碰撞检测 (不需要)")
        
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
        
        # 模型文件总结
        print(f"\\n💾 所有模型已保存到 models/ 目录:")
        for result in all_results:
            print(f"   - models/collision_aware_{result['num_joints']}joint_reacher.zip")
        
        # 详细统计
        print(f"\\n📋 详细统计信息:")
        print(f"   总训练时间: {sum(training_times)/60:.1f} 分钟")
        print(f"   平均成功率: {np.mean(success_rates):.1%}")
        print(f"   成功率标准差: {np.std(success_rates):.1%}")
        print(f"   成功阈值: {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
        
        # 结论
        print(f"\\n🎯 结论:")
        if best_model['success_rate'] > 0.5:
            print(f"   ✅ 训练成功！{best_model['num_joints']}关节模型表现最佳")
            if best_model.get('collision_rate', 0.0) < 0.1:
                print(f"   🔧 碰撞检测有效：碰撞率仅{best_model.get('collision_rate', 0.0):.1%}")
        elif max(success_rates) > 0.3:
            print(f"   ⚠️ 部分成功，最佳模型成功率为{max(success_rates):.1%}")
        else:
            print(f"   ❌ 整体表现较差，可能需要调整训练参数或成功阈值")
    
    print(f"\\n🎯 带碰撞检测的完整训练完成！每个关节数都有了物理真实的模型和详细的性能统计。")

if __name__ == "__main__":
    main()


