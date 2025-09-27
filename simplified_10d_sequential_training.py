#!/usr/bin/env python3
"""
简化的10维观测依次训练脚本：
🔧 将所有关节数的观测统一缩减到10维，减少复杂度
🎯 观测结构: [cos1, cos2, sin1, sin2, ee_x, ee_y, target_x, target_y, target_vec_x, target_vec_y]
💡 策略: 只保留前2个关节角度 + 末端位置 + 目标位置 + 目标向量
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import time
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 🔧 简化的配置参数
SUCCESS_THRESHOLD = 0.03  # 成功阈值 3cm
TARGET_MIN_RADIUS = 0.06  # 目标最小半径 6cm
PARALLEL_ENVS = 4  # 并行环境数量
TOTAL_TIMESTEPS = 100000  # 减少训练步数到10万
UNIFIED_OBS_DIM = 10  # 🎯 统一观测维度为10维

class Simplified10DExtractor(BaseFeaturesExtractor):
    """简化的10维特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(Simplified10DExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        assert obs_dim == UNIFIED_OBS_DIM, f"期望观测维度为{UNIFIED_OBS_DIM}，实际为{obs_dim}"
        
        print(f"🔧 Simplified10DExtractor: {obs_dim}维 -> {features_dim}维 (简化版)")
        
        # 🔧 简化的网络结构，无Dropout
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# 🔧 统一奖励计算的混入类
class UnifiedRewardMixin:
    """统一奖励计算的混入类"""
    
    def compute_unified_reward(self, fingertip_pos, target_pos, action=None):
        """统一的奖励计算函数"""
        distance = np.linalg.norm(fingertip_pos - target_pos)
        max_reach = getattr(self, 'max_reach', 0.3)
        
        # 标准化距离奖励
        reward = -distance / max_reach
        
        # 成功奖励
        if distance < SUCCESS_THRESHOLD:
            reward += 3.0
        
        # 轻微的控制代价
        if action is not None:
            control_cost = 0.01 * np.sum(np.square(action))
            reward -= control_cost
        
        return reward

# 简化的Reacher环境基类
class Simplified10DReacherEnv(MujocoEnv, UnifiedRewardMixin):
    """简化的10维Reacher环境基类"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        self.max_reach = sum(link_lengths)
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(xml_content)
        self.xml_file.flush()
        
        # 🎯 统一的10维观测空间
        obs_low = np.array([-1.0] * 10, dtype=np.float32)  # 所有分量都在[-1,1]范围
        obs_high = np.array([1.0] * 10, dtype=np.float32)
        observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.step_count = 0
        self.max_episode_steps = 100
        
        print(f"✅ {num_joints}关节Reacher创建完成 (10维简化版)")
        print(f"   链长: {link_lengths}")
        print(f"   最大可达距离: {self.max_reach:.3f}")
        print(f"   🎯 统一观测维度: {UNIFIED_OBS_DIM}维")
    
    def calculate_target_range(self):
        """计算目标生成的最大距离"""
        return self.max_reach * 0.85
    
    def generate_unified_target(self):
        """统一的目标生成策略"""
        max_target_distance = self.calculate_target_range()
        min_target_distance = TARGET_MIN_RADIUS
        
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def step(self, action):
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_10d_obs()
        
        # 使用统一的奖励计算
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
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_10d_obs(self):
        """🎯 统一的10维观测函数"""
        # 获取关节角度（只取前2个关节，不足的用0填充）
        theta = self.data.qpos.flat[:self.num_joints]
        if len(theta) < 2:
            theta = np.pad(theta, (0, 2 - len(theta)), 'constant')
        
        # 只保留前2个关节的角度信息
        joint1_angle = theta[0] if len(theta) > 0 else 0.0
        joint2_angle = theta[1] if len(theta) > 1 else 0.0
        
        # 获取位置信息
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        
        # 归一化位置到[-1, 1]范围
        normalized_fingertip = fingertip_pos / self.max_reach
        normalized_target = target_pos / self.max_reach
        
        # 计算目标向量（从末端执行器指向目标）
        target_vector = target_pos - fingertip_pos
        normalized_target_vector = target_vector / self.max_reach
        
        # 🎯 构建10维观测: [cos1, cos2, sin1, sin2, ee_x, ee_y, target_x, target_y, target_vec_x, target_vec_y]
        obs = np.array([
            np.cos(joint1_angle),           # cos1
            np.cos(joint2_angle),           # cos2  
            np.sin(joint1_angle),           # sin1
            np.sin(joint2_angle),           # sin2
            normalized_fingertip[0],        # ee_x (归一化)
            normalized_fingertip[1],        # ee_y (归一化)
            normalized_target[0],           # target_x (归一化)
            normalized_target[1],           # target_y (归一化)
            normalized_target_vector[0],    # target_vec_x (归一化)
            normalized_target_vector[1],    # target_vec_y (归一化)
        ], dtype=np.float32)
        
        # 确保所有值都在[-1, 1]范围内
        obs = np.clip(obs, -1.0, 1.0)
        
        return obs
    
    def reset_model(self):
        # 重置策略
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()
        
        # 只给机械臂关节添加随机速度
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        
        # 使用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_10d_obs()

# XML配置（保持不变，使用之前的XML）
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

# 环境类
class Simplified2JointReacherEnv(Simplified10DReacherEnv):
    """简化的2关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Simplified2JointReacherEnv 初始化 (10维)")
        
        super().__init__(
            xml_content=get_2joint_xml(),
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

class Simplified3JointReacherEnv(Simplified10DReacherEnv):
    """简化的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Simplified3JointReacherEnv 初始化 (10维)")
        
        super().__init__(
            xml_content=get_3joint_xml(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

# 🔧 统一奖励的2关节包装器
class Simplified2JointReacherWrapper(gym.Wrapper, UnifiedRewardMixin):
    """简化的2关节Reacher包装器 - 统一到10维观测"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]
        self.max_reach = sum(self.link_lengths)
        self.max_episode_steps = 100
        
        # 🎯 重新定义观测空间为10维
        obs_low = np.array([-1.0] * UNIFIED_OBS_DIM, dtype=np.float32)
        obs_high = np.array([1.0] * UNIFIED_OBS_DIM, dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        print("🌟 Simplified2JointReacherWrapper 初始化 (10维)")
        print(f"   链长: {self.link_lengths}")
        print(f"   最大可达距离: {self.max_reach:.3f}")
        print(f"   🎯 统一观测维度: {UNIFIED_OBS_DIM}维")
    
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
        
        # 🎯 获取10维观测
        obs = self._get_10d_obs()
        
        if info is None:
            info = {}
        info.update({
            'max_reach': self.max_reach,
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
        
        # 🎯 获取10维观测
        obs = self._get_10d_obs()
        
        if info is None:
            info = {}
        info.update({
            'max_reach': self.max_reach,
            'distance_to_target': distance,
            'is_success': is_success,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        })
        
        return obs, reward, terminated, truncated, info
    
    def _get_10d_obs(self):
        """🎯 获取10维观测"""
        reacher_env = self.env.unwrapped
        theta = reacher_env.data.qpos.flat[:2]  # 2个关节角度
        
        # 获取位置信息
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        
        # 归一化位置到[-1, 1]范围
        normalized_fingertip = fingertip_pos / self.max_reach
        normalized_target = target_pos / self.max_reach
        
        # 计算目标向量
        target_vector = target_pos - fingertip_pos
        normalized_target_vector = target_vector / self.max_reach
        
        # 🎯 构建10维观测
        obs = np.array([
            np.cos(theta[0]),               # cos1
            np.cos(theta[1]),               # cos2
            np.sin(theta[0]),               # sin1
            np.sin(theta[1]),               # sin2
            normalized_fingertip[0],        # ee_x
            normalized_fingertip[1],        # ee_y
            normalized_target[0],           # target_x
            normalized_target[1],           # target_y
            normalized_target_vector[0],    # target_vec_x
            normalized_target_vector[1],    # target_vec_y
        ], dtype=np.float32)
        
        # 确保所有值都在[-1, 1]范围内
        obs = np.clip(obs, -1.0, 1.0)
        
        return obs

def create_env(num_joints, render_mode=None):
    """创建指定关节数的简化环境"""
    if num_joints == 2:
        # 使用统一奖励的包装器
        env = gym.make('Reacher-v5', render_mode=render_mode)
        env = Simplified2JointReacherWrapper(env)
    elif num_joints == 3:
        env = Simplified3JointReacherEnv(render_mode=render_mode)
    else:
        raise ValueError(f"简化版本只支持2-3关节，不支持{num_joints}关节")
    
    env = Monitor(env)
    return env

def make_env(num_joints, render_mode=None):
    """创建环境的工厂函数"""
    def _init():
        return create_env(num_joints, render_mode)
    return _init

def train_simplified_joint_model(num_joints, total_timesteps=TOTAL_TIMESTEPS):
    """🔧 训练简化的单个关节数模型"""
    print(f"\\n🚀 开始训练简化的{num_joints}关节Reacher模型 (10维观测)")
    print(f"📊 训练步数: {total_timesteps}")
    print(f"🎯 成功阈值: {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
    print(f"🔧 并行环境数: {PARALLEL_ENVS}")
    print(f"🎯 统一观测维度: {UNIFIED_OBS_DIM}维")
    print("="*60)
    
    # 创建并行训练环境（不渲染）
    env_fns = [make_env(num_joints, render_mode=None) for _ in range(PARALLEL_ENVS)]
    vec_env = DummyVecEnv(env_fns)
    
    # 添加标准化
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 创建简化的SAC模型
    policy_kwargs = {
        'features_extractor_class': Simplified10DExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=5000,   # 减少learning_starts
        device='cpu',
        tensorboard_log=f"./tensorboard_logs/simplified_10d_{num_joints}joint/",
        batch_size=256,         
        buffer_size=500000,     # 减少buffer_size
        learning_rate=3e-4,     # 使用标准学习率
        gamma=0.99,
        tau=0.005,
        gradient_steps=1,
    )
    
    print(f"✅ 简化的{num_joints}关节SAC模型创建完成")
    print(f"   🎯 统一观测维度: {UNIFIED_OBS_DIM}维")
    print(f"   🔧 并行环境: {PARALLEL_ENVS}个")
    print(f"   🔧 标准化: 观测+奖励")
    print(f"   🔧 无Dropout特征提取器")
    print(f"   🔧 统一奖励函数")
    
    # 开始训练
    print(f"\\n🎯 开始简化的{num_joints}关节训练...")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\\n✅ 简化的{num_joints}关节训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        # 保存模型和标准化统计
        model_path = f"models/simplified_10d_{num_joints}joint_reacher"
        model.save(model_path)
        vec_env.save(f"{model_path}_vecnormalize.pkl")
        print(f"💾 模型已保存: {model_path}")
        
        return model, vec_env, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\n⚠️ 简化的{num_joints}关节训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model_path = f"models/simplified_10d_{num_joints}joint_reacher_interrupted"
        model.save(model_path)
        vec_env.save(f"{model_path}_vecnormalize.pkl")
        print(f"💾 中断模型已保存: {model_path}")
        return model, vec_env, training_time
    
    finally:
        vec_env.close()

def test_simplified_joint_model(num_joints, model, vec_env, n_eval_episodes=10):
    """🔧 测试简化的单个关节数模型"""
    print(f"\\n🧪 开始测试简化的{num_joints}关节模型 (10维观测)")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD} ({SUCCESS_THRESHOLD*100}cm)")
    print("-"*40)
    
    # 创建测试环境（带渲染）
    test_env = create_env(num_joints, render_mode='human')
    test_env = Monitor(test_env)
    
    # 应用相同的标准化统计
    test_vec_env = DummyVecEnv([lambda: test_env])
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=True, training=False)
    
    # 加载训练时的标准化统计
    try:
        test_vec_env.load_running_average(vec_env)
        print("✅ 已加载训练时的标准化统计")
    except:
        print("⚠️ 无法加载标准化统计，使用默认值")
    
    try:
        # 手动计算成功率
        success_episodes = 0
        episode_rewards = []
        episode_distances = []
        
        for episode in range(n_eval_episodes):
            obs = test_vec_env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            
            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_vec_env.step(action)
                episode_reward += reward[0]
                
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
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
            
            print(f"   Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_episodes / n_eval_episodes
        avg_reward = np.mean(episode_rewards)
        avg_min_distance = np.mean(episode_distances)
        
        print(f"\\n🎯 简化的{num_joints}关节模型测试结果:")
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
            'episode_rewards': episode_rewards,
            'episode_distances': episode_distances
        }
        
    except KeyboardInterrupt:
        print(f"\\n⚠️ 简化的{num_joints}关节测试被用户中断")
        return None
    
    finally:
        test_vec_env.close()

def main():
    """主函数：简化的10维观测依次训练和测试"""
    print("🌟 简化的10维观测依次训练和测试Reacher系统")
    print("🎯 将所有关节数的观测统一缩减到10维，减少复杂度")
    print(f"📊 配置: 每个模型训练{TOTAL_TIMESTEPS}步，{PARALLEL_ENVS}个并行环境，测试10个episodes")
    print(f"🎯 成功阈值: {SUCCESS_THRESHOLD*100}cm，目标最小半径: {TARGET_MIN_RADIUS*100}cm")
    print(f"🎯 统一观测维度: {UNIFIED_OBS_DIM}维")
    print()
    
    print("🎯 10维观测结构:")
    print("   [cos1, cos2, sin1, sin2, ee_x, ee_y, target_x, target_y, target_vec_x, target_vec_y]")
    print("   💡 只保留前2个关节角度 + 末端位置 + 目标位置 + 目标向量")
    print()
    
    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)
    
    # 存储所有结果
    all_results = []
    training_times = []
    
    # 简化版本只训练2-3关节
    joint_numbers = [2, 3]
    
    for num_joints in joint_numbers:
        try:
            print(f"\\n{'='*60}")
            print(f"🔄 当前进度: 简化的{num_joints}关节 Reacher ({joint_numbers.index(num_joints)+1}/{len(joint_numbers)})")
            print(f"{'='*60}")
            
            # 训练简化的模型
            model, vec_env, training_time = train_simplified_joint_model(num_joints, total_timesteps=TOTAL_TIMESTEPS)
            training_times.append(training_time)
            
            # 测试简化的模型
            test_result = test_simplified_joint_model(num_joints, model, vec_env, n_eval_episodes=10)
            
            if test_result:
                all_results.append(test_result)
            
            print(f"\\n✅ 简化的{num_joints}关节 Reacher 完成!")
            
        except KeyboardInterrupt:
            print(f"\\n⚠️ 在简化的{num_joints}关节训练时被用户中断")
            break
        except Exception as e:
            print(f"\\n❌ 简化的{num_joints}关节训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出最终总结
    print(f"\\n{'='*80}")
    print("🎉 简化的10维观测依次训练和测试完成!")
    print(f"{'='*80}")
    
    if all_results:
        print("\\n📊 简化模型性能总结:")
        print("-"*80)
        print(f"{'关节数':<8} {'成功率':<10} {'平均奖励':<12} {'平均最小距离':<15} {'训练时间':<10}")
        print("-"*80)
        
        for i, result in enumerate(all_results):
            training_time_min = training_times[i] / 60 if i < len(training_times) else 0
            print(f"{result['num_joints']:<8} {result['success_rate']:<10.1%} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<15.4f} {training_time_min:<10.1f}分钟")
        
        print("-"*80)
        
        # 简化效果分析
        print(f"\\n🎯 简化效果分析:")
        success_rates = [r['success_rate'] for r in all_results]
        avg_success_rate = np.mean(success_rates)
        print(f"   平均成功率: {avg_success_rate:.1%}")
        print(f"   观测维度: {UNIFIED_OBS_DIM}维 (相比原版19维减少了47%)")
        print(f"   训练步数: {TOTAL_TIMESTEPS} (相比原版20万步减少了50%)")
        print(f"   并行环境: {PARALLEL_ENVS}个")
        
        # 模型文件总结
        print(f"\\n💾 所有简化模型已保存到 models/ 目录:")
        for result in all_results:
            print(f"   - models/simplified_10d_{result['num_joints']}joint_reacher.zip")
            print(f"   - models/simplified_10d_{result['num_joints']}joint_reacher_vecnormalize.pkl")
        
        # 结论
        print(f"\\n🎯 结论:")
        if avg_success_rate > 0.6:
            print(f"   ✅ 简化非常成功！10维观测达到{avg_success_rate:.1%}成功率")
        elif avg_success_rate > 0.4:
            print(f"   ✅ 简化效果良好！10维观测达到{avg_success_rate:.1%}成功率")
        elif avg_success_rate > 0.2:
            print(f"   ⚠️ 简化有一定效果，10维观测达到{avg_success_rate:.1%}成功率")
        else:
            print(f"   ❌ 简化可能过度，10维观测仅达到{avg_success_rate:.1%}成功率")
    
    print(f"\\n🎯 简化的10维观测训练完成！大幅减少了模型复杂度。")

if __name__ == "__main__":
    main()


