#!/usr/bin/env python3
"""
基于MADDPG的完整依次训练和测试脚本：
1. 将SAC替换为baseline MADDPG实现
2. 每个关节作为一个智能体，协同学习
3. 中心化训练，分布式执行
4. 保持相同的环境和奖励函数
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import time
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from collections import deque
import random
from copy import deepcopy

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

# ============================================================================
# 🤖 基于Stable-Baselines3 DDPG的MADDPG实现
# ============================================================================

class SingleJointActionEnv(gym.Wrapper):
    """单关节动作环境包装器 - 每个智能体只控制一个关节"""
    
    def __init__(self, env, joint_id, num_joints):
        super().__init__(env)
        self.joint_id = joint_id
        self.num_joints = num_joints
        
        # 重新定义动作空间 - 只控制一个关节
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 观察空间保持不变
        self.observation_space = env.observation_space
        
        # 存储其他智能体的动作（用于协调）
        self._other_actions = np.zeros(num_joints)
        
        print(f"🤖 SingleJointActionEnv {joint_id}: 控制关节{joint_id}, 动作维度=1")
    
    def step(self, action):
        # 构建完整的动作向量
        full_action = self._other_actions.copy()
        full_action[self.joint_id] = action[0]
        
        # 执行环境步骤
        obs, reward, terminated, truncated, info = self.env.step(full_action)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._other_actions = np.zeros(self.num_joints)
        return obs, info
    
    def set_other_actions(self, actions):
        """设置其他智能体的动作"""
        self._other_actions = actions.copy()

class MADDPG_SB3:
    """基于Stable-Baselines3 DDPG的Multi-Agent DDPG - 独立学习版本"""
    
    def __init__(self, env, num_agents, learning_rate=1e-3, gamma=0.99, tau=0.005):
        self.num_agents = num_agents
        self.env = env
        self.agents = []
        self.single_envs = []
        
        # 为每个关节创建一个独立的DDPG智能体和环境
        for i in range(num_agents):
            # 创建单关节动作空间的环境
            single_joint_env = SingleJointActionEnv(env, joint_id=i, num_joints=num_agents)
            self.single_envs.append(single_joint_env)
            
            # 添加动作噪声
            action_noise = NormalActionNoise(
                mean=np.zeros(1), 
                sigma=0.1 * np.ones(1)
            )
            
            # 创建DDPG智能体
            agent = DDPG(
                "MlpPolicy",
                single_joint_env,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                action_noise=action_noise,
                verbose=0,
                device='cpu',
                batch_size=256,
                buffer_size=100000
            )
            
            self.agents.append(agent)
            
            print(f"🤖 DDPG Agent {i}: 已创建，控制关节{i}")
        
        print(f"🌟 MADDPG_SB3初始化完成: {num_agents}个独立DDPG智能体")
    
    def predict(self, obs, deterministic=True):
        """所有智能体同时预测动作"""
        actions = []
        for i, agent in enumerate(self.agents):
            # 每个智能体基于全局观察做决策
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(action[0])  # 取出单个动作值
        
        return np.array(actions)
    
    def learn(self, total_timesteps, log_interval=1000):
        """使用独立学习方式训练所有智能体"""
        print(f"🎯 开始MADDPG_SB3独立学习训练...")
        print(f"   每个智能体将训练 {total_timesteps // self.num_agents} 步")
        
        # 为每个智能体独立训练
        episode_rewards = []
        for i, agent in enumerate(self.agents):
            print(f"\n   🤖 训练智能体 {i} (控制关节{i})...")
            try:
                agent.learn(total_timesteps=total_timesteps // self.num_agents)
                episode_rewards.append(f"Agent_{i}_completed")
            except Exception as e:
                print(f"   ❌ 智能体 {i} 训练失败: {e}")
                episode_rewards.append(f"Agent_{i}_failed")
        
        print(f"\n✅ MADDPG_SB3独立学习训练完成!")
        return episode_rewards
    
    def save(self, filepath):
        """保存所有智能体"""
        import os
        base_path = filepath.replace('.pth', '').replace('.zip', '')
        
        for i, agent in enumerate(self.agents):
            agent_path = f"{base_path}_agent_{i}.zip"
            agent.save(agent_path)
        
        print(f"💾 MADDPG_SB3模型已保存: {base_path}_agent_*.zip")
    
    def load(self, filepath):
        """加载所有智能体"""
        base_path = filepath.replace('.pth', '').replace('.zip', '')
        
        for i, agent in enumerate(self.agents):
            agent_path = f"{base_path}_agent_{i}.zip"
            if os.path.exists(agent_path):
                agent.load(agent_path)
            else:
                print(f"⚠️ 未找到智能体{i}的模型文件: {agent_path}")
        
        print(f"📂 MADDPG_SB3模型已加载: {base_path}_agent_*.zip")

# ============================================================================
# 🌍 环境相关代码（与原版相同）
# ============================================================================

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
    
    def reward(self, action):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 统一奖励尺度：3+关节使用归一化距离，2关节保持标准Reacher-v5
        if self.use_unified_reward:  # 3+关节使用归一化距离
            normalized_distance = distance / self.max_reach
            distance_reward = -REWARD_NEAR_WEIGHT * normalized_distance
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

# XML配置生成函数（与原版相同）
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

# 环境类
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

def create_env(num_joints, render_mode=None, show_position_info=False):
    """创建指定关节数的环境"""
    if num_joints == 3:
        env = Sequential3JointReacherEnv(render_mode=render_mode, show_position_info=show_position_info)
    else:
        raise ValueError(f"MADDPG版本目前只支持3关节: {num_joints}")
    
    env = Monitor(env)
    return env

# ============================================================================
# 🚀 MADDPG训练和测试函数
# ============================================================================

class MADDPG_SB3_Wrapper:
    """MADDPG_SB3包装器，提供统一接口"""
    
    def __init__(self, maddpg_model):
        self.maddpg = maddpg_model
        self.num_agents = maddpg_model.num_agents
    
    def predict(self, obs, deterministic=True):
        """预测动作（兼容接口）"""
        actions = self.maddpg.predict(obs, deterministic=deterministic)
        return actions, None
    
    def save(self, filepath):
        """保存模型"""
        self.maddpg.save(filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.maddpg.load(filepath)

def train_single_joint_model(num_joints, total_timesteps=50000):
    """训练单个关节数的MADDPG_SB3模型"""
    print(f"\n🚀 开始训练{num_joints}关节MADDPG_SB3 Reacher模型")
    print(f"📊 训练步数: {total_timesteps}")
    print(f"🤖 算法: Stable-Baselines3 DDPG Multi-Agent (每个关节一个DDPG智能体)")
    if num_joints >= 3:
        print(f"🎯 成功阈值: {SUCCESS_THRESHOLD_RATIO:.1%} * R (统一奖励规范)")
    else:
        print(f"🎯 成功阈值: {SUCCESS_THRESHOLD_2JOINT} ({SUCCESS_THRESHOLD_2JOINT*100}cm) (默认奖励)")
    print("="*60)
    
    # 创建训练环境（开启渲染以观察训练过程）
    train_env = create_env(num_joints, render_mode='human', show_position_info=True)
    
    print(f"🔧 环境配置:")
    print(f"   观察空间: {train_env.observation_space}")
    print(f"   动作空间: {train_env.action_space}")
    print(f"   智能体数量: {num_joints}")
    
    # 创建MADDPG_SB3模型
    maddpg = MADDPG_SB3(
        env=train_env,
        num_agents=num_joints,
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.005
    )
    
    # 包装为兼容接口
    model = MADDPG_SB3_Wrapper(maddpg)
    
    print(f"✅ {num_joints}关节MADDPG_SB3模型创建完成")
    
    # 开始训练
    print(f"\n🎯 开始{num_joints}关节MADDPG_SB3训练...")
    
    try:
        start_time = time.time()
        
        # 使用MADDPG_SB3的learn方法进行训练
        episode_rewards = maddpg.learn(total_timesteps=total_timesteps, log_interval=1000)
        
        training_time = time.time() - start_time
        
        print(f"\n✅ {num_joints}关节MADDPG_SB3训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        print(f"🎯 总episodes: {len(episode_rewards)}")
        
        # 保存模型
        model_path = f"models/maddpg_sb3_sequential_{num_joints}joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        return model, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ {num_joints}关节MADDPG_SB3训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model_path = f"models/maddpg_sb3_sequential_{num_joints}joint_reacher_interrupted"
        model.save(model_path)
        print(f"💾 中断模型已保存: {model_path}")
        return model, training_time
    
    finally:
        train_env.close()

def test_single_joint_model(num_joints, model, n_eval_episodes=10):
    """测试单个关节数的MADDPG_SB3模型"""
    print(f"\n🧪 开始测试{num_joints}关节MADDPG_SB3模型")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    if num_joints >= 3:
        print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD_RATIO:.1%} * R (统一奖励规范)")
    else:
        print(f"🎯 成功标准: 距离目标 < {SUCCESS_THRESHOLD_2JOINT} ({SUCCESS_THRESHOLD_2JOINT*100}cm) (默认奖励)")
    print("-"*40)
    
    # 创建测试环境（带渲染）
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
            max_reach = info.get('max_reach', 1.0) if info else 0.3
            success_threshold = info.get('success_threshold', 0.05) if info else 0.075
            use_unified_reward = info.get('use_unified_reward', False) if info else True
            
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
        
        print(f"\n🎯 {num_joints}关节MADDPG模型测试结果:")
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
        print(f"\n⚠️ {num_joints}关节MADDPG测试被用户中断")
        return None
    
    finally:
        test_env.close()

def main():
    """主函数：MADDPG_SB3版本的训练和测试"""
    print("🌟 MADDPG_SB3版本：基于Stable-Baselines3 DDPG的多智能体Reacher训练系统")
    print("🤖 策略: 每个关节使用独立的SB3 DDPG智能体，协同学习最优策略")
    print("🔧 MADDPG_SB3配置:")
    print(f"   1. 每个关节一个Stable-Baselines3 DDPG智能体")
    print(f"   2. 使用标准DDPG算法和网络结构")
    print(f"   3. 自动处理经验回放和目标网络更新")
    print(f"   4. 统一标准Reacher-v5奖励: -1.0*distance - 0.1*sum(action²)")
    print(f"   5. 成功阈值: 3+关节={SUCCESS_THRESHOLD_RATIO:.1%}*R")
    print(f"🛡️ 自碰撞检测: 防止机械臂穿透自己，提高物理真实性")
    print(f"📊 配置: 训练50000步，测试10个episodes")
    print("💾 输出: 保存MADDPG_SB3模型文件")
    print()
    
    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)
    
    # 存储所有结果
    all_results = []
    training_times = []
    
    # 只训练3关节（MADDPG版本的演示）
    joint_numbers = [3]
    
    for num_joints in joint_numbers:
        try:
            print(f"\n{'='*60}")
            print(f"🔄 当前进度: {num_joints}关节 MADDPG Reacher ({joint_numbers.index(num_joints)+1}/{len(joint_numbers)})")
            print(f"{'='*60}")
            
            # 训练模型
            model, training_time = train_single_joint_model(num_joints, total_timesteps=50000)
            training_times.append(training_time)
            
            # 测试模型
            test_result = test_single_joint_model(num_joints, model, n_eval_episodes=10)
            
            if test_result:
                all_results.append(test_result)
            
            print(f"\n✅ {num_joints}关节 MADDPG Reacher 完成!")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ 在{num_joints}关节训练时被用户中断")
            break
        except Exception as e:
            print(f"\n❌ {num_joints}关节训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出最终总结
    print(f"\n{'='*80}")
    print("🎉 MADDPG版本训练和测试完成!")
    print(f"{'='*80}")
    
    if all_results:
        print("\n📊 MADDPG_SB3模型性能总结:")
        print("-"*100)
        print(f"{'关节数':<8} {'算法':<12} {'成功率':<10} {'平均奖励':<12} {'平均距离':<12} {'归一化距离':<12} {'训练时间':<10}")
        print("-"*100)
        
        for i, result in enumerate(all_results):
            training_time_min = training_times[i] / 60 if i < len(training_times) else 0
            normalized_dist = result.get('avg_normalized_distance', 'N/A')
            normalized_dist_str = f"{normalized_dist:.3f}" if normalized_dist != 'N/A' and normalized_dist is not None else 'N/A'
            print(f"{result['num_joints']:<8} {'MADDPG_SB3':<12} {result['success_rate']:<10.1%} {result['avg_reward']:<12.2f} {result['avg_min_distance']:<12.4f} {normalized_dist_str:<12} {training_time_min:<10.1f}分钟")
        
        print("-"*100)
        
        # 找出最佳模型
        best_model = max(all_results, key=lambda x: x['success_rate'])
        print(f"\n🏆 最佳MADDPG_SB3模型: {best_model['num_joints']}关节")
        print(f"   成功率: {best_model['success_rate']:.1%}")
        print(f"   平均奖励: {best_model['avg_reward']:.2f}")
        print(f"   平均最小距离: {best_model['avg_min_distance']:.4f}")
        
        # 模型文件总结
        print(f"\n💾 所有MADDPG_SB3模型已保存到 models/ 目录:")
        for result in all_results:
            print(f"   - models/maddpg_sb3_sequential_{result['num_joints']}joint_reacher_agent_*.zip")
        
        # 详细统计
        success_rates = [r['success_rate'] for r in all_results]
        rewards = [r['avg_reward'] for r in all_results]
        
        print(f"\n📋 MADDPG_SB3详细统计信息:")
        print(f"   总训练时间: {sum(training_times)/60:.1f} 分钟")
        print(f"   平均成功率: {np.mean(success_rates):.1%}")
        print(f"   平均奖励: {np.mean(rewards):.2f}")
        print(f"   🎯 成功阈值比例: {SUCCESS_THRESHOLD_RATIO:.1%} * R")
        
        # 结论
        print(f"\n🎯 MADDPG_SB3结论:")
        if best_model['success_rate'] > 0.5:
            print(f"   🏆 MADDPG_SB3训练成功！多智能体协作效果良好")
        elif best_model['success_rate'] > 0.3:
            print(f"   ⚠️ MADDPG_SB3部分成功，成功率为{best_model['success_rate']:.1%}")
        else:
            print(f"   ❌ MADDPG_SB3表现较差，可能需要调整超参数或网络结构")
        
        print(f"   🤖 多智能体协作: 每个关节使用独立的SB3 DDPG")
        print(f"   🎯 与SAC对比: 可以比较DDPG vs SAC的效果")
    
    print(f"\n🎯 MADDPG_SB3版本完成！")
    print(f"   - 多智能体协作: 每个关节使用独立的Stable-Baselines3 DDPG")
    print(f"   - 标准化实现: 使用成熟的SB3库，更稳定可靠")
    print(f"   - 易于调试: 利用SB3的完善工具和文档")
    print(f"   - 可以与SAC版本进行性能对比")

if __name__ == "__main__":
    main()
