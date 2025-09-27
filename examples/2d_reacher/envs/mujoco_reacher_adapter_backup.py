#!/usr/bin/env python3
"""
MuJoCo Reacher 适配器
将 OpenAI MuJoCo Reacher 环境适配为与当前 Reacher2DEnv 兼容的接口
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import math
import yaml
import os
from typing import Optional, Tuple, Dict, Any

class MuJoCoReacherAdapter:
    """
    MuJoCo Reacher 环境适配器
    提供与 Reacher2DEnv 相同的接口，但使用 MuJoCo 物理引擎
    """
    
    def __init__(self, num_links=2, link_lengths=None, render_mode=None, config_path=None, curriculum_stage=1, debug_level='SILENT'):
        """初始化适配器"""
        
        # 创建 MuJoCo Reacher 环境
        self.mujoco_env = gym.make('Reacher-v5', render_mode=render_mode)
        
        # 兼容性设置
        self.gym_api_version = "old"  # 保持与原环境兼容
        self.render_mode = render_mode
        
        # 加载配置（保持与原环境相同的配置系统）
        self.config = self._load_config(config_path)
        
        # 环境参数（保持与原环境兼容）
        self.num_links = num_links  # 支持自定义关节数，但 MuJoCo Reacher 固定为 2
        if self.num_links != 2:
            print(f"⚠️ MuJoCo Reacher 只支持 2 关节，将使用 2 关节而不是 {num_links}")
            self.num_links = 2
        
        # Link 长度设置
        if link_lengths is None:
            self.link_lengths = [0.1, 0.1]  # MuJoCo 中的默认 link 长度（米）
        else:
            self.link_lengths = link_lengths[:2]  # 只取前两个值
        self.max_torque = 100  # 保持与原环境相同的扭矩范围
        
        # 坐标系转换参数
        self.scale_factor = 600  # 将 MuJoCo 坐标缩放到像素坐标
        self.anchor_point = self.config.get("start", {}).get("position", [480, 620])
        self.goal_pos = np.array(self.config.get("goal", {}).get("position", [600, 550]))
        
        # 定义适配后的空间（保持与原环境相同）
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(2,), dtype=np.float32)
        
        # 添加 spec 属性以兼容向量化环境
        self.spec = None
        if hasattr(self.mujoco_env, 'spec'):
            self.spec = self.mujoco_env.spec
        
        # 状态变量
        self.step_count = 0
        self.collision_count = 0
        self.base_collision_count = 0
        self.self_collision_count = 0
        
        # 奖励组件
        self.current_reward = 0.0
        self.reward_components = {
            'distance_reward': 0.0,
            'reach_reward': 0.0,
            'progress_reward': 0.0,
            'collision_penalty': 0.0,
            'control_penalty': 0.0
        }
        
        # 维持系统（保持与原环境兼容）
        self.maintain_threshold = 150.0
        self.maintain_target_steps = 200
        self.maintain_counter = 0
        self.maintain_bonus_given = False
        self.in_maintain_zone = False
        self.maintain_history = []
        self.max_maintain_streak = 0
        
        # Episode 管理
        self.current_episode = 1
        self.episode_ended = False
        
        print(f"🎯 MuJoCo Reacher 适配器初始化完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
        print(f"   渲染模式: {render_mode}")
    
    def _load_config(self, config_path):
        """加载配置文件（与原环境相同）"""
        if config_path is None:
            # 使用默认配置
            return {
                "start": {"position": [480, 620]},
                "goal": {"position": [600, 550]},
                "obstacles": []
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️ 无法加载配置文件 {config_path}: {e}")
            return {
                "start": {"position": [480, 620]},
                "goal": {"position": [600, 550]},
                "obstacles": []
            }
    
    def _mujoco_to_custom_obs(self, mujoco_obs):
        """将 MuJoCo 观察转换为自定义格式"""
        # MuJoCo 观察格式: [cos(θ1), cos(θ2), sin(θ1), sin(θ2), 目标x, 目标y, ω1, ω2, 向量x, 向量y]
        
        # 提取角度（从 cos/sin 重构）
        cos_theta1, cos_theta2 = mujoco_obs[0], mujoco_obs[1]
        sin_theta1, sin_theta2 = mujoco_obs[2], mujoco_obs[3]
        
        theta1 = math.atan2(sin_theta1, cos_theta1)
        theta2 = math.atan2(sin_theta2, cos_theta2)
        
        # 提取角速度
        omega1, omega2 = mujoco_obs[6], mujoco_obs[7]
        
        # 计算末端执行器位置（基于 MuJoCo 的运动学）
        # MuJoCo 中的坐标需要转换到我们的像素坐标系
        end_effector_pos = self._calculate_end_effector_from_mujoco(theta1, theta2)
        
        # 目标位置（使用配置中的目标）
        goal_pos = self.goal_pos
        
        # 计算距离
        distance = np.linalg.norm(end_effector_pos - goal_pos)
        
        # 构建 12 维观察空间
        custom_obs = np.array([
            theta1,                    # 关节角度1
            theta2,                    # 关节角度2
            omega1,                    # 关节速度1
            omega2,                    # 关节速度2
            end_effector_pos[0],       # 末端位置x
            end_effector_pos[1],       # 末端位置y
            goal_pos[0],               # 目标位置x
            goal_pos[1],               # 目标位置y
            distance,                  # 距离
            0.0,                       # 碰撞状态（MuJoCo 环境中暂时设为0）
            float(self.collision_count),      # 碰撞计数
            float(self.base_collision_count)  # 基座碰撞计数
        ], dtype=np.float32)
        
        return custom_obs
    
    def _calculate_end_effector_from_mujoco(self, theta1, theta2):
        """根据关节角度计算末端执行器位置"""
        # 将 MuJoCo 的运动学转换为我们的坐标系
        # MuJoCo 中 link 长度通常是 0.1 米
        link1_length = self.link_lengths[0] * self.scale_factor  # 转换为像素
        link2_length = self.link_lengths[1] * self.scale_factor
        
        # 计算第一个 link 的末端位置
        x1 = self.anchor_point[0] + link1_length * np.cos(theta1)
        y1 = self.anchor_point[1] + link1_length * np.sin(theta1)
        
        # 计算第二个 link 的末端位置（末端执行器）
        x2 = x1 + link2_length * np.cos(theta1 + theta2)
        y2 = y1 + link2_length * np.sin(theta1 + theta2)
        
        return np.array([x2, y2])
    
    def _custom_to_mujoco_action(self, custom_action):
        """将自定义动作转换为 MuJoCo 动作"""
        # 自定义动作范围: [-100, 100]
        # MuJoCo 动作范围: [-1, 1]
        
        # 简单的线性映射
        mujoco_action = np.clip(custom_action / self.max_torque, -1.0, 1.0)
        return mujoco_action.astype(np.float32)
    
    def _compute_custom_reward(self, mujoco_obs, mujoco_reward, custom_obs):
        """计算自定义奖励（保持与原环境相同的奖励结构）"""
        end_pos = custom_obs[4:6]  # 末端位置
        goal_pos = custom_obs[6:8]  # 目标位置
        distance = custom_obs[8]   # 距离
        
        # 🎯 主要奖励：距离奖励
        distance_reward = -distance / 50.0
        
        # 🎯 到达奖励
        reach_reward = 0.0
        if distance < 100.0:  # 100px内给予到达奖励
            reach_reward = 50.0 - distance  # 越近奖励越大
        
        # 🎯 进步奖励（鼓励朝目标移动）
        progress_reward = 0.0
        if hasattr(self, 'prev_distance'):
            if distance < self.prev_distance:
                progress_reward = (self.prev_distance - distance) * 0.1
        self.prev_distance = distance
        
        # 🎯 控制惩罚（从 MuJoCo 奖励中提取）
        control_penalty = mujoco_reward - (-distance / 50.0)  # 估算控制惩罚
        
        # 🎯 碰撞惩罚（暂时为0，因为标准 MuJoCo Reacher 没有障碍物）
        collision_penalty = 0.0
        
        # 计算总奖励
        total_reward = distance_reward + reach_reward + progress_reward + control_penalty + collision_penalty
        
        # 存储奖励组成部分
        self.reward_components = {
            'distance_reward': distance_reward,
            'reach_reward': reach_reward,
            'progress_reward': progress_reward,
            'control_penalty': control_penalty,
            'collision_penalty': collision_penalty
        }
        
        self.current_reward = total_reward
        return total_reward
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        # 重置 MuJoCo 环境
        mujoco_obs, mujoco_info = self.mujoco_env.reset(seed=seed)
        
        # 重置适配器状态
        self.step_count = 0
        self.collision_count = 0
        self.base_collision_count = 0
        self.self_collision_count = 0
        
        # 重置维持状态
        self.maintain_counter = 0
        self.maintain_bonus_given = False
        self.in_maintain_zone = False
        
        # Episode 管理
        if not hasattr(self, 'current_episode'):
            self.current_episode = 1
            self.episode_ended = False
        elif hasattr(self, 'episode_ended') and self.episode_ended:
            self.current_episode += 1
            self.episode_ended = False
        
        # 转换观察
        custom_obs = self._mujoco_to_custom_obs(mujoco_obs)
        
        print(f"🔄 MuJoCo 适配器重置完成 - Episode {self.current_episode}")
        
        if self.gym_api_version == "new":
            info = self._get_info()
            return custom_obs, info
        else:
            return custom_obs
    
    def step(self, action):
        """执行一步"""
        # 转换动作
        mujoco_action = self._custom_to_mujoco_action(action)
        
        # 在 MuJoCo 环境中执行动作
        mujoco_obs, mujoco_reward, mujoco_terminated, mujoco_truncated, mujoco_info = self.mujoco_env.step(mujoco_action)
        
        # 转换观察
        custom_obs = self._mujoco_to_custom_obs(mujoco_obs)
        
        # 计算自定义奖励
        custom_reward = self._compute_custom_reward(mujoco_obs, mujoco_reward, custom_obs)
        
        # 检查终止条件（使用自定义逻辑）
        custom_done = self._is_done(custom_obs)
        
        self.step_count += 1
        
        # 获取信息
        info = self._get_info()
        
        if self.gym_api_version == "new":
            terminated = custom_done
            truncated = self.step_count >= 120000
            return custom_obs, custom_reward, terminated, truncated, info
        else:
            return custom_obs, custom_reward, custom_done, info
    
    def _is_done(self, obs):
        """检查是否完成（使用与原环境相同的逻辑）"""
        distance = obs[8]  # 距离在观察的第9个位置
        
        # 🎯 1. 到达目标就进入下一个episode
        if distance < 50.0:  # 50px内算到达目标
            print(f"🎯 到达目标! 距离: {distance:.1f}px，进入下一个episode")
            self.episode_ended = True
            return True
        
        # 🎯 2. 步数限制（防止无限循环）
        if self.step_count >= 2000:  # 每个episode最多2000步
            print(f"⏰ Episode步数限制: {self.step_count}步，进入下一个episode")
            self.episode_ended = True
            return True
        
        return False
    
    def _get_info(self):
        """获取额外信息（保持与原环境相同的格式）"""
        # 从当前观察中提取信息
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            end_pos = self._last_obs[4:6]
            goal_pos = self._last_obs[6:8]
            distance = self._last_obs[8]
        else:
            # 如果没有观察，使用默认值
            end_pos = np.array([0.0, 0.0])
            goal_pos = self.goal_pos
            distance = np.linalg.norm(end_pos - goal_pos)
        
        return {
            'end_effector_pos': end_pos,
            'goal_pos': goal_pos,
            'distance': float(distance),
            'collision_count': self.collision_count,
            'base_collision_count': self.base_collision_count,
            'step_count': self.step_count,
            'goal': {
                'distance_to_goal': float(distance),
                'goal_reached': distance < 20.0,
                'end_effector_position': end_pos,
                'goal_position': goal_pos,
            },
            'maintain': {
                'in_maintain_zone': self.in_maintain_zone,
                'maintain_counter': self.maintain_counter,
                'maintain_target': self.maintain_target_steps,
                'maintain_progress': self.maintain_counter / self.maintain_target_steps if self.maintain_target_steps > 0 else 0.0,
                'max_maintain_streak': self.max_maintain_streak,
                'maintain_completed': self.maintain_counter >= self.maintain_target_steps
            }
        }
    
    def render(self, mode='human'):
        """渲染环境（委托给 MuJoCo 环境）"""
        return self.mujoco_env.render()
    
    def seed(self, seed=None):
        """设置随机种子"""
        if hasattr(self.mujoco_env, 'seed'):
            return self.mujoco_env.seed(seed)
        elif hasattr(self.mujoco_env, 'reset'):
            # Gymnasium 环境使用 reset(seed=seed)
            return [seed]
        return [seed]
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'mujoco_env'):
            self.mujoco_env.close()

# 为了兼容性，提供一个别名
MuJoCoReacher2DEnv = MuJoCoReacherAdapter
