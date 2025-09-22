#!/usr/bin/env python3
"""
改进版2D Reacher环境 - 实现课程学习和改进奖励机制
解决SAC维持任务的探索vs利用冲突
"""

import numpy as np
import gym
from gym import spaces
import math
import random
import time
from collections import deque

class ImprovedReacher2DEnv(gym.Env):
    """
    改进版2D Reacher环境 - 专门为SAC维持任务优化
    """
    
    def __init__(self, num_links=3, link_lengths=None, render_mode=None, 
                 curriculum_stage=0, debug_level='INFO'):
        super(ImprovedReacher2DEnv, self).__init__()
        
        self.num_links = num_links
        self.link_lengths = link_lengths if link_lengths else [90.0] * num_links
        self.render_mode = render_mode
        self.debug_level = debug_level
        
        # 🆕 课程学习参数
        self.curriculum_stage = curriculum_stage
        self.setup_curriculum_parameters()
        
        # 动作和观察空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_links,), dtype=np.float32
        )
        
        # 观察空间：关节角度+速度+末端位置+目标位置+距离+维持状态
        obs_dim = num_links * 2 + 2 + 2 + 1 + 3  # +3 for curriculum info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 环境状态
        self.joint_angles = np.zeros(num_links)
        self.joint_velocities = np.zeros(num_links)
        self.goal_pos = np.array([200.0, 100.0])
        
        # 🆕 改进的维持系统
        self.in_maintain_zone = False
        self.maintain_counter = 0
        self.maintain_history = deque(maxlen=100)
        self.max_maintain_streak = 0
        self.consecutive_success_count = 0
        
        # 训练统计
        self.step_count = 0
        self.episode_count = 0
        self.total_success_episodes = 0
        
        # 🆕 奖励系统改进
        self.milestone_rewards_given = set()  # 防止重复给奖励
        
        self.reset()
        
    def setup_curriculum_parameters(self):
        """🆕 设置课程学习参数"""
        curriculum_configs = [
            # 阶段0: 容易 - 建立信心
            {
                'maintain_threshold': 40.0,
                'maintain_target_steps': 50,
                'leave_penalty': -2.0,
                'milestone_rewards': [5.0, 10.0, 20.0],
                'milestone_steps': [10, 25, 50],
                'description': '入门阶段 - 宽松要求'
            },
            # 阶段1: 中等 - 逐步提高
            {
                'maintain_threshold': 30.0,
                'maintain_target_steps': 150,
                'leave_penalty': -5.0,
                'milestone_rewards': [5.0, 10.0, 20.0, 30.0],
                'milestone_steps': [25, 50, 100, 150],
                'description': '进阶阶段 - 中等要求'
            },
            # 阶段2: 困难 - 最终目标
            {
                'maintain_threshold': 20.0,
                'maintain_target_steps': 300,
                'leave_penalty': -10.0,
                'milestone_rewards': [5.0, 10.0, 20.0, 30.0, 50.0],
                'milestone_steps': [50, 100, 150, 200, 300],
                'description': '专家阶段 - 严格要求'
            }
        ]
        
        # 选择当前课程配置
        config = curriculum_configs[min(self.curriculum_stage, len(curriculum_configs) - 1)]
        
        self.maintain_threshold = config['maintain_threshold']
        self.maintain_target_steps = config['maintain_target_steps']
        self.leave_penalty = config['leave_penalty']
        self.milestone_rewards = config['milestone_rewards']
        self.milestone_steps = config['milestone_steps']
        self.curriculum_description = config['description']
        
        if self.debug_level == 'INFO':
            print(f"🎓 课程学习阶段 {self.curriculum_stage}: {self.curriculum_description}")
            print(f"   维持阈值: {self.maintain_threshold}px, 目标步数: {self.maintain_target_steps}")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化关节角度
        self.joint_angles = np.random.uniform(-0.5, 0.5, self.num_links)
        self.joint_velocities = np.zeros(self.num_links)
        
        # 随机目标位置
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(100, 250)
        self.goal_pos = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])
        
        # 重置维持状态
        self.in_maintain_zone = False
        self.maintain_counter = 0
        self.milestone_rewards_given.clear()
        
        self.step_count = 0
        self.episode_count += 1
        
        return self._get_observation(), {}
    
    def step(self, action):
        """执行动作"""
        self.step_count += 1
        
        # 限制动作范围并应用
        action = np.clip(action, -1.0, 1.0)
        self.joint_velocities = action * 0.1  # 降低动作幅度，提高稳定性
        self.joint_angles += self.joint_velocities
        
        # 限制关节角度
        self.joint_angles = np.clip(self.joint_angles, -np.pi, np.pi)
        
        # 计算奖励
        reward = self._compute_improved_reward()
        
        # 检查episode结束条件
        terminated = self._is_terminated()
        truncated = self.step_count >= 1000  # 增加最大步数限制
        
        # 构建info
        info = self._build_info()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _compute_improved_reward(self):
        """🆕 改进的奖励函数 - 解决维持任务问题"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        
        total_reward = 0.0
        
        # 1. 基础距离奖励 - 温和的引导信号
        max_distance = 400.0
        distance_reward = -distance / max_distance * 1.0  # 降低权重
        total_reward += distance_reward
        
        # 2. 到达奖励
        if distance < 50.0:
            reach_reward = 2.0
            total_reward += reach_reward
        
        # 3. 🆕 改进的维持奖励系统
        maintain_reward = 0.0
        if distance < self.maintain_threshold:
            if not self.in_maintain_zone:
                # 刚进入维持区域
                self.in_maintain_zone = True
                self.maintain_counter = 1
                maintain_reward = 3.0  # 进入奖励
                if self.debug_level == 'INFO':
                    print(f"🎯 进入维持区域! 距离: {distance:.1f}px (阈值: {self.maintain_threshold}px)")
            else:
                # 继续在维持区域
                self.maintain_counter += 1
                
                # 基础维持奖励 - 每步都给
                maintain_reward = 1.0
                
                # 🆕 里程碑奖励 - 防止稀疏奖励问题
                for i, milestone_step in enumerate(self.milestone_steps):
                    if (self.maintain_counter == milestone_step and 
                        milestone_step not in self.milestone_rewards_given):
                        milestone_bonus = self.milestone_rewards[i]
                        maintain_reward += milestone_bonus
                        self.milestone_rewards_given.add(milestone_step)
                        if self.debug_level == 'INFO':
                            print(f"🏆 维持里程碑! {milestone_step}步 (+{milestone_bonus:.1f})")
                
                # 🆕 稳定性奖励 - 奖励小幅移动
                if hasattr(self, 'prev_distance'):
                    movement = abs(distance - self.prev_distance)
                    if movement < 1.0:  # 很稳定
                        maintain_reward += 2.0
                    elif movement < 3.0:  # 较稳定
                        maintain_reward += 1.0
                
                # 进度显示
                if self.maintain_counter % 25 == 0 and self.debug_level == 'INFO':
                    progress = (self.maintain_counter / self.maintain_target_steps) * 100
                    print(f"🏆 维持进度: {self.maintain_counter}/{self.maintain_target_steps} "
                          f"({progress:.1f}%) 维持奖励: +{maintain_reward:.2f}")
        else:
            # 离开维持区域
            if self.in_maintain_zone:
                # 🆕 温和的离开惩罚 - 不要太严厉
                maintain_reward = self.leave_penalty  # 从-20改为可配置的温和惩罚
                
                # 记录这次维持尝试
                self.maintain_history.append(self.maintain_counter)
                self.max_maintain_streak = max(self.max_maintain_streak, self.maintain_counter)
                
                if self.maintain_counter >= 25 and self.debug_level == 'INFO':
                    print(f"⚠️ 离开维持区域! 本次维持: {self.maintain_counter}步 "
                          f"(最佳: {self.max_maintain_streak}步) 惩罚: {self.leave_penalty}")
                
                # 重置维持状态
                self.in_maintain_zone = False
                self.maintain_counter = 0
                self.milestone_rewards_given.clear()
        
        total_reward += maintain_reward
        
        # 4. 🆕 控制平滑性奖励 - 鼓励小幅调整
        control_penalty = -0.1 * np.sum(np.square(self.joint_velocities))
        total_reward += control_penalty
        
        # 保存当前距离用于下一步计算
        self.prev_distance = distance
        
        return total_reward
    
    def _is_terminated(self):
        """检查episode是否结束"""
        # 🆕 只有达到维持目标才结束episode
        if self.maintain_counter >= self.maintain_target_steps:
            self.total_success_episodes += 1
            self.consecutive_success_count += 1
            if self.debug_level == 'INFO':
                print(f"🎉 维持任务完成! 连续成功: {self.consecutive_success_count}")
            return True
        
        # 连续失败太多次，提前结束
        if len(self.maintain_history) >= 5:
            recent_attempts = list(self.maintain_history)[-5:]
            if all(attempt < self.maintain_target_steps * 0.1 for attempt in recent_attempts):
                if self.debug_level == 'INFO':
                    print(f"⚠️ 连续失败，提前结束episode")
                self.consecutive_success_count = 0
                return True
        
        return False
    
    def _get_end_effector_position(self):
        """计算末端执行器位置"""
        x, y = 0.0, 0.0
        angle_sum = 0.0
        
        for i in range(self.num_links):
            angle_sum += self.joint_angles[i]
            x += self.link_lengths[i] * np.cos(angle_sum)
            y += self.link_lengths[i] * np.sin(angle_sum)
        
        return np.array([x, y])
    
    def _get_observation(self):
        """获取观察"""
        obs = []
        
        # 关节角度和速度
        obs.extend(self.joint_angles)
        obs.extend(self.joint_velocities)
        
        # 末端执行器位置
        end_pos = self._get_end_effector_position()
        obs.extend(end_pos)
        
        # 目标位置
        obs.extend(self.goal_pos)
        
        # 距离
        distance = np.linalg.norm(end_pos - self.goal_pos)
        obs.append(distance)
        
        # 🆕 维持相关状态信息 - 帮助SAC理解任务状态
        obs.append(float(self.in_maintain_zone))  # 是否在维持区域
        obs.append(self.maintain_counter / self.maintain_target_steps)  # 维持进度
        obs.append(self.max_maintain_streak / self.maintain_target_steps)  # 历史最佳
        
        return np.array(obs, dtype=np.float32)
    
    def _build_info(self):
        """构建info字典"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        
        return {
            'distance_to_goal': float(distance),
            'goal_reached': distance <= self.maintain_threshold,
            'maintain_completed': self.maintain_counter >= self.maintain_target_steps,
            'maintain_progress': self.maintain_counter / self.maintain_target_steps,
            'curriculum_stage': self.curriculum_stage,
            'success_rate': self.total_success_episodes / max(self.episode_count, 1),
            'consecutive_successes': self.consecutive_success_count
        }
    
    def set_curriculum_stage(self, stage):
        """🆕 设置课程学习阶段"""
        if stage != self.curriculum_stage:
            self.curriculum_stage = stage
            self.setup_curriculum_parameters()
            print(f"🎓 课程升级到阶段 {stage}: {self.curriculum_description}")
    
    def get_curriculum_progress(self):
        """🆕 获取课程学习进度"""
        return {
            'stage': self.curriculum_stage,
            'success_rate': self.total_success_episodes / max(self.episode_count, 1),
            'consecutive_successes': self.consecutive_success_count,
            'max_maintain_streak': self.max_maintain_streak,
            'ready_for_next_stage': self.consecutive_success_count >= 5
        }

# 注册环境
if __name__ == "__main__":
    # 测试环境
    env = ImprovedReacher2DEnv(num_links=5, curriculum_stage=0, debug_level='INFO')
    
    print("🧪 测试改进版Reacher环境...")
    obs, info = env.reset()
    print(f"观察空间维度: {obs.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    
    # 运行几步测试
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.3f}, distance={info['distance_to_goal']:.1f}")
        
        if terminated or truncated:
            break
    
    print("✅ 环境测试完成!")
