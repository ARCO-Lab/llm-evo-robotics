#!/usr/bin/env python3
"""
MuJoCo Reacher 适配器
将 OpenAI MuJoCo Reacher 环境适配为与当前 Reacher2DEnv 兼容的接口
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import Env
import math
import yaml
import os
from typing import Optional, Tuple, Dict, Any

class MuJoCoReacherAdapter(Env):
    """
    MuJoCo Reacher 环境适配器
    提供与 Reacher2DEnv 相同的接口，但使用 MuJoCo 物理引擎
    """
    
    def __init__(self, num_links=2, link_lengths=None, render_mode=None, config_path=None, curriculum_stage=1, debug_level='SILENT'):
        """初始化适配器"""
        
        # 创建 MuJoCo Reacher 环境
        self.mujoco_env = gym.make('Reacher-v5', render_mode=render_mode)
        
        # 兼容性设置
        self.gym_api_version = "old"  # 使用旧 API 以兼容向量化环境
        self.render_mode = render_mode
        
        # 设置 Gymnasium 标准属性
        super().__init__()
        
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
        self.max_torque = 0.05  # 增加扭矩，确保能产生有效运动
        
        # 坐标系转换参数
        self.scale_factor = 600  # 将 MuJoCo 坐标缩放到像素坐标
        self.anchor_point = self.config.get("start", {}).get("position", [480, 620])
        self.goal_pos = np.array(self.config.get("goal", {}).get("position", [600, 550]))
        self.initial_goal_pos = self.goal_pos.copy()  # 保存初始目标位置
        
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
    
    def _init_rotation_monitor(self):
        """初始化疯狂旋转检测系统"""
        self.rotation_monitor = {
            'joint_velocities': [],  # 关节速度历史
            'action_magnitudes': [],  # 动作幅度历史
            'position_changes': [],  # 位置变化历史
            'crazy_rotation_count': 0,  # 疯狂旋转计数
            'last_joint_positions': None,  # 上一步的关节位置
            'high_velocity_steps': 0,  # 连续高速度步数
            'detection_window': 20,  # 检测窗口大小
            'velocity_threshold': 1.0,  # 速度阈值（rad/s）- 极低阈值
            'action_threshold': 1.0,  # 动作幅度阈值 - 降低
            'crazy_threshold': 10,  # 连续异常步数阈值
        }
    
    def _detect_crazy_rotation(self, mujoco_obs, custom_action):
        """检测疯狂旋转行为"""
        monitor = self.rotation_monitor
        
        # 提取关节角度和速度
        joint_positions = mujoco_obs[:2]  # 关节位置
        joint_velocities = mujoco_obs[2:4]  # 关节速度
        
        # 计算动作幅度
        action_magnitude = np.linalg.norm(custom_action)
        
        # 计算关节速度幅度
        velocity_magnitude = np.linalg.norm(joint_velocities)
        
        # 更新历史记录
        monitor['joint_velocities'].append(velocity_magnitude)
        monitor['action_magnitudes'].append(action_magnitude)
        
        # 保持窗口大小
        if len(monitor['joint_velocities']) > monitor['detection_window']:
            monitor['joint_velocities'].pop(0)
            monitor['action_magnitudes'].pop(0)
        
        # 检测异常条件
        is_crazy = False
        crazy_reasons = []
        
        # 条件1: 关节速度过高
        if velocity_magnitude > monitor['velocity_threshold']:
            monitor['high_velocity_steps'] += 1
            crazy_reasons.append(f"高速度: {velocity_magnitude:.2f} rad/s")
        else:
            monitor['high_velocity_steps'] = 0
        
        # 条件2: 动作幅度过大
        if action_magnitude > monitor['action_threshold']:
            crazy_reasons.append(f"大动作: {action_magnitude:.2f}")
        
        # 条件3: 连续高速度
        if monitor['high_velocity_steps'] > 5:
            crazy_reasons.append(f"连续高速: {monitor['high_velocity_steps']}步")
            is_crazy = True
        
        # 条件4: 平均速度过高
        if len(monitor['joint_velocities']) >= 10:
            avg_velocity = np.mean(monitor['joint_velocities'][-10:])
            if avg_velocity > monitor['velocity_threshold'] * 0.7:
                crazy_reasons.append(f"平均高速: {avg_velocity:.2f}")
                is_crazy = True
        
        # 更新疯狂旋转计数
        if is_crazy:
            monitor['crazy_rotation_count'] += 1
            if monitor['crazy_rotation_count'] % 20 == 1:  # 每20步报告一次
                print(f"🌪️ 检测到疯狂旋转! 原因: {', '.join(crazy_reasons)}")
                print(f"   关节速度: {joint_velocities}")
                print(f"   动作输入: {custom_action}")
                print(f"   累计异常: {monitor['crazy_rotation_count']}步")
        else:
            # 重置计数器（允许偶尔的异常）
            if monitor['crazy_rotation_count'] > 0:
                monitor['crazy_rotation_count'] = max(0, monitor['crazy_rotation_count'] - 1)
        
        return is_crazy, crazy_reasons
    
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
        # 自定义动作范围: [-max_torque, max_torque]
        # MuJoCo 动作范围: [-1, 1]
        
        # 添加调试信息（每100步打印一次）
        if not hasattr(self, '_action_debug_counter'):
            self._action_debug_counter = 0
        self._action_debug_counter += 1
        
        if self._action_debug_counter % 100 == 0:
            print(f"🎮 [Step {self._action_debug_counter}] SAC动作: {custom_action}, 范围: [{np.min(custom_action):.3f}, {np.max(custom_action):.3f}]")
        
        # 修正的线性映射 - 直接使用 custom_action，不再除以 max_torque
        # 因为 custom_action 已经在 [-max_torque, max_torque] 范围内
        # 我们需要将其映射到 [-1, 1] 范围，但保持比例关系
        mujoco_action = np.clip(custom_action, -self.max_torque, self.max_torque)
        
        if self._action_debug_counter % 100 == 0:
            print(f"🎮 [Step {self._action_debug_counter}] MuJoCo动作: {mujoco_action}, 缩放比例: 1/{self.max_torque}")
        
        return mujoco_action.astype(np.float32)
    
    def _compute_custom_reward(self, mujoco_obs, mujoco_reward, custom_obs):
        """🎯 改进的奖励函数，提供更好的学习信号"""
        end_pos = custom_obs[4:6]  # 末端位置
        goal_pos = custom_obs[6:8]  # 目标位置
        distance = custom_obs[8]   # 距离（像素）
        
        # 将距离转换为标准化单位 (0-1)
        normalized_distance = min(distance / 200.0, 1.0)  # 200px 为最大有意义距离
        
        # 🎯 1. 距离奖励 - 指数衰减，越近奖励越大
        distance_reward = np.exp(-normalized_distance * 3.0) - 1.0  # 范围 [-1, 0]
        
        # 🎯 2. 到达奖励 - 大幅奖励成功
        reach_reward = 0.0
        if distance < 10.0:  # 10px 内算真正成功
            reach_reward = 20.0  # 超大奖励
        elif distance < 25.0:  # 25px 内给予较大奖励
            reach_reward = 10.0 * (25.0 - distance) / 15.0
        elif distance < 50.0:  # 50px 内给予小奖励
            reach_reward = 2.0 * (50.0 - distance) / 25.0
        
        # 🎯 3. 进步奖励 - 鼓励朝目标移动
        progress_reward = 0.0
        if hasattr(self, 'prev_distance'):
            progress = self.prev_distance - distance
            if progress > 0:  # 距离减小
                progress_reward = progress * 0.02  # 每像素进步给 0.02 奖励
            else:  # 距离增大
                progress_reward = progress * 0.01  # 轻微惩罚退步
        self.prev_distance = distance
        
        # 🎯 4. 速度奖励 - 鼓励合理的移动速度
        joint_velocities = mujoco_obs[2:4]  # 关节速度
        velocity_magnitude = np.linalg.norm(joint_velocities)
        
        # 适中的速度最好，太快或太慢都不好
        if velocity_magnitude < 0.5:
            velocity_reward = -0.1  # 太慢惩罚
        elif velocity_magnitude > 10.0:
            velocity_reward = -0.5  # 太快惩罚（疯狂旋转）
        else:
            velocity_reward = 0.1  # 合理速度奖励
        
        # 🎯 5. 控制平滑度奖励
        if hasattr(self, 'prev_action'):
            action_change = np.linalg.norm(self.current_action - self.prev_action)
            if action_change > 2.0:
                smoothness_reward = -0.2  # 动作变化太大惩罚
            else:
                smoothness_reward = 0.05  # 平滑控制奖励
        else:
            smoothness_reward = 0.0
        
        # 🎯 6. 控制惩罚 - 鼓励小的控制输入
        control_penalty = -0.01 * np.sum(np.square(self.current_action))
        
        # 计算总奖励
        total_reward = (distance_reward + reach_reward + progress_reward + 
                       velocity_reward + smoothness_reward + control_penalty)
        
        # 🎯 调试信息（每100步打印一次）
        if not hasattr(self, '_reward_debug_counter'):
            self._reward_debug_counter = 0
        self._reward_debug_counter += 1
        
        if self._reward_debug_counter % 100 == 0:
            print(f"🎁 [Step {self._reward_debug_counter}] 奖励分解:")
            print(f"   距离奖励: {distance_reward:.3f} (距离: {distance:.1f}px)")
            print(f"   到达奖励: {reach_reward:.3f}")
            print(f"   进步奖励: {progress_reward:.3f}")
            print(f"   速度奖励: {velocity_reward:.3f} (速度: {velocity_magnitude:.2f})")
            print(f"   平滑奖励: {smoothness_reward:.3f}")
            print(f"   控制惩罚: {control_penalty:.3f}")
            print(f"   总奖励: {total_reward:.3f}")
        
        # 存储当前动作用于下次计算平滑度
        if hasattr(self, 'current_action'):
            self.prev_action = self.current_action.copy()
        
        # 存储奖励组成部分
        self.reward_components = {
            'distance_reward': distance_reward,
            'reach_reward': reach_reward,
            'progress_reward': progress_reward,
            'velocity_reward': velocity_reward,
            'smoothness_reward': smoothness_reward,
            'control_penalty': control_penalty,
            'total_reward': total_reward
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
        
        # 初始化疯狂旋转检测系统
        self._init_rotation_monitor()
        
        # 转换观察
        custom_obs = self._mujoco_to_custom_obs(mujoco_obs)
        # 确保观察值是 float32 类型
        custom_obs = custom_obs.astype(np.float32)
        
        print(f"🔄 MuJoCo 适配器重置完成 - Episode {self.current_episode}")
        
        if self.gym_api_version == "new":
            info = self._get_info()
            return custom_obs, info
        else:
            return custom_obs
    
    def step(self, action):
        """执行一步"""
        # 存储当前动作用于奖励计算
        self.current_action = np.array(action)
        
        # 转换动作
        mujoco_action = self._custom_to_mujoco_action(action)
        
        # 在 MuJoCo 环境中执行动作
        mujoco_obs, mujoco_reward, mujoco_terminated, mujoco_truncated, mujoco_info = self.mujoco_env.step(mujoco_action)
        
        # 检测疯狂旋转
        is_crazy, crazy_reasons = self._detect_crazy_rotation(mujoco_obs, action)
        
        # 疯狂旋转检测（仅记录，不强制重置）
        if is_crazy and self.rotation_monitor['crazy_rotation_count'] % 20 == 0:
            print(f"⚠️ 检测到高速旋转，但继续训练... (计数: {self.rotation_monitor['crazy_rotation_count']})")
        
        # 转换观察
        custom_obs = self._mujoco_to_custom_obs(mujoco_obs)
        # 确保观察值是 float32 类型
        custom_obs = custom_obs.astype(np.float32)
        
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
        if distance < 10.0:  # 10px内算真正到达目标（更严格的标准）
            print(f"🎯 真正到达目标! 距离: {distance:.1f}px，进入下一个episode")
            self.episode_ended = True
            return True
        
        # 🎯 2. 步数限制（防止无限循环）
        if self.step_count >= 200:  # 每个episode最多200步，加快学习
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
    
    def set_goal_position(self, goal_pos):
        """动态设置目标位置"""
        self.goal_pos = np.array(goal_pos)
        print(f"🎯 目标位置更新为: [{self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f}]")
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'mujoco_env'):
            self.mujoco_env.close()

# 为了兼容性，提供一个别名
MuJoCoReacher2DEnv = MuJoCoReacherAdapter
