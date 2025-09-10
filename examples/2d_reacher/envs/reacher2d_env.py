#!/usr/bin/env python3
"""
Gymnasium版本的Reacher2D环境
兼容原有接口，但使用稳定的数学模型替代PyMunk物理仿真
解决关节分离和穿透问题
"""

import gym
from gym import Env
from gym.spaces import Box

import numpy as np
import pygame
import math
import yaml
import os
import sys
import logging
from typing import Optional, Tuple, Dict, Any

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/configs'))

class Reacher2DEnv(Env):
    """
    稳定的2D Reacher环境 - Gymnasium版本
    保持与原版相同的接口，但使用解析几何替代物理仿真
    """
    
    def __init__(self, num_links=3, link_lengths=None, render_mode=None, config_path=None, curriculum_stage=1, debug_level='SILENT'):
        super().__init__()
        
        # 设置日志
        self._set_logging(debug_level)
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.logger.info(f"self.config: {self.config}")
        
        # 从配置获取参数
        self.anchor_point = self.config["start"]["position"]
        self.gym_api_version = "old"  # 保持兼容性
        
        # 课程学习参数
        self.curriculum_stage = curriculum_stage
        self.base_goal_pos = np.array(self.config["goal"]["position"]) if "goal" in self.config else np.array([600, 575])
        print(f"🔍 [__init__] base_goal_pos from config: {self.base_goal_pos}")
        print(f"🔍 [__init__] anchor_point: {self.anchor_point}")
        print(f"🔍 [__init__] curriculum_stage: {curriculum_stage}")

        # 机器人参数
        self.num_links = num_links
        if link_lengths is None:
            self.link_lengths = [60] * num_links
        else:
            assert len(link_lengths) == num_links
            self.link_lengths = link_lengths
        
        self.render_mode = render_mode
        
        # 物理参数
        self.dt = 1/120.0  # 保持与原版相同
        self.max_torque = 100  # 保持与原版相同
        
        # 状态变量
        self.joint_angles = np.zeros(num_links)
        self.joint_velocities = np.zeros(num_links)
        self.step_count = 0

        
        # 目标位置（从配置或课程学习）
        self.goal_pos = self.base_goal_pos.copy()
        
        # 障碍物（从配置加载）
        self.obstacles = self._load_obstacles()
        
        # 定义空间
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(self.num_links,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 8,), dtype=np.float32)
        
        # 渲染相关
        self.screen = None
        self.clock = None
        self.draw_options = None
        
        # 🎯 实时显示变量
        self.current_reward = 0.0
        self.reward_components = {
            'distance_reward': 0.0,
            'reach_reward': 0.0,
            'collision_penalty': 0.0,
            'control_penalty': 0.0
        }
        
        # ✅ 为了兼容训练系统，添加PyMunk兼容属性
        self._create_compatibility_bodies()  # 创建模拟的body对象
        self._create_compatibility_space()  # 创建模拟的space对象
        
        # 统计信息
        self.collision_count = 0
        self.base_collision_count = 0
        self.self_collision_count = 0
        if self.render_mode:
            self._init_rendering()

    def _create_compatibility_bodies(self):
        """创建兼容训练系统的模拟body对象"""
        class MockBody:
            def __init__(self, x=0, y=0, angle=0):
                self.position = (x, y)
                self.angle = angle
                self.shapes = []  # 空的shapes列表，兼容渲染系统
                self.velocity = (0, 0)  # 空的velocity，兼容渲染系统
                self.angular_velocity = 0  # 空的angular_velocity，兼容渲染系统
        
        # 为每个link创建一个模拟body
        self.bodies = [MockBody() for _ in range(self.num_links)]
    
    def _create_compatibility_space(self):
        """创建兼容渲染系统的模拟space对象"""
        class MockSpace:
            def debug_draw(self, draw_options):
                # 空的debug_draw方法，兼容渲染系统
                pass
        
        self.space = MockSpace()
    
    def _update_compatibility_bodies(self):
        """更新模拟body的位置和角度"""
        link_positions = self._calculate_link_positions()
        
        for i, body in enumerate(self.bodies):
            if i < len(link_positions):
                x, y = link_positions[i]
                body.position = (x, y)
                body.angle = self.joint_angles[i] if i < len(self.joint_angles) else 0
    
    def _set_logging(self, debug_level):
        """设置日志系统"""
        self.logger = logging.getLogger(f"Reacher2DEnv_{id(self)}")
        self.logger.handlers = []  # 清除现有处理器
        
        if debug_level != 'SILENT':
            # 设置日志级别
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR
            }
            log_level = level_map.get(debug_level, logging.INFO)
            self.logger.setLevel(log_level)

            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            formatter = logging.Formatter('%(levelname)s [Reacher2D]: %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.log_level = self.logger.level
            self.is_debug = self.log_level <= logging.DEBUG
            self.is_info = self.log_level <= logging.INFO
            self.is_warning = self.log_level <= logging.WARNING
        self.is_silent = debug_level == 'SILENT'
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            # 默认配置
            return {
                "start": {"position": [300, 300]},
                "goal": {"position": [600, 550]},
                "obstacles": []
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.warning(f"无法加载配置文件 {config_path}: {e}")
            return {
                "start": {"position": [300, 300]},
                "goal": {"position": [600, 550]},
                "obstacles": []
            }
    
    def _load_obstacles(self):
        """从配置加载障碍物"""
        obstacles = []
        if "obstacles" in self.config:
            for obs_config in self.config["obstacles"]:
                if "type" in obs_config:
                    # 新格式：有type字段
                    if obs_config["type"] == "circle":
                        obstacles.append({
                            'center': obs_config["center"],
                            'radius': obs_config["radius"]
                        })
                    elif obs_config["type"] == "zigzag":
                        # 处理之字形障碍物
                        start = obs_config["start"]
                        end = obs_config["end"]
                        segments = obs_config.get("segments", 3)
                        width = obs_config.get("width", 20)
                        
                        # 简化为多个圆形障碍物
                        for i in range(segments + 1):
                            t = i / segments
                            x = start[0] + t * (end[0] - start[0])
                            y = start[1] + t * (end[1] - start[1])
                            # 添加之字形偏移
                            if i % 2 == 1:
                                y += width
                            obstacles.append({
                                'center': [x, y],
                                'radius': width // 2
                            })
                elif "shape" in obs_config:
                    # 旧格式：segment形状
                    if obs_config["shape"] == "segment":
                        points = obs_config["points"]
                        # 保持线段障碍物的原始形式
                        obstacles.append({
                            'type': 'segment',
                            'start': points[0],
                            'end': points[1],
                            'thickness': 8  # 线段的厚度（像link一样）
                        })
        
        return obstacles
    
    def _init_rendering(self):
        """初始化渲染"""
        pygame.init()
        self.width, self.height = 1200, 1200
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Stable Reacher2D (Gymnasium)")
        self.clock = pygame.time.Clock()
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 🔍 调试：确认这个reset方法被调用
        print(f"🔍 [RESET] reacher2d_env.py reset方法被调用")
        
        # 重置关节状态 - 竖直向下
        # 第一个关节(基座)指向下方(π/2), 其他关节为0(直线)
        self.joint_angles = np.zeros(self.num_links)
        self.joint_angles[0] = np.pi/2  # 90度，垂直向下
        # 其他关节保持0，形成一条直线
        
        print(f"🔍 [RESET] 设置后的关节角度: {self.joint_angles}")
        print(f"🔍 [RESET] 第一个关节: {self.joint_angles[0]:.4f} 弧度 = {np.degrees(self.joint_angles[0]):.2f}°")
        
        # 添加小的随机扰动，避免完全相同的初始状态
        noise = np.random.uniform(-np.pi/36, np.pi/36, self.num_links)  # ±5度的小扰动
        self.joint_angles += noise
        
        self.joint_velocities = np.zeros(self.num_links)
        self.step_count = 0
        
        # 重置统计
        self.collision_count = 0
        self.base_collision_count = 0
        self.self_collision_count = 0  # 🆕 添加自碰撞计数重置
        
        # 🆕 重置关节使用历史记录 - 用于joint_usage_reward
        if hasattr(self, 'prev_joint_angles'):
            delattr(self, 'prev_joint_angles')

        # 课程学习目标位置
        self.goal_pos = self._get_curriculum_goal()

        # ✅ 更新兼容性body对象
        self._update_compatibility_bodies()

        observation = self._get_observation()
        
        if self.gym_api_version == "new":
            info = self._get_info()
            return observation, info
        else:
            return observation
    def step(self, action):
        """执行一步"""
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # 保存当前状态
        old_joint_angles = self.joint_angles.copy()
        old_joint_velocities = self.joint_velocities.copy()
        
        # 简化的动力学模型
        # 扭矩转换为角加速度（简化的转动惯量为1）
        torque_to_acceleration = 0.1  # 转换系数
        angular_acceleration = action * torque_to_acceleration
        
        # 更新角速度和角度
        new_joint_velocities = self.joint_velocities + angular_acceleration * self.dt
        new_joint_velocities *= 0.98  # 阻尼
        new_joint_angles = self.joint_angles + new_joint_velocities * self.dt
        
        # 角度限制
        for i in range(self.num_links):
            if i == 0:  # 基座关节可以360度旋转
                new_joint_angles[i] = new_joint_angles[i] % (2 * np.pi)
            else:  # 其他关节限制
                new_joint_angles[i] = np.clip(new_joint_angles[i], -np.pi * 7 / 8, np.pi * 7 / 8)
        
        # 临时设置新状态以检查碰撞
        self.joint_angles = new_joint_angles
        self.joint_velocities = new_joint_velocities
        
        # 检查碰撞
        collision_detected = self._check_collision()
        
        if collision_detected:
            # 如果发生碰撞，恢复到原来的状态并停止运动
            self.joint_angles = old_joint_angles
            self.joint_velocities = old_joint_velocities * 0.3  # 大幅减少速度，模拟碰撞阻尼
        # 如果没有碰撞，保持新状态
        
        # 计算奖励（传入碰撞状态避免重复检查）
        reward = self._compute_reward(collision_detected)
        
        # 检查终止条件
        done = self._is_done()
        
        self.step_count += 1
        
        # ✅ 更新兼容性body对象
        self._update_compatibility_bodies()

        observation = self._get_observation()
        info = self._get_info()
        
        if self.gym_api_version == "new":
            terminated = done
            truncated = self.step_count >= 500
            return observation, reward, terminated, truncated, info
        else:
            return observation, reward, done, info
    
    def _get_curriculum_goal(self):
        """获取课程学习的目标位置"""
        # 根据课程阶段调整目标位置
        if self.curriculum_stage == 0:
            # 简单目标：靠近基座
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(100, 150)
            return np.array(self.anchor_point) + distance * np.array([np.cos(angle), np.sin(angle)])
        else:
            # 使用配置中的目标位置
            return self.base_goal_pos.copy()
    
    def _get_end_effector_position(self):
        """计算末端执行器位置"""
        pos = np.array(self.anchor_point, dtype=float)
        current_angle = 0.0
        
        for i in range(self.num_links):
            current_angle += self.joint_angles[i]
            pos += self.link_lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
        
        return pos
    
    def _get_link_positions(self):
        """获取所有link的位置"""
        positions = [np.array(self.anchor_point, dtype=float)]
        current_angle = 0.0
        
        for i in range(self.num_links):
            current_angle += self.joint_angles[i]
            next_pos = positions[-1] + self.link_lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
            positions.append(next_pos)
        
        return positions
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """计算点到线段的最短距离"""
        # 将点和线段转换为numpy数组
        p = np.array(point)
        a = np.array(seg_start)
        b = np.array(seg_end)
        
        # 向量AB和AP
        ab = b - a
        ap = p - a
        
        # 如果线段长度为0，返回点到点的距离
        ab_length_sq = np.dot(ab, ab)
        if ab_length_sq == 0:
            return np.linalg.norm(ap)
        
        # 计算投影参数t
        t = np.dot(ap, ab) / ab_length_sq
        t = max(0, min(1, t))  # 限制t在[0,1]范围内
        
        # 计算线段上最近的点
        closest_point = a + t * ab
        
        # 返回距离
        return np.linalg.norm(p - closest_point)
    
    def _segment_to_segment_distance(self, seg1_start, seg1_end, seg2_start, seg2_end):
        """计算两个线段之间的最短距离"""
        # 检查四个点到对方线段的距离
        distances = [
            self._point_to_segment_distance(seg1_start, seg2_start, seg2_end),
            self._point_to_segment_distance(seg1_end, seg2_start, seg2_end),
            self._point_to_segment_distance(seg2_start, seg1_start, seg1_end),
            self._point_to_segment_distance(seg2_end, seg1_start, seg1_end)
        ]
        
        return min(distances)
    
    def _check_collision(self):
        """检查碰撞"""
        link_positions = self._calculate_link_positions()
        collision_detected = False
        
        # 检查每个link与障碍物的碰撞
        for i in range(len(link_positions) - 1):
            link_start = link_positions[i]
            link_end = link_positions[i + 1]
            
            # 检查这个link与所有障碍物的碰撞
            for obstacle in self.obstacles:
                if obstacle.get('type') == 'segment':
                    # 线段障碍物：计算线段到线段的距离
                    obs_start = obstacle['start']
                    obs_end = obstacle['end']
                    thickness = obstacle.get('thickness', 8)
                    
                    distance = self._segment_to_segment_distance(
                        link_start, link_end, obs_start, obs_end
                    )
                    
                    # 如果距离小于厚度的一半，则发生碰撞
                    if distance < thickness / 2 + 2:  # +2像素安全边距
                        collision_detected = True
                        if i == 0:  # 基座关节碰撞
                            self.base_collision_count += 1
                        else:
                            self.collision_count += 1
                else:
                    # 圆形障碍物：使用原来的方法
                    if 'center' in obstacle and 'radius' in obstacle:
                        mid_point = (link_start + link_end) / 2
                        dist = np.linalg.norm(mid_point - np.array(obstacle['center']))
                        if dist < obstacle['radius'] + 5:  # 5像素的安全边距
                            collision_detected = True
                            if i == 0:  # 基座关节碰撞
                                self.base_collision_count += 1
                            else:
                                self.collision_count += 1

        for i in range(len(link_positions) - 1):
            for j in range(i + 2, len(link_positions) - 1):  # 跳过相邻的link
                link1_start = link_positions[i]
                link1_end = link_positions[i + 1]
                link2_start = link_positions[j]
                link2_end = link_positions[j + 1]
                
                # 计算两个link之间的距离
                distance = self._segment_to_segment_distance(
                    link1_start, link1_end, link2_start, link2_end
                )
                
                # 如果距离小于link的厚度，则发生自碰撞
                link_thickness = 8  # 与渲染时的线宽一致
                if distance < link_thickness + 2:  # +2像素安全边距
                    collision_detected = True
                    self.collision_count += 1
                    
                    # 可以添加自碰撞的调试信息
                    if not self.is_silent:
                        print(f"🔴 自碰撞检测: Link{i} 与 Link{j} 碰撞，距离: {distance:.1f}px")
        
        return collision_detected
    
    def _get_observation(self):
        """获取观察"""
        # 保持与原版相同的观察格式
        obs = []
        
        # 关节角度和角速度
        for i in range(self.num_links):
            obs.extend([self.joint_angles[i], self.joint_velocities[i]])
        
        # 末端执行器位置
        end_pos = self._get_end_effector_position()
        obs.extend(end_pos)
        
        # 目标位置
        obs.extend(self.goal_pos)
        
        # 距离
        distance = np.linalg.norm(end_pos - self.goal_pos)
        obs.append(distance)
        
        # 碰撞状态
        collision = self._check_collision()
        obs.extend([float(collision), float(self.collision_count)])
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self, collision_detected=None):
        """计算奖励"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        
        # 距离奖励
        distance_reward = -distance / 150.0
        
        # 到达奖励
        reach_reward = 0.0
        if distance < 20.0:
            reach_reward = 10.0
        
        # 碰撞惩罚
        collision_penalty = 0.0
        if collision_detected is None:
            collision_detected = self._check_collision()
        if collision_detected:
            collision_penalty = -2.0  # 增加碰撞惩罚
        
        # 控制平滑性
        control_penalty = -0.01 * np.sum(np.square(self.joint_velocities))
        midline_reward = self._compute_midline_reward(end_pos)
        joint_usage_reward = self._compute_joint_usage_reward()
        # 🎯 存储奖励组成部分用于实时显示
        self.reward_components = {
            'distance_reward': distance_reward,
            'reach_reward': reach_reward,
            'collision_penalty': collision_penalty,
            'control_penalty': control_penalty,
            'midline_reward': midline_reward,
            'joint_usage_reward': joint_usage_reward  # 🆕 添加这个
        }
        
        total_reward = distance_reward + reach_reward + collision_penalty + control_penalty + midline_reward + joint_usage_reward
        self.current_reward = total_reward
        
        return total_reward
    def _compute_midline_reward(self, end_pos):
        """计算中线奖励 - 负数的垂直距离"""
        # 获取中线信息
        midline_info = self._calculate_channel_midline()
        if not midline_info:
            return 0.0
        
        # 计算end-effector到中线的垂直距离（只考虑y方向）
        distance_to_midline = abs(end_pos[1] - midline_info['midline_y'])
        
        # 🎯 中线奖励 = 负数的距离（距离越近奖励越大）
        midline_reward = -distance_to_midline / 300.0  # 除以100进行缩放
        
        # 🔍 调试信息
        if self.step_count <= 10:
            print(f"🔍 [MIDLINE] 中线y={midline_info['midline_y']:.1f}, 末端y={end_pos[1]:.1f}, 距离={distance_to_midline:.1f}, 奖励={midline_reward:.3f}")
        
        return midline_reward

    def _compute_joint_usage_reward(self):
        """奖励所有关节的平衡使用"""
        # 初始化历史关节角度
        if not hasattr(self, 'prev_joint_angles'):
            self.prev_joint_angles = self.joint_angles.copy()
            return 0.0
        
        # 计算每个关节的角度变化（活跃度）
        joint_changes = np.abs(self.joint_angles - self.prev_joint_angles)
        
        # 🎯 特别关注第一个关节是否过度固化
        first_joint_change = joint_changes[0]
        other_joints_change = np.mean(joint_changes[1:]) if len(joint_changes) > 1 else 0.0
        
        # 计算关节使用的平衡性
        usage_balance_reward = 0.0
        
        # 1. 奖励第一个关节的适度活跃（防止固化）
        if first_joint_change > 0.01:  # 如果第一个关节有明显变化
            usage_balance_reward += 0.02
        elif first_joint_change < 0.005:  # 如果第一个关节几乎不动（固化）
            usage_balance_reward -= 0.01
        
        # 2. 奖励所有关节的协调使用
        if len(joint_changes) > 1:
            # 标准差越小说明关节使用越平衡
            joint_std = np.std(joint_changes)
            balance_score = max(0, 1.0 - joint_std * 10)  # 缩放标准差
            usage_balance_reward += balance_score * 0.03
        
        # 更新历史关节角度
        self.prev_joint_angles = self.joint_angles.copy()
        
        # 限制奖励范围
        usage_balance_reward = np.clip(usage_balance_reward, -0.05, 0.1)
        
        # 🔍 调试信息
        if self.step_count <= 10 or self.step_count % 100 == 0:
            print(f"🔍 [JOINT_USAGE] 第一关节变化={first_joint_change:.4f}, 平衡奖励={usage_balance_reward:.3f}")
        
        return usage_balance_reward


    def _is_done(self):
        """检查是否完成"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        
        # 到达目标
        if distance < 20.0:
            return True
        
        # 步数限制
        if self.step_count >= 500:
            return True
        
        return False
    
    def _calculate_link_positions(self):
        """计算所有link的位置"""
        positions = [np.array(self.anchor_point)]  # 基座位置
        
        current_angle = 0
        current_pos = np.array(self.anchor_point)
        
        # 🔍 调试：位置计算过程
        if self.step_count <= 1:  # 只在前两步显示
            print(f"🔍 [CALC_POS] 开始位置计算")
            print(f"🔍 [CALC_POS] 基座位置: {self.anchor_point}")
            print(f"🔍 [CALC_POS] 关节角度: {self.joint_angles}")
        
        for i in range(self.num_links):
            current_angle += self.joint_angles[i]
            
            # 计算下一个关节位置
            dx = self.link_lengths[i] * np.cos(current_angle)
            dy = self.link_lengths[i] * np.sin(current_angle)
            current_pos = current_pos + np.array([dx, dy])
            
            if self.step_count <= 1:  # 调试信息
                print(f"🔍 [CALC_POS] Link {i}: angle={current_angle:.4f}, dx={dx:.1f}, dy={dy:.1f}, pos=[{current_pos[0]:.1f}, {current_pos[1]:.1f}]")
            
            positions.append(current_pos.copy())
        
        if self.step_count <= 1:
            print(f"🔍 [CALC_POS] 最终末端位置: [{positions[-1][0]:.1f}, {positions[-1][1]:.1f}]")
        
        return positions
    
    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        link_positions = self._calculate_link_positions()
        return link_positions[-1]  # 最后一个位置
    
    def _get_observation(self):
        """获取观察值"""
        # 关节角度和角速度
        obs = np.concatenate([self.joint_angles, self.joint_velocities])
        
        # 末端执行器位置
        end_pos = self._get_end_effector_position()
        obs = np.concatenate([obs, end_pos])
        
        # 目标位置
        obs = np.concatenate([obs, self.goal_pos])
        
        # 到目标的距离
        distance = np.linalg.norm(end_pos - self.goal_pos)
        obs = np.concatenate([obs, [distance]])
        
        # 碰撞信息
        obs = np.concatenate([obs, [self.collision_count, self.base_collision_count, self.self_collision_count]])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """获取额外信息"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        return {
            'end_effector_pos': end_pos,
            'goal_pos': self.goal_pos,
            'distance': float(distance),
            'collision_count': self.collision_count,
            'base_collision_count': self.base_collision_count,
            'step_count': self.step_count,
            'goal': {
                'distance_to_goal': float(distance),
                'goal_reached': distance < 20.0,
                'end_effector_position': end_pos,
                'goal_position': self.goal_pos,
            }
        }

    def _calculate_channel_midline(self):
        """计算通道中线位置 - 独立函数"""
        if not self.obstacles:
            return None
        
        # 收集所有线段障碍物的y坐标
        y_positions = []
        for obstacle in self.obstacles:
            if obstacle.get('type') == 'segment':
                y_positions.extend([obstacle['start'][1], obstacle['end'][1]])
        
        if len(y_positions) < 4:  # 需要至少4个点
            return None
        
        # 找到上下两组的边界
        y_positions.sort()
        upper_max_y = max(y_positions[:len(y_positions)//2])  # 上半部分的最大值
        lower_min_y = min(y_positions[len(y_positions)//2:])  # 下半部分的最小值
        
        # 计算中线位置
        channel_midline_y = (upper_max_y + lower_min_y) / 2
        
        return {
            'midline_y': channel_midline_y,
            'upper_boundary': upper_max_y,
            'lower_boundary': lower_min_y,
            'channel_width': lower_min_y - upper_max_y
        }

    def _draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=10):
        """绘制虚线"""
        start = np.array(start_pos)
        end = np.array(end_pos)
        
        total_vector = end - start
        total_length = np.linalg.norm(total_vector)
        
        if total_length == 0:
            return
        
        unit_vector = total_vector / total_length
        
        current_pos = start
        drawn_length = 0
        is_dash = True
        
        while drawn_length < total_length:
            remaining_length = total_length - drawn_length
            current_dash_length = min(dash_length, remaining_length)
            
            segment_end = current_pos + unit_vector * current_dash_length
            
            if is_dash:
                pygame.draw.line(surface, color, current_pos.astype(int), segment_end.astype(int), width)
            
            current_pos = segment_end
            drawn_length += current_dash_length
            is_dash = not is_dash
    def _render_midline_visualization(self):
        """渲染中线可视化 - 一直显示"""
        # 获取中线信息
        midline_info = self._calculate_channel_midline()
        if not midline_info:
            return
        
        midline_y = int(midline_info['midline_y'])
        
        # 🎨 绘制水平中线（青色实线）- 一直显示
        pygame.draw.line(self.screen, (0, 255, 255), (450, midline_y), (750, midline_y), 3)
        
        # 🎨 在中线上标记几个点 - 一直显示
        for x in range(500, 700, 50):
            pygame.draw.circle(self.screen, (0, 255, 255), (x, midline_y), 4, 2)
        
        # 🎨 绘制末端执行器到中线的垂直连接 - 一直显示
        end_pos = self._get_end_effector_position()
        distance_to_midline = abs(end_pos[1] - midline_info['midline_y'])
        
        end_pos_int = end_pos.astype(int)
        midline_point = (int(end_pos[0]), midline_y)
        
        # 根据距离选择颜色
        if distance_to_midline < 15:
            color = (0, 255, 0)  # 绿色 - 很近，奖励高
        elif distance_to_midline < 30:
            color = (255, 255, 0)  # 黄色 - 中等
        elif distance_to_midline < 50:
            color = (255, 165, 0)  # 橙色 - 较远
        else:
            color = (255, 0, 0)  # 红色 - 很远，惩罚大
        
        # 绘制垂直连接线 - 一直显示
        self._draw_dashed_line(self.screen, color, end_pos_int, midline_point, 2, 8)
        pygame.draw.circle(self.screen, color, midline_point, 4, 2)
        
        # 显示距离数字 - 一直显示
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 18)
            distance_text = f"{distance_to_midline:.0f}"
            text_surface = font.render(distance_text, True, color)
            self.screen.blit(text_surface, (midline_point[0] + 8, midline_point[1] - 10))
        
        # 🔍 调试信息（前几步显示）
        if self.step_count <= 5:
            print(f"🔍 [RENDER] 中线位置: y={midline_info['midline_y']:.1f}, 垂直距离: {distance_to_midline:.1f}")
    
    def render(self, mode='human'):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            self._init_rendering()
        
        # 清屏
        self.screen.fill((240, 240, 240))
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            if obstacle.get('type') == 'segment':
                # 绘制线段障碍物（像link一样）
                start = [int(obstacle['start'][0]), int(obstacle['start'][1])]
                end = [int(obstacle['end'][0]), int(obstacle['end'][1])]
                thickness = obstacle.get('thickness', 8)
                
                # 绘制粗线段
                pygame.draw.line(self.screen, (150, 75, 75), start, end, thickness)
                
                # 在两端绘制圆形端点（像link的关节）
                pygame.draw.circle(self.screen, (120, 60, 60), start, thickness // 2)
                pygame.draw.circle(self.screen, (120, 60, 60), end, thickness // 2)
            else:
                # 绘制圆形障碍物
                if 'center' in obstacle and 'radius' in obstacle:
                    center = [int(obstacle['center'][0]), int(obstacle['center'][1])]
                    radius = int(obstacle['radius'])
                    pygame.draw.circle(self.screen, (200, 100, 100), center, radius)
        self._render_midline_visualization()
        # 绘制机器人
        link_positions = self._calculate_link_positions()
        
        for i in range(len(link_positions) - 1):
            start = link_positions[i].astype(int)
            end = link_positions[i + 1].astype(int)
            
            # 绘制link
            pygame.draw.line(self.screen, (50, 50, 200), start, end, 8)
            
            # 绘制关节
            pygame.draw.circle(self.screen, (100, 100, 100), start, 6)
        
        # 绘制末端执行器 - 增强显示效果
        end_pos = link_positions[-1].astype(int)
        # 外圈红色圆圈
        pygame.draw.circle(self.screen, (255, 0, 0), end_pos, 12)
        # 内圈白色圆圈作为对比
        pygame.draw.circle(self.screen, (255, 255, 255), end_pos, 8)
        # 中心红点
        pygame.draw.circle(self.screen, (200, 0, 0), end_pos, 4)
        
        # 绘制目标
        goal_int = self.goal_pos.astype(int)
        pygame.draw.circle(self.screen, (50, 200, 50), goal_int, 15, 3)
        
        # 绘制信息
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 20)
            info = self._get_info()
            
            # 🎯 基本信息
            basic_texts = [
                f"Step: {self.step_count}",
                f"Distance: {info['distance']:.1f}",
                f"End-Effector: ({info['end_effector_pos'][0]:.1f}, {info['end_effector_pos'][1]:.1f})",
                f"Goal: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})"
            ]
            
            for i, text in enumerate(basic_texts):
                text_surface = font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 10 + i * 25))
            
            # 🎯 奖励信息 - 右侧显示
            reward_y_start = 10
            reward_texts = [
                f"Total Reward: {self.current_reward:.3f}",
                f"  Distance: {self.reward_components['distance_reward']:.3f}",
                f"  Reach: {self.reward_components['reach_reward']:.3f}",
                f"  Collision: {self.reward_components['collision_penalty']:.3f}",
                f"  Control: {self.reward_components['control_penalty']:.3f}",
                f"  Midline: {self.reward_components['midline_reward']:.3f}",
                f"  Joint Usage: {self.reward_components['joint_usage_reward']:.3f}" 
            ]
            
            # 绘制奖励背景框
            reward_bg_rect = pygame.Rect(self.width - 250, reward_y_start - 5, 240, len(reward_texts) * 22 + 10)
            pygame.draw.rect(self.screen, (240, 240, 240), reward_bg_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), reward_bg_rect, 2)
            
            for i, text in enumerate(reward_texts):
                # 根据奖励值选择颜色
                if i == 0:  # 总奖励
                    color = (0, 150, 0) if self.current_reward >= 0 else (150, 0, 0)
                    text_surface = font.render(text, True, color)
                else:  # 组成部分
                    value = list(self.reward_components.values())[i-1]
                    if value > 0:
                        color = (0, 120, 0)  # 绿色表示正奖励
                    elif value < 0:
                        color = (120, 0, 0)  # 红色表示惩罚
                    else:
                        color = (80, 80, 80)  # 灰色表示零
                    text_surface = small_font.render(text, True, color)
                
                self.screen.blit(text_surface, (self.width - 245, reward_y_start + i * 22))
            
            # 🎯 碰撞信息 - 左下角
            collision_texts = [
                f"Collisions: {info['collision_count']}",
                f"Base Collisions: {info['base_collision_count']}"
            ]
            
            for i, text in enumerate(collision_texts):
                text_surface = font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 110 + i * 25))
        
        if mode == 'human':
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)
        
        # 🔍 自动截图：保存前5步的截图用于对比分析
        if self.step_count <= 5:
            import os
            screenshot_dir = 'screenshots/enhanced_train_auto11111'
            os.makedirs(screenshot_dir, exist_ok=True)
            filename = f'{screenshot_dir}/step_{self.step_count:02d}.png'
            pygame.image.save(self.screen, filename)
            print(f"🖼️ [Step {self.step_count}] 自动保存截图: {filename}")
            
            # 显示详细信息
            end_pos = self._get_end_effector_position()
            print(f"    📍 末端位置: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
            print(f"    📐 关节角度: [{', '.join([f'{a:.3f}' for a in self.joint_angles])}]")
            print(f"    🎯 目标位置: [{self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f}]")
    
        return None

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
