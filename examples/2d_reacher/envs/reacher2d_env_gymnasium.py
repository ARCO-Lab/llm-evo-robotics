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
    
    def __init__(self, num_links=3, link_lengths=None, render_mode=None, config_path=None, curriculum_stage=0, debug_level='SILENT'):
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
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 7,), dtype=np.float32)
        
        # 渲染相关
        self.screen = None
        self.clock = None
        self.draw_options = None
        
        # 统计信息
        self.collision_count = 0
        self.base_collision_count = 0
        
        if self.render_mode:
            self._init_rendering()
    
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
                        # 将线段转换为圆形障碍物
                        start = np.array(points[0])
                        end = np.array(points[1])
                        mid_point = (start + end) / 2
                        obstacles.append({
                            'center': mid_point.tolist(),
                            'radius': 15  # 线段宽度的一半
                        })
        
        return obstacles
    
    def _init_rendering(self):
        """初始化渲染"""
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 1200))
        pygame.display.set_caption("Stable Reacher2D (Gymnasium)")
        self.clock = pygame.time.Clock()
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 重置关节状态
        self.joint_angles = np.random.uniform(-np.pi/6, np.pi/6, self.num_links)
        self.joint_velocities = np.zeros(self.num_links)
        self.step_count = 0
        
        # 重置统计
        self.collision_count = 0
        self.base_collision_count = 0
        
        # 课程学习目标位置
        self.goal_pos = self._get_curriculum_goal()
        
        observation = self._get_observation()
        
        if self.gym_api_version == "new":
            info = self._get_info()
            return observation, info
        else:
            return observation
    
    def step(self, action):
        """执行一步"""
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # 简化的动力学模型
        # 扭矩转换为角加速度（简化的转动惯量为1）
        torque_to_acceleration = 0.1  # 转换系数
        angular_acceleration = action * torque_to_acceleration
        
        # 更新角速度和角度
        self.joint_velocities += angular_acceleration * self.dt
        self.joint_velocities *= 0.98  # 阻尼
        self.joint_angles += self.joint_velocities * self.dt
        
        # 角度限制
        for i in range(self.num_links):
            if i == 0:  # 基座关节可以360度旋转
                self.joint_angles[i] = self.joint_angles[i] % (2 * np.pi)
            else:  # 其他关节限制
                self.joint_angles[i] = np.clip(self.joint_angles[i], -np.pi*2/3, np.pi*2/3)
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止条件
        done = self._is_done()
        
        self.step_count += 1
        
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
    
    def _check_collision(self):
        """检查碰撞"""
        link_positions = self._get_link_positions()
        collision_detected = False
        
        # 检查每个link与障碍物的碰撞
        for i in range(len(link_positions) - 1):
            start = link_positions[i]
            end = link_positions[i + 1]
            mid_point = (start + end) / 2
            
            for obstacle in self.obstacles:
                dist = np.linalg.norm(mid_point - np.array(obstacle['center']))
                if dist < obstacle['radius'] + 5:  # 5像素的安全边距
                    collision_detected = True
                    if i == 0:  # 基座关节碰撞
                        self.base_collision_count += 1
                    else:
                        self.collision_count += 1
        
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
    
    def _compute_reward(self):
        """计算奖励"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        
        # 距离奖励
        distance_reward = -distance / 100.0
        
        # 到达奖励
        reach_reward = 0.0
        if distance < 20.0:
            reach_reward = 10.0
        
        # 碰撞惩罚
        collision_penalty = 0.0
        if self._check_collision():
            collision_penalty = -2.0
        
        # 控制平滑性
        control_penalty = -0.01 * np.sum(np.square(self.joint_velocities))
        
        return distance_reward + reach_reward + collision_penalty + control_penalty
    
    def _is_done(self):
        """检查是否完成"""
        end_pos = self._get_end_effector_position()
        distance = np.linalg.norm(end_pos - self.goal_pos)
        
        # 到达目标
        if distance < 15.0:
            return True
        
        # 步数限制
        if self.step_count >= 500:
            return True
        
        return False
    
    def _get_info(self):
        """获取额外信息"""
        end_pos = self._get_end_effector_position()
        return {
            'end_effector_pos': end_pos,
            'goal_pos': self.goal_pos,
            'distance': np.linalg.norm(end_pos - self.goal_pos),
            'collision_count': self.collision_count,
            'base_collision_count': self.base_collision_count,
            'step_count': self.step_count
        }
    
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
            center = [int(obstacle['center'][0]), int(obstacle['center'][1])]
            radius = int(obstacle['radius'])
            pygame.draw.circle(self.screen, (200, 100, 100), center, radius)
        
        # 绘制机器人
        link_positions = self._get_link_positions()
        
        for i in range(len(link_positions) - 1):
            start = link_positions[i].astype(int)
            end = link_positions[i + 1].astype(int)
            
            # 绘制link
            pygame.draw.line(self.screen, (50, 50, 200), start, end, 8)
            
            # 绘制关节
            pygame.draw.circle(self.screen, (100, 100, 100), start, 6)
        
        # 绘制末端执行器
        end_pos = link_positions[-1].astype(int)
        pygame.draw.circle(self.screen, (200, 50, 50), end_pos, 8)
        
        # 绘制目标
        goal_int = self.goal_pos.astype(int)
        pygame.draw.circle(self.screen, (50, 200, 50), goal_int, 15, 3)
        
        # 绘制信息
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            info = self._get_info()
            
            texts = [
                f"Step: {self.step_count}",
                f"Distance: {info['distance']:.1f}",
                f"Collisions: {info['collision_count']}",
                f"Base Collisions: {info['base_collision_count']}"
            ]
            
            for i, text in enumerate(texts):
                text_surface = font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 10 + i * 25))
        
        if mode == 'human':
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)
        
        return None
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
