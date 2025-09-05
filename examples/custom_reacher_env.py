#!/usr/bin/env python3
"""
自定义2D Reacher环境 - 基于Gymnasium
使用简单但稳定的数学模型，避免复杂物理引擎的问题
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional, Tuple, Dict, Any

class SimpleReacher2DEnv(gym.Env):
    """
    简单的2D Reacher环境
    - 使用解析几何而非物理仿真
    - 避免关节分离和穿透问题
    - 支持障碍物碰撞检测
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, 
                 num_links: int = 3,
                 link_lengths: Optional[list] = None,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 500):
        
        super().__init__()
        
        # 环境参数
        self.num_links = num_links
        self.link_lengths = link_lengths or [60.0] * num_links
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # 物理参数
        self.dt = 0.02  # 时间步长
        self.max_torque = 2.0  # 最大扭矩
        self.damping = 0.95  # 阻尼系数
        
        # 状态变量
        self.joint_angles = np.zeros(num_links)  # 关节角度
        self.joint_velocities = np.zeros(num_links)  # 关节角速度
        self.step_count = 0
        
        # 目标和障碍物
        self.target_pos = np.array([200.0, 100.0])
        self.obstacles = [
            {'center': [150, 200], 'radius': 30},
            {'center': [250, 150], 'radius': 25},
        ]
        
        # 工作空间
        self.workspace_center = np.array([300.0, 300.0])
        self.workspace_size = 400.0
        
        # Gymnasium空间定义
        # 观察空间: [cos(θ), sin(θ), θ_dot] for each joint + [target_x, target_y, end_x, end_y]
        obs_dim = num_links * 3 + 4  # 每个关节3个值 + 目标位置2个 + 末端位置2个
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动作空间: 每个关节的扭矩
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, 
            shape=(num_links,), dtype=np.float32
        )
        
        # 渲染相关
        self.screen = None
        self.clock = None
        self.screen_size = 600
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机初始化关节角度
        self.joint_angles = self.np_random.uniform(-np.pi/4, np.pi/4, self.num_links)
        self.joint_velocities = np.zeros(self.num_links)
        self.step_count = 0
        
        # 随机目标位置
        angle = self.np_random.uniform(0, 2*np.pi)
        distance = self.np_random.uniform(50, 150)
        self.target_pos = self.workspace_center + distance * np.array([np.cos(angle), np.sin(angle)])
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        action = np.clip(action, -self.max_torque, self.max_torque)
        
        # 简单的动力学模型（无需复杂物理仿真）
        # 扭矩直接影响角加速度
        angular_acceleration = action  # 简化：扭矩 = 角加速度
        
        # 更新角速度和角度
        self.joint_velocities += angular_acceleration * self.dt
        self.joint_velocities *= self.damping  # 应用阻尼
        self.joint_angles += self.joint_velocities * self.dt
        
        # 角度限制（防止过度旋转）
        for i in range(self.num_links):
            if i == 0:  # 基座关节可以360度旋转
                self.joint_angles[i] = self.joint_angles[i] % (2 * np.pi)
            else:  # 其他关节有限制
                self.joint_angles[i] = np.clip(self.joint_angles[i], -np.pi*2/3, np.pi*2/3)
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止条件
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        self.step_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        obs = []
        
        # 每个关节的状态: cos(θ), sin(θ), θ_dot
        for i in range(self.num_links):
            obs.extend([
                np.cos(self.joint_angles[i]),
                np.sin(self.joint_angles[i]),
                self.joint_velocities[i]
            ])
        
        # 目标位置
        obs.extend(self.target_pos)
        
        # 末端执行器位置
        end_pos = self._get_end_effector_position()
        obs.extend(end_pos)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """计算末端执行器位置（解析几何）"""
        pos = np.array(self.workspace_center)  # 基座位置
        current_angle = 0.0
        
        for i in range(self.num_links):
            current_angle += self.joint_angles[i]
            pos += self.link_lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
        
        return pos
    
    def _get_link_positions(self) -> list:
        """获取所有link的位置（用于渲染和碰撞检测）"""
        positions = [np.array(self.workspace_center)]  # 基座位置
        current_angle = 0.0
        
        for i in range(self.num_links):
            current_angle += self.joint_angles[i]
            next_pos = positions[-1] + self.link_lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
            positions.append(next_pos)
        
        return positions
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        end_pos = self._get_end_effector_position()
        
        # 距离奖励
        distance_to_target = np.linalg.norm(end_pos - self.target_pos)
        distance_reward = -distance_to_target / 100.0
        
        # 到达目标奖励
        reach_reward = 0.0
        if distance_to_target < 20.0:
            reach_reward = 10.0
        
        # 碰撞惩罚
        collision_penalty = 0.0
        if self._check_collision():
            collision_penalty = -5.0
        
        # 控制惩罚（鼓励平滑运动）
        control_penalty = -0.01 * np.sum(np.square(self.joint_velocities))
        
        return distance_reward + reach_reward + collision_penalty + control_penalty
    
    def _check_collision(self) -> bool:
        """检查碰撞（简单的点-圆碰撞）"""
        link_positions = self._get_link_positions()
        
        # 检查每个link中点是否与障碍物碰撞
        for i in range(len(link_positions) - 1):
            start = link_positions[i]
            end = link_positions[i + 1]
            mid_point = (start + end) / 2  # link中点
            
            for obstacle in self.obstacles:
                dist = np.linalg.norm(mid_point - np.array(obstacle['center']))
                if dist < obstacle['radius']:
                    return True
        
        return False
    
    def _is_terminated(self) -> bool:
        """检查是否终止"""
        end_pos = self._get_end_effector_position()
        distance_to_target = np.linalg.norm(end_pos - self.target_pos)
        
        # 到达目标
        if distance_to_target < 15.0:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        end_pos = self._get_end_effector_position()
        distance_to_target = np.linalg.norm(end_pos - self.target_pos)
        
        return {
            'end_effector_pos': end_pos,
            'target_pos': self.target_pos,
            'distance_to_target': distance_to_target,
            'collision': self._check_collision(),
            'step_count': self.step_count
        }
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                pygame.display.set_caption("Simple 2D Reacher")
            else:
                self.screen = pygame.Surface((self.screen_size, self.screen_size))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # 清屏
        self.screen.fill((240, 240, 240))  # 浅灰色背景
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            center = obstacle['center']
            radius = obstacle['radius']
            pygame.draw.circle(self.screen, (200, 100, 100), center, radius)
        
        # 绘制机器人链接
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
        target_int = self.target_pos.astype(int)
        pygame.draw.circle(self.screen, (50, 200, 50), target_int, 15, 3)
        
        # 绘制信息
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            info = self._get_info()
            
            texts = [
                f"Step: {self.step_count}",
                f"Distance: {info['distance_to_target']:.1f}",
                f"Collision: {info['collision']}"
            ]
            
            for i, text in enumerate(texts):
                text_surface = font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 10 + i * 25))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

# 手动控制测试函数
def manual_control_test():
    """手动控制测试"""
    # 初始化pygame
    pygame.init()
    pygame.display.init()
    
    env = SimpleReacher2DEnv(render_mode="human")
    observation, info = env.reset()
    
    print("🎮 手动控制测试")
    print("按键说明:")
    print("  W/S: 控制第1个关节")
    print("  A/D: 控制第2个关节") 
    print("  I/K: 控制第3个关节")
    print("  ESC: 退出")
    
    running = True
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 获取按键状态
        keys = pygame.key.get_pressed()
        action = np.zeros(env.num_links)
        
        # 根据按键设置动作
        if keys[pygame.K_w]:
            action[0] = 1.0
        elif keys[pygame.K_s]:
            action[0] = -1.0
            
        if keys[pygame.K_a]:
            action[1] = -1.0
        elif keys[pygame.K_d]:
            action[1] = 1.0
            
        if keys[pygame.K_i]:
            action[2] = 1.0
        elif keys[pygame.K_k]:
            action[2] = -1.0
        
        # 执行步骤
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        # 打印信息
        if env.step_count % 60 == 0:  # 每秒打印一次
            print(f"步数: {info['step_count']}, 距离: {info['distance_to_target']:.1f}, 奖励: {reward:.2f}")
        
        # 检查重置
        if terminated or truncated:
            print("🔄 环境重置")
            observation, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    manual_control_test()
