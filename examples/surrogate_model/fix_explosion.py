#!/usr/bin/env python3
"""
修复机器人"炸开"现象的解决方案
- 实现渐进式分离力
- 添加速度限制
- 改进碰撞处理器
"""

import sys
import os
import numpy as np
import pygame
import time
import math

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def create_anti_explosion_env():
    """创建防炸开的环境"""
    
    class AntiExplosionReacher2DEnv(Reacher2DEnv):
        """防炸开版本的Reacher2D环境"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_separation_impulse = 50.0  # 限制最大分离冲量
            self.gentle_separation = True       # 启用温和分离
            self.explosion_detection = True     # 启用炸开检测
            
        def _setup_collision_handlers(self):
            """改进的碰撞处理器 - 防止炸开"""
            try:
                # 🎯 1. 温和的Link间碰撞处理
                def gentle_link_collision_handler(arbiter, space, data):
                    """温和的Link间碰撞处理 - 防止炸开"""
                    
                    # 获取碰撞的两个shape
                    shape_a, shape_b = arbiter.shapes
                    body_a, body_b = shape_a.body, shape_b.body
                    
                    # 🔧 关键1：设置很低的弹性系数
                    arbiter.restitution = 0.01  # 几乎无弹性
                    arbiter.friction = 0.9      # 高摩擦力
                    
                    # 🔧 关键2：限制分离速度
                    contact_set = arbiter.contact_point_set
                    for i in range(contact_set.count):
                        contact = contact_set.points[i]
                        
                        # 如果穿透深度很大，使用温和的分离
                        if contact.distance < -10.0:  # 深度穿透
                            # 计算温和的分离冲量
                            penetration_depth = abs(contact.distance)
                            
                            # 🎯 关键：渐进式分离而非瞬间分离
                            gentle_impulse = min(penetration_depth * 0.1, self.max_separation_impulse)
                            separation_impulse = contact.normal * gentle_impulse
                            
                            # 分别对两个body施加相反的温和冲量
                            body_a.apply_impulse_at_world_point(separation_impulse, contact.point_a)
                            body_b.apply_impulse_at_world_point(-separation_impulse, contact.point_b)
                            
                            # 🔧 限制分离后的速度
                            max_velocity = 100.0  # 最大分离速度
                            if np.linalg.norm(body_a.velocity) > max_velocity:
                                body_a.velocity = body_a.velocity / np.linalg.norm(body_a.velocity) * max_velocity
                            if np.linalg.norm(body_b.velocity) > max_velocity:
                                body_b.velocity = body_b.velocity / np.linalg.norm(body_b.velocity) * max_velocity
                    
                    return True  # 允许碰撞处理，但已经调整了参数
                
                # 🎯 2. 为所有Link对设置温和碰撞处理
                for i in range(self.num_links):
                    for j in range(i + 2, self.num_links):  # 跳过相邻Link
                        try:
                            self.space.on_collision(
                                collision_type_a=i + 1,
                                collision_type_b=j + 1,
                                begin=gentle_link_collision_handler
                            )
                            self.logger.debug(f"✅ 设置温和碰撞处理: Link{i+1} vs Link{j+1}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ 设置温和碰撞处理器失败: {e}")
                
                # 🎯 3. 机器人与障碍物碰撞处理（保持原有逻辑）
                def robot_obstacle_collision_handler(arbiter, space, data):
                    """处理机器人与障碍物的碰撞"""
                    if not hasattr(self, 'collision_count'):
                        self.collision_count = 0
                    self.collision_count += 1
                    
                    # 设置适中的碰撞参数
                    arbiter.restitution = 0.3
                    arbiter.friction = 0.8
                    
                    return True
                
                # 为每个机器人链接设置与障碍物的碰撞检测
                OBSTACLE_COLLISION_TYPE = 100
                for i in range(self.num_links):
                    robot_link_type = i + 1
                    try:
                        self.space.on_collision(
                            collision_type_a=robot_link_type,
                            collision_type_b=OBSTACLE_COLLISION_TYPE,
                            begin=robot_obstacle_collision_handler
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ 设置机器人-障碍物碰撞处理器失败: {e}")
                        
            except Exception as e:
                self.logger.warning(f"⚠️ 碰撞处理器设置失败: {e}")
        
        def step(self, actions):
            """增强版step - 包含炸开检测和速度限制"""
            
            # 🔧 在step前记录速度
            pre_step_velocities = []
            for body in self.bodies:
                pre_step_velocities.append({
                    'velocity': np.array(body.velocity),
                    'angular_velocity': body.angular_velocity
                })
            
            # 执行原有step逻辑
            result = super().step(actions)
            
            # 🎯 炸开检测和修正
            if self.explosion_detection:
                self._detect_and_fix_explosion(pre_step_velocities)
            
            return result
        
        def _detect_and_fix_explosion(self, pre_step_velocities):
            """检测和修正炸开现象"""
            explosion_detected = False
            max_safe_velocity = 200.0      # 安全最大线速度
            max_safe_angular_velocity = 10.0  # 安全最大角速度
            
            for i, body in enumerate(self.bodies):
                if i < len(pre_step_velocities):
                    pre_vel = pre_step_velocities[i]
                    
                    # 检查速度突变
                    velocity_change = np.linalg.norm(np.array(body.velocity) - pre_vel['velocity'])
                    angular_velocity_change = abs(body.angular_velocity - pre_vel['angular_velocity'])
                    
                    # 🚨 炸开检测：速度突然大幅增加
                    if (velocity_change > 150.0 or 
                        angular_velocity_change > 8.0 or
                        np.linalg.norm(body.velocity) > max_safe_velocity or
                        abs(body.angular_velocity) > max_safe_angular_velocity):
                        
                        explosion_detected = True
                        
                        # 🔧 温和修正：不是直接设为0，而是渐进减少
                        if np.linalg.norm(body.velocity) > max_safe_velocity:
                            # 限制线速度
                            vel_direction = np.array(body.velocity) / (np.linalg.norm(body.velocity) + 1e-6)
                            body.velocity = (vel_direction * max_safe_velocity * 0.5).tolist()
                        
                        if abs(body.angular_velocity) > max_safe_angular_velocity:
                            # 限制角速度
                            body.angular_velocity = np.sign(body.angular_velocity) * max_safe_angular_velocity * 0.5
                        
                        self.logger.warning(f"🚨 检测到Link{i}炸开倾向，已修正速度")
            
            if explosion_detected:
                self.logger.warning("🔴 检测到炸开现象，已进行速度修正")
    
    return AntiExplosionReacher2DEnv

def test_anti_explosion():
    """测试防炸开解决方案"""
    print("🛡️ 测试防炸开机器人环境")
    
    # 创建防炸开环境
    AntiExplosionEnv = create_anti_explosion_env()
    env = AntiExplosionEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'
    )
    
    env.reset()
    
    # 初始化pygame
    pygame.init()
    clock = pygame.time.Clock()
    
    running = True
    step_count = 0
    explosion_count = 0
    
    print("\n🎮 测试说明:")
    print("  A: 持续按住让机器人折叠")
    print("  然后突然释放 - 观察是否还会炸开")
    print("  Space: 自动测试模式")
    print("  Q: 退出")
    
    auto_test = False
    fold_phase = True
    fold_counter = 0
    
    while running and step_count < 3000:
        # 处理事件
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_test = not auto_test
                    print(f"🔄 {'启用' if auto_test else '禁用'}自动测试模式")
        
        # 生成动作
        if auto_test:
            # 自动测试：模拟持续折叠然后突然停止
            if fold_phase:
                # 折叠阶段：让机器人折叠
                actions = np.array([50, -50, 50, -50])  # 交替方向
                fold_counter += 1
                if fold_counter > 100:  # 折叠100步
                    fold_phase = False
                    fold_counter = 0
                    print("🔄 切换到释放阶段")
            else:
                # 释放阶段：突然停止
                actions = np.array([0, 0, 0, 0])
                fold_counter += 1
                if fold_counter > 50:  # 释放50步
                    fold_phase = True
                    fold_counter = 0
                    print("🔄 切换到折叠阶段")
        else:
            # 手动控制
            actions = np.zeros(4)
            if keys[pygame.K_a]:
                actions[1] = 50  # 持续折叠
            if keys[pygame.K_d]:
                actions[1] = -50
            if keys[pygame.K_w]:
                actions[0] = 50
            if keys[pygame.K_s]:
                actions[0] = -50
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 检测爆炸
        max_velocity = max([np.linalg.norm(body.velocity) for body in env.bodies])
        max_angular_velocity = max([abs(body.angular_velocity) for body in env.bodies])
        
        if max_velocity > 300 or max_angular_velocity > 15:
            explosion_count += 1
            print(f"🚨 检测到可能的炸开！步数: {step_count}, 最大速度: {max_velocity:.1f}")
        
        # 渲染
        env.render()
        
        # 在屏幕上显示统计信息
        font = pygame.font.Font(None, 36)
        stats_text = [
            f"步数: {step_count}",
            f"炸开次数: {explosion_count}",
            f"最大速度: {max_velocity:.1f}",
            f"最大角速度: {max_angular_velocity:.1f}",
            f"模式: {'自动测试' if auto_test else '手动控制'}",
            f"阶段: {'折叠' if fold_phase else '释放'}" if auto_test else ""
        ]
        
        for i, text in enumerate(stats_text):
            if text:  # 跳过空字符串
                surface = font.render(text, True, (255, 0, 0))
                env.screen.blit(surface, (10, 10 + i * 30))
        
        pygame.display.flip()
        
        step_count += 1
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    print(f"\n📊 测试结果:")
    print(f"  总步数: {step_count}")
    print(f"  炸开次数: {explosion_count}")
    print(f"  炸开率: {explosion_count/step_count*100:.2f}%")
    
    env.close()

if __name__ == "__main__":
    test_anti_explosion()

