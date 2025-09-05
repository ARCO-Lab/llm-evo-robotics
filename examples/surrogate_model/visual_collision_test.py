#!/usr/bin/env python3
"""
可视化碰撞测试 - 可以看到渲染窗口的Joint稳定性测试
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

def visual_collision_test():
    """可视化碰撞测试Joint稳定性"""
    print("👀 可视化碰撞测试 - 观察Joint稳定性")
    print("=" * 50)
    print("🎮 手动控制说明:")
    print("   W/S键: 基座关节 逆时针/顺时针旋转")
    print("   A/D键: 第二关节 逆时针/顺时针旋转") 
    print("   Q/E键: 第三关节 逆时针/顺时针旋转")
    print("   Z/C键: 第四关节 逆时针/顺时针旋转")
    print("   空格键: 暂停/继续")
    print("   ESC键: 退出")
    print("   1键: 强力模式开关 (增强控制力度)")
    print()
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )
    
    env.reset()
    
    print(f"🔍 Joint配置:")
    for i, joint in enumerate(env.joints):
        print(f"   Joint {i}: max_force = {joint.max_force}")
    
    print(f"\n🎮 开始可视化测试...")
    
    # 初始化pygame事件处理
    pygame.init()
    clock = pygame.time.Clock()
    
    # 测试参数
    running = True
    paused = False
    step_count = 0
    explosion_count = 0
    collision_count = 0
    power_mode = False  # 强力模式开关
    
    # 记录初始Joint距离
    initial_distances = []
    for i in range(len(env.bodies) - 1):
        body1 = env.bodies[i]
        body2 = env.bodies[i + 1]
        distance = (body1.position - body2.position).length
        initial_distances.append(distance)
    
    while running:
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'⏸️  暂停' if paused else '▶️  继续'}")
                elif event.key == pygame.K_1:
                    power_mode = not power_mode
                    print(f"{'💪 强力模式开启' if power_mode else '🤏 普通模式开启'}")
        
        if not paused:
            # 获取当前按键状态
            keys = pygame.key.get_pressed()
            
            # 设置控制力度
            normal_force = 3.0
            power_force = 8.0
            force = power_force if power_mode else normal_force
            
            # 根据按键设置动作 - WASD + QE + ZC控制
            action = [0.0, 0.0, 0.0, 0.0]
            
            # W/S控制基座关节 (Joint 0)
            if keys[pygame.K_w]:
                action[0] = -force  # 逆时针
            elif keys[pygame.K_s]:
                action[0] = force   # 顺时针
            
            # A/D控制第二关节 (Joint 1) 
            if keys[pygame.K_a]:
                action[1] = -force  # 逆时针
            elif keys[pygame.K_d]:
                action[1] = force   # 顺时针
            
            # Q/E控制第三关节 (Joint 2)
            if keys[pygame.K_q]:
                action[2] = -force  # 逆时针
            elif keys[pygame.K_e]:
                action[2] = force   # 顺时针
            
            # Z/C控制第四关节 (Joint 3)
            if keys[pygame.K_z]:
                action[3] = -force  # 逆时针
            elif keys[pygame.K_c]:
                action[3] = force   # 顺时针
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # 检查碰撞
            if 'collisions' in info and len(info['collisions']) > 0:
                collision_count += len(info['collisions'])
                if len(info['collisions']) > 2:  # 只报告较多的碰撞
                    print(f"💥 步数{step_count}: {len(info['collisions'])} 次碰撞")
            
            # 检查Joint稳定性
            joint_broken = False
            for i in range(len(env.bodies) - 1):
                body1 = env.bodies[i]
                body2 = env.bodies[i + 1]
                distance = (body1.position - body2.position).length
                expected_distance = env.link_lengths[i+1] if i+1 < len(env.link_lengths) else env.link_lengths[i]
                
                # 检测Joint散架 - 距离超过预期的2倍
                if distance > expected_distance * 2:
                    if not joint_broken:
                        explosion_count += 1
                        print(f"❌ 步数{step_count}: Joint {i}-{i+1} 散架!")
                        print(f"   当前距离: {distance:.2f}px")
                        print(f"   预期距离: {expected_distance}px")
                        print(f"   偏差倍数: {distance/expected_distance:.2f}x")
                        joint_broken = True
            
            # 渲染
            env.render()
            
            # 每100步报告状态
            if step_count % 100 == 0:
                end_effector = env._get_end_effector_position()
                base_pos = env.bodies[0].position
                print(f"📊 步数{step_count}:")
                print(f"   基座位置: ({base_pos.x:.1f}, {base_pos.y:.1f})")
                print(f"   末端位置: [{end_effector[0]:.1f}, {end_effector[1]:.1f}]")
                print(f"   总碰撞数: {collision_count}")
                print(f"   Joint散架数: {explosion_count}")
                print(f"   Joint稳定性: {'❌ 不稳定' if explosion_count > 0 else '✅ 稳定'}")
        
        # 控制帧率
        clock.tick(60)  # 60 FPS
        
        # 如果没有按键，稍微休息一下
        if not any(pygame.key.get_pressed()):
            time.sleep(0.01)
    
    print(f"\n📊 最终统计:")
    print(f"   总步数: {step_count}")
    print(f"   总碰撞数: {collision_count}")
    print(f"   Joint散架数: {explosion_count}")
    print(f"   Joint稳定性: {'❌ 不稳定' if explosion_count > 0 else '✅ 稳定'}")
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    visual_collision_test()
