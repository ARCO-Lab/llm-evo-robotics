#!/usr/bin/env python3
"""
Joint稳定性手动测试
基于manual_control_test.py，添加Joint稳定性监测
"""

import sys
import os
import numpy as np
import pygame
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def joint_stability_manual_test():
    """Joint稳定性手动测试"""
    print("🔧 Joint稳定性手动测试")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'
    )
    
    env.reset()
    
    print(f"\n🎮 手动控制说明:")
    print("  W/S: 基座关节 逆时针/顺时针")
    print("  A/D: 第二关节 逆时针/顺时针") 
    print("  Q/E: 第三关节 逆时针/顺时针")
    print("  Z/C: 第四关节 逆时针/顺时针")
    print("  1: 强力模式开关")
    print("  R: 重置")
    print("  ESC: 退出")
    print("  目标: 观察碰撞时Joint是否散架")
    
    # 检查Joint配置
    print(f"\n🔍 Joint配置:")
    for i, joint in enumerate(env.joints):
        print(f"   Joint {i}: max_force = {joint.max_force}")
    
    # 记录初始Joint距离
    initial_distances = []
    for i in range(len(env.bodies) - 1):
        body1 = env.bodies[i]
        body2 = env.bodies[i + 1]
        distance = (body1.position - body2.position).length
        initial_distances.append(distance)
        print(f"   初始Joint {i}-{i+1}距离: {distance:.2f}px")
    
    pygame.init()
    clock = pygame.time.Clock()
    
    # 测试参数
    running = True
    step_count = 0
    collision_count = 0
    explosion_count = 0
    power_mode = False
    
    print(f"\n🎮 开始手动控制测试...")
    
    while running:
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    step_count = 0
                    collision_count = 0
                    explosion_count = 0
                    print("🔄 环境已重置")
                elif event.key == pygame.K_1:
                    power_mode = not power_mode
                    print(f"{'💪 强力模式' if power_mode else '🤏 普通模式'}")
        
        # 获取按键状态
        keys = pygame.key.get_pressed()
        
        # 设置控制力度
        normal_force = 3.0
        power_force = 10.0  # 更大的力度来测试Joint稳定性
        force = power_force if power_mode else normal_force
        
        # 根据按键设置动作
        action = [0.0, 0.0, 0.0, 0.0]
        
        # W/S控制基座关节
        if keys[pygame.K_w]:
            action[0] = -force
        elif keys[pygame.K_s]:
            action[0] = force
        
        # A/D控制第二关节
        if keys[pygame.K_a]:
            action[1] = -force
        elif keys[pygame.K_d]:
            action[1] = force
        
        # Q/E控制第三关节
        if keys[pygame.K_q]:
            action[2] = -force
        elif keys[pygame.K_e]:
            action[2] = force
        
        # Z/C控制第四关节
        if keys[pygame.K_z]:
            action[3] = -force
        elif keys[pygame.K_c]:
            action[3] = force
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # 检查碰撞
        if 'collisions' in info and len(info['collisions']) > 0:
            collision_count += len(info['collisions'])
            if len(info['collisions']) > 2:
                print(f"💥 步数{step_count}: {len(info['collisions'])} 次碰撞")
        
        # 检查Joint稳定性
        joint_broken = False
        max_distance_change = 0
        
        for i in range(len(env.bodies) - 1):
            body1 = env.bodies[i]
            body2 = env.bodies[i + 1]
            current_distance = (body1.position - body2.position).length
            expected_distance = initial_distances[i]
            
            distance_change = abs(current_distance - expected_distance)
            max_distance_change = max(max_distance_change, distance_change)
            
            # 检测Joint散架 - 距离变化超过50%
            if distance_change > expected_distance * 0.5:
                if not joint_broken:
                    explosion_count += 1
                    print(f"❌ 步数{step_count}: Joint {i}-{i+1} 可能散架!")
                    print(f"   当前距离: {current_distance:.2f}px")
                    print(f"   预期距离: {expected_distance:.2f}px") 
                    print(f"   变化: {distance_change:.2f}px ({distance_change/expected_distance*100:.1f}%)")
                    joint_broken = True
        
        # 渲染
        env.render()
        
        # 每100步报告状态
        if step_count % 100 == 0 and step_count > 0:
            base_pos = env.bodies[0].position
            end_effector = env._get_end_effector_position()
            print(f"📊 步数{step_count}:")
            print(f"   基座位置: ({base_pos.x:.1f}, {base_pos.y:.1f})")
            print(f"   末端位置: [{end_effector[0]:.1f}, {end_effector[1]:.1f}]")
            print(f"   总碰撞: {collision_count}, Joint异常: {explosion_count}")
            print(f"   最大Joint距离变化: {max_distance_change:.2f}px")
            print(f"   强力模式: {'开启' if power_mode else '关闭'}")
        
        # 控制帧率
        clock.tick(60)
    
    print(f"\n📊 最终统计:")
    print(f"   总步数: {step_count}")
    print(f"   总碰撞数: {collision_count}")
    print(f"   Joint异常数: {explosion_count}")
    print(f"   Joint稳定性: {'❌ 不稳定' if explosion_count > 0 else '✅ 稳定'}")
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    joint_stability_manual_test()

