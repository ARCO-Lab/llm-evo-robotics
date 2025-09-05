#!/usr/bin/env python3
"""
强制碰撞测试 - 直接让机器人撞墙
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

def force_collision_test():
    """强制产生碰撞测试Joint稳定性"""
    print("💥 强制碰撞测试")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )
    
    env.reset()
    
    print(f"🔍 障碍物位置:")
    for i, obstacle in enumerate(env.obstacles):
        print(f"   障碍物 {i}: {obstacle}")
    
    print(f"\n🔍 初始机器人位置:")
    for i, body in enumerate(env.bodies):
        print(f"   Link {i}: position = {body.position}")
    
    print(f"\n🎮 强制向右大幅运动 - 撞向右侧障碍物:")
    
    explosion_count = 0
    collision_count = 0
    
    for step in range(300):
        # 极大的动作值，强制机器人快速运动
        action = [0.0, 10.0, 10.0, 10.0]  # 所有关节都大力运动
        obs, reward, done, info = env.step(action)
        
        # 检查是否有碰撞信息
        if 'collisions' in info:
            collision_count += len(info['collisions'])
            if len(info['collisions']) > 0:
                print(f"💥 步数{step}: 检测到 {len(info['collisions'])} 次碰撞")
        
        # 检查Joint距离
        joint_broken = False
        for i in range(len(env.bodies) - 1):
            body1 = env.bodies[i]
            body2 = env.bodies[i + 1]
            distance = (body1.position - body2.position).length
            expected_distance = env.link_lengths[i+1] if i+1 < len(env.link_lengths) else env.link_lengths[i]
            
            # 如果距离超过预期的3倍，认为散架了
            if distance > expected_distance * 3:
                if not joint_broken:
                    explosion_count += 1
                    print(f"❌ 步数{step}: Joint {i}-{i+1} 散架! 距离: {distance:.2f}px (预期: {expected_distance}px)")
                    joint_broken = True
        
        # 检查是否有body飞出屏幕
        for i, body in enumerate(env.bodies):
            pos = body.position
            if abs(pos.x) > 2000 or abs(pos.y) > 2000:
                print(f"🚀 步数{step}: Link {i} 飞出屏幕! 位置: {pos}")
        
        # 每50步报告状态
        if step % 50 == 49:
            end_effector_pos = env._get_end_effector_position()
            print(f"📊 步数{step+1}: 末端执行器位置: {end_effector_pos}")
            
            # 检查机器人是否还在合理位置
            base_pos = env.bodies[0].position
            if abs(base_pos.x - 450) > 10 or abs(base_pos.y - 620) > 100:
                print(f"⚠️ 基座位置异常: {base_pos} (应该在 450,620 附近)")
        
        time.sleep(0.005)  # 稍微慢一点便于观察
    
    print(f"\n📊 最终统计:")
    print(f"   总碰撞次数: {collision_count}")
    print(f"   Joint散架次数: {explosion_count}")
    print(f"   Joint稳定性: {'❌ 不稳定' if explosion_count > 0 else '✅ 稳定'}")

if __name__ == "__main__":
    force_collision_test()

