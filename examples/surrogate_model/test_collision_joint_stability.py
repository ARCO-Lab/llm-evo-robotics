#!/usr/bin/env python3
"""
测试碰撞时Joint稳定性
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

def test_collision_stability():
    """测试碰撞时的Joint稳定性"""
    print("🔧 测试碰撞时Joint稳定性")
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
    
    print(f"🔍 Joint配置检查:")
    for i, joint in enumerate(env.joints):
        print(f"   Joint {i}: max_force = {joint.max_force}")
    
    print(f"\n🔍 Link质量检查:")
    total_mass = 0
    for i, body in enumerate(env.bodies):
        mass = body.mass
        total_mass += mass
        print(f"   Link {i}: mass = {mass:.2f}")
    print(f"   总质量: {total_mass:.2f}")
    
    print(f"\n🔍 重力检查:")
    gravity = env.space.gravity
    total_weight = total_mass * abs(gravity.y)
    print(f"   重力: {gravity}")
    print(f"   总重力: {total_weight:.2f} N")
    print(f"   Joint最大约束力: {env.joints[0].max_force} N")
    print(f"   约束力是否足够: {'✅' if env.joints[0].max_force > total_weight * 2 else '❌'}")
    
    # 检查初始Joint距离
    print(f"\n🔍 初始Joint距离:")
    initial_distances = []
    for i in range(len(env.bodies) - 1):
        body1 = env.bodies[i]
        body2 = env.bodies[i + 1]
        distance = (body1.position - body2.position).length
        initial_distances.append(distance)
        print(f"   Joint {i}-{i+1}: {distance:.2f}px")
    
    # 强制碰撞测试
    print(f"\n🎮 强制碰撞测试 - 让机器人撞墙:")
    collision_detected = False
    explosion_detected = False
    
    for step in range(200):
        # 强制向右运动，撞向障碍物
        action = [0.0, 5.0, 0.0, 0.0]  # 只让第二个关节强力运动
        obs, reward, done, info = env.step(action)
        
        # 检查Joint距离变化
        current_distances = []
        max_distance_change = 0
        
        for i in range(len(env.bodies) - 1):
            body1 = env.bodies[i]
            body2 = env.bodies[i + 1]
            distance = (body1.position - body2.position).length
            current_distances.append(distance)
            
            distance_change = abs(distance - initial_distances[i])
            max_distance_change = max(max_distance_change, distance_change)
            
            # 检测Joint散架
            if distance > initial_distances[i] * 2:  # 距离增加超过2倍认为散架
                explosion_detected = True
                print(f"❌ 步数{step}: Joint {i}-{i+1} 散架! 距离: {distance:.2f}px (初始: {initial_distances[i]:.2f}px)")
        
        # 检测碰撞
        if hasattr(env, 'collision_count') and env.collision_count > 0:
            if not collision_detected:
                collision_detected = True
                print(f"💥 步数{step}: 检测到碰撞!")
        
        # 每50步报告状态
        if step % 50 == 49:
            print(f"📊 步数{step+1}: 最大Joint距离变化: {max_distance_change:.2f}px")
            
            if explosion_detected:
                print(f"❌ 检测到Joint散架!")
                break
        
        time.sleep(0.01)  # 控制速度便于观察
    
    print(f"\n📊 最终结果:")
    print(f"   碰撞检测: {'✅' if collision_detected else '❌'}")
    print(f"   Joint稳定性: {'❌ 散架' if explosion_detected else '✅ 稳定'}")

if __name__ == "__main__":
    test_collision_stability()
