#!/usr/bin/env python3
"""
深度调试基座关节连接问题
"""

import sys
import os
import numpy as np
import pygame
import time
import pymunk

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def debug_joint_connection():
    """深度调试基座关节连接"""
    print("🔍 深度调试基座关节连接问题")
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
    
    # 检查基座关节的详细信息
    base_body = env.bodies[0]
    base_joint = env.joints[0]
    
    print(f"🎯 基座锚点位置: {env.anchor_point}")
    print(f"🤖 基座关节初始位置: {base_body.position}")
    print(f"🔗 基座Joint信息:")
    print(f"   类型: {type(base_joint).__name__}")
    print(f"   连接体A: {base_joint.a}")
    print(f"   连接体B: {base_joint.b}")
    print(f"   锚点A: {base_joint.anchor_a}")
    print(f"   锚点B: {base_joint.anchor_b}")
    print(f"   max_force: {base_joint.max_force}")
    
    # 检查static_body
    print(f"🏗️ Static body信息:")
    print(f"   类型: {type(env.space.static_body).__name__}")
    print(f"   body_type: {env.space.static_body.body_type}")
    print(f"   是否是STATIC: {env.space.static_body.body_type == pymunk.Body.STATIC}")
    
    # 运行物理仿真并监控位置变化
    print(f"\n⏳ 运行物理仿真监控位置变化...")
    for i in range(50):
        # 运行一步物理仿真
        env.space.step(1/60.0)
        
        current_pos = base_body.position
        distance_from_anchor = ((current_pos[0] - env.anchor_point[0])**2 + 
                               (current_pos[1] - env.anchor_point[1])**2)**0.5
        
        if i % 10 == 0:
            print(f"步骤 {i}: 基座位置 = ({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                  f"距离锚点 = {distance_from_anchor:.2f}px")
        
        # 如果距离超过1像素，说明基座没有固定
        if distance_from_anchor > 1.0:
            print(f"❌ 步骤 {i}: 基座位置偏离锚点 {distance_from_anchor:.2f}px！")
            print(f"   基座速度: {base_body.velocity}")
            print(f"   基座角速度: {base_body.angular_velocity}")
            
            # 检查关节约束力
            print(f"   关节约束力: {base_joint.impulse}")
            break
    else:
        print(f"✅ 基座位置在50步内保持稳定")
    
    env.close()

if __name__ == "__main__":
    debug_joint_connection()

