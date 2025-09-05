#!/usr/bin/env python3
"""
检查基座关节的实际位置
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

def check_base_position():
    """检查基座关节的实际位置"""
    print("🔍 检查基座关节位置")
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
    
    # 检查基座关节位置
    base_body = env.bodies[0]  # 基座关节
    anchor_point = env.anchor_point
    
    print(f"🎯 预期基座锚点位置: {anchor_point}")
    print(f"🤖 实际基座关节位置: {base_body.position}")
    print(f"📏 距离锚点的偏差: {abs(base_body.position[0] - anchor_point[0]):.2f}px (X轴)")
    print(f"📏 距离锚点的偏差: {abs(base_body.position[1] - anchor_point[1]):.2f}px (Y轴)")
    
    # 检查关节连接
    if len(env.joints) > 0:
        base_joint = env.joints[0]
        print(f"🔗 基座关节类型: {type(base_joint).__name__}")
        print(f"🔗 基座关节连接: {base_joint.a} <-> {base_body}")
        print(f"🔗 基座关节是否连接到static_body: {base_joint.a == env.space.static_body}")
        
        # 检查关节的锚点
        print(f"🔗 关节锚点A (static_body): {base_joint.anchor_a}")
        print(f"🔗 关节锚点B (base_body): {base_joint.anchor_b}")
    
    # 运行几步看位置变化
    print(f"\n⏳ 运行10步检查位置变化...")
    for i in range(10):
        env.step([0, 0, 0, 0])  # 无动作
        current_pos = base_body.position
        print(f"步骤 {i+1}: 基座位置 = ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
        
        # 如果位置变化超过1像素，说明基座没有固定
        if abs(current_pos[0] - anchor_point[0]) > 1 or abs(current_pos[1] - anchor_point[1]) > 1:
            print(f"❌ 警告：基座位置偏离锚点超过1像素！")
            break
    else:
        print(f"✅ 基座位置稳定，固定正常")
    
    env.close()

if __name__ == "__main__":
    check_base_position()

