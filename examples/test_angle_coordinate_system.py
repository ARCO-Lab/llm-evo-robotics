#!/usr/bin/env python3
"""
测试角度坐标系
验证不同角度值对应的实际方向
"""

import sys
import os
import numpy as np
import math

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv

def test_angle_directions():
    """测试不同角度对应的实际方向"""
    print("=" * 60)
    print("🧭 测试角度坐标系")
    print("=" * 60)
    
    # 测试角度
    test_angles = [
        (0, "0° (数学中的正东)"),
        (math.pi/2, "π/2 (数学中的正北)"), 
        (math.pi, "π (数学中的正西)"),
        (3*math.pi/2, "3π/2 (数学中的正南)")
    ]
    
    for angle, description in test_angles:
        print(f"\n🔍 测试角度: {angle:.4f} 弧度 ({description})")
        
        # 创建环境
        env = Reacher2DEnv(
            num_links=4,
            link_lengths=[80, 80, 80, 60],
            render_mode=None,
            config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        # 重置环境
        env.reset()
        
        # 手动设置第一个关节角度
        env.joint_angles[0] = angle
        env.joint_angles[1:] = 0  # 其他关节设为0
        
        # 计算位置
        positions = env._calculate_link_positions()
        
        print(f"   📍 基座位置: [{positions[0][0]:.1f}, {positions[0][1]:.1f}]")
        print(f"   📍 第一个Link末端: [{positions[1][0]:.1f}, {positions[1][1]:.1f}]")
        
        # 计算方向
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        
        print(f"   📐 位移: dx={dx:.1f}, dy={dy:.1f}")
        
        # 判断方向
        if abs(dx) > abs(dy):
            if dx > 0:
                direction = "→ 水平向右"
            else:
                direction = "← 水平向左"
        else:
            if dy > 0:
                direction = "↓ 垂直向下"
            else:
                direction = "↑ 垂直向上"
        
        print(f"   🧭 实际方向: {direction}")
        
        # 计算实际角度
        actual_angle = math.atan2(dy, dx)
        actual_degrees = math.degrees(actual_angle)
        print(f"   📊 实际角度: {actual_angle:.4f} 弧度 = {actual_degrees:.2f}°")
        
        env.close()

def test_current_reset_behavior():
    """测试当前reset()的行为"""
    print("\n" + "=" * 60)
    print("🔄 测试当前reset()的实际行为")
    print("=" * 60)
    
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode=None,
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    for i in range(3):
        print(f"\n🔄 重置 #{i+1}")
        env.reset()
        
        print(f"   📐 关节角度: {[f'{a:.4f}' for a in env.joint_angles]}")
        
        positions = env._calculate_link_positions()
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        
        actual_angle = math.atan2(dy, dx)
        actual_degrees = math.degrees(actual_angle)
        
        print(f"   📍 第一个Link: [{positions[0][0]:.1f}, {positions[0][1]:.1f}] → [{positions[1][0]:.1f}, {positions[1][1]:.1f}]")
        print(f"   🧭 实际方向角度: {actual_degrees:.2f}°")
        
        if abs(actual_degrees) < 30:
            direction = "水平向右"
        elif abs(actual_degrees - 90) < 30:
            direction = "垂直向下"
        elif abs(actual_degrees - 180) < 30 or abs(actual_degrees + 180) < 30:
            direction = "水平向左"
        elif abs(actual_degrees - 270) < 30 or abs(actual_degrees + 90) < 30:
            direction = "垂直向上"
        else:
            direction = f"其他方向 ({actual_degrees:.1f}°)"
        
        print(f"   🎯 方向判断: {direction}")
    
    env.close()

def main():
    """主测试函数"""
    print("🔍 角度坐标系测试")
    print("目标：理解为什么设置π/2会显示为水平向右")
    
    test_angle_directions()
    test_current_reset_behavior()
    
    print("\n" + "=" * 60)
    print("📊 分析结果，找出角度坐标系的问题")
    print("=" * 60)

if __name__ == "__main__":
    main()
