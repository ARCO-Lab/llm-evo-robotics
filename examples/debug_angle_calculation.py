#!/usr/bin/env python3
"""
调试角度计算的脚本
验证为什么base link朝向水平向右
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

def debug_angle_calculation():
    """调试角度计算"""
    print("=" * 60)
    print("🔍 调试机器人角度计算")
    print("=" * 60)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[60, 60, 60, 60],
        render_mode=None,  # 不渲染，只计算
        debug_level='SILENT'
    )
    
    print(f"📍 基座位置: {env.anchor_point}")
    
    # 重置环境
    env.reset()
    
    print(f"\n📐 初始关节角度:")
    for i, angle in enumerate(env.joint_angles):
        degrees = math.degrees(angle)
        print(f"   关节{i}: {angle:.4f} 弧度 = {degrees:.2f}°")
    
    # 计算位置
    positions = env._calculate_link_positions()
    
    print(f"\n📍 Link位置计算:")
    for i, pos in enumerate(positions):
        if i == 0:
            print(f"   基座: [{pos[0]:.1f}, {pos[1]:.1f}]")
        else:
            print(f"   Link{i}: [{pos[0]:.1f}, {pos[1]:.1f}]")
    
    # 手动验证第一个link的计算
    print(f"\n🧮 手动验证第一个Link计算:")
    base_angle = env.joint_angles[0]
    link_length = env.link_lengths[0]
    
    print(f"   基座角度: {base_angle:.4f} 弧度 = {math.degrees(base_angle):.2f}°")
    print(f"   Link长度: {link_length}")
    
    # 计算第一个link的终点
    dx = link_length * np.cos(base_angle)
    dy = link_length * np.sin(base_angle)
    
    print(f"   dx = {link_length} * cos({base_angle:.4f}) = {dx:.2f}")
    print(f"   dy = {link_length} * sin({base_angle:.4f}) = {dy:.2f}")
    
    end_pos = np.array(env.anchor_point) + np.array([dx, dy])
    print(f"   第一个Link终点: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
    
    # 验证方向
    print(f"\n🧭 方向分析:")
    if abs(dx) > abs(dy):
        if dx > 0:
            direction = "主要向右"
        else:
            direction = "主要向左"
    else:
        if dy > 0:
            direction = "主要向下"
        else:
            direction = "主要向上"
    
    print(f"   第一个Link方向: {direction}")
    print(f"   水平分量: {dx:.2f} (正数=向右, 负数=向左)")
    print(f"   垂直分量: {dy:.2f} (正数=向下, 负数=向上)")
    
    # 测试不同角度
    print(f"\n🔄 测试不同角度的方向:")
    test_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    angle_names = ["0° (右)", "45° (右下)", "90° (下)", "135° (左下)", 
                   "180° (左)", "225° (左上)", "270° (上)", "315° (右上)"]
    
    for angle, name in zip(test_angles, angle_names):
        test_dx = 60 * np.cos(angle)
        test_dy = 60 * np.sin(angle)
        print(f"   {name}: dx={test_dx:+6.1f}, dy={test_dy:+6.1f}")
    
    print(f"\n💡 结论:")
    if abs(dx) > abs(dy) and dx > 0:
        print(f"   ❌ 当前设置确实让第一个Link主要朝向水平向右!")
        print(f"   📊 水平分量({dx:.1f}) > 垂直分量({dy:.1f})")
        print(f"   🔧 要让Link垂直向下，基座角度应该设置为 π/2 = 90°")
        
        # 检查当前角度设置
        current_base_angle_degrees = math.degrees(env.joint_angles[0])
        print(f"   🔍 当前基座角度: {current_base_angle_degrees:.2f}°")
        
        if abs(current_base_angle_degrees - 90) > 10:  # 如果偏离90度超过10度
            print(f"   ⚠️ 角度设置可能有问题！应该接近90°")
    else:
        print(f"   ✅ Link方向正确")

if __name__ == "__main__":
    debug_angle_calculation()

