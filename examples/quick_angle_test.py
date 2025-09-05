#!/usr/bin/env python3
"""
快速角度测试 - 直接检查问题
"""

import sys
import os
import numpy as np

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv

def quick_test():
    """快速测试当前环境的行为"""
    print("🔍 快速角度测试")
    
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode=None,
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    # 重置环境
    env.reset()
    
    # 直接检查关节角度
    print(f"📐 关节角度: {env.joint_angles}")
    print(f"📐 第一个关节: {env.joint_angles[0]:.4f} 弧度")
    
    # 计算末端位置
    end_pos = env._get_end_effector_position()
    print(f"📍 末端位置: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
    print(f"📍 基座位置: {env.anchor_point}")
    
    # 如果末端位置的x坐标接近450，说明是水平向右
    # 如果末端位置的y坐标接近920，说明是垂直向下
    if abs(end_pos[0] - 450) < 50:
        print("🚨 检测到水平向右！")
        print("   这意味着第一个关节角度实际上是0，不是π/2")
        
        # 检查reset方法中的代码
        print("\n🔍 检查reset方法...")
        print(f"   self.joint_angles[0] 应该设置为 π/2 = {np.pi/2:.4f}")
        print(f"   但实际值是: {env.joint_angles[0]:.4f}")
        
        if abs(env.joint_angles[0]) < 0.1:  # 接近0
            print("❌ 问题确认：第一个关节角度被设置为0而不是π/2")
        elif abs(env.joint_angles[0] - np.pi/2) < 0.2:  # 接近π/2
            print("❓ 奇怪：角度设置正确，但位置计算有问题")
    
    elif abs(end_pos[1] - 920) < 50:
        print("✅ 检测到垂直向下，这是正确的")
    
    else:
        print(f"❓ 未知方向：末端位置 [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
    
    env.close()

if __name__ == "__main__":
    quick_test()
