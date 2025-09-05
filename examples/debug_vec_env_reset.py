#!/usr/bin/env python3
"""
调试向量化环境重置行为的脚本
验证为什么enhanced_train.py中角度修改不生效
"""

import sys
import os
import numpy as np
import time
import torch

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/env_config'))

from reacher2d_env import Reacher2DEnv
from env_wrapper import make_reacher2d_vec_envs

def test_direct_env():
    """测试直接环境"""
    print("=" * 60)
    print("🔍 测试1: 直接 Reacher2DEnv")
    print("=" * 60)
    
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode=None,  # 不渲染，只看数据
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    for reset_count in range(3):
        print(f"\n🔄 直接环境重置 #{reset_count + 1}")
        obs = env.reset()
        
        print(f"📐 关节角度: {[f'{angle:.4f}' for angle in env.joint_angles]}")
        positions = env._calculate_link_positions()
        print(f"📍 第一个Link方向: [{positions[0][0]:.1f}, {positions[0][1]:.1f}] -> [{positions[1][0]:.1f}, {positions[1][1]:.1f}]")
        
        # 计算第一个link的方向
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        print(f"🧭 第一个Link实际角度: {angle_rad:.4f} 弧度 = {angle_deg:.2f}°")
    
    env.close()

def test_vec_env():
    """测试向量化环境"""
    print("\n" + "=" * 60)
    print("🔍 测试2: 向量化环境")
    print("=" * 60)
    
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': None,
        'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    # 创建向量化环境（单进程）
    envs = make_reacher2d_vec_envs(
        env_params=env_params,
        seed=42,
        num_processes=1,  # 单进程便于调试
        gamma=0.99,
        log_dir=None,
        device=torch.device('cpu'),
        allow_early_resets=False
    )
    
    print(f"📋 向量化环境类型: {type(envs)}")
    
    # 初始重置
    print(f"\n🔄 向量化环境初始重置")
    obs = envs.reset()
    print(f"📊 观察空间形状: {obs.shape}")
    
    # 尝试访问底层环境
    if hasattr(envs, 'envs'):
        print(f"📋 底层环境数量: {len(envs.envs)}")
        if hasattr(envs.envs[0], 'joint_angles'):
            print(f"📐 底层环境关节角度: {[f'{angle:.4f}' for angle in envs.envs[0].joint_angles]}")
        elif hasattr(envs.envs[0], 'env') and hasattr(envs.envs[0].env, 'joint_angles'):
            print(f"📐 底层环境关节角度: {[f'{angle:.4f}' for angle in envs.envs[0].env.joint_angles]}")
        else:
            print("⚠️ 无法访问底层环境的关节角度")
    
    # 模拟训练步骤
    print(f"\n🚀 模拟训练步骤...")
    for step in range(5):
        # 随机动作
        action = torch.from_numpy(np.array([envs.action_space.sample() for _ in range(1)]))
        
        obs, reward, done, info = envs.step(action)
        
        print(f"Step {step}: done={done}, reward={reward}")
        
        # 检查是否有环境重置
        if done.any():
            print(f"  🔄 环境在step {step}重置了")
            # 这里向量化环境会自动重置
    
    envs.close()

def test_manual_reset():
    """测试手动重置"""
    print("\n" + "=" * 60)
    print("🔍 测试3: 手动重置对比")
    print("=" * 60)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode=None,
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    print("📝 测试不同角度设置的效果...")
    
    # 测试不同的初始角度
    test_angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    test_names = ["0° (右)", "45° (右下)", "90° (下)", "180° (左)", "270° (上)"]
    
    for i, (angle, name) in enumerate(zip(test_angles, test_names)):
        print(f"\n🔧 设置 joint_angles[0] = {angle:.4f} ({name})")
        
        # 手动设置角度
        env.joint_angles[0] = angle
        
        # 重置环境
        obs = env.reset()
        
        # 检查结果
        print(f"📐 重置后关节角度: {[f'{a:.4f}' for a in env.joint_angles]}")
        
        positions = env._calculate_link_positions()
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        actual_angle = np.arctan2(dy, dx)
        actual_deg = np.degrees(actual_angle)
        
        print(f"🧭 实际第一个Link角度: {actual_angle:.4f} 弧度 = {actual_deg:.2f}°")
        
        # 验证是否匹配
        angle_diff = abs(actual_angle - angle)
        if angle_diff > 0.1:
            print(f"⚠️ 角度不匹配! 期望: {np.degrees(angle):.2f}°, 实际: {actual_deg:.2f}°")
        else:
            print(f"✅ 角度匹配!")
    
    env.close()

def main():
    """主测试函数"""
    print("🔍 调试向量化环境重置行为")
    print("目标：理解为什么enhanced_train.py中修改角度不生效")
    
    test_direct_env()
    test_vec_env() 
    test_manual_reset()
    
    print("\n" + "=" * 60)
    print("📊 结论分析:")
    print("1. 直接环境: 每次reset()都会重新设置角度")
    print("2. 向量化环境: 在训练过程中很少调用reset()")
    print("3. enhanced_train.py看不到角度变化的原因:")
    print("   - 环境创建后，大部分时间在执行step()") 
    print("   - 只有episode结束时才会重置")
    print("   - 而且重置时会重新计算角度")
    print("=" * 60)

if __name__ == "__main__":
    main()
