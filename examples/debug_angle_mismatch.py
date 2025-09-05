#!/usr/bin/env python3
"""
调试角度不匹配问题
对比 enhanced_train.py 和 test_initial_pose.py 中的实际角度
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
    """测试直接环境创建"""
    print("=" * 60)
    print("🔍 测试1: 直接创建 Reacher2DEnv")
    print("=" * 60)
    
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode=None,  # 不渲染，只看数据
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    print("🔄 重置环境...")
    obs = env.reset()
    
    print(f"📐 关节角度: {[f'{angle:.4f}' for angle in env.joint_angles]}")
    print(f"📐 第一个关节角度: {env.joint_angles[0]:.4f} 弧度 = {np.degrees(env.joint_angles[0]):.2f}°")
    
    positions = env._calculate_link_positions()
    print(f"📍 基座位置: [{positions[0][0]:.1f}, {positions[0][1]:.1f}]")
    print(f"📍 末端位置: [{positions[-1][0]:.1f}, {positions[-1][1]:.1f}]")
    
    # 计算第一个link的方向
    dx = positions[1][0] - positions[0][0]
    dy = positions[1][1] - positions[0][1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    print(f"🧭 第一个Link实际方向角度: {angle_rad:.4f} 弧度 = {angle_deg:.2f}°")
    
    if abs(angle_deg) < 30:  # 接近水平右
        print("   → 水平向右")
    elif abs(angle_deg - 90) < 30:  # 接近垂直下
        print("   → 垂直向下")
    elif abs(angle_deg - 180) < 30:  # 接近水平左
        print("   → 水平向左")
    else:
        print(f"   → 其他方向")
    
    env.close()

def test_enhanced_train_style():
    """模拟 enhanced_train.py 的环境创建方式"""
    print("\n" + "=" * 60)
    print("🔍 测试2: 模拟 enhanced_train.py 的环境创建")
    print("=" * 60)
    
    # 使用与 enhanced_train.py 相同的参数
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    # 创建向量化环境
    envs = make_reacher2d_vec_envs(
        env_params=env_params,
        seed=42,  # enhanced_train.py 中的默认种子
        num_processes=1,
        gamma=0.99,
        log_dir=None,
        device=torch.device('cpu'),
        allow_early_resets=False
    )
    
    # 创建同步渲染环境 (就像 enhanced_train.py 中的 sync_env)
    render_env_params = env_params.copy()
    sync_env = Reacher2DEnv(**render_env_params)
    
    print("🔄 重置环境...")
    
    # 重置向量化环境
    current_obs = envs.reset()
    print(f"📊 向量化环境观察: {current_obs[0][:4]}")  # 前4个值是关节角度
    print(f"📐 向量化环境第一个关节角度: {current_obs[0][0]:.4f} 弧度 = {np.degrees(current_obs[0][0].item()):.2f}°")
    
    # 重置同步环境
    sync_env.reset()
    print(f"📐 同步环境关节角度: {[f'{angle:.4f}' for angle in sync_env.joint_angles]}")
    print(f"📐 同步环境第一个关节角度: {sync_env.joint_angles[0]:.4f} 弧度 = {np.degrees(sync_env.joint_angles[0]):.2f}°")
    
    # 检查同步环境的位置
    positions = sync_env._calculate_link_positions()
    print(f"📍 同步环境基座位置: [{positions[0][0]:.1f}, {positions[0][1]:.1f}]")
    print(f"📍 同步环境末端位置: [{positions[-1][0]:.1f}, {positions[-1][1]:.1f}]")
    
    # 计算第一个link的方向
    dx = positions[1][0] - positions[0][0]
    dy = positions[1][1] - positions[0][1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    print(f"🧭 同步环境第一个Link实际方向角度: {angle_rad:.4f} 弧度 = {angle_deg:.2f}°")
    
    if abs(angle_deg) < 30:  # 接近水平右
        print("   → 水平向右")
    elif abs(angle_deg - 90) < 30:  # 接近垂直下
        print("   → 垂直向下")
    elif abs(angle_deg - 180) < 30:  # 接近水平左
        print("   → 水平向左")
    else:
        print(f"   → 其他方向")
    
    envs.close()
    sync_env.close()

def test_seed_effect():
    """测试不同种子的影响"""
    print("\n" + "=" * 60)
    print("🔍 测试3: 测试不同种子对初始角度的影响")
    print("=" * 60)
    
    seeds = [42, 0, 123, 999]
    
    for seed in seeds:
        print(f"\n🌱 种子: {seed}")
        
        env = Reacher2DEnv(
            num_links=4,
            link_lengths=[80, 80, 80, 60],
            render_mode=None,
            config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        # 设置种子
        env.seed(seed)
        np.random.seed(seed)
        
        obs = env.reset()
        
        print(f"   📐 第一个关节角度: {env.joint_angles[0]:.4f} 弧度 = {np.degrees(env.joint_angles[0]):.2f}°")
        
        positions = env._calculate_link_positions()
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        if abs(angle_deg) < 30:
            direction = "水平向右"
        elif abs(angle_deg - 90) < 30:
            direction = "垂直向下"
        elif abs(angle_deg - 180) < 30:
            direction = "水平向左"
        else:
            direction = f"其他 ({angle_deg:.1f}°)"
        
        print(f"   🧭 方向: {direction}")
        
        env.close()

def main():
    """主测试函数"""
    print("🔍 调试角度不匹配问题")
    print("目标：找出为什么 enhanced_train.py 显示水平向右而不是垂直向下")
    
    test_direct_env()
    test_enhanced_train_style()
    test_seed_effect()
    
    print("\n" + "=" * 60)
    print("📊 总结:")
    print("通过对比不同环境创建方式和种子设置，")
    print("找出 enhanced_train.py 中角度设置不生效的原因")
    print("=" * 60)

if __name__ == "__main__":
    main()
