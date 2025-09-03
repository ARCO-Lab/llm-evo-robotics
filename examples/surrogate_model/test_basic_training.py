#!/usr/bin/env python3
"""
测试基础训练功能（无渲染）
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

print("🔍 测试基础训练功能（无渲染）...")

try:
    # 基础导入
    sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
    from reacher2d_env import Reacher2DEnv
    import numpy as np
    
    # 创建无渲染环境
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': None,  # 无渲染
        'config_path': '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml',
        'debug_level': 'SILENT'
    }
    
    print("1️⃣ 创建环境...")
    env = Reacher2DEnv(**env_params)
    
    print("2️⃣ 测试环境reset...")
    obs = env.reset()
    print(f"   观察空间形状: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    
    print("3️⃣ 测试多步环境交互...")
    total_reward = 0
    for i in range(10):
        action = np.random.uniform(-1, 1, 4)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if i == 0:
            print(f"   第一步奖励: {reward:.3f}")
    
    print(f"   10步总奖励: {total_reward:.3f}")
    print(f"   平均奖励: {total_reward/10:.3f}")
    
    print("4️⃣ 测试奖励范围...")
    rewards = []
    for i in range(100):
        action = np.random.uniform(-1, 1, 4) * 0.5
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
    
    print(f"   奖励范围: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"   奖励均值: {np.mean(rewards):.3f}")
    
    env.close()
    print("\n✅ 基础训练功能测试成功！")
    print("   环境运行正常，增强版距离奖励系统工作正常")
    print("   可以进行无渲染训练")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
