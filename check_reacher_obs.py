#!/usr/bin/env python3
"""
检查 MuJoCo Reacher-v5 的观察空间格式
"""

import gymnasium as gym
import numpy as np

def check_reacher_observation_format():
    """检查 Reacher-v5 的观察空间格式"""
    print("🔍 检查 MuJoCo Reacher-v5 观察空间格式")
    print("=" * 50)
    
    # 创建环境
    env = gym.make('Reacher-v5')
    
    print(f"📊 环境信息:")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print(f"   观察维度: {env.observation_space.shape[0]}")
    print(f"   动作维度: {env.action_space.shape[0]}")
    
    print("\n🎯 采样观察空间...")
    
    # 重置环境并获取观察
    obs, info = env.reset(seed=42)
    
    print(f"\n📋 观察向量 (10维):")
    for i, val in enumerate(obs):
        print(f"   [{i}]: {val:.6f}")
    
    print(f"\n🔍 分析观察空间结构...")
    
    # 执行几步以观察变化
    for step in range(3):
        action = env.action_space.sample()
        obs_new, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n📊 Step {step+1}:")
        print(f"   动作: [{action[0]:.3f}, {action[1]:.3f}]")
        print(f"   奖励: {reward:.3f}")
        
        # 分析观察变化
        obs_diff = obs_new - obs
        print(f"   观察变化:")
        for i, (old_val, new_val, diff) in enumerate(zip(obs, obs_new, obs_diff)):
            if abs(diff) > 1e-6:
                print(f"     [{i}]: {old_val:.6f} → {new_val:.6f} (Δ{diff:+.6f})")
        
        obs = obs_new
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    print(f"\n📚 MuJoCo Reacher-v5 观察空间文档:")
    print(f"   根据 MuJoCo 文档，Reacher-v5 的 10 维观察应该是:")
    print(f"   [0-1]: cos/sin of joint 1 angle")
    print(f"   [2-3]: cos/sin of joint 2 angle") 
    print(f"   [4-5]: joint 1 and joint 2 velocities")
    print(f"   [6-7]: end effector position (x, y)")
    print(f"   [8-9]: target position (x, y)")
    print(f"   注意：没有 'vector from target to end effector'")
    
    print(f"\n✅ 观察空间格式检查完成")

if __name__ == "__main__":
    check_reacher_observation_format()


