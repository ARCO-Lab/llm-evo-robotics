#!/usr/bin/env python3
"""
测试 SAC verbose 输出
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

def test_verbose_sac():
    print("🧪 测试 SAC verbose 输出")
    
    # 创建简单环境
    env = gym.make('Reacher-v5')
    env = Monitor(env)
    
    print("✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建 SAC 模型 - 使用最大 verbose
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=100,      # 很早开始学习
        batch_size=64,
        verbose=2,                # 最大 verbose
        device='cpu'
    )
    
    print("✅ SAC 模型创建完成")
    print("🚀 开始训练 (2000 steps)...")
    
    # 训练
    model.learn(
        total_timesteps=2000,
        progress_bar=True
    )
    
    print("✅ 训练完成")
    env.close()

if __name__ == "__main__":
    test_verbose_sac()


