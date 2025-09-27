#!/usr/bin/env python3
"""
简单的Baseline SAC训练3关节Reacher
使用现有环境但移除所有自定义组件
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from complete_sequential_training_with_evaluation import create_env

def train_simple_baseline():
    """训练简单baseline SAC，移除所有自定义组件"""
    print("🚀 开始训练简单Baseline SAC 3关节Reacher")
    print("📋 配置:")
    print("  - 使用现有3关节环境")
    print("  - 移除自定义特征提取器")
    print("  - 使用标准SAC默认参数")
    print("  - 标准MlpPolicy")
    
    # 创建环境
    env = create_env(3, render_mode=None)
    print(f"✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space.shape}")
    print(f"   动作空间: {env.action_space.shape}")
    
    # 创建纯baseline SAC模型 - 不使用任何自定义组件
    model = SAC(
        'MlpPolicy',  # 标准MLP策略，不使用自定义特征提取器
        env,
        verbose=1,
        learning_rate=3e-4,    # 标准学习率
        buffer_size=1000000,   # 标准buffer
        batch_size=256,        # 标准batch大小
        tau=0.005,            # 标准软更新参数
        gamma=0.99,           # 标准折扣因子
        train_freq=1,         # 标准训练频率
        gradient_steps=1,     # 标准梯度步数
        # 不传入任何policy_kwargs，使用完全默认的网络结构
    )
    
    print("✅ 简单Baseline SAC模型创建完成")
    print("   - 使用标准MlpPolicy")
    print("   - 默认网络结构: [256, 256]")
    print("   - 无自定义特征提取器")
    print("   - 所有参数为SAC默认值")
    
    # 训练
    print("\n🎯 开始训练10000步...")
    model.learn(total_timesteps=10000, progress_bar=True)
    
    # 保存模型
    model.save('models/simple_baseline_3joint_sac')
    print("💾 模型已保存: models/simple_baseline_3joint_sac.zip")
    
    # 测试
    print("\n🧪 测试简单baseline模型...")
    success_count = 0
    rewards = []
    distances = []
    
    for i in range(10):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        min_distance = float('inf')
        
        target_pos = info.get('target_pos', [0, 0])
        initial_distance = info.get('distance_to_target', 0)
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            distance = info.get('distance_to_target', 0)
            min_distance = min(min_distance, distance)
            
            if info.get('is_success', False):
                episode_success = True
            
            if terminated or truncated:
                break
        
        if episode_success:
            success_count += 1
        
        rewards.append(episode_reward)
        distances.append(min_distance)
        
        target_dist = np.linalg.norm(target_pos)
        improvement = initial_distance - min_distance
        print(f"  Episode {i+1}: 目标距离={target_dist:.3f}, 改善={improvement:.4f}, 奖励={episode_reward:.1f}, 成功={'✅' if episode_success else '❌'}")
    
    print(f"\n📊 简单Baseline测试结果:")
    print(f"  成功率: {success_count/10:.1%}")
    print(f"  平均奖励: {np.mean(rewards):.1f}")
    print(f"  平均最小距离: {np.mean(distances):.4f}")
    print(f"  奖励范围: [{min(rewards):.1f}, {max(rewards):.1f}]")
    
    env.close()
    return model

def compare_with_custom_model():
    """对比自定义模型和baseline模型"""
    print("\n🔍 对比分析:")
    print("=" * 60)
    
    # 加载并测试自定义模型
    print("\n1️⃣ 测试自定义模型 (SpecializedJointExtractor)")
    try:
        from stable_baselines3 import SAC
        custom_model = SAC.load('models/complete_sequential_3joint_reacher.zip')
        env = create_env(3, render_mode=None)
        
        custom_rewards = []
        custom_success = 0
        
        for i in range(5):
            obs, info = env.reset()
            episode_reward = 0
            episode_success = False
            
            for step in range(100):
                action, _ = custom_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if info.get('is_success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                custom_success += 1
            custom_rewards.append(episode_reward)
        
        print(f"   自定义模型成功率: {custom_success/5:.1%}")
        print(f"   自定义模型平均奖励: {np.mean(custom_rewards):.1f}")
        
        env.close()
        
    except Exception as e:
        print(f"   ❌ 自定义模型测试失败: {e}")
    
    # 测试baseline模型
    print("\n2️⃣ 测试Baseline模型 (标准MlpPolicy)")
    try:
        baseline_model = SAC.load('models/simple_baseline_3joint_sac.zip')
        env = create_env(3, render_mode=None)
        
        baseline_rewards = []
        baseline_success = 0
        
        for i in range(5):
            obs, info = env.reset()
            episode_reward = 0
            episode_success = False
            
            for step in range(100):
                action, _ = baseline_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if info.get('is_success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                baseline_success += 1
            baseline_rewards.append(episode_reward)
        
        print(f"   Baseline模型成功率: {baseline_success/5:.1%}")
        print(f"   Baseline模型平均奖励: {np.mean(baseline_rewards):.1f}")
        
        env.close()
        
    except Exception as e:
        print(f"   ❌ Baseline模型测试失败: {e}")
    
    print("\n📋 对比总结:")
    print("   如果Baseline模型表现更好，说明自定义特征提取器可能有问题")
    print("   如果自定义模型表现更好，说明特征提取器是有效的")

if __name__ == "__main__":
    # 训练baseline模型
    train_simple_baseline()
    
    # 对比分析
    compare_with_custom_model()

