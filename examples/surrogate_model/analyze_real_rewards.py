#!/usr/bin/env python3
"""
分析真实奖励范围和Critic loss计算
找出为什么Critic loss还是偏高的原因
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "examples/2d_reacher/envs"))
sys.path.append(os.path.join(base_dir, "examples/2d_reacher/utils"))

from reacher2d_env import Reacher2DEnv
from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
from sac.universal_ppo_model import UniversalAttnModel, UniversalPPOWithBuffer

def analyze_real_rewards_and_critic():
    """分析真实奖励范围和Critic预测"""
    
    print("🔍 分析真实奖励范围和Critic Loss计算...")
    print("=" * 60)
    
    # === 1. 创建环境并收集真实奖励 ===
    print("\n📊 步骤1: 收集真实环境奖励")
    
    env = Reacher2DEnv(num_links=5, render_mode=None)
    gnn_encoder = Reacher2D_GNN_Encoder()
    
    rewards_collected = []
    episode_returns = []
    
    # 收集多个episode的奖励
    for episode in range(3):
        obs = env.reset()
        episode_rewards = []
        episode_return = 0
        
        for step in range(100):  # 每个episode 100步
            # 随机动作
            action = np.random.uniform(-10, 10, size=env.num_links)
            obs, reward, done, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_return += reward
            rewards_collected.append(reward)
            
            if done:
                break
        
        episode_returns.append(episode_return)
        print(f"   Episode {episode+1}: 单步奖励范围 [{min(episode_rewards):.3f}, {max(episode_rewards):.3f}], 累积奖励: {episode_return:.3f}")
    
    rewards_array = np.array(rewards_collected)
    returns_array = np.array(episode_returns)
    
    print(f"\n📊 真实奖励统计:")
    print(f"   单步奖励范围: [{rewards_array.min():.3f}, {rewards_array.max():.3f}]")
    print(f"   单步奖励均值: {rewards_array.mean():.3f}")
    print(f"   单步奖励标准差: {rewards_array.std():.3f}")
    print(f"   Episode累积奖励范围: [{returns_array.min():.3f}, {returns_array.max():.3f}]")
    print(f"   Episode累积奖励均值: {returns_array.mean():.3f}")
    
    # === 2. 分析Critic网络预测范围 ===
    print(f"\n🏛️ 步骤2: 分析Critic网络预测范围")
    
    device = 'cpu'
    ppo_agent = UniversalPPOWithBuffer(
        buffer_size=1024,
        batch_size=32,
        device=device
    )
    
    # 创建测试数据
    obs = env.reset()
    gnn_embeds = gnn_encoder.encode(obs)
    joint_q, vertex_k, vertex_v = ppo_agent._prepare_inputs(obs, gnn_embeds, env.num_links)
    
    # 测试Critic预测范围
    with torch.no_grad():
        critic_values = []
        for _ in range(50):  # 多次随机测试
            # 随机扰动输入
            noise = torch.randn_like(joint_q) * 0.1
            noisy_input = joint_q + noise
            
            value = ppo_agent.critic(noisy_input, vertex_k, vertex_v)
            critic_values.append(value.item())
    
    critic_array = np.array(critic_values)
    print(f"   Critic预测范围: [{critic_array.min():.3f}, {critic_array.max():.3f}]")
    print(f"   Critic预测均值: {critic_array.mean():.3f}")
    print(f"   Critic预测标准差: {critic_array.std():.3f}")
    print(f"   Critic Value Scale: {ppo_agent.critic.value_scale}")
    
    # === 3. 模拟Returns计算 ===
    print(f"\n🔄 步骤3: 模拟Returns计算过程")
    
    # 模拟一个简单的trajectory
    simulated_rewards = rewards_array[:10]  # 取前10个真实奖励
    simulated_values = critic_array[:10]    # 取前10个Critic预测
    
    # 计算returns（简化版GAE）
    gamma = 0.99
    gae_lambda = 0.95
    
    # 奖励归一化（和代码中一样）
    if len(simulated_rewards) > 1:
        reward_mean = simulated_rewards.mean()
        reward_std = simulated_rewards.std() + 1e-8
        normalized_rewards = (simulated_rewards - reward_mean) / reward_std
    else:
        normalized_rewards = simulated_rewards
    
    print(f"   原始奖励范围: [{simulated_rewards.min():.3f}, {simulated_rewards.max():.3f}]")
    print(f"   归一化后奖励范围: [{normalized_rewards.min():.3f}, {normalized_rewards.max():.3f}]")
    
    # 计算returns
    returns = np.zeros_like(normalized_rewards)
    advantages = np.zeros_like(normalized_rewards)
    gae = 0
    
    for step in reversed(range(len(normalized_rewards))):
        if step == len(normalized_rewards) - 1:
            next_value = 0  # 最后一步
        else:
            next_value = simulated_values[step + 1]
        
        delta = normalized_rewards[step] + gamma * next_value - simulated_values[step]
        gae = delta + gamma * gae_lambda * gae
        advantages[step] = gae
        returns[step] = advantages[step] + simulated_values[step]
    
    print(f"   计算出的Returns范围: [{returns.min():.3f}, {returns.max():.3f}]")
    print(f"   计算出的Advantages范围: [{advantages.min():.3f}, {advantages.max():.3f}]")
    
    # === 4. 分析Critic Loss ===
    print(f"\n📊 步骤4: 分析Critic Loss大小")
    
    # 模拟critic loss计算
    predicted_values = np.array(simulated_values[:len(returns)])
    actual_returns = returns
    
    # 不同loss函数的结果
    mse_loss = np.mean((predicted_values - actual_returns) ** 2)
    mae_loss = np.mean(np.abs(predicted_values - actual_returns))
    smooth_l1_loss = np.mean(np.where(
        np.abs(predicted_values - actual_returns) < 1.0,
        0.5 * (predicted_values - actual_returns) ** 2,
        np.abs(predicted_values - actual_returns) - 0.5
    ))
    
    print(f"   预测值范围: [{predicted_values.min():.3f}, {predicted_values.max():.3f}]")
    print(f"   真实Returns范围: [{actual_returns.min():.3f}, {actual_returns.max():.3f}]")
    print(f"   预测误差(绝对值): [{np.abs(predicted_values - actual_returns).min():.3f}, {np.abs(predicted_values - actual_returns).max():.3f}]")
    print(f"   MSE Loss: {mse_loss:.3f}")
    print(f"   MAE Loss: {mae_loss:.3f}")
    print(f"   Smooth L1 Loss: {smooth_l1_loss:.3f}")
    
    # === 5. 诊断问题 ===
    print(f"\n🔍 步骤5: 问题诊断")
    
    issues = []
    
    # 检查数值范围匹配
    reward_range = rewards_array.max() - rewards_array.min()
    critic_range = critic_array.max() - critic_array.min()
    range_ratio = critic_range / reward_range if reward_range > 0 else float('inf')
    
    print(f"   环境奖励范围: {reward_range:.3f}")
    print(f"   Critic预测范围: {critic_range:.3f}")
    print(f"   范围比例: {range_ratio:.3f}")
    
    if range_ratio > 5 or range_ratio < 0.2:
        issues.append(f"Critic预测范围与环境奖励范围不匹配 (比例: {range_ratio:.2f})")
    
    # 检查奖励归一化的影响
    if reward_std > 1.0:
        issues.append(f"奖励标准差过大 ({reward_std:.3f})，归一化可能导致数值失真")
    
    # 检查loss大小
    if smooth_l1_loss > 5.0:
        issues.append(f"Smooth L1 Loss仍然过大 ({smooth_l1_loss:.3f})")
    
    if len(issues) == 0:
        print("   ✅ 未发现明显问题")
    else:
        print("   ⚠️ 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
    
    # === 6. 推荐的进一步修复 ===
    print(f"\n🔧 推荐的进一步修复:")
    
    if range_ratio > 5:
        print("   1. 进一步降低value_scale")
        new_scale = max(5.0, reward_range * 1.5)
        print(f"      推荐: self.value_scale = {new_scale:.1f}")
    
    if smooth_l1_loss > 5.0:
        print("   2. 调整loss函数或添加更严格的输出限制")
        print("      - 使用MSE loss替代smooth_l1_loss")
        print("      - 或者进一步限制value输出范围")
    
    print("   3. 检查奖励归一化是否合适")
    print("      - 考虑不进行奖励归一化")
    print("      - 或者使用更温和的归一化方法")
    
    return {
        'reward_range': [rewards_array.min(), rewards_array.max()],
        'critic_range': [critic_array.min(), critic_array.max()],
        'predicted_loss': smooth_l1_loss,
        'issues': issues
    }

if __name__ == "__main__":
    try:
        results = analyze_real_rewards_and_critic()
        print(f"\n🎉 分析完成！")
        print(f"主要发现: {len(results['issues'])} 个潜在问题")
    except Exception as e:
        print(f"\n❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
