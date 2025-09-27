#!/usr/bin/env python3
"""
简化的注意力机制 SAC
轻量级设计，更适合 Reacher 任务
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, List, Tuple, Type, Union

class SimpleAttentionLayer(nn.Module):
    """
    简化的注意力层 - 专门为 Reacher 任务设计
    不使用多头注意力，而是使用简单的特征加权机制
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(SimpleAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简化的注意力机制：只使用一个线性层计算权重
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # 输出与输入维度相同
            nn.Softmax(dim=-1)  # 归一化权重
        )
        
        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"🧠 SimpleAttentionLayer 初始化: input_dim={input_dim}, hidden_dim={hidden_dim}")
        print(f"   参数量大幅减少，计算更高效")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的前向传播
        x: [batch_size, input_dim]
        """
        # 计算注意力权重
        attention_weights = self.attention_weights(x)  # [batch_size, input_dim]
        
        # 应用注意力权重到输入特征
        weighted_features = x * attention_weights  # 元素级别的加权
        
        # 特征变换
        output = self.feature_transform(weighted_features)
        
        return output

class ReacherSpecificAttention(nn.Module):
    """
    专门为 Reacher 任务设计的注意力机制
    考虑 Reacher 观察空间的特定结构
    """
    def __init__(self, obs_dim: int = 10, hidden_dim: int = 64):
        super(ReacherSpecificAttention, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        # MuJoCo Reacher-v5 观察空间结构 (10维):
        # [0:2] - cos/sin of joint angles (关节角度)
        # [2:4] - joint velocities (关节速度)  
        # [4:6] - end effector position (末端位置 x,y)
        # [6:8] - target position (目标位置 x,y)
        # [8:10] - vector from target to end effector (目标到末端的向量)
        
        # 为不同类型的特征设计不同的注意力权重
        self.joint_attention = nn.Linear(4, 4)      # 关节相关 (角度+速度)
        self.position_attention = nn.Linear(4, 4)   # 位置相关 (末端+目标)
        self.vector_attention = nn.Linear(2, 2)     # 向量相关 (目标到末端)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        print(f"🎯 ReacherSpecificAttention 初始化:")
        print(f"   专门针对 Reacher {obs_dim}维观察空间设计")
        print(f"   分别处理关节、位置、向量信息")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        针对 Reacher 结构的注意力处理
        """
        batch_size = x.size(0)
        
        # 分解观察空间 (10维)
        joint_features = x[:, :4]      # 关节角度和速度
        position_features = x[:, 4:8]  # 末端和目标位置
        vector_features = x[:, 8:10]   # 目标到末端的向量
        
        # 分别计算注意力权重
        joint_weights = torch.sigmoid(self.joint_attention(joint_features))
        position_weights = torch.sigmoid(self.position_attention(position_features))
        vector_weights = torch.sigmoid(self.vector_attention(vector_features))
        
        # 应用注意力权重
        weighted_joint = joint_features * joint_weights
        weighted_position = position_features * position_weights
        weighted_vector = vector_features * vector_weights
        
        # 重新组合
        weighted_obs = torch.cat([weighted_joint, weighted_position, weighted_vector], dim=1)
        
        # 融合处理
        output = self.fusion(weighted_obs)
        
        return output

class SimplifiedAttentionExtractor(BaseFeaturesExtractor):
    """
    简化的注意力特征提取器
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, attention_type: str = "simple"):
        super(SimplifiedAttentionExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        self.attention_type = attention_type
        
        print(f"🔍 SimplifiedAttentionExtractor 初始化:")
        print(f"   观察空间维度: {obs_dim}")
        print(f"   输出特征维度: {features_dim}")
        print(f"   注意力类型: {attention_type}")
        
        if attention_type == "simple":
            # 简单注意力机制
            self.attention = SimpleAttentionLayer(obs_dim, features_dim)
        elif attention_type == "reacher_specific":
            # Reacher 专用注意力机制
            self.attention = ReacherSpecificAttention(obs_dim, features_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # 输出层 (如果需要调整维度)
        if attention_type == "simple":
            self.output_layer = nn.Identity()  # SimpleAttentionLayer 已经输出正确维度
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(features_dim, features_dim),
                nn.ReLU()
            )
        
        print(f"✅ SimplifiedAttentionExtractor 构建完成")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 注意力处理
        features = self.attention(observations)
        
        # 输出处理
        output = self.output_layer(features)
        
        return output

def sac_with_simplified_attention_training(attention_type: str = "reacher_specific"):
    print("🚀 SAC + 简化注意力机制训练")
    print(f"🧠 注意力类型: {attention_type}")
    print("⚡ 轻量级设计，减少计算开销，提高稳定性")
    print("=" * 70)
    
    # 创建原生 MuJoCo Reacher 环境
    print("🏭 创建 MuJoCo Reacher-v5 环境...")
    env = gym.make('Reacher-v5', render_mode='human')
    env = Monitor(env)
    
    print(f"✅ 环境创建完成")
    print(f"🎮 动作空间: {env.action_space}")
    print(f"👁️ 观察空间: {env.observation_space}")
    print(f"📏 观察维度: {env.observation_space.shape}")
    
    # 创建评估环境
    eval_env = gym.make('Reacher-v5')
    eval_env = Monitor(eval_env)
    
    print("=" * 70)
    
    # 创建带简化注意力的 SAC 模型
    print("🤖 创建 SAC + 简化注意力模型...")
    
    # 定义策略参数
    policy_kwargs = {
        "features_extractor_class": SimplifiedAttentionExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "attention_type": attention_type
        },
        "net_arch": [256, 256],  # 保持与之前相同的网络架构
        "activation_fn": torch.nn.ReLU,
    }
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,          # 与 baseline 相同
        buffer_size=1000000,         # 与 baseline 相同
        learning_starts=100,         # 与 baseline 相同
        batch_size=256,              # 与 baseline 相同
        tau=0.005,                   # 与 baseline 相同
        gamma=0.99,                  # 与 baseline 相同
        train_freq=1,                # 与 baseline 相同
        gradient_steps=1,            # 与 baseline 相同
        ent_coef='auto',             # 与 baseline 相同
        target_update_interval=1,    # 与 baseline 相同
        use_sde=False,               # 与 baseline 相同
        policy_kwargs=policy_kwargs, # 简化注意力机制
        verbose=1,
        device='cpu'
    )
    
    print("✅ SAC + 简化注意力模型创建完成")
    print(f"📊 模型参数:")
    print(f"   策略: MlpPolicy + SimplifiedAttentionExtractor")
    print(f"   注意力类型: {attention_type}")
    print(f"   特征维度: 128")
    print(f"   网络架构: [256, 256]")
    print(f"   参数量: 大幅减少")
    print(f"   计算开销: 显著降低")
    
    print("=" * 70)
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./sac_simple_attention_{attention_type}_best/',
        log_path=f'./sac_simple_attention_{attention_type}_logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("🎯 开始训练...")
    print("📊 训练配置:")
    print("   总步数: 50,000")
    print("   评估频率: 每 5,000 步")
    print("   预期: 更快收敛，更稳定性能")
    print("=" * 70)
    
    start_time = time.time()
    
    # 训练模型
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏆 训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model.save(f"sac_simple_attention_{attention_type}_final")
    print(f"💾 模型已保存为: sac_simple_attention_{attention_type}_final.zip")
    
    # 最终评估
    print("\n🔍 最终评估 (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    print(f"📊 最终评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 与之前结果对比
    baseline_reward = -4.86
    complex_attention_reward = -4.45
    
    improvement_vs_baseline = mean_reward - baseline_reward
    improvement_vs_complex = mean_reward - complex_attention_reward
    
    print(f"\n📈 性能对比:")
    print(f"   Baseline SAC: {baseline_reward:.2f}")
    print(f"   复杂注意力: {complex_attention_reward:.2f}")
    print(f"   简化注意力: {mean_reward:.2f}")
    print(f"   vs Baseline: {improvement_vs_baseline:+.2f}")
    print(f"   vs 复杂注意力: {improvement_vs_complex:+.2f}")
    
    if improvement_vs_baseline > 0.3 and improvement_vs_complex > 0:
        print("   🎉 简化注意力效果最好!")
    elif improvement_vs_baseline > 0.1:
        print("   👍 简化注意力有效改进!")
    elif improvement_vs_baseline > -0.1:
        print("   ⚖️ 简化注意力效果相当")
    else:
        print("   ⚠️ 简化注意力仍需优化")
    
    # 演示训练好的模型
    print("\n🎮 演示训练好的模型 (10 episodes)...")
    demo_env = gym.make('Reacher-v5', render_mode='human')
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(10):
        obs, info = demo_env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 判断成功
        if episode_reward > -5:
            success_count += 1
            print(f"🎯 Episode {episode+1}: 成功! 奖励={episode_reward:.2f}, 长度={episode_length}")
        else:
            print(f"📊 Episode {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")
    
    demo_env.close()
    
    # 演示统计
    demo_success_rate = success_count / 10
    demo_avg_reward = np.mean(episode_rewards)
    
    print("\n" + "=" * 70)
    print("📊 演示统计:")
    print(f"   成功率: {demo_success_rate:.1%} ({success_count}/10)")
    print(f"   平均奖励: {demo_avg_reward:.2f}")
    print(f"   平均长度: {np.mean(episode_lengths):.1f}")
    print(f"   奖励标准差: {np.std(episode_rewards):.2f}")
    
    # 与之前演示对比
    baseline_demo_success = 0.9
    complex_demo_success = 0.4
    baseline_demo_reward = -4.82
    complex_demo_reward = -5.18
    
    print(f"\n📈 演示效果对比:")
    print(f"   Baseline 成功率: {baseline_demo_success:.1%}")
    print(f"   复杂注意力成功率: {complex_demo_success:.1%}")
    print(f"   简化注意力成功率: {demo_success_rate:.1%}")
    print(f"   ")
    print(f"   Baseline 平均奖励: {baseline_demo_reward:.2f}")
    print(f"   复杂注意力平均奖励: {complex_demo_reward:.2f}")
    print(f"   简化注意力平均奖励: {demo_avg_reward:.2f}")
    
    # 训练时间对比
    baseline_time = 14.3
    complex_time = 19.3
    time_vs_baseline = training_time/60 - baseline_time
    time_vs_complex = training_time/60 - complex_time
    
    print(f"\n⏱️ 训练时间对比:")
    print(f"   Baseline: {baseline_time:.1f} 分钟")
    print(f"   复杂注意力: {complex_time:.1f} 分钟")
    print(f"   简化注意力: {training_time/60:.1f} 分钟")
    print(f"   vs Baseline: {time_vs_baseline:+.1f} 分钟")
    print(f"   vs 复杂注意力: {time_vs_complex:+.1f} 分钟")
    
    if abs(time_vs_baseline) < 2:
        print("   ✅ 训练时间与 Baseline 相当，开销可接受")
    elif time_vs_complex < -2:
        print("   🚀 训练时间显著减少，简化效果明显")
    
    print("\n✅ SAC + 简化注意力训练完成!")
    
    # 清理
    env.close()
    eval_env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'demo_success_rate': demo_success_rate,
        'demo_avg_reward': demo_avg_reward,
        'improvement_vs_baseline': improvement_vs_baseline,
        'improvement_vs_complex': improvement_vs_complex,
        'time_vs_baseline': time_vs_baseline,
        'time_vs_complex': time_vs_complex
    }

if __name__ == "__main__":
    print("🔥 开始 SAC + 简化注意力机制训练")
    print("⚡ 轻量级设计，专门针对 Reacher 任务优化")
    print("🎯 目标: 保持性能提升的同时提高稳定性和效率")
    print()
    
    # 可以选择不同的注意力类型
    attention_types = ["reacher_specific", "simple"]
    
    for attention_type in attention_types[:1]:  # 先测试 reacher_specific
        print(f"\n{'='*50}")
        print(f"🧠 测试注意力类型: {attention_type}")
        print(f"{'='*50}")
        
        try:
            results = sac_with_simplified_attention_training(attention_type)
            
            print(f"\n🎊 {attention_type} 注意力训练结果总结:")
            print(f"   最终评估奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   训练时间: {results['training_time']/60:.1f} 分钟")
            print(f"   演示成功率: {results['demo_success_rate']:.1%}")
            print(f"   演示平均奖励: {results['demo_avg_reward']:.2f}")
            print(f"   vs Baseline 改进: {results['improvement_vs_baseline']:+.2f}")
            print(f"   vs 复杂注意力改进: {results['improvement_vs_complex']:+.2f}")
            print(f"   训练时间 vs Baseline: {results['time_vs_baseline']:+.1f} 分钟")
            print(f"   训练时间 vs 复杂注意力: {results['time_vs_complex']:+.1f} 分钟")
            
            # 总体评估
            if (results['improvement_vs_baseline'] > 0.2 and 
                results['demo_success_rate'] > 0.7 and 
                results['time_vs_complex'] < 0):
                print(f"\n🏆 {attention_type} 注意力机制表现优秀!")
                print("   性能提升 + 高成功率 + 训练效率提高")
            elif results['improvement_vs_baseline'] > 0.1:
                print(f"\n👍 {attention_type} 注意力机制有效!")
            else:
                print(f"\n⚠️ {attention_type} 注意力机制需要进一步优化")
            
        except Exception as e:
            print(f"❌ {attention_type} 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
