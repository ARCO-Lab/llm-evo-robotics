#!/usr/bin/env python3
"""
在 Baseline SAC 基础上添加 Attention Layer
逐步集成自定义架构
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

class AttentionLayer(nn.Module):
    """
    自注意力层 - 用于处理观察特征
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 线性变换层
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        print(f"🧠 AttentionLayer 初始化: input_dim={input_dim}, hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]
        """
        batch_size = x.size(0)
        
        # 如果输入是2D，扩展为3D (添加序列维度)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        seq_len = x.size(1)
        
        # 计算 Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在形状为: [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, seq_len, head_dim]
        
        # 重新组合多头
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # 输出投影
        output = self.output_proj(attended)
        
        # 残差连接 + Layer Norm (需要先投影输入到相同维度)
        if x.size(-1) != self.hidden_dim:
            # 如果输入维度不匹配，使用一个投影层
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x.size(-1), self.hidden_dim).to(x.device)
            x_proj = self.input_proj(x)
        else:
            x_proj = x
        
        output = self.layer_norm(output + x_proj)
        
        # 如果原始输入是2D，压缩回2D
        if squeeze_output:
            output = output.squeeze(1)
        
        return output

class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    带有注意力机制的特征提取器
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(AttentionFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # 获取观察空间维度
        obs_dim = observation_space.shape[0]
        
        print(f"🔍 AttentionFeaturesExtractor 初始化:")
        print(f"   观察空间维度: {obs_dim}")
        print(f"   输出特征维度: {features_dim}")
        
        # 输入预处理层
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # 注意力层
        self.attention = AttentionLayer(
            input_dim=64,
            hidden_dim=features_dim,
            num_heads=4
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"✅ AttentionFeaturesExtractor 构建完成")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        observations: [batch_size, obs_dim]
        return: [batch_size, features_dim]
        """
        # 输入预处理
        x = self.input_layer(observations)
        
        # 注意力处理
        x = self.attention(x)
        
        # 输出处理
        features = self.output_layer(x)
        
        return features

def sac_with_attention_training():
    print("🚀 SAC + Attention Layer 训练")
    print("🧠 在成功的 Baseline SAC 基础上添加注意力机制")
    print("📊 对比训练效果和性能变化")
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
    
    # 创建带注意力的 SAC 模型
    print("🤖 创建 SAC + Attention 模型...")
    
    # 定义策略参数，使用自定义特征提取器
    policy_kwargs = {
        "features_extractor_class": AttentionFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [256, 256],  # Actor 和 Critic 网络架构
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
        policy_kwargs=policy_kwargs, # 添加注意力机制
        verbose=1,
        device='cpu'
    )
    
    print("✅ SAC + Attention 模型创建完成")
    print(f"📊 模型参数:")
    print(f"   策略: MlpPolicy + AttentionFeaturesExtractor")
    print(f"   特征维度: 128")
    print(f"   注意力头数: 4")
    print(f"   网络架构: [256, 256]")
    print(f"   其他参数与 baseline 相同")
    
    print("=" * 70)
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./sac_attention_best/',
        log_path='./sac_attention_logs/',
        eval_freq=5000,              # 每5000步评估一次
        n_eval_episodes=10,          # 每次评估10个episodes
        deterministic=True,          # 评估时使用确定性策略
        render=False                 # 评估时不渲染
    )
    
    # 开始训练
    print("🎯 开始训练...")
    print("📊 训练配置:")
    print("   总步数: 50,000 (与 baseline 相同)")
    print("   评估频率: 每 5,000 步")
    print("   日志间隔: 每 1,000 步")
    print("=" * 70)
    
    start_time = time.time()
    
    # 训练模型
    model.learn(
        total_timesteps=50000,       # 与 baseline 相同的训练步数
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
    model.save("sac_attention_final")
    print("💾 模型已保存为: sac_attention_final.zip")
    
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
    
    # 与 baseline 对比
    baseline_reward = -4.86  # baseline SAC 的结果
    improvement = mean_reward - baseline_reward
    
    print(f"\n📈 与 Baseline SAC 对比:")
    print(f"   Baseline SAC: {baseline_reward:.2f}")
    print(f"   SAC + Attention: {mean_reward:.2f}")
    print(f"   改进幅度: {improvement:+.2f}")
    
    if improvement > 0.5:
        print("   🎉 显著改进! 注意力机制效果很好")
    elif improvement > 0.1:
        print("   👍 有改进! 注意力机制有积极作用")
    elif improvement > -0.1:
        print("   ⚖️ 效果相当，注意力机制没有负面影响")
    else:
        print("   ⚠️ 性能下降，可能需要调整注意力机制参数")
    
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
        
        # 判断成功 (与 baseline 相同的标准)
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
    
    # 与 baseline 演示对比
    baseline_demo_success = 0.9  # baseline 的演示成功率
    baseline_demo_reward = -4.82  # baseline 的演示平均奖励
    
    print(f"\n📈 演示效果对比:")
    print(f"   Baseline 成功率: {baseline_demo_success:.1%}")
    print(f"   SAC + Attention 成功率: {demo_success_rate:.1%}")
    print(f"   成功率变化: {demo_success_rate - baseline_demo_success:+.1%}")
    print(f"   ")
    print(f"   Baseline 平均奖励: {baseline_demo_reward:.2f}")
    print(f"   SAC + Attention 平均奖励: {demo_avg_reward:.2f}")
    print(f"   奖励变化: {demo_avg_reward - baseline_demo_reward:+.2f}")
    
    # 训练时间对比
    baseline_time = 14.3  # baseline 的训练时间(分钟)
    time_change = training_time/60 - baseline_time
    
    print(f"\n⏱️ 训练时间对比:")
    print(f"   Baseline 训练时间: {baseline_time:.1f} 分钟")
    print(f"   SAC + Attention 训练时间: {training_time/60:.1f} 分钟")
    print(f"   时间变化: {time_change:+.1f} 分钟")
    
    if abs(time_change) < 2:
        print("   ✅ 训练时间基本相同，注意力机制开销可接受")
    elif time_change > 0:
        print("   ⚠️ 训练时间增加，注意力机制有一定计算开销")
    else:
        print("   🚀 训练时间减少，可能是随机因素")
    
    print("\n✅ SAC + Attention 训练完成!")
    
    # 清理
    env.close()
    eval_env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'demo_success_rate': demo_success_rate,
        'demo_avg_reward': demo_avg_reward,
        'improvement_vs_baseline': improvement,
        'time_vs_baseline': time_change
    }

if __name__ == "__main__":
    print("🔥 开始 SAC + Attention Layer 训练")
    print("🧠 在成功的 Baseline SAC 基础上添加注意力机制")
    print("📈 观察注意力机制对训练效果的影响")
    print()
    
    try:
        results = sac_with_attention_training()
        
        print(f"\n🎊 SAC + Attention 训练结果总结:")
        print(f"   最终评估奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   训练时间: {results['training_time']/60:.1f} 分钟")
        print(f"   演示成功率: {results['demo_success_rate']:.1%}")
        print(f"   演示平均奖励: {results['demo_avg_reward']:.2f}")
        print(f"   相比 Baseline 改进: {results['improvement_vs_baseline']:+.2f}")
        print(f"   训练时间变化: {results['time_vs_baseline']:+.1f} 分钟")
        
        # 总体评估
        if results['improvement_vs_baseline'] > 0.1:
            print(f"\n🎉 注意力机制集成成功! 性能有明显提升")
        elif results['improvement_vs_baseline'] > -0.1:
            print(f"\n👍 注意力机制集成良好! 性能保持稳定")
        else:
            print(f"\n⚠️ 注意力机制可能需要进一步调优")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("💡 请检查注意力机制实现和参数设置")
        import traceback
        traceback.print_exc()
