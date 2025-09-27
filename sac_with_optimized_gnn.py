#!/usr/bin/env python3
"""
优化版 SAC + GNN + 注意力机制
解决性能下降问题
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
from typing import Dict, List, Tuple, Type, Union, Optional

class SimpleGraphProcessor(nn.Module):
    """
    简化的图处理器 - 专门为 2 关节 Reacher 优化
    避免过度复杂化
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        super(SimpleGraphProcessor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简单的关节间交互层
        self.joint_interaction = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 两个关节的特征拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2),  # 输出增强的关节特征
        )
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        print(f"🔗 SimpleGraphProcessor 初始化: {input_dim}→{hidden_dim}→{input_dim*2}")
    
    def forward(self, joint1_features: torch.Tensor, joint2_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理两个关节之间的交互
        joint1_features: [batch_size, input_dim]
        joint2_features: [batch_size, input_dim]
        return: (enhanced_joint1, enhanced_joint2)
        """
        # 拼接两个关节的特征
        combined = torch.cat([joint1_features, joint2_features], dim=1)
        
        # 通过交互层
        enhanced_combined = self.joint_interaction(combined)
        
        # 分离增强后的特征
        enhanced_joint1 = enhanced_combined[:, :self.input_dim]
        enhanced_joint2 = enhanced_combined[:, self.input_dim:]
        
        # 残差连接
        output_joint1 = self.residual_weight * enhanced_joint1 + (1 - self.residual_weight) * joint1_features
        output_joint2 = self.residual_weight * enhanced_joint2 + (1 - self.residual_weight) * joint2_features
        
        return output_joint1, output_joint2

class LightweightAttention(nn.Module):
    """
    轻量级注意力机制 - 减少参数数量
    """
    def __init__(self, feature_dim: int = 32):
        super(LightweightAttention, self).__init__()
        self.feature_dim = feature_dim
        
        # 简化的注意力计算
        self.attention = nn.Linear(feature_dim, 1, bias=False)
        
        print(f"🎯 LightweightAttention 初始化: {feature_dim}→1")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算轻量级注意力
        features: [batch_size, feature_dim]
        return: [batch_size, feature_dim]
        """
        # 计算注意力权重
        attention_weight = torch.sigmoid(self.attention(features))  # [batch_size, 1]
        
        # 应用注意力
        attended_features = features * attention_weight
        
        return attended_features

class OptimizedGNNExtractor(BaseFeaturesExtractor):
    """
    优化版 GNN 特征提取器
    解决性能下降问题
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(OptimizedGNNExtractor, self).__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        
        print(f"🔍 OptimizedGNNExtractor 初始化:")
        print(f"   观察空间维度: {self.obs_dim}")
        print(f"   输出特征维度: {features_dim}")
        print(f"   优化目标: 减少复杂度，提升性能")
        
        # 关节特征处理 (简化版)
        joint_feature_dim = 16  # 减少特征维度
        
        self.joint1_encoder = nn.Sequential(
            nn.Linear(2, joint_feature_dim),  # angle + velocity
            nn.ReLU(),
            nn.LayerNorm(joint_feature_dim)
        )
        
        self.joint2_encoder = nn.Sequential(
            nn.Linear(2, joint_feature_dim),  # angle + velocity
            nn.ReLU(),
            nn.LayerNorm(joint_feature_dim)
        )
        
        # 简化的图处理
        self.graph_processor = SimpleGraphProcessor(
            input_dim=joint_feature_dim,
            hidden_dim=32
        )
        
        # 轻量级注意力
        self.joint1_attention = LightweightAttention(joint_feature_dim)
        self.joint2_attention = LightweightAttention(joint_feature_dim)
        
        # 全局特征处理 (位置信息)
        # MuJoCo Reacher-v5: [4:6] end effector, [6:8] target, [8:10] vector
        global_feature_dim = 6
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # 特征融合 (简化版)
        fusion_input_dim = joint_feature_dim * 2 + 32  # 两个关节 + 全局特征
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 减少 dropout
            nn.Linear(features_dim, features_dim)
        )
        
        print(f"✅ OptimizedGNNExtractor 构建完成")
        print(f"   关节特征维度: 2→{joint_feature_dim}")
        print(f"   全局特征维度: {global_feature_dim}→32")
        print(f"   融合输入维度: {fusion_input_dim}")
        print(f"   总参数数量显著减少")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        优化的前向传播
        observations: [batch_size, obs_dim]
        return: [batch_size, features_dim]
        """
        batch_size = observations.size(0)
        
        # 1. 提取关节特征 (MuJoCo Reacher-v5 格式)
        # [0:2] - cos/sin of joint angles
        # [2:4] - joint velocities
        joint1_raw = torch.cat([observations[:, 0:1], observations[:, 2:3]], dim=1)  # [angle, velocity]
        joint2_raw = torch.cat([observations[:, 1:2], observations[:, 3:4]], dim=1)  # [angle, velocity]
        
        # 2. 编码关节特征
        joint1_encoded = self.joint1_encoder(joint1_raw)  # [batch_size, joint_feature_dim]
        joint2_encoded = self.joint2_encoder(joint2_raw)  # [batch_size, joint_feature_dim]
        
        # 3. 图处理 (关节间交互)
        joint1_enhanced, joint2_enhanced = self.graph_processor(joint1_encoded, joint2_encoded)
        
        # 4. 轻量级注意力
        joint1_attended = self.joint1_attention(joint1_enhanced)
        joint2_attended = self.joint2_attention(joint2_enhanced)
        
        # 5. 处理全局特征
        global_features = observations[:, 4:]  # [end_effector, target, vector]
        global_encoded = self.global_encoder(global_features)  # [batch_size, 32]
        
        # 6. 特征融合
        fused_features = torch.cat([joint1_attended, joint2_attended, global_encoded], dim=1)
        output = self.fusion_net(fused_features)
        
        return output

def optimized_sac_training():
    print("🚀 优化版 SAC + GNN + 注意力机制训练")
    print("🎯 目标: 解决性能下降问题")
    print("🔧 优化策略:")
    print("   - 简化 GNN 架构")
    print("   - 减少参数数量")
    print("   - 轻量级注意力机制")
    print("   - 针对 2 关节 Reacher 优化")
    print("=" * 70)
    
    # 创建环境
    print("🏭 创建 MuJoCo Reacher-v5 环境...")
    env = gym.make('Reacher-v5', render_mode='human')
    env = Monitor(env)
    
    eval_env = gym.make('Reacher-v5')
    eval_env = Monitor(eval_env)
    
    print(f"✅ 环境创建完成")
    print(f"🎮 动作空间: {env.action_space}")
    print(f"👁️ 观察空间: {env.observation_space}")
    
    print("=" * 70)
    
    # 创建优化版模型
    print("🤖 创建优化版 SAC 模型...")
    
    policy_kwargs = {
        "features_extractor_class": OptimizedGNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
        },
        "net_arch": [128, 128],  # 减少网络大小
        "activation_fn": torch.nn.ReLU,
    }
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,         # 减少缓冲区大小
        learning_starts=100,        # 更早开始学习
        batch_size=128,             # 减少批次大小
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        use_sde=False,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device='cpu'
    )
    
    print("✅ 优化版 SAC 模型创建完成")
    print(f"📊 优化配置:")
    print(f"   特征维度: 128 (vs 之前的复杂架构)")
    print(f"   网络架构: [128, 128] (vs [256, 256])")
    print(f"   批次大小: 128 (vs 256)")
    print(f"   缓冲区: 100K (vs 1M)")
    print(f"   参数数量显著减少")
    
    print("=" * 70)
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./sac_optimized_gnn_best/',
        log_path='./sac_optimized_gnn_logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("🎯 开始优化训练...")
    print("📊 训练配置:")
    print("   总步数: 30,000 (减少训练步数)")
    print("   评估频率: 每 5,000 步")
    print("   预期: 更快收敛，更好性能")
    print("=" * 70)
    
    start_time = time.time()
    
    # 训练模型 (减少训练步数)
    model.learn(
        total_timesteps=30000,  # 减少到 30K
        callback=eval_callback,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏆 优化训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model.save("sac_optimized_gnn_final")
    print(f"💾 模型已保存为: sac_optimized_gnn_final.zip")
    
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
    simple_attention_reward = -4.69
    complex_gnn_reward = -5.56
    
    improvement_vs_baseline = mean_reward - baseline_reward
    improvement_vs_simple = mean_reward - simple_attention_reward
    improvement_vs_complex = mean_reward - complex_gnn_reward
    
    print(f"\n📈 性能对比:")
    print(f"   Baseline SAC: {baseline_reward:.2f}")
    print(f"   简化注意力: {simple_attention_reward:.2f}")
    print(f"   复杂 GNN: {complex_gnn_reward:.2f}")
    print(f"   优化 GNN: {mean_reward:.2f}")
    print(f"   vs Baseline: {improvement_vs_baseline:+.2f}")
    print(f"   vs 简化注意力: {improvement_vs_simple:+.2f}")
    print(f"   vs 复杂 GNN: {improvement_vs_complex:+.2f}")
    
    if improvement_vs_baseline > -0.2:
        print("   🎉 优化 GNN 成功接近 Baseline 性能!")
    elif improvement_vs_complex > 0.5:
        print("   👍 优化 GNN 显著优于复杂版本!")
    else:
        print("   ⚠️ 仍需进一步优化")
    
    # 演示训练好的模型
    print("\n🎮 演示优化后的模型 (10 episodes)...")
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
    simple_demo_success = 0.7
    complex_demo_success = 0.4
    
    print(f"\n📈 演示效果对比:")
    print(f"   Baseline 成功率: {baseline_demo_success:.1%}")
    print(f"   简化注意力成功率: {simple_demo_success:.1%}")
    print(f"   复杂 GNN 成功率: {complex_demo_success:.1%}")
    print(f"   优化 GNN 成功率: {demo_success_rate:.1%}")
    
    if demo_success_rate >= 0.8:
        print("   🏆 优化 GNN 成功恢复高性能!")
    elif demo_success_rate >= 0.6:
        print("   👍 优化 GNN 显著改善!")
    elif demo_success_rate > complex_demo_success:
        print("   📈 优化 GNN 有所改善")
    else:
        print("   ⚠️ 仍需进一步优化")
    
    # 训练时间对比
    baseline_time = 14.3
    simple_time = 16.4
    complex_time = 35.7
    time_vs_baseline = training_time/60 - baseline_time
    time_vs_complex = training_time/60 - complex_time
    
    print(f"\n⏱️ 训练时间对比:")
    print(f"   Baseline: {baseline_time:.1f} 分钟")
    print(f"   简化注意力: {simple_time:.1f} 分钟")
    print(f"   复杂 GNN: {complex_time:.1f} 分钟")
    print(f"   优化 GNN: {training_time/60:.1f} 分钟")
    print(f"   vs Baseline: {time_vs_baseline:+.1f} 分钟")
    print(f"   vs 复杂 GNN: {time_vs_complex:+.1f} 分钟")
    
    if abs(time_vs_baseline) < 5:
        print("   ✅ 训练时间接近 Baseline，优化成功!")
    elif time_vs_complex < -10:
        print("   🚀 训练时间显著减少!")
    
    print("\n✅ 优化版 SAC + GNN 训练完成!")
    
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
    print("🔥 开始优化版 SAC + GNN 训练")
    print("🎯 解决性能下降问题")
    print("🔧 关键优化:")
    print("   1. 简化 GNN 架构")
    print("   2. 减少参数数量") 
    print("   3. 轻量级注意力")
    print("   4. 针对任务优化")
    print()
    
    try:
        results = optimized_sac_training()
        
        print(f"\n🎊 优化版训练结果总结:")
        print(f"   最终评估奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   训练时间: {results['training_time']/60:.1f} 分钟")
        print(f"   演示成功率: {results['demo_success_rate']:.1%}")
        print(f"   演示平均奖励: {results['demo_avg_reward']:.2f}")
        print(f"   vs Baseline 改进: {results['improvement_vs_baseline']:+.2f}")
        print(f"   vs 复杂 GNN 改进: {results['improvement_vs_complex']:+.2f}")
        print(f"   训练时间 vs Baseline: {results['time_vs_baseline']:+.1f} 分钟")
        print(f"   训练时间 vs 复杂 GNN: {results['time_vs_complex']:+.1f} 分钟")
        
        # 总体评估
        if (results['improvement_vs_baseline'] > -0.3 and 
            results['demo_success_rate'] > 0.7 and 
            results['time_vs_baseline'] < 10):
            print(f"\n🏆 优化成功!")
            print("   性能接近 Baseline + 合理训练时间 + 高成功率")
        elif results['improvement_vs_complex'] > 0.5:
            print(f"\n👍 显著改善!")
            print("   相比复杂 GNN 有明显提升")
        else:
            print(f"\n📈 有所改善，但仍需进一步优化")
        
        print(f"\n🔗 优化版模型已准备好扩展到多关节任务!")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
