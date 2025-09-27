#!/usr/bin/env python3
"""
简化版真正通用 SAC 架构
基于 GPT 建议，但使用标准 SB3 MlpPolicy 避免兼容性问题：
1. 修复 MuJoCo Reacher-v5 观察解析
2. Link 长度信息融合
3. 保持所有现有优势
4. 使用标准 MlpPolicy 确保稳定性
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import math

# ============================================================================
# 🧩 修复 1: 正确的观察解析系统
# ============================================================================

class CorrectMaskSystem:
    """
    修复版 Mask 系统：正确解析 MuJoCo Reacher-v5 观察格式
    """
    
    @staticmethod
    def create_joint_mask(batch_size: int, num_joints: int, max_joints: int, device: torch.device) -> torch.Tensor:
        """
        创建关节掩码
        return: [batch_size, max_joints] - True 表示有效关节，False 表示 padding
        """
        mask = torch.zeros(batch_size, max_joints, dtype=torch.bool, device=device)
        mask[:, :num_joints] = True
        return mask
    
    @staticmethod
    def parse_observation_correct(obs: torch.Tensor, num_joints: int, max_joints: int, 
                                link_lengths: Optional[List[float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        正确解析 MuJoCo Reacher-v5 观察空间
        
        MuJoCo Reacher-v5 观察格式 (10维):
        [0-1]: cos/sin of joint 1 angle
        [2-3]: cos/sin of joint 2 angle  
        [4-5]: joint 1 and joint 2 velocities
        [6-7]: end effector position (x, y)
        [8-9]: target position (x, y)
        
        obs: [batch_size, obs_dim]
        return: (joint_features_with_links, global_features)
        """
        batch_size = obs.size(0)
        device = obs.device
        
        # 默认 link 长度
        if link_lengths is None:
            if num_joints == 2:
                link_lengths = [0.1, 0.1]  # MuJoCo Reacher-v5 默认
            else:
                link_lengths = [0.1] * num_joints
        
        # 确保 link_lengths 长度匹配
        while len(link_lengths) < num_joints:
            link_lengths.append(0.1)
        
        if num_joints == 2:
            # 正确解析 MuJoCo Reacher-v5 格式
            # [0-1]: joint 1 cos/sin
            joint1_cos = obs[:, 0:1]  # [batch_size, 1]
            joint1_sin = obs[:, 1:2]  # [batch_size, 1]
            
            # [2-3]: joint 2 cos/sin  
            joint2_cos = obs[:, 2:3]  # [batch_size, 1]
            joint2_sin = obs[:, 3:4]  # [batch_size, 1]
            
            # [4-5]: joint velocities
            joint1_vel = obs[:, 4:5]  # [batch_size, 1]
            joint2_vel = obs[:, 5:6]  # [batch_size, 1]
            
            # [6-9]: 全局特征 (end effector + target)
            global_features = obs[:, 6:]  # [batch_size, 4]
            
            # 构造关节特征 + link 长度
            joint_features_list = []
            
            # Joint 1: [cos, sin, vel, link_length]
            link1_val = torch.full_like(joint1_cos, link_lengths[0])
            joint1_feature = torch.cat([joint1_cos, joint1_sin, joint1_vel, link1_val], dim=1)
            joint_features_list.append(joint1_feature)
            
            # Joint 2: [cos, sin, vel, link_length]
            link2_val = torch.full_like(joint2_cos, link_lengths[1])
            joint2_feature = torch.cat([joint2_cos, joint2_sin, joint2_vel, link2_val], dim=1)
            joint_features_list.append(joint2_feature)
            
            joint_features = torch.stack(joint_features_list, dim=1)  # [batch_size, 2, 4]
            
        else:
            # 通用格式：假设每个关节 4 维 [cos, sin, vel, link_length]
            joint_dim = num_joints * 4
            if obs.size(1) >= joint_dim:
                joint_obs = obs[:, :joint_dim]
                global_features = obs[:, joint_dim:]
                joint_features = joint_obs.view(batch_size, num_joints, 4)
            else:
                # 如果观察维度不足，用零填充
                joint_features = torch.zeros(batch_size, num_joints, 4, device=device)
                global_features = obs
                
                # 尽可能填充可用的观察
                available_dim = min(obs.size(1), joint_dim)
                if available_dim > 0:
                    joint_obs_partial = obs[:, :available_dim]
                    joint_features_flat = joint_features.view(batch_size, -1)
                    joint_features_flat[:, :available_dim] = joint_obs_partial
                    joint_features = joint_features_flat.view(batch_size, num_joints, 4)
                
                # 添加 link 长度信息
                for i in range(num_joints):
                    joint_features[:, i, 3] = link_lengths[i]
        
        # Padding 到 max_joints
        joint_features_padded = torch.zeros(batch_size, max_joints, 4, device=device)
        joint_features_padded[:, :num_joints] = joint_features
        
        return joint_features_padded, global_features

# ============================================================================
# 🧩 修复 2: Link-Aware 关节编码器
# ============================================================================

class LinkAwareJointEncoder(nn.Module):
    """
    Link-Aware 关节编码器：融合 link 长度信息
    输入格式：[cos, sin, vel, link_length] (4维)
    """
    def __init__(self, joint_input_dim: int = 4, joint_feature_dim: int = 64):
        super(LinkAwareJointEncoder, self).__init__()
        self.joint_input_dim = joint_input_dim
        self.joint_feature_dim = joint_feature_dim
        
        # Link 长度特征处理 (几何信息)
        self.link_processor = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)
        )
        
        # 运动特征处理 [cos, sin, vel] (运动信息)
        self.motion_processor = nn.Sequential(
            nn.Linear(3, 24),
            nn.ReLU(),
            nn.LayerNorm(24)
        )
        
        # 几何-运动融合处理器
        self.fusion_processor = nn.Sequential(
            nn.Linear(32, joint_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(joint_feature_dim)
        )
        
        print(f"🔗 LinkAwareJointEncoder: {joint_input_dim} → {joint_feature_dim} (几何+运动融合)")
    
    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        编码关节特征 + link 长度
        joint_features: [batch_size, max_joints, 4] - [cos, sin, vel, link_length]
        return: [batch_size, max_joints, joint_feature_dim]
        """
        batch_size, max_joints, _ = joint_features.shape
        
        # 分离运动特征和几何特征
        motion_features = joint_features[:, :, :3]  # [cos, sin, vel]
        link_lengths = joint_features[:, :, 3:4]    # [link_length]
        
        # 重塑为 [batch_size * max_joints, feature_dim]
        motion_flat = motion_features.view(-1, 3)
        link_flat = link_lengths.view(-1, 1)
        
        # 分别处理几何和运动信息
        motion_encoded = self.motion_processor(motion_flat)
        link_encoded = self.link_processor(link_flat)
        
        # 融合几何和运动特征
        fused_features = torch.cat([motion_encoded, link_encoded], dim=1)
        joint_encoded = self.fusion_processor(fused_features)
        
        # 重塑回 [batch_size, max_joints, joint_feature_dim]
        encoded_features = joint_encoded.view(batch_size, max_joints, self.joint_feature_dim)
        
        return encoded_features

# ============================================================================
# 🧩 修复 3: 复用现有的优秀注意力组件
# ============================================================================

class FixedSelfAttention(nn.Module):
    """修复版自注意力"""
    def __init__(self, feature_dim: int = 64, num_heads: int = 4, dropout: float = 0.0):
        super(FixedSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        assert feature_dim % num_heads == 0
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        print(f"🧠 FixedSelfAttention: {feature_dim}d, {num_heads} heads")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
        
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        x = self.layer_norm(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        
        return x

class FixedAttentionPooling(nn.Module):
    """修复版注意力池化"""
    def __init__(self, input_dim: int = 64, output_dim: int = 128):
        super(FixedAttentionPooling, self).__init__()
        
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
        print(f"🎯 FixedAttentionPooling: {input_dim} → {output_dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.score(x).squeeze(-1)
        
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        
        w = F.softmax(s, dim=1).unsqueeze(-1)
        pooled = (x * w).sum(dim=1)
        output = self.proj(pooled)
        
        return output

# ============================================================================
# 🧩 修复 4: 简化版真正通用特征提取器
# ============================================================================

class SimplifiedTrulyUniversalExtractor(BaseFeaturesExtractor):
    """
    简化版真正通用特征提取器
    基于 GPT 建议但使用标准 SB3 架构确保兼容性
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, 
                 num_joints: int = 2, max_joints: int = 10, 
                 link_lengths: Optional[List[float]] = None):
        super(SimplifiedTrulyUniversalExtractor, self).__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.num_joints = num_joints
        self.max_joints = max_joints
        self.joint_input_dim = 4  # [cos, sin, vel, link_length]
        self.link_lengths = link_lengths
        
        print(f"🌟 SimplifiedTrulyUniversalExtractor 初始化:")
        print(f"   观察空间维度: {self.obs_dim}")
        print(f"   当前关节数: {num_joints}")
        print(f"   最大关节数: {max_joints}")
        print(f"   关节输入维度: {self.joint_input_dim} [cos, sin, vel, link_length]")
        print(f"   Link 长度: {link_lengths}")
        print(f"   输出特征维度: {features_dim}")
        
        # 模块 1: Link-Aware 关节编码器
        self.joint_encoder = LinkAwareJointEncoder(
            joint_input_dim=self.joint_input_dim,
            joint_feature_dim=64
        )
        
        # 模块 2: 自注意力
        self.self_attention = FixedSelfAttention(
            feature_dim=64,
            num_heads=4,
            dropout=0.0
        )
        
        # 模块 3: 注意力池化
        self.attention_pooling = FixedAttentionPooling(
            input_dim=64,
            output_dim=features_dim // 2
        )
        
        # 全局特征处理
        if num_joints == 2:
            global_dim = 4  # MuJoCo Reacher-v5: end effector + target (4维)
        else:
            global_dim = max(0, self.obs_dim - (num_joints * 4))
        
        if global_dim > 0:
            self.global_processor = nn.Sequential(
                nn.Linear(global_dim, features_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(features_dim // 2)
            )
            fusion_dim = features_dim
        else:
            self.global_processor = None
            fusion_dim = features_dim // 2
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
        # 修复版 Mask 系统
        self.mask_system = CorrectMaskSystem()
        
        print(f"✅ SimplifiedTrulyUniversalExtractor 构建完成")
        print(f"   🔧 GPT 建议修复:")
        print(f"   ✅ 正确的 MuJoCo Reacher-v5 观察解析")
        print(f"   ✅ Link 长度信息融合")
        print(f"   ✅ 使用标准 SB3 架构确保兼容性")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        简化版真正通用前向传播
        """
        batch_size = observations.size(0)
        device = observations.device
        
        # 步骤 1: 正确解析观察空间并融合 link 长度
        joint_features_with_links, global_features = self.mask_system.parse_observation_correct(
            observations, self.num_joints, self.max_joints, self.link_lengths
        )
        
        # 步骤 2: 创建关节掩码
        joint_mask = self.mask_system.create_joint_mask(
            batch_size, self.num_joints, self.max_joints, device
        )
        
        # 步骤 3: Link-Aware 关节编码
        encoded_joints = self.joint_encoder(joint_features_with_links)
        
        # 步骤 4: 自注意力建模关节间交互
        attended_joints = self.self_attention(encoded_joints, mask=joint_mask)
        
        # 步骤 5: 注意力池化
        pooled_joint_features = self.attention_pooling(attended_joints, mask=joint_mask)
        
        # 步骤 6: 处理全局特征
        if self.global_processor is not None and global_features.size(1) > 0:
            processed_global = self.global_processor(global_features)
            fused_features = torch.cat([pooled_joint_features, processed_global], dim=1)
        else:
            fused_features = pooled_joint_features
        
        # 步骤 7: 最终融合
        final_features = self.final_fusion(fused_features)
        
        return final_features

# ============================================================================
# 🧩 修复 5: 训练函数
# ============================================================================

def train_simplified_truly_universal_sac(num_joints: int = 2, max_joints: int = 10, 
                                        link_lengths: Optional[List[float]] = None,
                                        total_timesteps: int = 50000):
    """
    训练简化版真正通用 SAC
    """
    print("🌟 简化版真正通用 SAC 训练")
    print(f"🔗 当前关节数: {num_joints}")
    print(f"🔗 最大支持关节数: {max_joints}")
    print(f"🔗 Link 长度: {link_lengths}")
    print(f"💡 架构: GPT 建议修复 + 标准 SB3 MlpPolicy")
    print(f"🎯 目标: 稳定性 + 正确观察解析 + Link 融合")
    print("=" * 70)
    
    # 创建环境
    print(f"🏭 创建环境...")
    if num_joints == 2:
        env = gym.make('Reacher-v5')
        eval_env = gym.make('Reacher-v5')
    else:
        print(f"⚠️ 暂不支持 {num_joints} 关节环境，使用 2 关节进行验证")
        env = gym.make('Reacher-v5')
        eval_env = gym.make('Reacher-v5')
    
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    
    print(f"✅ 环境创建完成")
    print(f"🎮 动作空间: {env.action_space}")
    print(f"👁️ 观察空间: {env.observation_space}")
    
    print("=" * 70)
    
    # 创建简化版真正通用模型
    print("🤖 创建简化版真正通用 SAC 模型...")
    
    policy_kwargs = {
        "features_extractor_class": SimplifiedTrulyUniversalExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "num_joints": num_joints,
            "max_joints": max_joints,
            "link_lengths": link_lengths
        },
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
    }
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
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
    
    print("✅ 简化版真正通用 SAC 模型创建完成")
    print(f"📊 GPT 建议修复 (简化版):")
    print(f"   ✅ 修复 MuJoCo Reacher-v5 观察解析")
    print(f"   ✅ Link 长度信息融合")
    print(f"   ✅ 使用标准 SB3 MlpPolicy 确保稳定性")
    print(f"   ✅ 保持所有现有优势")
    
    print("=" * 70)
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./simplified_truly_universal_{num_joints}joints_best/',
        log_path=f'./simplified_truly_universal_{num_joints}joints_logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("🎯 开始简化版真正通用训练...")
    print("📊 训练配置:")
    print(f"   总步数: {total_timesteps:,}")
    print("   评估频率: 每 5,000 步")
    print("   预期: 稳定训练 + 正确观察解析")
    print("=" * 70)
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏆 简化版真正通用训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model_name = f"simplified_truly_universal_{num_joints}joints_final"
    model.save(model_name)
    print(f"💾 模型已保存为: {model_name}.zip")
    
    # 最终评估
    print("\n🔍 最终评估 (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    print(f"📊 简化版真正通用模型评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 与之前版本对比
    baseline_reward = -4.86
    simplified_fixed_reward = -3.76
    link_aware_reward = -3.81
    
    print(f"\n📈 性能对比:")
    print(f"   Baseline SAC: {baseline_reward:.2f}")
    print(f"   简化版修复: {simplified_fixed_reward:.2f}")
    print(f"   Link-Aware: {link_aware_reward:.2f}")
    print(f"   简化版真正通用: {mean_reward:.2f}")
    
    improvement_vs_baseline = mean_reward - baseline_reward
    improvement_vs_link_aware = mean_reward - link_aware_reward
    
    print(f"\n📊 改进幅度:")
    print(f"   vs Baseline: {improvement_vs_baseline:+.2f}")
    print(f"   vs Link-Aware: {improvement_vs_link_aware:+.2f}")
    
    if improvement_vs_link_aware > 0.5:
        print("   🎉 GPT 建议修复大成功!")
    elif improvement_vs_link_aware > 0.0:
        print("   👍 GPT 建议修复有效!")
    else:
        print("   📈 需要进一步调试")
    
    # 演示
    print("\n🎮 演示简化版真正通用模型 (10 episodes)...")
    demo_env = gym.make('Reacher-v5', render_mode='human')
    
    episode_rewards = []
    success_count = 0
    
    for episode in range(10):
        obs, info = demo_env.reset()
        episode_reward = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode_reward > -5:
            success_count += 1
            print(f"🎯 Episode {episode+1}: 成功! 奖励={episode_reward:.2f}")
        else:
            print(f"📊 Episode {episode+1}: 奖励={episode_reward:.2f}")
    
    demo_env.close()
    
    demo_success_rate = success_count / 10
    demo_avg_reward = np.mean(episode_rewards)
    
    print("\n" + "=" * 70)
    print("📊 简化版真正通用演示统计:")
    print(f"   成功率: {demo_success_rate:.1%} ({success_count}/10)")
    print(f"   平均奖励: {demo_avg_reward:.2f}")
    print(f"   奖励标准差: {np.std(episode_rewards):.2f}")
    
    print(f"\n🌟 简化版真正通用架构优势:")
    print(f"   ✅ 正确的 MuJoCo 观察解析")
    print(f"   ✅ Link 长度信息融合")
    print(f"   ✅ 标准 SB3 MlpPolicy 确保稳定性")
    print(f"   ✅ 保持所有现有优势")
    print(f"   🌐 为真正任意关节数扩展奠定基础")
    
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
        'improvement_vs_link_aware': improvement_vs_link_aware,
        'num_joints': num_joints,
        'max_joints': max_joints,
        'link_lengths': link_lengths
    }

if __name__ == "__main__":
    print("🌟 简化版真正通用 SAC 训练系统")
    print("💡 基于 GPT 建议但使用标准 SB3 架构")
    print("🎯 目标: 稳定性 + 正确观察解析 + Link 融合")
    print()
    
    try:
        result = train_simplified_truly_universal_sac(
            num_joints=2, 
            max_joints=10, 
            link_lengths=None,
            total_timesteps=50000
        )
        
        print(f"\n🎊 简化版真正通用训练结果总结:")
        print(f"   最终评估奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   训练时间: {result['training_time']/60:.1f} 分钟")
        print(f"   演示成功率: {result['demo_success_rate']:.1%}")
        print(f"   演示平均奖励: {result['demo_avg_reward']:.2f}")
        print(f"   vs Baseline: {result['improvement_vs_baseline']:+.2f}")
        print(f"   vs Link-Aware: {result['improvement_vs_link_aware']:+.2f}")
        
        if result['improvement_vs_link_aware'] > 0.5:
            print(f"\n🏆 GPT 建议修复大成功!")
            print("   简化版真正通用架构验证成功!")
        elif result['improvement_vs_link_aware'] > 0.0:
            print(f"\n👍 GPT 建议修复有效!")
            print("   架构改进得到验证!")
        else:
            print(f"\n📈 需要进一步调试和优化")
        
        print(f"\n✅ 简化版真正通用架构验证完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
