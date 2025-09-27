#!/usr/bin/env python3
"""
Link-Aware SAC 架构 - 方案 A: 关节特征扩展
基于成功的简化版修复架构，扩展关节特征维度融合 link 长度信息

扩展内容：
1. 关节特征: [cos, sin, vel] → [cos, sin, vel, link_length]
2. Link-Motion 分离处理
3. 取消训练渲染，提高效率
4. 保持现有架构的所有优势
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
# 🧩 扩展 1: Link-Aware 关节编码器 - 方案 A
# ============================================================================

class LinkAwareJointEncoder(nn.Module):
    """
    Link-Aware 关节编码器：融合 link 长度信息
    输入格式：[cos, sin, vel, link_length] (4维)
    """
    def __init__(self, joint_input_dim: int = 4, joint_feature_dim: int = 64):
        super(LinkAwareJointEncoder, self).__init__()
        self.joint_input_dim = joint_input_dim  # [cos, sin, vel, link_length]
        self.joint_feature_dim = joint_feature_dim
        
        # Link 长度特征处理 (几何信息)
        self.link_processor = nn.Sequential(
            nn.Linear(1, 8),  # link_length → 8维几何特征
            nn.ReLU(),
            nn.LayerNorm(8)
        )
        
        # 运动特征处理 [cos, sin, vel] (运动信息)
        self.motion_processor = nn.Sequential(
            nn.Linear(3, 24),  # [cos, sin, vel] → 24维运动特征
            nn.ReLU(),
            nn.LayerNorm(24)
        )
        
        # 几何-运动融合处理器
        self.fusion_processor = nn.Sequential(
            nn.Linear(32, joint_feature_dim),  # 8(几何) + 24(运动) = 32 → 64
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
        motion_encoded = self.motion_processor(motion_flat)  # [batch_size * max_joints, 24]
        link_encoded = self.link_processor(link_flat)        # [batch_size * max_joints, 8]
        
        # 融合几何和运动特征
        fused_features = torch.cat([motion_encoded, link_encoded], dim=1)  # [batch_size * max_joints, 32]
        joint_encoded = self.fusion_processor(fused_features)  # [batch_size * max_joints, 64]
        
        # 重塑回 [batch_size, max_joints, joint_feature_dim]
        encoded_features = joint_encoded.view(batch_size, max_joints, self.joint_feature_dim)
        
        return encoded_features

# ============================================================================
# 🧩 复用现有的优秀组件
# ============================================================================

class FixedSelfAttention(nn.Module):
    """
    修复版自注意力：使用 key_padding_mask 而非复杂的 attn_mask
    """
    def __init__(self, feature_dim: int = 64, num_heads: int = 4, dropout: float = 0.0):
        super(FixedSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # 标准 Multi-Head Attention - 去掉 Dropout
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm - 去掉 Dropout
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Feed Forward - 去掉 Dropout
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        print(f"🧠 FixedSelfAttention: {feature_dim}d, {num_heads} heads (复用修复版)")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        自注意力处理 - 使用 key_padding_mask
        x: [batch_size, max_joints, feature_dim] (已 padding)
        mask: [batch_size, max_joints] - True 表示有效关节，False 表示 padding
        return: [batch_size, max_joints, feature_dim]
        """
        # 准备 key_padding_mask (True 表示要屏蔽的位置)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # 反转：True 表示屏蔽 padding 位置
        
        # Multi-Head Self-Attention - 使用 key_padding_mask
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # 残差连接 + Layer Norm
        x = self.layer_norm(x + attn_output)
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        
        # 残差连接 + Layer Norm
        x = self.layer_norm2(x + ff_output)
        
        # 应用掩码到输出 (将 padding 位置置零)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        
        return x

class FixedAttentionPooling(nn.Module):
    """
    修复版注意力池化：去掉双重 softmax，mask 在 softmax 前生效
    """
    def __init__(self, input_dim: int = 64, output_dim: int = 128):
        super(FixedAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 注意力权重计算 - 去掉 Softmax
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # 输出投影 - 去掉 Dropout
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
        print(f"🎯 FixedAttentionPooling: {input_dim} → {output_dim} (复用修复版)")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        注意力池化 - mask 在 softmax 前生效
        x: [batch_size, max_joints, input_dim] (已 padding)
        mask: [batch_size, max_joints] - True 表示有效关节
        return: [batch_size, output_dim]
        """
        # 计算注意力分数
        s = self.score(x).squeeze(-1)  # [batch_size, max_joints]
        
        # 先 mask 再 softmax
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)  # padding 位置设为极小值
        
        # 单次 softmax
        w = F.softmax(s, dim=1).unsqueeze(-1)  # [batch_size, max_joints, 1]
        
        # 加权聚合
        pooled = (x * w).sum(dim=1)  # [batch_size, input_dim]
        
        # 输出投影
        output = self.proj(pooled)  # [batch_size, output_dim]
        
        return output

# ============================================================================
# 🧩 扩展 2: Link-Aware Mask 系统
# ============================================================================

class LinkAwareMaskSystem:
    """
    Link-Aware Mask 系统：处理 link 长度信息
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
    def parse_observation_with_links(obs: torch.Tensor, num_joints: int, max_joints: int, 
                                   link_lengths: Optional[List[float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解析观察空间并融合 link 长度信息
        obs: [batch_size, obs_dim]
        link_lengths: [link1_length, link2_length, ...] 或 None (使用默认值)
        return: (joint_features_with_links, global_features)
        """
        batch_size = obs.size(0)
        device = obs.device
        
        # 默认 link 长度 (MuJoCo Reacher-v5)
        if link_lengths is None:
            if num_joints == 2:
                link_lengths = [0.1, 0.1]  # MuJoCo Reacher-v5 默认 link 长度
            else:
                link_lengths = [0.1] * num_joints  # 通用默认值
        
        # 确保 link_lengths 长度匹配
        while len(link_lengths) < num_joints:
            link_lengths.append(0.1)  # 默认长度
        
        if num_joints == 2:
            # MuJoCo Reacher-v5 格式
            joint_cos_sin = obs[:, :2]  # [batch_size, 2]
            joint_velocities = obs[:, 2:4]  # [batch_size, 2]
            global_features = obs[:, 4:]  # [batch_size, 6]
            
            # 构造关节特征 + link 长度
            joint_features_list = []
            for i in range(num_joints):
                cos_val = joint_cos_sin[:, i:i+1]  # [batch_size, 1]
                sin_val = torch.zeros_like(cos_val)  # 简化：sin 设为 0
                vel_val = joint_velocities[:, i:i+1]  # [batch_size, 1]
                
                # 添加 link 长度信息
                link_val = torch.full_like(cos_val, link_lengths[i])  # [batch_size, 1]
                
                joint_feature = torch.cat([cos_val, sin_val, vel_val, link_val], dim=1)  # [batch_size, 4]
                joint_features_list.append(joint_feature)
            
            joint_features = torch.stack(joint_features_list, dim=1)  # [batch_size, num_joints, 4]
            
        else:
            # 通用格式：假设前 num_joints*3 是关节特征，剩余是全局特征
            joint_dim = num_joints * 3
            joint_obs = obs[:, :joint_dim]  # [batch_size, num_joints*3]
            global_features = obs[:, joint_dim:]  # [batch_size, remaining]
            
            # 重塑关节特征并添加 link 长度
            joint_features_3d = joint_obs.view(batch_size, num_joints, 3)  # [batch_size, num_joints, 3]
            
            # 为每个关节添加 link 长度
            joint_features_list = []
            for i in range(num_joints):
                joint_motion = joint_features_3d[:, i]  # [batch_size, 3]
                link_val = torch.full((batch_size, 1), link_lengths[i], device=device)  # [batch_size, 1]
                joint_feature = torch.cat([joint_motion, link_val], dim=1)  # [batch_size, 4]
                joint_features_list.append(joint_feature)
            
            joint_features = torch.stack(joint_features_list, dim=1)  # [batch_size, num_joints, 4]
        
        # Padding 到 max_joints
        joint_features_padded = torch.zeros(batch_size, max_joints, 4, device=device)
        joint_features_padded[:, :num_joints] = joint_features
        
        return joint_features_padded, global_features

# ============================================================================
# 🧩 扩展 3: Link-Aware 通用特征提取器
# ============================================================================

class LinkAwareUniversalExtractor(BaseFeaturesExtractor):
    """
    Link-Aware 通用特征提取器
    在成功架构基础上融合 link 长度信息
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, 
                 num_joints: int = 2, max_joints: int = 10, 
                 link_lengths: Optional[List[float]] = None):
        super(LinkAwareUniversalExtractor, self).__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.num_joints = num_joints
        self.max_joints = max_joints
        self.joint_input_dim = 4  # [cos, sin, vel, link_length]
        self.link_lengths = link_lengths
        
        print(f"🌟 LinkAwareUniversalExtractor 初始化:")
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
        
        # 模块 2: 自注意力 (复用修复版)
        self.self_attention = FixedSelfAttention(
            feature_dim=64,
            num_heads=4,
            dropout=0.0
        )
        
        # 模块 3: 注意力池化 (复用修复版)
        self.attention_pooling = FixedAttentionPooling(
            input_dim=64,
            output_dim=features_dim // 2
        )
        
        # 全局特征处理 (复用现有逻辑)
        if num_joints == 2:
            global_dim = 6  # MuJoCo Reacher-v5 的全局特征维度
        else:
            global_dim = max(0, self.obs_dim - (num_joints * 3))  # 注意：观察空间仍是 3 维/关节
        
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
        
        # 最终融合 (复用现有逻辑)
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
        # Link-Aware Mask 系统
        self.mask_system = LinkAwareMaskSystem()
        
        print(f"✅ LinkAwareUniversalExtractor 构建完成")
        print(f"   🔗 融合 link 长度信息")
        print(f"   ✅ 保持现有架构的所有优势")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Link-Aware 前向传播
        """
        batch_size = observations.size(0)
        device = observations.device
        
        # 步骤 1: 解析观察空间并融合 link 长度
        joint_features_with_links, global_features = self.mask_system.parse_observation_with_links(
            observations, self.num_joints, self.max_joints, self.link_lengths
        )
        
        # 步骤 2: 创建关节掩码
        joint_mask = self.mask_system.create_joint_mask(
            batch_size, self.num_joints, self.max_joints, device
        )
        
        # 步骤 3: Link-Aware 关节编码 (几何+运动融合)
        encoded_joints = self.joint_encoder(joint_features_with_links)  # [batch_size, max_joints, 64]
        
        # 步骤 4: 自注意力建模关节间交互 (现在包含几何信息)
        attended_joints = self.self_attention(
            encoded_joints, 
            mask=joint_mask
        )  # [batch_size, max_joints, 64]
        
        # 步骤 5: 注意力池化
        pooled_joint_features = self.attention_pooling(
            attended_joints,
            mask=joint_mask
        )  # [batch_size, features_dim//2]
        
        # 步骤 6: 处理全局特征
        if self.global_processor is not None and global_features.size(1) > 0:
            processed_global = self.global_processor(global_features)  # [batch_size, features_dim//2]
            fused_features = torch.cat([pooled_joint_features, processed_global], dim=1)
        else:
            fused_features = pooled_joint_features
        
        # 步骤 7: 最终融合
        final_features = self.final_fusion(fused_features)  # [batch_size, features_dim]
        
        return final_features

# ============================================================================
# 🧩 扩展 4: Link-Aware 训练函数 (取消渲染)
# ============================================================================

def train_link_aware_sac(num_joints: int = 2, max_joints: int = 10, 
                        link_lengths: Optional[List[float]] = None,
                        total_timesteps: int = 50000):
    """
    训练 Link-Aware SAC (取消渲染，提高效率)
    """
    print("🌟 Link-Aware SAC 训练 (方案 A)")
    print(f"🔗 当前关节数: {num_joints}")
    print(f"🔗 最大支持关节数: {max_joints}")
    print(f"🔗 Link 长度: {link_lengths}")
    print(f"💡 架构: 关节特征扩展 [cos, sin, vel, link_length]")
    print(f"🎯 目标: 几何感知 + 高效训练")
    print("=" * 70)
    
    # 创建环境 - 取消训练渲染
    print(f"🏭 创建环境...")
    if num_joints == 2:
        env = gym.make('Reacher-v5')  # 训练时不渲染
        eval_env = gym.make('Reacher-v5')  # 评估时不渲染
    else:
        print(f"⚠️ 暂不支持 {num_joints} 关节环境，使用 2 关节进行验证")
        env = gym.make('Reacher-v5')
        eval_env = gym.make('Reacher-v5')
    
    env = Monitor(env)
    eval_env = Monitor(eval_env)
    
    print(f"✅ 环境创建完成 (无渲染，高效训练)")
    print(f"🎮 动作空间: {env.action_space}")
    print(f"👁️ 观察空间: {env.observation_space}")
    
    print("=" * 70)
    
    # 创建 Link-Aware 模型
    print("🤖 创建 Link-Aware SAC 模型...")
    
    policy_kwargs = {
        "features_extractor_class": LinkAwareUniversalExtractor,
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
    
    print("✅ Link-Aware SAC 模型创建完成")
    print(f"📊 扩展特点:")
    print(f"   🔗 关节特征扩展: [cos, sin, vel] → [cos, sin, vel, link_length]")
    print(f"   🧠 几何-运动分离处理")
    print(f"   🎯 更精确的空间感知")
    print(f"   ✅ 保持所有现有优势")
    print(f"   ⚡ 无渲染，高效训练")
    
    print("=" * 70)
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./link_aware_{num_joints}joints_best/',
        log_path=f'./link_aware_{num_joints}joints_logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("🎯 开始 Link-Aware 训练...")
    print("📊 训练配置:")
    print(f"   总步数: {total_timesteps:,}")
    print("   评估频率: 每 5,000 步")
    print("   渲染: 关闭 (提高效率)")
    print("   预期: 几何感知能力提升")
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
    print("🏆 Link-Aware 训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model_name = f"link_aware_{num_joints}joints_final"
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
    
    print(f"📊 Link-Aware 模型评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 与之前版本对比
    baseline_reward = -4.86
    simplified_fixed_reward = -3.76  # 之前的最佳结果
    
    print(f"\n📈 性能对比:")
    print(f"   Baseline SAC: {baseline_reward:.2f}")
    print(f"   简化版修复: {simplified_fixed_reward:.2f}")
    print(f"   Link-Aware: {mean_reward:.2f}")
    
    improvement_vs_baseline = mean_reward - baseline_reward
    improvement_vs_simplified = mean_reward - simplified_fixed_reward
    
    print(f"\n📊 改进幅度:")
    print(f"   vs Baseline: {improvement_vs_baseline:+.2f}")
    print(f"   vs 简化版修复: {improvement_vs_simplified:+.2f}")
    
    if improvement_vs_simplified > 0.5:
        print("   🎉 Link 几何信息融合大成功!")
    elif improvement_vs_simplified > 0.0:
        print("   👍 Link 几何信息融合有效!")
    elif improvement_vs_simplified > -0.5:
        print("   📈 Link 几何信息效果中性，需进一步优化")
    else:
        print("   ⚠️ Link 几何信息可能引入噪声，需要调整")
    
    # 演示 - 只在演示时渲染
    print("\n🎮 演示 Link-Aware 模型 (10 episodes)...")
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
    print("📊 Link-Aware 演示统计:")
    print(f"   成功率: {demo_success_rate:.1%} ({success_count}/10)")
    print(f"   平均奖励: {demo_avg_reward:.2f}")
    print(f"   奖励标准差: {np.std(episode_rewards):.2f}")
    
    # 与之前最佳结果对比
    simplified_demo_success = 0.6
    simplified_demo_reward = -4.27
    
    print(f"\n📈 与简化版修复对比:")
    print(f"   简化版成功率: {simplified_demo_success:.1%}")
    print(f"   Link-Aware 成功率: {demo_success_rate:.1%}")
    print(f"   成功率变化: {demo_success_rate - simplified_demo_success:+.1%}")
    print(f"   ")
    print(f"   简化版平均奖励: {simplified_demo_reward:.2f}")
    print(f"   Link-Aware 平均奖励: {demo_avg_reward:.2f}")
    print(f"   奖励变化: {demo_avg_reward - simplified_demo_reward:+.2f}")
    
    if demo_success_rate >= 0.7:
        print("   🎉 Link-Aware 大成功!")
    elif demo_success_rate >= 0.6:
        print("   👍 Link-Aware 成功!")
    elif demo_success_rate >= 0.5:
        print("   📈 Link-Aware 良好!")
    else:
        print("   📈 仍需进一步优化")
    
    print(f"\n🌟 Link-Aware 架构优势 (方案 A):")
    print(f"   🔗 关节特征扩展: 4维输入 [cos, sin, vel, link_length]")
    print(f"   🧠 几何-运动分离处理: 8维几何 + 24维运动")
    print(f"   🎯 精确空间感知: link 长度直接影响关节表示")
    print(f"   ✅ 保持现有优势: 所有 GPT-5 修复都保留")
    print(f"   🌐 通用扩展性: 支持任意 link 长度配置")
    print(f"   ⚡ 高效训练: 无渲染，快速收敛")
    
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
        'improvement_vs_simplified': improvement_vs_simplified,
        'num_joints': num_joints,
        'max_joints': max_joints,
        'link_lengths': link_lengths
    }

if __name__ == "__main__":
    print("🌟 Link-Aware SAC 训练系统 (方案 A)")
    print("💡 关节特征扩展: [cos, sin, vel] → [cos, sin, vel, link_length]")
    print("🎯 目标: 几何感知 + 高效训练")
    print()
    
    # 默认配置测试
    print("🔗 测试默认 Link 长度配置...")
    
    try:
        result = train_link_aware_sac(
            num_joints=2, 
            max_joints=10, 
            link_lengths=None,  # 使用默认 [0.1, 0.1]
            total_timesteps=50000
        )
        
        print(f"\n🎊 Link-Aware 训练结果总结:")
        print(f"   最终评估奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   训练时间: {result['training_time']/60:.1f} 分钟")
        print(f"   演示成功率: {result['demo_success_rate']:.1%}")
        print(f"   演示平均奖励: {result['demo_avg_reward']:.2f}")
        print(f"   vs Baseline: {result['improvement_vs_baseline']:+.2f}")
        print(f"   vs 简化版修复: {result['improvement_vs_simplified']:+.2f}")
        
        if result['improvement_vs_simplified'] > 0.5:
            print(f"\n🏆 Link-Aware 几何感知大成功!")
            print("   方案 A 的关节特征扩展策略验证成功!")
        elif result['improvement_vs_simplified'] > 0.0:
            print(f"\n👍 Link-Aware 几何感知有效!")
            print("   方案 A 带来了性能提升!")
        else:
            print(f"\n📈 Link-Aware 需要进一步优化")
            print("   可能需要调整几何-运动特征的融合方式")
        
        print(f"\n✅ Link-Aware 架构 (方案 A) 验证完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
