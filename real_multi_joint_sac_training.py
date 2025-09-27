#!/usr/bin/env python3
"""
真实多关节 Reacher SAC 训练脚本
基于 GPT-5 建议：使用真实的 N 关节 MuJoCo 环境进行训练
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# 导入真实多关节环境
from real_multi_joint_reacher import RealMultiJointWrapper

# ============================================================================
# 🧩 Link-Aware 通用特征提取器 (适配真实多关节环境)
# ============================================================================

class RealMultiJointMaskSystem:
    """真实多关节环境的掩码系统"""
    
    @staticmethod
    def parse_observation_for_real_multi_joint(obs: np.ndarray, 
                                             num_joints: int, 
                                             link_lengths: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析真实多关节环境的观察空间
        
        obs格式: [joint_features×N, global_features×6]
        joint_features: [cos, sin, vel, link_length] × num_joints (4*N 维)
        global_features: [ee_x, ee_y, target_x, target_y, target_vec_x, target_vec_y] (6维)
        
        Returns:
            joint_features: [num_joints, 4] (cos, sin, vel, link_length)
            global_features: [6,] (ee_pos, target_pos, target_vec)
        """
        # 关节特征：前 4*num_joints 维
        joint_features_flat = obs[:4 * num_joints]
        joint_features = joint_features_flat.reshape(num_joints, 4)
        
        # 全局特征：后 6 维
        global_features = obs[4 * num_joints:]
        
        return joint_features, global_features

class RealLinkAwareJointEncoder(nn.Module):
    """真实多关节环境的关节编码器"""
    
    def __init__(self, joint_input_dim: int = 4, joint_hidden_dim: int = 32):
        super().__init__()
        self.joint_input_dim = joint_input_dim
        self.joint_hidden_dim = joint_hidden_dim
        
        # 几何特征处理 (cos, sin, link_length)
        self.geometric_processor = nn.Sequential(
            nn.Linear(3, 16),  # cos, sin, link_length
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # 运动特征处理 (vel)
        self.kinematic_processor = nn.Sequential(
            nn.Linear(1, 8),   # vel
            nn.ReLU(),
            nn.Linear(8, 8)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(16 + 8, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim)
        )
    
    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: [batch_size, num_joints, 4] (cos, sin, vel, link_length)
        Returns:
            encoded_features: [batch_size, num_joints, joint_hidden_dim]
        """
        batch_size, num_joints, _ = joint_features.shape
        
        # 分离几何和运动特征
        geometric_features = joint_features[:, :, [0, 1, 3]]  # cos, sin, link_length
        kinematic_features = joint_features[:, :, [2]]        # vel
        
        # 处理几何特征
        geometric_encoded = self.geometric_processor(
            geometric_features.reshape(-1, 3)
        ).reshape(batch_size, num_joints, 16)
        
        # 处理运动特征
        kinematic_encoded = self.kinematic_processor(
            kinematic_features.reshape(-1, 1)
        ).reshape(batch_size, num_joints, 8)
        
        # 融合特征
        combined_features = torch.cat([geometric_encoded, kinematic_encoded], dim=-1)
        encoded_features = self.fusion(
            combined_features.reshape(-1, 24)
        ).reshape(batch_size, num_joints, self.joint_hidden_dim)
        
        return encoded_features

class RealFixedSelfAttention(nn.Module):
    """真实多关节环境的自注意力机制"""
    
    def __init__(self, feature_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, joint_features: torch.Tensor, 
                joint_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            joint_features: [batch_size, num_joints, feature_dim]
            joint_mask: [batch_size, num_joints] (True for valid joints)
        Returns:
            attended_features: [batch_size, num_joints, feature_dim]
        """
        # 准备 key_padding_mask (True for padded positions)
        key_padding_mask = None
        if joint_mask is not None:
            key_padding_mask = ~joint_mask  # 反转掩码
        
        # 自注意力
        attended_features, _ = self.multihead_attn(
            query=joint_features,
            key=joint_features,
            value=joint_features,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和层归一化
        output = self.layer_norm(joint_features + attended_features)
        
        return output

class RealFixedAttentionPooling(nn.Module):
    """真实多关节环境的注意力池化"""
    
    def __init__(self, feature_dim: int = 32, pooled_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.pooled_dim = pooled_dim
        
        self.attention_weights = nn.Linear(feature_dim, 1)
        self.output_projection = nn.Linear(feature_dim, pooled_dim)
    
    def forward(self, joint_features: torch.Tensor, 
                joint_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            joint_features: [batch_size, num_joints, feature_dim]
            joint_mask: [batch_size, num_joints] (True for valid joints)
        Returns:
            pooled_features: [batch_size, pooled_dim]
        """
        batch_size, num_joints, feature_dim = joint_features.shape
        
        # 计算注意力权重
        attention_scores = self.attention_weights(joint_features)  # [batch_size, num_joints, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, num_joints]
        
        # 应用掩码
        if joint_mask is not None:
            attention_scores = attention_scores.masked_fill(~joint_mask, float('-inf'))
        
        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_joints]
        
        # 加权池化
        pooled_features = torch.sum(
            joint_features * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, feature_dim]
        
        # 输出投影
        output = self.output_projection(pooled_features)  # [batch_size, pooled_dim]
        
        return output

class RealMultiJointUniversalExtractor(BaseFeaturesExtractor):
    """真实多关节环境的通用特征提取器"""
    
    def __init__(self, observation_space: gym.Space, 
                 num_joints: int = 3,
                 joint_hidden_dim: int = 32,
                 pooled_dim: int = 64,
                 global_hidden_dim: int = 32,
                 features_dim: int = 128):
        
        super(RealMultiJointUniversalExtractor, self).__init__(observation_space, features_dim)
        
        self.num_joints = num_joints
        self.joint_hidden_dim = joint_hidden_dim
        self.pooled_dim = pooled_dim
        self.global_hidden_dim = global_hidden_dim
        
        print(f"🔧 RealMultiJointUniversalExtractor 初始化:")
        print(f"   关节数: {num_joints}")
        print(f"   观察空间: {observation_space}")
        print(f"   特征维度: {features_dim}")
        
        # 关节编码器
        self.joint_encoder = RealLinkAwareJointEncoder(
            joint_input_dim=4,
            joint_hidden_dim=joint_hidden_dim
        )
        
        # 自注意力
        self.self_attention = RealFixedSelfAttention(
            feature_dim=joint_hidden_dim,
            num_heads=4
        )
        
        # 注意力池化
        self.attention_pooling = RealFixedAttentionPooling(
            feature_dim=joint_hidden_dim,
            pooled_dim=pooled_dim
        )
        
        # 全局特征处理器
        self.global_processor = nn.Sequential(
            nn.Linear(6, global_hidden_dim),  # ee_pos, target_pos, target_vec
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim)
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(pooled_dim + global_hidden_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, obs_dim]
        Returns:
            features: [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # 直接在 tensor 上解析观察 (避免 CPU-GPU 转换)
        # 观察格式: [joint_features×N, global_features×6]
        # joint_features: [cos, sin, vel, link_length] × num_joints (4*N 维)
        # global_features: [ee_x, ee_y, target_x, target_y, target_vec_x, target_vec_y] (6维)
        
        joint_features_flat = observations[:, :4 * self.num_joints]  # [batch_size, 4*num_joints]
        joint_features_tensor = joint_features_flat.reshape(batch_size, self.num_joints, 4)  # [batch_size, num_joints, 4]
        
        global_features_tensor = observations[:, 4 * self.num_joints:]  # [batch_size, 6]
        
        # 关节特征编码
        encoded_joint_features = self.joint_encoder(joint_features_tensor)
        
        # 自注意力 (真实多关节环境不需要掩码，所有关节都是真实的)
        attended_joint_features = self.self_attention(encoded_joint_features)
        
        # 注意力池化
        pooled_joint_features = self.attention_pooling(attended_joint_features)
        
        # 全局特征处理
        processed_global_features = self.global_processor(global_features_tensor)
        
        # 最终融合
        combined_features = torch.cat([pooled_joint_features, processed_global_features], dim=-1)
        final_features = self.final_fusion(combined_features)
        
        return final_features

# ============================================================================
# 🧩 训练函数
# ============================================================================

def train_real_multi_joint_sac(num_joints: int = 3,
                              link_lengths: List[float] = None,
                              total_timesteps: int = 50000,
                              render_mode: str = None) -> Dict[str, Any]:
    """
    训练真实多关节 Reacher SAC 模型
    
    Args:
        num_joints: 关节数量
        link_lengths: 每个关节的 link 长度
        total_timesteps: 总训练步数
        render_mode: 渲染模式 ('human' 或 None)
    
    Returns:
        训练结果字典
    """
    if link_lengths is None:
        link_lengths = [0.1] * num_joints
    
    print(f"\n{'='*60}")
    print(f"🚀 开始训练真实 {num_joints} 关节 Reacher SAC")
    print(f"{'='*60}")
    print(f"📊 训练配置:")
    print(f"   关节数: {num_joints}")
    print(f"   Link 长度: {link_lengths}")
    print(f"   总步数: {total_timesteps}")
    print(f"   渲染模式: {render_mode}")
    
    # 创建环境
    print(f"\n🌍 创建真实多关节环境...")
    env = RealMultiJointWrapper(
        num_joints=num_joints,
        link_lengths=link_lengths,
        render_mode=render_mode
    )
    
    # 包装监控
    env = Monitor(env)
    
    # 创建评估环境
    eval_env = RealMultiJointWrapper(
        num_joints=num_joints,
        link_lengths=link_lengths,
        render_mode=None  # 评估时不渲染
    )
    eval_env = Monitor(eval_env)
    
    print(f"✅ 环境创建完成")
    print(f"   训练环境观察空间: {env.observation_space}")
    print(f"   训练环境动作空间: {env.action_space}")
    
    # 创建 SAC 模型
    print(f"\n🤖 创建 SAC 模型...")
    
    policy_kwargs = {
        'features_extractor_class': RealMultiJointUniversalExtractor,
        'features_extractor_kwargs': {
            'num_joints': num_joints,
            'joint_hidden_dim': 32,
            'pooled_dim': 64,
            'global_hidden_dim': 32,
            'features_dim': 128
        },
        'net_arch': [256, 256]
    }
    
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=100,         # 更早开始学习，更快看到训练日志
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        verbose=2,                   # 增加到 verbose=2 获得更详细的日志
        device='auto',
        tensorboard_log="./tensorboard_logs/"  # 添加 tensorboard 日志
    )
    
    print(f"✅ SAC 模型创建完成")
    print(f"   策略网络: MlpPolicy + RealMultiJointUniversalExtractor")
    print(f"   学习率: 3e-4")
    print(f"   缓冲区大小: 100000")
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./logs/real_{num_joints}joint_sac/',
        log_path=f'./logs/real_{num_joints}joint_sac/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print(f"\n🎯 开始训练...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"✅ 训练完成！用时: {training_time:.1f} 秒")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"⚠️ 训练被中断！已训练时间: {training_time:.1f} 秒")
    
    # 评估模型
    print(f"\n📈 评估模型性能...")
    
    try:
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        print(f"📊 评估结果:")
        print(f"   平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
        
        # 计算成功率 (假设奖励 > -1 为成功)
        episode_rewards = []
        for _ in range(20):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        
        success_rate = sum(1 for r in episode_rewards if r > -1) / len(episode_rewards)
        
        print(f"   成功率: {success_rate:.1%}")
        print(f"   训练时间: {training_time:.1f} 秒")
        
        # 保存模型
        model_path = f"real_{num_joints}joint_sac_model"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        results = {
            'num_joints': num_joints,
            'link_lengths': link_lengths,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'model_path': model_path,
            'is_real_multi_joint': True
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return {
            'num_joints': num_joints,
            'training_time': training_time,
            'error': str(e),
            'is_real_multi_joint': True
        }
    
    finally:
        env.close()
        eval_env.close()

# ============================================================================
# 🧩 主函数
# ============================================================================

def main():
    """主函数：测试不同关节数的训练效果"""
    
    print("🌟 真实多关节 Reacher SAC 训练测试")
    print("💡 基于 GPT-5 建议：真实的 N 关节 MuJoCo 动力学")
    print()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 测试配置 (减少训练步数以便快速看到效果)
    test_configs = [
        {'num_joints': 2, 'link_lengths': [0.1, 0.1], 'timesteps': 10000},
        {'num_joints': 3, 'link_lengths': [0.1, 0.1, 0.1], 'timesteps': 15000},
        # {'num_joints': 4, 'link_lengths': [0.1, 0.1, 0.1, 0.1], 'timesteps': 20000},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"🧪 测试配置: {config['num_joints']} 关节")
        print(f"{'='*80}")
        
        try:
            result = train_real_multi_joint_sac(
                num_joints=config['num_joints'],
                link_lengths=config['link_lengths'],
                total_timesteps=config['timesteps'],
                render_mode=None  # 训练时不渲染
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ {config['num_joints']} 关节训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'num_joints': config['num_joints'],
                'error': str(e),
                'is_real_multi_joint': True
            })
    
    # 总结结果
    print(f"\n{'='*80}")
    print(f"📊 训练结果总结")
    print(f"{'='*80}")
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['num_joints']} 关节: 训练失败 - {result['error']}")
        else:
            print(f"✅ {result['num_joints']} 关节:")
            print(f"   平均奖励: {result.get('mean_reward', 'N/A'):.3f}")
            print(f"   成功率: {result.get('success_rate', 0):.1%}")
            print(f"   训练时间: {result.get('training_time', 0):.1f} 秒")
            print(f"   真实多关节: {result.get('is_real_multi_joint', False)}")

if __name__ == "__main__":
    main()
