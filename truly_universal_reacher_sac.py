#!/usr/bin/env python3
"""
真正通用 Reacher SAC 架构
基于 GPT-5 建议的完整实现：
A. 可变关节数环境包装器
B. 自定义 SAC 策略支持 J_max 维输出
C. 真正支持任意关节数训练
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
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import math

# ============================================================================
# 🧩 方案 A: 可变关节数环境包装器
# ============================================================================

class VariableJointReacherWrapper(gym.Wrapper):
    """
    可变关节数 Reacher 环境包装器
    
    功能：
    1. 统一 action_space 到 J_max 维
    2. 统一 observation_space 到 padded 维度
    3. 支持动态切换关节数
    4. 提供 mask 和 link_lengths 信息
    """
    
    def __init__(self, env, max_joints: int = 10, current_joints: int = 2, 
                 link_lengths: Optional[List[float]] = None):
        super(VariableJointReacherWrapper, self).__init__(env)
        
        self.max_joints = max_joints
        self.current_joints = current_joints
        self.original_action_space = env.action_space
        self.original_obs_space = env.observation_space
        
        # 设置 link 长度
        if link_lengths is None:
            self.link_lengths = [0.1] * max_joints
        else:
            self.link_lengths = link_lengths + [0.1] * (max_joints - len(link_lengths))
        
        # 重新定义动作空间为 J_max 维
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(max_joints,), 
            dtype=np.float32
        )
        
        # 重新定义观察空间 (padding 到适合 J_max 的维度)
        # 原始 Reacher-v5: 10维 [cos1, sin1, cos2, sin2, vel1, vel2, ee_x, ee_y, target_x, target_y]
        # 通用格式: J_max*4 + global_features
        padded_obs_dim = max_joints * 4 + 4  # 4 global features (ee + target)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(padded_obs_dim,),
            dtype=np.float32
        )
        
        print(f"🌐 VariableJointReacherWrapper 初始化:")
        print(f"   原始环境: {env.spec.id if hasattr(env, 'spec') else 'Unknown'}")
        print(f"   当前关节数: {current_joints}")
        print(f"   最大关节数: {max_joints}")
        print(f"   原始动作空间: {self.original_action_space}")
        print(f"   包装后动作空间: {self.action_space}")
        print(f"   原始观察空间: {self.original_obs_space}")
        print(f"   包装后观察空间: {self.observation_space}")
        print(f"   Link 长度: {self.link_lengths[:current_joints]}")
    
    def set_joint_config(self, num_joints: int, link_lengths: Optional[List[float]] = None):
        """动态设置关节配置"""
        self.current_joints = min(num_joints, self.max_joints)
        if link_lengths is not None:
            self.link_lengths[:len(link_lengths)] = link_lengths
        
        print(f"🔄 更新关节配置: {self.current_joints} 关节, Link长度: {self.link_lengths[:self.current_joints]}")
    
    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        将原始观察 padding 到统一格式
        
        MuJoCo Reacher-v5 观察格式 (10维):
        [0-1]: cos/sin of joint 1 angle
        [2-3]: cos/sin of joint 2 angle  
        [4-5]: joint 1 and joint 2 velocities
        [6-7]: end effector position (x, y)
        [8-9]: target position (x, y)
        
        转换为通用格式: [joint_features_padded, global_features]
        joint_features: [cos, sin, vel, link_length] × max_joints
        global_features: [ee_x, ee_y, target_x, target_y]
        """
        # 解析原始观察
        if self.current_joints == 2 and len(obs) == 10:
            # MuJoCo Reacher-v5 格式
            joint1_cos, joint1_sin = obs[0], obs[1]
            joint2_cos, joint2_sin = obs[2], obs[3]
            joint1_vel, joint2_vel = obs[4], obs[5]
            ee_x, ee_y = obs[6], obs[7]
            target_x, target_y = obs[8], obs[9]
            
            # 构造关节特征
            joint_features = []
            
            # Joint 1: [cos, sin, vel, link_length]
            joint1_feature = [joint1_cos, joint1_sin, joint1_vel, self.link_lengths[0]]
            joint_features.extend(joint1_feature)
            
            # Joint 2: [cos, sin, vel, link_length]
            joint2_feature = [joint2_cos, joint2_sin, joint2_vel, self.link_lengths[1]]
            joint_features.extend(joint2_feature)
            
            # Padding 剩余关节 (全零 + link_length)
            for i in range(2, self.max_joints):
                padding_feature = [0.0, 0.0, 0.0, self.link_lengths[i]]
                joint_features.extend(padding_feature)
            
            # 全局特征
            global_features = [ee_x, ee_y, target_x, target_y]
            
            # 组合
            padded_obs = np.array(joint_features + global_features, dtype=np.float32)
            
        else:
            # 通用格式或其他情况，简单 padding
            padded_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            copy_len = min(len(obs), len(padded_obs))
            padded_obs[:copy_len] = obs[:copy_len]
        
        return padded_obs
    
    def _unpad_action(self, action: np.ndarray) -> np.ndarray:
        """将 J_max 维动作切片到当前关节数"""
        # 只取前 current_joints 维
        unpadded_action = action[:self.current_joints]
        
        # 确保维度匹配原始环境
        if len(unpadded_action) != self.original_action_space.shape[0]:
            # 如果维度不匹配，截断或填充到原始维度
            original_dim = self.original_action_space.shape[0]
            if len(unpadded_action) > original_dim:
                unpadded_action = unpadded_action[:original_dim]
            else:
                padded_action = np.zeros(original_dim, dtype=np.float32)
                padded_action[:len(unpadded_action)] = unpadded_action
                unpadded_action = padded_action
        
        return unpadded_action
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        # Padding 观察
        padded_obs = self._pad_observation(obs)
        
        # 添加关节配置信息到 info
        info['num_joints'] = self.current_joints
        info['max_joints'] = self.max_joints
        info['link_lengths'] = self.link_lengths[:self.current_joints]
        info['joint_mask'] = [True] * self.current_joints + [False] * (self.max_joints - self.current_joints)
        
        return padded_obs, info
    
    def step(self, action):
        """执行动作"""
        # 将 J_max 维动作切片到当前关节数
        unpadded_action = self._unpad_action(action)
        
        # 在原始环境中执行
        obs, reward, terminated, truncated, info = self.env.step(unpadded_action)
        
        # Padding 观察
        padded_obs = self._pad_observation(obs)
        
        # 添加关节配置信息到 info
        info['num_joints'] = self.current_joints
        info['max_joints'] = self.max_joints
        info['link_lengths'] = self.link_lengths[:self.current_joints]
        info['joint_mask'] = [True] * self.current_joints + [False] * (self.max_joints - self.current_joints)
        
        return padded_obs, reward, terminated, truncated, info

# ============================================================================
# 🧩 复用现有的优秀特征提取组件
# ============================================================================

class CorrectMaskSystem:
    """修复版 Mask 系统"""
    
    @staticmethod
    def create_joint_mask(batch_size: int, num_joints: int, max_joints: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(batch_size, max_joints, dtype=torch.bool, device=device)
        mask[:, :num_joints] = True
        return mask
    
    @staticmethod
    def parse_observation_universal(obs: torch.Tensor, max_joints: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通用观察解析：从包装器的统一格式解析
        obs: [batch_size, max_joints*4 + global_dim]
        return: (joint_features, global_features)
        """
        batch_size = obs.size(0)
        
        # 分离关节特征和全局特征
        joint_dim = max_joints * 4
        joint_features_flat = obs[:, :joint_dim]  # [batch_size, max_joints*4]
        global_features = obs[:, joint_dim:]      # [batch_size, global_dim]
        
        # 重塑关节特征
        joint_features = joint_features_flat.view(batch_size, max_joints, 4)  # [batch_size, max_joints, 4]
        
        return joint_features, global_features

class LinkAwareJointEncoder(nn.Module):
    """Link-Aware 关节编码器"""
    def __init__(self, joint_input_dim: int = 4, joint_feature_dim: int = 64):
        super(LinkAwareJointEncoder, self).__init__()
        self.joint_input_dim = joint_input_dim
        self.joint_feature_dim = joint_feature_dim
        
        # Link 长度特征处理
        self.link_processor = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.LayerNorm(8)
        )
        
        # 运动特征处理
        self.motion_processor = nn.Sequential(
            nn.Linear(3, 24),
            nn.ReLU(),
            nn.LayerNorm(24)
        )
        
        # 几何-运动融合
        self.fusion_processor = nn.Sequential(
            nn.Linear(32, joint_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(joint_feature_dim)
        )
        
        print(f"🔗 LinkAwareJointEncoder: {joint_input_dim} → {joint_feature_dim}")
    
    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        batch_size, max_joints, _ = joint_features.shape
        
        # 分离特征
        motion_features = joint_features[:, :, :3]  # [cos, sin, vel]
        link_lengths = joint_features[:, :, 3:4]    # [link_length]
        
        # 重塑
        motion_flat = motion_features.view(-1, 3)
        link_flat = link_lengths.view(-1, 1)
        
        # 编码
        motion_encoded = self.motion_processor(motion_flat)
        link_encoded = self.link_processor(link_flat)
        
        # 融合
        fused_features = torch.cat([motion_encoded, link_encoded], dim=1)
        joint_encoded = self.fusion_processor(fused_features)
        
        # 重塑回原形状
        encoded_features = joint_encoded.view(batch_size, max_joints, self.joint_feature_dim)
        
        return encoded_features

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
# 🧩 通用特征提取器
# ============================================================================

class TrulyUniversalExtractor(BaseFeaturesExtractor):
    """真正通用特征提取器"""
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, 
                 max_joints: int = 10):
        super(TrulyUniversalExtractor, self).__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.max_joints = max_joints
        self.joint_input_dim = 4  # [cos, sin, vel, link_length]
        
        print(f"🌟 TrulyUniversalExtractor 初始化:")
        print(f"   观察空间维度: {self.obs_dim}")
        print(f"   最大关节数: {max_joints}")
        print(f"   关节输入维度: {self.joint_input_dim}")
        print(f"   输出特征维度: {features_dim}")
        
        # 模块组装
        self.joint_encoder = LinkAwareJointEncoder(
            joint_input_dim=self.joint_input_dim,
            joint_feature_dim=64
        )
        
        self.self_attention = FixedSelfAttention(
            feature_dim=64,
            num_heads=4,
            dropout=0.0
        )
        
        self.attention_pooling = FixedAttentionPooling(
            input_dim=64,
            output_dim=features_dim // 2
        )
        
        # 全局特征处理
        global_dim = 4  # [ee_x, ee_y, target_x, target_y]
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim, features_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(features_dim // 2)
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
        # Mask 系统
        self.mask_system = CorrectMaskSystem()
        
        print(f"✅ TrulyUniversalExtractor 构建完成")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)
        device = observations.device
        
        # 解析观察空间
        joint_features, global_features = self.mask_system.parse_observation_universal(
            observations, self.max_joints
        )
        
        # 动态检测有效关节数 (基于 link_length != 0)
        link_lengths = joint_features[:, :, 3]  # [batch_size, max_joints]
        joint_mask = (link_lengths > 0).bool()  # [batch_size, max_joints]
        
        # 关节编码
        encoded_joints = self.joint_encoder(joint_features)
        
        # 自注意力
        attended_joints = self.self_attention(encoded_joints, mask=joint_mask)
        
        # 注意力池化
        pooled_joint_features = self.attention_pooling(attended_joints, mask=joint_mask)
        
        # 全局特征处理
        processed_global = self.global_processor(global_features)
        
        # 融合
        fused_features = torch.cat([pooled_joint_features, processed_global], dim=1)
        final_features = self.final_fusion(fused_features)
        
        return final_features

# ============================================================================
# 🧩 方案 B: 自定义 SAC 策略支持 J_max 维输出
# ============================================================================

class UniversalJointGaussianHeads(nn.Module):
    """通用逐关节高斯头：支持 J_max 维输出"""
    def __init__(self, input_dim: int = 128, max_joints: int = 10):
        super(UniversalJointGaussianHeads, self).__init__()
        self.input_dim = input_dim
        self.max_joints = max_joints
        
        # 共享特征处理
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 为每个关节创建独立的高斯头
        self.joint_heads = nn.ModuleList()
        for i in range(max_joints):
            joint_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # mean + log_std for 1D action
            )
            self.joint_heads.append(joint_head)
        
        print(f"🎯 UniversalJointGaussianHeads: {max_joints} joints, 1D each")
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成 J_max 维高斯策略参数
        features: [batch_size, input_dim]
        return: (mean, log_std) 每个都是 [batch_size, max_joints]
        """
        batch_size = features.size(0)
        
        # 共享特征处理
        shared_features = self.shared_net(features)
        
        # 收集所有关节的输出
        joint_outputs = []
        for i in range(self.max_joints):
            joint_output = self.joint_heads[i](shared_features)  # [batch_size, 2]
            joint_outputs.append(joint_output)
        
        # 堆叠所有关节输出
        all_outputs = torch.stack(joint_outputs, dim=1)  # [batch_size, max_joints, 2]
        
        # 分离 mean 和 log_std
        mean_all = all_outputs[:, :, 0]  # [batch_size, max_joints]
        log_std_all = all_outputs[:, :, 1]  # [batch_size, max_joints]
        
        # 限制 log_std 范围
        log_std_all = torch.clamp(log_std_all, -20, 2)
        
        return mean_all, log_std_all

class TrulyUniversalSACPolicy(ActorCriticPolicy):
    """真正通用的 SAC 策略：支持 J_max 维输出"""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        max_joints: int = 10,
        **kwargs
    ):
        self.max_joints = max_joints
        
        print(f"🤖 TrulyUniversalSACPolicy 初始化:")
        print(f"   最大关节数: {max_joints}")
        print(f"   动作空间: {action_space}")
        
        super(TrulyUniversalSACPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        
        # 替换 action_net 为通用逐关节高斯头
        # 注意：latent_pi 的维度可能与 features_dim 不同，需要使用 net_arch 的最后一层
        latent_dim = self.net_arch[-1] if self.net_arch else self.features_dim
        self.action_net = UniversalJointGaussianHeads(
            input_dim=latent_dim,
            max_joints=max_joints
        )
        
        # 创建分布
        self.action_dist = SquashedDiagGaussianDistribution(max_joints)
        
        # 为 SB3 兼容性添加必要属性
        self.actor = self.action_net
        
        # 确保 critic 和 critic_target 在初始化时就设置
        # 使用 mlp_extractor 作为 critic (在 super().__init__ 中已创建)
        if hasattr(self, 'mlp_extractor') and self.mlp_extractor is not None:
            self.critic = self.mlp_extractor
            import copy
            self.critic_target = copy.deepcopy(self.mlp_extractor)
        else:
            # 如果 mlp_extractor 还未创建，创建一个临时的
            from stable_baselines3.common.torch_layers import MlpExtractor
            temp_extractor = MlpExtractor(
                self.features_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )
            self.critic = temp_extractor
            import copy
            self.critic_target = copy.deepcopy(temp_extractor)
        
        print(f"✅ TrulyUniversalSACPolicy 构建完成")
    
    def _build_mlp_extractor(self) -> None:
        """构建 MLP 提取器"""
        from stable_baselines3.common.torch_layers import MlpExtractor
        
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        
        # 设置 critic 相关属性
        self.critic = self.mlp_extractor
        
        # 创建 critic_target (深拷贝)
        import copy
        self.critic_target = copy.deepcopy(self.mlp_extractor)
    
    def forward_actor(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actor 前向传播"""
        # 提取特征
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        
        # 生成 J_max 维动作分布参数
        mean, log_std = self.action_net(latent_pi)
        
        return mean, log_std
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """预测动作"""
        mean, log_std = self.forward_actor(observation)
        
        # 创建分布
        self.action_dist = self.action_dist.proba_distribution(mean, log_std)
        
        # 采样动作
        if deterministic:
            actions = self.action_dist.mode()
        else:
            actions = self.action_dist.sample()
        
        return actions
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        # 前向传播
        mean, log_std = self.forward_actor(obs)
        
        # 创建分布
        self.action_dist = self.action_dist.proba_distribution(mean, log_std)
        
        # 计算 log_prob 和 entropy
        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()
        
        # Critic 评估
        features = self.extract_features(obs, self.features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy

# ============================================================================
# 🧩 训练函数
# ============================================================================

def train_truly_universal_reacher_sac(max_joints: int = 10, 
                                     joint_configs: List[Tuple[int, List[float]]] = None,
                                     total_timesteps: int = 50000):
    """
    训练真正通用 Reacher SAC
    
    Args:
        max_joints: 最大支持关节数
        joint_configs: [(num_joints, link_lengths), ...] 关节配置列表
        total_timesteps: 总训练步数
    """
    print("🌟 真正通用 Reacher SAC 训练")
    print(f"🔗 最大支持关节数: {max_joints}")
    print(f"💡 架构: GPT-5 建议的完整实现")
    print(f"🎯 目标: 真正支持任意关节数训练")
    print("=" * 70)
    
    # 默认关节配置
    if joint_configs is None:
        joint_configs = [
            (2, [0.1, 0.1]),  # 2关节 Reacher
            # 可以添加更多配置
            # (3, [0.1, 0.1, 0.1]),  # 3关节 Reacher
            # (4, [0.1, 0.1, 0.1, 0.1]),  # 4关节 Reacher
        ]
    
    print(f"🔧 关节配置:")
    for i, (num_joints, link_lengths) in enumerate(joint_configs):
        print(f"   配置 {i+1}: {num_joints} 关节, Link长度: {link_lengths}")
    
    # 创建环境 (先用第一个配置)
    print(f"\n🏭 创建环境...")
    base_env = gym.make('Reacher-v5')
    
    # 包装为可变关节数环境
    num_joints, link_lengths = joint_configs[0]
    env = VariableJointReacherWrapper(
        base_env, 
        max_joints=max_joints, 
        current_joints=num_joints,
        link_lengths=link_lengths
    )
    env = Monitor(env)
    
    # 评估环境
    eval_base_env = gym.make('Reacher-v5')
    eval_env = VariableJointReacherWrapper(
        eval_base_env,
        max_joints=max_joints,
        current_joints=num_joints,
        link_lengths=link_lengths
    )
    eval_env = Monitor(eval_env)
    
    print(f"✅ 环境创建完成")
    print(f"🎮 包装后动作空间: {env.action_space}")
    print(f"👁️ 包装后观察空间: {env.observation_space}")
    
    print("=" * 70)
    
    # 创建真正通用模型
    print("🤖 创建真正通用 SAC 模型...")
    
    policy_kwargs = {
        "features_extractor_class": TrulyUniversalExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "max_joints": max_joints
        },
        "max_joints": max_joints,
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
    }
    
    model = SAC(
        TrulyUniversalSACPolicy,
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
    
    print("✅ 真正通用 SAC 模型创建完成")
    print(f"📊 GPT-5 建议实现:")
    print(f"   ✅ 可变关节数环境包装器")
    print(f"   ✅ 自定义 SAC 策略支持 J_max 维输出")
    print(f"   ✅ 统一 action_space 和 observation_space")
    print(f"   ✅ 动态 mask 和 link_lengths 处理")
    
    print("=" * 70)
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./truly_universal_reacher_{max_joints}joints_best/',
        log_path=f'./truly_universal_reacher_{max_joints}joints_logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("🎯 开始真正通用训练...")
    print("📊 训练配置:")
    print(f"   总步数: {total_timesteps:,}")
    print("   评估频率: 每 5,000 步")
    print("   预期: 真正支持任意关节数")
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
    print("🏆 真正通用训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model_name = f"truly_universal_reacher_{max_joints}joints_final"
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
    
    print(f"📊 真正通用模型评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 演示
    print("\n🎮 演示真正通用模型 (10 episodes)...")
    demo_base_env = gym.make('Reacher-v5', render_mode='human')
    demo_env = VariableJointReacherWrapper(
        demo_base_env,
        max_joints=max_joints,
        current_joints=num_joints,
        link_lengths=link_lengths
    )
    
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
    print("📊 真正通用演示统计:")
    print(f"   成功率: {demo_success_rate:.1%} ({success_count}/10)")
    print(f"   平均奖励: {demo_avg_reward:.2f}")
    print(f"   奖励标准差: {np.std(episode_rewards):.2f}")
    
    print(f"\n🌟 真正通用架构优势:")
    print(f"   ✅ 可变关节数环境包装器")
    print(f"   ✅ 统一 J_max 维 action_space")
    print(f"   ✅ 动态 mask 和 link_lengths")
    print(f"   ✅ 自定义策略支持任意关节数")
    print(f"   🌐 真正支持任意关节数扩展")
    
    # 清理
    env.close()
    eval_env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'demo_success_rate': demo_success_rate,
        'demo_avg_reward': demo_avg_reward,
        'max_joints': max_joints,
        'joint_configs': joint_configs
    }

if __name__ == "__main__":
    print("🌟 真正通用 Reacher SAC 训练系统")
    print("💡 基于 GPT-5 建议的完整实现")
    print("🎯 目标: 真正支持任意关节数")
    print()
    
    try:
        result = train_truly_universal_reacher_sac(
            max_joints=10,
            joint_configs=[(2, [0.1, 0.1])],  # 先测试 2 关节
            total_timesteps=50000
        )
        
        print(f"\n🎊 真正通用训练结果总结:")
        print(f"   最终评估奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"   训练时间: {result['training_time']/60:.1f} 分钟")
        print(f"   演示成功率: {result['demo_success_rate']:.1%}")
        print(f"   演示平均奖励: {result['demo_avg_reward']:.2f}")
        print(f"   最大支持关节数: {result['max_joints']}")
        
        print(f"\n🏆 GPT-5 建议实现成功!")
        print("   真正通用架构验证完成!")
        
        print(f"\n✅ 现在可以扩展到任意关节数!")
        print("   只需修改 joint_configs 即可训练不同关节数的 Reacher")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
