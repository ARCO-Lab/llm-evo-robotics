#!/usr/bin/env python3
"""
可变关节 Reacher SAC 训练系统
基于 multi_joint_reacher_sac.py 修改，实现真正意义上可以兼容可变关节数量训练的模型

核心改进：
1. 使用真实的多关节 MuJoCo 环境 (不是基于 Reacher-v5 的包装器)
2. 保持通用特征提取器架构
3. 支持在同一个模型中训练不同关节数
4. 使用 J_max 策略，支持真正的可变关节数量
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

# 导入真实多关节环境
from real_multi_joint_reacher import RealMultiJointWrapper

# ============================================================================
# 🧩 可变关节掩码系统
# ============================================================================

class VariableJointMaskSystem:
    """可变关节掩码系统，支持 J_max 策略"""
    
    @staticmethod
    def create_joint_mask(num_joints: int, max_joints: int) -> torch.Tensor:
        """
        创建关节掩码
        Args:
            num_joints: 实际关节数
            max_joints: 最大关节数 (J_max)
        Returns:
            joint_mask: [max_joints,] (True for valid joints)
        """
        mask = torch.zeros(max_joints, dtype=torch.bool)
        mask[:num_joints] = True
        return mask
    
    @staticmethod
    def pad_observation(obs: np.ndarray, num_joints: int, max_joints: int, 
                       link_lengths: List[float]) -> np.ndarray:
        """
        将观察填充到 J_max 维度
        
        Args:
            obs: 原始观察 [joint_features×N + global_features×6]
            num_joints: 实际关节数
            max_joints: 最大关节数
            link_lengths: 关节长度列表
        
        Returns:
            padded_obs: 填充后的观察 [joint_features×J_max + global_features×6]
        """
        # 解析原始观察
        joint_features_flat = obs[:4 * num_joints]  # [cos, sin, vel, link_length] × num_joints
        global_features = obs[4 * num_joints:]      # [ee_pos, target_pos, target_vec]
        
        # 重塑关节特征
        joint_features = joint_features_flat.reshape(num_joints, 4)
        
        # 创建填充后的关节特征
        padded_joint_features = np.zeros((max_joints, 4), dtype=np.float32)
        padded_joint_features[:num_joints] = joint_features
        
        # 对于填充的关节，使用默认值
        for i in range(num_joints, max_joints):
            padded_joint_features[i] = [1.0, 0.0, 0.0, 0.05]  # [cos=1, sin=0, vel=0, link_len=0.05]
        
        # 重新组合
        padded_obs = np.concatenate([
            padded_joint_features.flatten(),  # 4 * max_joints
            global_features                   # 6
        ])
        
        return padded_obs
    
    @staticmethod
    def pad_action(action: np.ndarray, num_joints: int, max_joints: int) -> np.ndarray:
        """
        将动作填充到 J_max 维度
        
        Args:
            action: 原始动作 [num_joints,]
            num_joints: 实际关节数
            max_joints: 最大关节数
        
        Returns:
            padded_action: 填充后的动作 [max_joints,]
        """
        padded_action = np.zeros(max_joints, dtype=np.float32)
        padded_action[:num_joints] = action
        return padded_action
    
    @staticmethod
    def unpad_action(padded_action: np.ndarray, num_joints: int) -> np.ndarray:
        """
        从填充的动作中提取实际动作
        
        Args:
            padded_action: 填充后的动作 [max_joints,]
            num_joints: 实际关节数
        
        Returns:
            action: 实际动作 [num_joints,]
        """
        return padded_action[:num_joints]

# ============================================================================
# 🧩 可变关节环境包装器
# ============================================================================

class VariableJointReacherWrapper(gym.Wrapper):
    """
    可变关节 Reacher 环境包装器
    支持 J_max 策略，可以在训练过程中动态改变关节数
    """
    
    def __init__(self, max_joints: int = 4, 
                 joint_configs: List[Dict] = None,
                 current_config_idx: int = 0):
        """
        Args:
            max_joints: 最大关节数 (J_max)
            joint_configs: 关节配置列表 [{'num_joints': 2, 'link_lengths': [0.1, 0.1]}, ...]
            current_config_idx: 当前配置索引
        """
        
        self.max_joints = max_joints
        self.joint_configs = joint_configs or [
            {'num_joints': 2, 'link_lengths': [0.1, 0.1]},
            {'num_joints': 3, 'link_lengths': [0.1, 0.1, 0.1]},
            {'num_joints': 4, 'link_lengths': [0.1, 0.1, 0.1, 0.1]}
        ]
        self.current_config_idx = current_config_idx
        self.current_config = self.joint_configs[current_config_idx]
        
        print(f"🌟 VariableJointReacherWrapper 初始化:")
        print(f"   最大关节数 (J_max): {max_joints}")
        print(f"   关节配置数量: {len(self.joint_configs)}")
        print(f"   当前配置: {self.current_config}")
        
        # 创建当前配置的环境
        self._create_current_env()
        
        # 初始化 wrapper
        super(VariableJointReacherWrapper, self).__init__(self.current_env)
        
        # 设置统一的观察和动作空间 (基于 J_max)
        obs_dim = 4 * max_joints + 6  # joint_features×J_max + global_features
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(max_joints,), dtype=np.float32
        )
        
        print(f"✅ 统一空间设置完成:")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def _create_current_env(self):
        """创建当前配置的环境"""
        config = self.current_config
        self.current_env = RealMultiJointWrapper(
            num_joints=config['num_joints'],
            link_lengths=config['link_lengths'],
            render_mode=None
        )
        self.current_num_joints = config['num_joints']
        self.current_link_lengths = config['link_lengths']
    
    def switch_config(self, config_idx: int):
        """切换到不同的关节配置"""
        if 0 <= config_idx < len(self.joint_configs):
            print(f"🔄 切换关节配置: {self.joint_configs[config_idx]}")
            
            # 关闭当前环境
            if hasattr(self, 'current_env'):
                self.current_env.close()
            
            # 更新配置
            self.current_config_idx = config_idx
            self.current_config = self.joint_configs[config_idx]
            
            # 创建新环境
            self._create_current_env()
            
            # 更新 wrapper 的环境
            self.env = self.current_env
        else:
            raise ValueError(f"Invalid config_idx: {config_idx}")
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.current_env.reset(**kwargs)
        
        # 填充观察到 J_max
        padded_obs = VariableJointMaskSystem.pad_observation(
            obs, self.current_num_joints, self.max_joints, self.current_link_lengths
        )
        
        # 添加关节信息
        info['num_joints'] = self.current_num_joints
        info['max_joints'] = self.max_joints
        info['link_lengths'] = self.current_link_lengths
        info['joint_mask'] = VariableJointMaskSystem.create_joint_mask(
            self.current_num_joints, self.max_joints
        ).numpy()
        
        return padded_obs, info
    
    def step(self, action):
        """执行动作"""
        # 从 J_max 动作中提取实际动作
        real_action = VariableJointMaskSystem.unpad_action(action, self.current_num_joints)
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.current_env.step(real_action)
        
        # 填充观察到 J_max
        padded_obs = VariableJointMaskSystem.pad_observation(
            obs, self.current_num_joints, self.max_joints, self.current_link_lengths
        )
        
        # 添加关节信息
        info['num_joints'] = self.current_num_joints
        info['max_joints'] = self.max_joints
        info['link_lengths'] = self.current_link_lengths
        info['joint_mask'] = VariableJointMaskSystem.create_joint_mask(
            self.current_num_joints, self.max_joints
        ).numpy()
        
        return padded_obs, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'current_env'):
            self.current_env.close()

# ============================================================================
# 🧩 可变关节通用特征提取器
# ============================================================================

class VariableJointUniversalExtractor(BaseFeaturesExtractor):
    """可变关节通用特征提取器，支持 J_max 策略"""
    
    def __init__(self, observation_space: gym.Space, 
                 max_joints: int = 4,
                 joint_hidden_dim: int = 64,
                 pooled_dim: int = 128,
                 global_hidden_dim: int = 64,
                 features_dim: int = 256):
        
        super(VariableJointUniversalExtractor, self).__init__(observation_space, features_dim)
        
        self.max_joints = max_joints
        self.joint_hidden_dim = joint_hidden_dim
        self.pooled_dim = pooled_dim
        self.global_hidden_dim = global_hidden_dim
        
        print(f"🔧 VariableJointUniversalExtractor 初始化:")
        print(f"   最大关节数: {max_joints}")
        print(f"   观察空间: {observation_space}")
        print(f"   特征维度: {features_dim}")
        
        # 关节编码器 (处理 [cos, sin, vel, link_length])
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim),
            nn.ReLU()
        )
        
        # 多头自注意力 (关节间交互)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=joint_hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 注意力池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(joint_hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 全局特征处理器
        self.global_processor = nn.Sequential(
            nn.Linear(6, global_hidden_dim),  # [ee_pos, target_pos, target_vec]
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.ReLU()
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(joint_hidden_dim + global_hidden_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, obs_dim] (obs_dim = 4*max_joints + 6)
        Returns:
            features: [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # 解析观察
        joint_features_flat = observations[:, :4 * self.max_joints]  # [batch_size, 4*max_joints]
        joint_features = joint_features_flat.reshape(batch_size, self.max_joints, 4)  # [batch_size, max_joints, 4]
        
        global_features = observations[:, 4 * self.max_joints:]  # [batch_size, 6]
        
        # 关节特征编码
        encoded_joints = self.joint_encoder(joint_features)  # [batch_size, max_joints, joint_hidden_dim]
        
        # 多头自注意力 (关节间交互)
        attended_joints, _ = self.multihead_attention(
            query=encoded_joints,
            key=encoded_joints,
            value=encoded_joints
        )  # [batch_size, max_joints, joint_hidden_dim]
        
        # 注意力池化
        attention_weights = self.attention_pooling(attended_joints)  # [batch_size, max_joints, 1]
        pooled_joints = torch.sum(attended_joints * attention_weights, dim=1)  # [batch_size, joint_hidden_dim]
        
        # 全局特征处理
        processed_global = self.global_processor(global_features)  # [batch_size, global_hidden_dim]
        
        # 最终融合
        combined_features = torch.cat([pooled_joints, processed_global], dim=-1)
        final_features = self.final_fusion(combined_features)
        
        return final_features

# ============================================================================
# 🧩 训练函数
# ============================================================================

def train_variable_joint_sac(max_joints: int = 4,
                            joint_configs: List[Dict] = None,
                            total_timesteps: int = 50000,
                            config_switch_freq: int = 10000) -> Dict[str, Any]:
    """
    训练可变关节 SAC 模型
    
    Args:
        max_joints: 最大关节数 (J_max)
        joint_configs: 关节配置列表
        total_timesteps: 总训练步数
        config_switch_freq: 配置切换频率
    
    Returns:
        训练结果
    """
    
    if joint_configs is None:
        joint_configs = [
            {'num_joints': 2, 'link_lengths': [0.1, 0.1]},
            {'num_joints': 3, 'link_lengths': [0.1, 0.1, 0.1]},
            {'num_joints': 4, 'link_lengths': [0.1, 0.1, 0.1, 0.1]}
        ]
    
    print(f"\n{'='*70}")
    print(f"🚀 可变关节 Reacher SAC 训练")
    print(f"{'='*70}")
    print(f"📊 训练配置:")
    print(f"   最大关节数 (J_max): {max_joints}")
    print(f"   关节配置: {joint_configs}")
    print(f"   总训练步数: {total_timesteps}")
    print(f"   配置切换频率: {config_switch_freq}")
    
    # 创建可变关节环境
    print(f"\n🌍 创建可变关节环境...")
    base_env = VariableJointReacherWrapper(
        max_joints=max_joints,
        joint_configs=joint_configs,
        current_config_idx=0
    )
    env = Monitor(base_env)
    
    # 创建评估环境
    base_eval_env = VariableJointReacherWrapper(
        max_joints=max_joints,
        joint_configs=joint_configs,
        current_config_idx=0
    )
    eval_env = Monitor(base_eval_env)
    
    print(f"✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建 SAC 模型
    print(f"\n🤖 创建可变关节 SAC 模型...")
    
    policy_kwargs = {
        'features_extractor_class': VariableJointUniversalExtractor,
        'features_extractor_kwargs': {
            'max_joints': max_joints,
            'joint_hidden_dim': 64,
            'pooled_dim': 128,
            'global_hidden_dim': 64,
            'features_dim': 256
        },
        'net_arch': [512, 512]
    }
    
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        verbose=2,
        device='auto',
        tensorboard_log="./variable_joint_logs/"
    )
    
    print(f"✅ SAC 模型创建完成")
    print(f"   策略: MlpPolicy + VariableJointUniversalExtractor")
    print(f"   网络架构: [512, 512]")
    print(f"   缓冲区大小: 100,000")
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./logs/variable_joint_sac/',
        log_path=f'./logs/variable_joint_sac/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print(f"\n🎯 开始可变关节训练...")
    start_time = time.time()
    
    try:
        # 训练循环，定期切换关节配置
        steps_trained = 0
        config_idx = 0
        
        while steps_trained < total_timesteps:
            # 计算这次训练的步数
            steps_to_train = min(config_switch_freq, total_timesteps - steps_trained)
            
            print(f"\n🔄 训练阶段 {steps_trained//config_switch_freq + 1}:")
            print(f"   当前配置: {joint_configs[config_idx]}")
            print(f"   训练步数: {steps_to_train}")
            
            # 切换环境配置
            base_env.switch_config(config_idx)
            base_eval_env.switch_config(config_idx)
            
            # 训练
            model.learn(
                total_timesteps=steps_to_train,
                callback=eval_callback,
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            steps_trained += steps_to_train
            
            # 切换到下一个配置
            config_idx = (config_idx + 1) % len(joint_configs)
        
        training_time = time.time() - start_time
        print(f"✅ 可变关节训练完成！用时: {training_time:.1f} 秒")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"⚠️ 训练被中断！已训练时间: {training_time:.1f} 秒")
    
    # 评估所有配置
    print(f"\n📈 评估所有关节配置...")
    
    results = {
        'max_joints': max_joints,
        'joint_configs': joint_configs,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'config_results': []
    }
    
    for i, config in enumerate(joint_configs):
        print(f"\n🧪 评估配置 {i+1}: {config}")
        
        # 切换到当前配置
        base_eval_env.switch_config(i)
        
        try:
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True
            )
            
            print(f"   平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
            
            results['config_results'].append({
                'config': config,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            })
            
        except Exception as e:
            print(f"   评估失败: {e}")
            results['config_results'].append({
                'config': config,
                'error': str(e)
            })
    
    # 保存模型
    model_path = f"variable_joint_sac_j{max_joints}_model"
    model.save(model_path)
    print(f"\n💾 模型已保存: {model_path}")
    
    results['model_path'] = model_path
    
    env.close()
    eval_env.close()
    
    return results

# ============================================================================
# 🧩 主函数
# ============================================================================

def main():
    """主函数"""
    print("🌟 可变关节 Reacher SAC 训练系统")
    print("💡 真正意义上可以兼容可变关节数量训练的模型")
    print("🎯 基于 J_max 策略，支持动态关节数切换")
    print()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 训练配置
    joint_configs = [
        {'num_joints': 2, 'link_lengths': [0.1, 0.1]},
        {'num_joints': 3, 'link_lengths': [0.1, 0.1, 0.1]},
        {'num_joints': 4, 'link_lengths': [0.1, 0.1, 0.1, 0.1]}
    ]
    
    try:
        results = train_variable_joint_sac(
            max_joints=4,
            joint_configs=joint_configs,
            total_timesteps=30000,  # 减少训练步数以便测试
            config_switch_freq=5000  # 每5000步切换一次配置
        )
        
        # 打印结果
        print(f"\n{'='*70}")
        print(f"📊 可变关节训练结果")
        print(f"{'='*70}")
        
        for i, result in enumerate(results['config_results']):
            if 'error' in result:
                print(f"❌ 配置 {i+1}: {result['config']} - 失败: {result['error']}")
            else:
                print(f"✅ 配置 {i+1}: {result['config']}")
                print(f"   平均奖励: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        
        print(f"\n🎉 可变关节模型训练成功！")
        print(f"   训练时间: {results['training_time']:.1f} 秒")
        print(f"   模型路径: {results['model_path']}")
        
    except Exception as e:
        print(f"❌ 可变关节训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
