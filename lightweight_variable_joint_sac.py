#!/usr/bin/env python3
"""
轻量级可变关节 SAC 训练系统
解决训练卡住的问题，简化注意力机制和环境切换
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
# 🧩 轻量级可变关节特征提取器
# ============================================================================

class LightweightVariableJointExtractor(BaseFeaturesExtractor):
    """轻量级可变关节特征提取器，避免复杂的注意力机制"""
    
    def __init__(self, observation_space: gym.Space, 
                 max_joints: int = 4,
                 joint_hidden_dim: int = 32,
                 global_hidden_dim: int = 32,
                 features_dim: int = 128):
        
        super(LightweightVariableJointExtractor, self).__init__(observation_space, features_dim)
        
        self.max_joints = max_joints
        self.joint_hidden_dim = joint_hidden_dim
        self.global_hidden_dim = global_hidden_dim
        
        print(f"🔧 LightweightVariableJointExtractor 初始化:")
        print(f"   最大关节数: {max_joints}")
        print(f"   观察空间: {observation_space}")
        print(f"   特征维度: {features_dim}")
        
        # 简化的关节编码器
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, joint_hidden_dim),  # [cos, sin, vel, link_length]
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, joint_hidden_dim)
        )
        
        # 简单的池化层 (避免复杂的注意力机制)
        self.joint_pooling = nn.Sequential(
            nn.Linear(joint_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 全局特征处理器
        self.global_processor = nn.Sequential(
            nn.Linear(6, global_hidden_dim),  # [ee_pos, target_pos, target_vec]
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim)
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
        
        # 简单加权池化 (避免复杂的多头注意力)
        joint_weights = self.joint_pooling(encoded_joints)  # [batch_size, max_joints, 1]
        joint_weights = F.softmax(joint_weights.squeeze(-1), dim=1)  # [batch_size, max_joints]
        
        pooled_joints = torch.sum(encoded_joints * joint_weights.unsqueeze(-1), dim=1)  # [batch_size, joint_hidden_dim]
        
        # 全局特征处理
        processed_global = self.global_processor(global_features)  # [batch_size, global_hidden_dim]
        
        # 最终融合
        combined_features = torch.cat([pooled_joints, processed_global], dim=-1)
        final_features = self.final_fusion(combined_features)
        
        return final_features

# ============================================================================
# 🧩 简化的可变关节环境包装器
# ============================================================================

class SimpleVariableJointWrapper(gym.Wrapper):
    """简化的可变关节环境包装器，减少环境切换开销"""
    
    def __init__(self, joint_configs: List[Dict], max_joints: int = 4):
        """
        Args:
            joint_configs: 关节配置列表
            max_joints: 最大关节数
        """
        
        self.joint_configs = joint_configs
        self.max_joints = max_joints
        self.current_config_idx = 0
        self.current_config = joint_configs[0]
        
        print(f"🌟 SimpleVariableJointWrapper 初始化:")
        print(f"   最大关节数: {max_joints}")
        print(f"   关节配置: {joint_configs}")
        
        # 创建当前环境
        self._create_current_env()
        
        # 初始化 wrapper
        super(SimpleVariableJointWrapper, self).__init__(self.current_env)
        
        # 设置统一的观察和动作空间
        obs_dim = 4 * max_joints + 6
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(max_joints,), dtype=np.float32
        )
        
        print(f"✅ 统一空间设置:")
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
    
    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """填充观察到 J_max 维度"""
        # 解析原始观察
        joint_features_flat = obs[:4 * self.current_num_joints]
        global_features = obs[4 * self.current_num_joints:]
        
        # 重塑关节特征
        joint_features = joint_features_flat.reshape(self.current_num_joints, 4)
        
        # 创建填充后的关节特征
        padded_joint_features = np.zeros((self.max_joints, 4), dtype=np.float32)
        padded_joint_features[:self.current_num_joints] = joint_features
        
        # 填充默认值
        for i in range(self.current_num_joints, self.max_joints):
            padded_joint_features[i] = [1.0, 0.0, 0.0, 0.05]
        
        # 重新组合
        padded_obs = np.concatenate([
            padded_joint_features.flatten(),
            global_features
        ])
        
        return padded_obs
    
    def _unpad_action(self, action: np.ndarray) -> np.ndarray:
        """从填充的动作中提取实际动作"""
        return action[:self.current_num_joints]
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.current_env.reset(**kwargs)
        padded_obs = self._pad_observation(obs)
        
        info['num_joints'] = self.current_num_joints
        info['max_joints'] = self.max_joints
        
        return padded_obs, info
    
    def step(self, action):
        """执行动作"""
        real_action = self._unpad_action(action)
        obs, reward, terminated, truncated, info = self.current_env.step(real_action)
        padded_obs = self._pad_observation(obs)
        
        info['num_joints'] = self.current_num_joints
        info['max_joints'] = self.max_joints
        
        return padded_obs, reward, terminated, truncated, info
    
    def switch_config(self, config_idx: int):
        """切换配置 (简化版，避免频繁创建环境)"""
        if 0 <= config_idx < len(self.joint_configs):
            new_config = self.joint_configs[config_idx]
            
            # 只有当配置真的改变时才切换
            if new_config != self.current_config:
                print(f"🔄 切换关节配置: {new_config}")
                
                # 关闭当前环境
                if hasattr(self, 'current_env'):
                    self.current_env.close()
                
                # 更新配置
                self.current_config_idx = config_idx
                self.current_config = new_config
                
                # 创建新环境
                self._create_current_env()
                self.env = self.current_env
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'current_env'):
            self.current_env.close()

# ============================================================================
# 🧩 轻量级训练函数
# ============================================================================

def train_lightweight_variable_joint_sac(max_joints: int = 4,
                                        joint_configs: List[Dict] = None,
                                        total_timesteps: int = 15000) -> Dict[str, Any]:
    """轻量级可变关节 SAC 训练"""
    
    if joint_configs is None:
        joint_configs = [
            {'num_joints': 2, 'link_lengths': [0.1, 0.1]},
            {'num_joints': 3, 'link_lengths': [0.1, 0.1, 0.1]},
        ]
    
    print(f"\n{'='*60}")
    print(f"🚀 轻量级可变关节 SAC 训练")
    print(f"{'='*60}")
    print(f"📊 配置: J_max={max_joints}, 步数={total_timesteps}")
    
    # 创建环境
    print(f"\n🌍 创建轻量级环境...")
    base_env = SimpleVariableJointWrapper(
        joint_configs=joint_configs,
        max_joints=max_joints
    )
    env = Monitor(base_env)
    
    # 创建 SAC 模型 (简化配置)
    print(f"\n🤖 创建轻量级 SAC 模型...")
    
    policy_kwargs = {
        'features_extractor_class': LightweightVariableJointExtractor,
        'features_extractor_kwargs': {
            'max_joints': max_joints,
            'joint_hidden_dim': 32,
            'global_hidden_dim': 32,
            'features_dim': 128
        },
        'net_arch': [256, 256]  # 简化网络
    }
    
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=50000,      # 减小缓冲区
        learning_starts=500,    # 更早开始学习
        batch_size=128,         # 减小批次大小
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        verbose=2,
        device='cuda'
    )
    
    print(f"✅ 模型创建完成")
    
    # 开始训练
    print(f"\n🎯 开始轻量级训练...")
    start_time = time.time()
    
    try:
        # 简化训练：每个配置训练一段时间
        steps_per_config = total_timesteps // len(joint_configs)
        
        for i, config in enumerate(joint_configs):
            print(f"\n📍 训练配置 {i+1}/{len(joint_configs)}: {config}")
            
            # 切换配置
            base_env.switch_config(i)
            
            # 训练
            model.learn(
                total_timesteps=steps_per_config,
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            print(f"✅ 配置 {i+1} 训练完成")
        
        training_time = time.time() - start_time
        print(f"✅ 轻量级训练完成！用时: {training_time:.1f} 秒")
        
        # 简单评估
        print(f"\n📈 快速评估...")
        results = {'training_time': training_time, 'configs': []}
        
        for i, config in enumerate(joint_configs):
            base_env.switch_config(i)
            
            try:
                mean_reward, std_reward = evaluate_policy(
                    model, env, n_eval_episodes=3, deterministic=True
                )
                
                print(f"   配置 {i+1}: {mean_reward:.3f} ± {std_reward:.3f}")
                results['configs'].append({
                    'config': config,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward
                })
                
            except Exception as e:
                print(f"   配置 {i+1}: 评估失败 - {e}")
                results['configs'].append({
                    'config': config,
                    'error': str(e)
                })
        
        # 保存模型
        model_path = f"lightweight_variable_joint_sac_j{max_joints}"
        model.save(model_path)
        results['model_path'] = model_path
        
        return results
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"⚠️ 训练被中断！用时: {training_time:.1f} 秒")
        return {'training_time': training_time, 'interrupted': True}
    
    finally:
        env.close()

# ============================================================================
# 🧩 主函数
# ============================================================================

def main():
    """主函数"""
    print("🌟 轻量级可变关节 SAC 训练系统")
    print("💡 解决训练卡住问题，简化架构")
    print()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 简化配置
    joint_configs = [
        {'num_joints': 2, 'link_lengths': [0.1, 0.1]},
        {'num_joints': 3, 'link_lengths': [0.1, 0.1, 0.1]},
    ]
    
    try:
        results = train_lightweight_variable_joint_sac(
            max_joints=4,
            joint_configs=joint_configs,
            total_timesteps=10000  # 减少训练步数
        )
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"📊 轻量级训练结果")
        print(f"{'='*60}")
        
        if 'interrupted' in results:
            print(f"⚠️ 训练被中断，用时: {results['training_time']:.1f} 秒")
        else:
            for i, result in enumerate(results.get('configs', [])):
                if 'error' in result:
                    print(f"❌ 配置 {i+1}: {result['config']} - 失败")
                else:
                    print(f"✅ 配置 {i+1}: {result['config']}")
                    print(f"   平均奖励: {result['mean_reward']:.3f}")
            
            print(f"\n🎉 轻量级训练成功！")
            print(f"   训练时间: {results['training_time']:.1f} 秒")
            print(f"   模型路径: {results.get('model_path', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 轻量级训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


