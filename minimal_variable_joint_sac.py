#!/usr/bin/env python3
"""
最简化可变关节 SAC 训练
完全避免卡住问题：
1. 不使用复杂的环境切换
2. 不使用 evaluate_policy
3. 使用最简单的特征提取器
4. 最小化资源使用
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入真实多关节环境
from real_multi_joint_reacher import RealMultiJointWrapper

# ============================================================================
# 🧩 最简化特征提取器
# ============================================================================

class MinimalVariableJointExtractor(BaseFeaturesExtractor):
    """最简化可变关节特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super(MinimalVariableJointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        print(f"🔧 MinimalVariableJointExtractor: {obs_dim} -> {features_dim}")
        
        # 最简单的全连接网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# ============================================================================
# 🧩 最简化训练函数
# ============================================================================

def train_minimal_variable_joint_sac(num_joints: int = 2, 
                                   total_timesteps: int = 5000) -> Dict:
    """最简化可变关节 SAC 训练"""
    
    print(f"\n{'='*50}")
    print(f"🚀 最简化 {num_joints} 关节 SAC 训练")
    print(f"{'='*50}")
    
    # 创建单一环境 (不切换)
    print(f"🌍 创建 {num_joints} 关节环境...")
    env = RealMultiJointWrapper(
        num_joints=num_joints,
        link_lengths=[0.1] * num_joints,
        render_mode=None
    )
    env = Monitor(env)
    
    print(f"✅ 环境创建完成: {env.observation_space}")
    
    # 创建最简化 SAC 模型
    print(f"🤖 创建最简化 SAC 模型...")
    
    policy_kwargs = {
        'features_extractor_class': MinimalVariableJointExtractor,
        'features_extractor_kwargs': {
            'features_dim': 64
        },
        'net_arch': [128, 128]  # 最小网络
    }
    
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=10000,      # 最小缓冲区
        learning_starts=200,    # 很早开始
        batch_size=64,          # 最小批次
        verbose=2,
        device='cpu'            # 强制使用 CPU 避免 GPU 问题
    )
    
    print(f"✅ 模型创建完成")
    
    # 开始训练
    print(f"🎯 开始训练 {total_timesteps} 步...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,          # 每4个episodes输出一次训练统计表格
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"✅ 训练完成！用时: {training_time:.1f} 秒")
        
        # 最简化评估 (不使用 evaluate_policy)
        print(f"📈 简单测试...")
        
        total_reward = 0
        num_episodes = 3
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 50:  # 限制步数避免卡住
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            total_reward += episode_reward
            print(f"   Episode {episode+1}: {episode_reward:.3f} ({step_count} steps)")
        
        avg_reward = total_reward / num_episodes
        print(f"📊 平均奖励: {avg_reward:.3f}")
        
        # 保存模型
        model_path = f"minimal_{num_joints}joint_sac"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        results = {
            'num_joints': num_joints,
            'training_time': training_time,
            'avg_reward': avg_reward,
            'model_path': model_path,
            'success': True
        }
        
        return results
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"❌ 训练失败: {e}")
        return {
            'num_joints': num_joints,
            'training_time': training_time,
            'error': str(e),
            'success': False
        }
    
    finally:
        env.close()
        print(f"🔒 环境已关闭")

# ============================================================================
# 🧩 主函数
# ============================================================================

def main():
    """主函数：逐个测试不同关节数"""
    
    print("🌟 最简化可变关节 SAC 训练系统")
    print("💡 完全避免卡住问题")
    print()
    
    # 逐个测试不同关节数
    joint_configs = [2, 3]  # 先测试 2 和 3 关节
    results = []
    
    for num_joints in joint_configs:
        print(f"\n{'='*60}")
        print(f"🧪 测试 {num_joints} 关节配置")
        print(f"{'='*60}")
        
        try:
            result = train_minimal_variable_joint_sac(
                num_joints=num_joints,
                total_timesteps=3000  # 减少到 3000 步
            )
            results.append(result)
            
            if result['success']:
                print(f"✅ {num_joints} 关节训练成功")
                print(f"   平均奖励: {result['avg_reward']:.3f}")
                print(f"   训练时间: {result['training_time']:.1f} 秒")
            else:
                print(f"❌ {num_joints} 关节训练失败: {result['error']}")
                
        except KeyboardInterrupt:
            print(f"⚠️ {num_joints} 关节训练被中断")
            results.append({
                'num_joints': num_joints,
                'interrupted': True
            })
            break  # 如果用户中断，停止后续测试
            
        except Exception as e:
            print(f"❌ {num_joints} 关节训练异常: {e}")
            results.append({
                'num_joints': num_joints,
                'error': str(e),
                'success': False
            })
    
    # 总结结果
    print(f"\n{'='*60}")
    print(f"📊 最简化训练总结")
    print(f"{'='*60}")
    
    for result in results:
        if 'interrupted' in result:
            print(f"⚠️ {result['num_joints']} 关节: 被中断")
        elif result.get('success', False):
            print(f"✅ {result['num_joints']} 关节: 成功")
            print(f"   奖励: {result['avg_reward']:.3f}")
            print(f"   时间: {result['training_time']:.1f}s")
        else:
            print(f"❌ {result['num_joints']} 关节: 失败")
    
    print(f"\n🎉 测试完成！")

if __name__ == "__main__":
    main()
