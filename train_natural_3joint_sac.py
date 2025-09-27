#!/usr/bin/env python3
"""
使用自然3关节Reacher环境训练SAC
基于标准MuJoCo Reacher，渲染效果更好
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 导入自然3关节环境
from natural_3joint_reacher import Natural3JointReacherEnv

# ============================================================================
# 🧩 自然3关节特征提取器
# ============================================================================

class Natural3JointExtractor(BaseFeaturesExtractor):
    """针对自然3关节Reacher优化的特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(Natural3JointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]  # 15维
        print(f"🔧 Natural3JointExtractor: {obs_dim} -> {features_dim}")
        
        # 针对15维观察空间的网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# ============================================================================
# 🧩 训练函数
# ============================================================================

def train_natural_3joint_sac(total_timesteps: int = 20000, 
                            model_name: str = "natural_3joint_sac",
                            render_during_training: bool = True):
    """训练自然3关节SAC模型"""
    
    print(f"\n{'='*70}")
    print(f"🚀 自然3关节 Reacher SAC 训练")
    print(f"{'='*70}")
    
    # 创建训练环境
    print(f"🌍 创建自然3关节训练环境...")
    env = Natural3JointReacherEnv(
        render_mode='human' if render_during_training else None
    )
    env = Monitor(env)
    
    print(f"✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print(f"   渲染模式: {'启用' if render_during_training else '禁用'}")
    
    # 创建SAC模型
    print(f"🤖 创建SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': Natural3JointExtractor,
        'features_extractor_kwargs': {
            'features_dim': 128
        },
        'net_arch': [256, 256]
    }
    
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=500,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        verbose=2,
        device='cpu'
    )
    
    print(f"✅ SAC模型创建完成")
    
    # 开始训练
    print(f"\n🎯 开始训练 ({total_timesteps} 步)...")
    if render_during_training:
        print(f"💡 请观察MuJoCo窗口中的3关节机器人训练过程")
        print(f"🎮 场地大小已优化，机械臂不会伸出边界")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 训练完成！")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        
        # 保存模型
        model.save(model_name)
        print(f"💾 模型已保存: {model_name}")
        
        # 立即测试
        print(f"\n🎮 立即测试训练好的模型...")
        test_results = test_natural_3joint_model(model, env)
        
        return model, training_time, test_results
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        training_time = time.time() - start_time
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        # 保存中断的模型
        model.save(f"{model_name}_interrupted")
        print(f"💾 中断模型已保存: {model_name}_interrupted")
        
        return model, training_time, None
    
    finally:
        env.close()

# ============================================================================
# 🧩 测试函数
# ============================================================================

def test_natural_3joint_model(model, env=None, num_episodes: int = 5):
    """测试自然3关节模型"""
    
    print(f"\n{'='*50}")
    print(f"🎮 测试自然3关节模型")
    print(f"{'='*50}")
    
    # 如果没有提供环境，创建一个新的
    if env is None:
        env = Natural3JointReacherEnv(render_mode='human')
        env = Monitor(env)
        should_close = True
    else:
        should_close = False
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        min_distance = float('inf')
        
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        while True:
            # 使用训练好的策略
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 记录最小距离
            distance = info['distance_to_target']
            min_distance = min(min_distance, distance)
            
            # 每10步显示一次信息
            if episode_length % 10 == 0:
                print(f"   Step {episode_length}: 距离={distance:.3f}, 奖励={reward:.3f}")
            
            # 稍微慢一点让您观察
            time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 判断成功 (距离小于2cm)
        success = min_distance < 0.02
        if success:
            success_count += 1
            print(f"   ✅ 成功! 奖励={episode_reward:.3f}, 最小距离={min_distance:.3f}")
        else:
            print(f"   ❌ 失败. 奖励={episode_reward:.3f}, 最小距离={min_distance:.3f}")
    
    # 统计结果
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    print(f"\n📊 自然3关节测试结果:")
    print(f"   成功率: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"   平均奖励: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
    print(f"   平均长度: {np.mean(episode_lengths):.1f}")
    
    if should_close:
        env.close()
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'episode_rewards': episode_rewards
    }

# ============================================================================
# 🧩 主函数
# ============================================================================

def main():
    """主函数"""
    print("🌟 自然3关节 Reacher SAC 训练系统")
    print("💡 基于标准MuJoCo Reacher，优化的场地和渲染效果")
    print("🎯 3关节机械臂，总长度约24cm，适合的工作空间")
    print()
    
    try:
        # 训练模型
        model, training_time, test_results = train_natural_3joint_sac(
            total_timesteps=20000,
            model_name="models/natural_3joint_sac",
            render_during_training=True  # 训练时显示渲染
        )
        
        print(f"\n🎉 自然3关节训练完成！")
        print(f"⏱️ 总用时: {training_time/60:.1f} 分钟")
        
        if test_results:
            print(f"📊 测试结果: 成功率 {test_results['success_rate']:.1%}")
            print(f"           平均奖励 {test_results['avg_reward']:.3f}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
    
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 🧩 独立测试函数
# ============================================================================

def test_saved_natural_3joint_model():
    """测试已保存的自然3关节模型"""
    print("🎮 测试已保存的自然3关节模型")
    
    model_path = "models/natural_3joint_sac"
    
    if os.path.exists(f"{model_path}.zip"):
        try:
            # 创建环境
            env = Natural3JointReacherEnv(render_mode='human')
            
            # 加载模型
            model = SAC.load(model_path, env=env)
            print("✅ 模型加载成功")
            
            # 测试模型
            results = test_natural_3joint_model(model, env, num_episodes=3)
            
            print(f"✅ 测试完成")
            print(f"   成功率: {results['success_rate']:.1%}")
            print(f"   平均奖励: {results['avg_reward']:.3f}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    else:
        print(f"⚠️ 模型文件不存在: {model_path}.zip")
        print("   请先运行训练: python train_natural_3joint_sac.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 仅测试已保存的模型
        test_saved_natural_3joint_model()
    else:
        # 完整的训练流程
        main()


