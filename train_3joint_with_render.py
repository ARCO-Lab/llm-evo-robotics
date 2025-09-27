#!/usr/bin/env python3
"""
3关节 Reacher 训练并显示渲染
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
from stable_baselines3.common.callbacks import BaseCallback

# 导入真实多关节环境
from real_multi_joint_reacher import RealMultiJointWrapper

# ============================================================================
# 🧩 3关节特征提取器
# ============================================================================

class ThreeJointExtractor(BaseFeaturesExtractor):
    """3关节特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(ThreeJointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        print(f"🔧 ThreeJointExtractor: {obs_dim} -> {features_dim}")
        
        # 针对3关节优化的网络结构
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# ============================================================================
# 🧩 渲染回调
# ============================================================================

class RenderCallback(BaseCallback):
    """渲染回调，在训练过程中显示进度"""
    
    def __init__(self, render_freq: int = 100, verbose: int = 0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # 每隔一定步数显示信息
        if self.step_count % self.render_freq == 0:
            if hasattr(self.locals, 'infos') and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                distance = info.get('distance_to_target', 'N/A')
                reward = self.locals.get('rewards', [0])[0]
                print(f"🎯 Step {self.step_count}: 距离={distance:.3f}, 奖励={reward:.3f}")
        
        return True

# ============================================================================
# 🧩 训练函数
# ============================================================================

def train_3joint_with_render(total_timesteps: int = 15000, 
                           model_name: str = "3joint_render_sac"):
    """训练3关节模型并显示渲染"""
    
    print(f"\n{'='*70}")
    print(f"🚀 3关节 Reacher 训练 (带渲染)")
    print(f"{'='*70}")
    
    # 强制启用渲染
    os.environ['MUJOCO_GL'] = 'glfw'
    
    # 创建3关节环境 (带渲染)
    print(f"🌍 创建3关节训练环境 (带渲染)...")
    env = RealMultiJointWrapper(
        num_joints=3,
        link_lengths=[0.1, 0.1, 0.1],
        render_mode='human'  # 训练时也渲染
    )
    env = Monitor(env)
    
    print(f"✅ 3关节环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建SAC模型 (针对3关节优化)
    print(f"🤖 创建3关节优化SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': ThreeJointExtractor,
        'features_extractor_kwargs': {
            'features_dim': 128
        },
        'net_arch': [256, 256, 128]  # 更深的网络
    }
    
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,          # 较小的学习率
        buffer_size=100000,          # 更大的缓冲区
        learning_starts=1000,        # 更多的预热步数
        batch_size=256,              # 更大的批次
        tau=0.005,                   # 软更新系数
        gamma=0.99,                  # 折扣因子
        ent_coef='auto',             # 自动熵系数
        verbose=2,                   # 详细输出
        device='cpu'
    )
    
    print(f"✅ SAC模型创建完成")
    print(f"📊 模型配置:")
    print(f"   学习率: 1e-4")
    print(f"   缓冲区: 100,000")
    print(f"   批次大小: 256")
    print(f"   网络结构: [256, 256, 128]")
    
    # 创建渲染回调
    render_callback = RenderCallback(render_freq=200, verbose=1)
    
    # 开始训练
    print(f"\n🎯 开始3关节训练 ({total_timesteps} 步)...")
    print(f"💡 请观察MuJoCo窗口中的3关节机器人训练过程")
    print(f"🎮 窗口标题: MuJoCo")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=render_callback,
            log_interval=2,              # 更频繁的日志
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 3关节训练完成！")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        
        # 保存模型
        model.save(model_name)
        print(f"💾 模型已保存: {model_name}")
        
        # 立即测试训练好的模型
        print(f"\n🎮 立即测试训练好的3关节模型...")
        test_trained_3joint_model(model, env)
        
        return model, training_time
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        training_time = time.time() - start_time
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        # 保存中断时的模型
        model.save(f"{model_name}_interrupted")
        print(f"💾 中断模型已保存: {model_name}_interrupted")
        
        return model, training_time
    
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return None, 0
    
    finally:
        env.close()

# ============================================================================
# 🧩 测试函数
# ============================================================================

def test_trained_3joint_model(model, env, num_episodes: int = 3):
    """测试训练好的3关节模型"""
    
    print(f"\n{'='*50}")
    print(f"🎮 测试训练好的3关节模型")
    print(f"{'='*50}")
    
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
            distance = info.get('distance_to_target', float('inf'))
            min_distance = min(min_distance, distance)
            
            # 每10步显示一次信息
            if episode_length % 10 == 0:
                print(f"   Step {episode_length}: 距离={distance:.3f}, 奖励={reward:.3f}")
            
            # 稍微慢一点让您观察
            time.sleep(0.1)
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 判断成功
        success = min_distance < 0.05 or episode_reward > -5
        if success:
            success_count += 1
            print(f"   ✅ 成功! 奖励={episode_reward:.3f}, 最小距离={min_distance:.3f}")
        else:
            print(f"   ❌ 失败. 奖励={episode_reward:.3f}, 最小距离={min_distance:.3f}")
    
    # 统计结果
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    print(f"\n📊 3关节测试结果:")
    print(f"   成功率: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"   平均奖励: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
    print(f"   平均长度: {np.mean(episode_lengths):.1f}")
    
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
    print("🌟 3关节 Reacher 训练渲染演示")
    print("💡 训练过程中将显示MuJoCo渲染窗口")
    print("🎯 3关节机器人比2关节更复杂，需要更长时间训练")
    print()
    
    try:
        # 训练3关节模型
        model, training_time = train_3joint_with_render(
            total_timesteps=15000,  # 增加训练步数
            model_name="models/3joint_render_sac"
        )
        
        if model is not None:
            print(f"\n🎉 3关节训练演示完成！")
            print(f"⏱️ 总用时: {training_time/60:.1f} 分钟")
        else:
            print(f"\n❌ 3关节训练失败")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 演示被用户中断")
    
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


