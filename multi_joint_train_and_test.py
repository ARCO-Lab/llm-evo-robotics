#!/usr/bin/env python3
"""
多关节 Reacher 训练和测试系统
- 训练时不渲染（提高速度）
- 测试时显示渲染结果
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
# 🧩 多关节特征提取器
# ============================================================================

class MultiJointExtractor(BaseFeaturesExtractor):
    """多关节特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(MultiJointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        print(f"🔧 MultiJointExtractor: {obs_dim} -> {features_dim}")
        
        # 特征提取网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
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

def train_multi_joint_sac(num_joints: int = 2, 
                         total_timesteps: int = 10000,
                         model_name: str = None) -> Dict:
    """训练多关节 SAC 模型"""
    
    if model_name is None:
        model_name = f"multi_joint_{num_joints}j_sac"
    
    print(f"\n{'='*60}")
    print(f"🚀 训练 {num_joints} 关节 Reacher SAC")
    print(f"{'='*60}")
    
    # 创建训练环境 (不渲染)
    print(f"🌍 创建 {num_joints} 关节训练环境...")
    env = RealMultiJointWrapper(
        num_joints=num_joints,
        link_lengths=[0.1] * num_joints,
        render_mode=None  # 训练时不渲染
    )
    env = Monitor(env)
    
    print(f"✅ 训练环境创建完成: {env.observation_space}")
    
    # 创建 SAC 模型
    print(f"🤖 创建 SAC 模型...")
    
    policy_kwargs = {
        'features_extractor_class': MultiJointExtractor,
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
        verbose=2,
        device='cpu'
    )
    
    print(f"✅ SAC 模型创建完成")
    
    # 开始训练
    print(f"🎯 开始训练 {total_timesteps} 步...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"✅ 训练完成！用时: {training_time:.1f} 秒")
        
        # 保存模型
        model.save(model_name)
        print(f"💾 模型已保存: {model_name}")
        
        results = {
            'num_joints': num_joints,
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'model_path': model_name,
            'success': True
        }
        
        return results, model
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"❌ 训练失败: {e}")
        return {
            'num_joints': num_joints,
            'training_time': training_time,
            'error': str(e),
            'success': False
        }, None
    
    finally:
        env.close()

# ============================================================================
# 🧩 测试函数 (带渲染)
# ============================================================================

def test_multi_joint_sac(model, num_joints: int = 2, 
                        num_episodes: int = 5,
                        render: bool = True) -> Dict:
    """测试多关节 SAC 模型 (带渲染)"""
    
    print(f"\n{'='*60}")
    print(f"🎮 测试 {num_joints} 关节 Reacher SAC")
    print(f"{'='*60}")
    
    # 创建测试环境 (带渲染)
    print(f"🌍 创建 {num_joints} 关节测试环境...")
    test_env = RealMultiJointWrapper(
        num_joints=num_joints,
        link_lengths=[0.1] * num_joints,
        render_mode='human' if render else None  # 测试时渲染
    )
    
    print(f"✅ 测试环境创建完成")
    print(f"🎯 开始测试 {num_episodes} 个 episodes...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_length = 0
            min_distance = float('inf')
            
            print(f"\n📍 Episode {episode + 1}/{num_episodes}")
            
            while True:
                # 使用训练好的模型预测动作
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # 记录最小距离
                distance = info.get('distance_to_target', float('inf'))
                min_distance = min(min_distance, distance)
                
                # 显示实时信息
                if episode_length % 10 == 0:
                    print(f"   Step {episode_length}: 奖励={reward:.3f}, 距离={distance:.3f}")
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 判断成功 (距离小于5cm或奖励足够高)
            success = min_distance < 0.05 or episode_reward > -5
            if success:
                success_count += 1
                print(f"   ✅ 成功! 奖励={episode_reward:.3f}, 长度={episode_length}, 最小距离={min_distance:.3f}")
            else:
                print(f"   ❌ 失败. 奖励={episode_reward:.3f}, 长度={episode_length}, 最小距离={min_distance:.3f}")
        
        # 统计结果
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = success_count / num_episodes
        
        print(f"\n📊 测试结果统计:")
        print(f"   成功率: {success_rate:.1%} ({success_count}/{num_episodes})")
        print(f"   平均奖励: {avg_reward:.3f} ± {np.std(episode_rewards):.3f}")
        print(f"   平均长度: {avg_length:.1f}")
        
        results = {
            'num_joints': num_joints,
            'num_episodes': num_episodes,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return {'error': str(e)}
    
    finally:
        test_env.close()
        print(f"🔒 测试环境已关闭")

# ============================================================================
# 🧩 加载已保存的模型进行测试
# ============================================================================

def load_and_test_model(model_path: str, num_joints: int = 2, 
                       num_episodes: int = 5, render: bool = True):
    """加载已保存的模型并测试"""
    
    print(f"\n{'='*60}")
    print(f"📂 加载模型进行测试")
    print(f"{'='*60}")
    
    try:
        # 创建临时环境以获取观察空间
        temp_env = RealMultiJointWrapper(
            num_joints=num_joints,
            link_lengths=[0.1] * num_joints,
            render_mode=None
        )
        
        # 加载模型
        print(f"📂 加载模型: {model_path}")
        model = SAC.load(model_path, env=temp_env)
        print(f"✅ 模型加载成功")
        
        temp_env.close()
        
        # 测试模型
        results = test_multi_joint_sac(
            model=model,
            num_joints=num_joints,
            num_episodes=num_episodes,
            render=render
        )
        
        return results
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return {'error': str(e)}

# ============================================================================
# 🧩 主函数
# ============================================================================

def main():
    """主函数：训练和测试多关节模型"""
    
    print("🌟 多关节 Reacher 训练和测试系统")
    print("💡 训练时不渲染，测试时显示渲染结果")
    print()
    
    # 创建日志目录
    os.makedirs('models', exist_ok=True)
    
    # 配置
    joint_configs = [
        {'num_joints': 2, 'timesteps': 8000, 'test_episodes': 3},
        {'num_joints': 3, 'timesteps': 10000, 'test_episodes': 3},
    ]
    
    all_results = []
    
    for config in joint_configs:
        num_joints = config['num_joints']
        timesteps = config['timesteps']
        test_episodes = config['test_episodes']
        
        print(f"\n{'='*80}")
        print(f"🧪 处理 {num_joints} 关节配置")
        print(f"{'='*80}")
        
        model_name = f"models/multi_joint_{num_joints}j_sac"
        
        try:
            # 1. 训练模型
            print(f"🎯 第1步: 训练 {num_joints} 关节模型")
            train_results, trained_model = train_multi_joint_sac(
                num_joints=num_joints,
                total_timesteps=timesteps,
                model_name=model_name
            )
            
            if not train_results.get('success', False):
                print(f"❌ {num_joints} 关节训练失败，跳过测试")
                continue
            
            # 2. 测试模型 (带渲染)
            print(f"\n🎮 第2步: 测试 {num_joints} 关节模型 (带渲染)")
            test_results = test_multi_joint_sac(
                model=trained_model,
                num_joints=num_joints,
                num_episodes=test_episodes,
                render=True
            )
            
            # 3. 合并结果
            combined_results = {
                'num_joints': num_joints,
                'train_results': train_results,
                'test_results': test_results
            }
            all_results.append(combined_results)
            
            print(f"✅ {num_joints} 关节完成")
            
        except KeyboardInterrupt:
            print(f"⚠️ {num_joints} 关节被用户中断")
            break
            
        except Exception as e:
            print(f"❌ {num_joints} 关节处理失败: {e}")
            continue
    
    # 总结所有结果
    print(f"\n{'='*80}")
    print(f"📊 多关节训练和测试总结")
    print(f"{'='*80}")
    
    for result in all_results:
        num_joints = result['num_joints']
        train_res = result['train_results']
        test_res = result['test_results']
        
        print(f"\n🔧 {num_joints} 关节:")
        
        if train_res.get('success', False):
            print(f"   ✅ 训练: 成功 ({train_res['training_time']:.1f}s)")
        else:
            print(f"   ❌ 训练: 失败")
        
        if 'error' not in test_res:
            print(f"   🎮 测试: 成功率 {test_res['success_rate']:.1%}")
            print(f"        平均奖励 {test_res['avg_reward']:.3f}")
        else:
            print(f"   ❌ 测试: 失败")
    
    print(f"\n🎉 多关节系统测试完成！")

# ============================================================================
# 🧩 独立测试函数 (仅测试已保存的模型)
# ============================================================================

def test_saved_models():
    """测试已保存的模型"""
    
    print("🎮 测试已保存的多关节模型")
    print()
    
    # 测试配置
    test_configs = [
        {'model_path': 'models/multi_joint_2j_sac', 'num_joints': 2},
        {'model_path': 'models/multi_joint_3j_sac', 'num_joints': 3},
    ]
    
    for config in test_configs:
        model_path = config['model_path']
        num_joints = config['num_joints']
        
        print(f"\n🧪 测试 {num_joints} 关节模型...")
        
        if os.path.exists(f"{model_path}.zip"):
            results = load_and_test_model(
                model_path=model_path,
                num_joints=num_joints,
                num_episodes=3,
                render=True
            )
            
            if 'error' not in results:
                print(f"✅ {num_joints} 关节测试完成")
                print(f"   成功率: {results['success_rate']:.1%}")
                print(f"   平均奖励: {results['avg_reward']:.3f}")
            else:
                print(f"❌ {num_joints} 关节测试失败")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}.zip")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 仅测试已保存的模型
        test_saved_models()
    else:
        # 完整的训练和测试流程
        main()


