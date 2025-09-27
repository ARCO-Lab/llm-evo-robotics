#!/usr/bin/env python3
"""
可视化SAC训练脚本：
1. 使用单个SAC智能体控制3关节Reacher
2. 开启渲染以观察训练过程
3. 与MADDPG进行对比
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import time

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 导入环境
from maddpg_complete_sequential_training import (
    create_env, 
    SUCCESS_THRESHOLD_RATIO, 
    SUCCESS_THRESHOLD_2JOINT
)

def train_visual_sac(num_joints=3, total_timesteps=10000):
    """训练可视化SAC模型"""
    print(f"🌟 可视化SAC训练系统")
    print(f"🤖 策略: 单个SAC智能体控制{num_joints}关节Reacher")
    print(f"📊 训练步数: {total_timesteps}")
    print(f"🎯 成功阈值: {SUCCESS_THRESHOLD_RATIO:.1%} * R")
    print("="*60)
    
    # 创建训练环境（开启渲染）
    train_env = create_env(num_joints, render_mode='human', show_position_info=True)
    
    print(f"🔧 环境配置:")
    print(f"   观察空间: {train_env.observation_space}")
    print(f"   动作空间: {train_env.action_space}")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=1,
        learning_starts=1000,
        device='cpu',
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=100000
    )
    
    print(f"✅ SAC模型创建完成")
    
    # 开始训练
    print(f"\n🎯 开始SAC训练...")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=10,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ SAC训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        return model, training_time
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ SAC训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        return model, training_time
    
    finally:
        train_env.close()

def test_visual_sac(model, num_joints=3, n_episodes=5):
    """测试SAC模型"""
    print(f"\n🧪 开始测试SAC模型 {n_episodes} episodes...")
    
    # 创建测试环境（开启渲染）
    test_env = create_env(num_joints, render_mode='human', show_position_info=True)
    
    success_count = 0
    episode_rewards = []
    episode_distances = []
    
    try:
        for episode in range(n_episodes):
            obs, info = test_env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            
            for step in range(100):  # 每个episode最多100步
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                
                # 检查成功
                distance = info.get('distance_to_target', float('inf'))
                min_distance = min(min_distance, distance)
                
                if info.get('is_success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_count += 1
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
            
            print(f"   Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_count / n_episodes
        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances)
        
        print(f"\n🎯 SAC测试结果:")
        print(f"   成功率: {success_rate:.1%} ({success_count}/{n_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均最小距离: {avg_distance:.4f}")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_distance': avg_distance,
            'episode_rewards': episode_rewards,
            'episode_distances': episode_distances
        }
        
    except KeyboardInterrupt:
        print(f"\n⚠️ SAC测试被用户中断")
        return None
    
    finally:
        test_env.close()

def main():
    """主函数"""
    print("🌟 可视化SAC vs MADDPG对比训练系统")
    print("🎯 目标: 观察不同算法在3关节Reacher上的表现")
    print()
    
    # 训练SAC
    model, training_time = train_visual_sac(num_joints=3, total_timesteps=5000)
    
    # 测试SAC
    result = test_visual_sac(model, num_joints=3, n_episodes=5)
    
    if result:
        print(f"\n🎉 可视化SAC训练完成!")
        print(f"   训练时间: {training_time/60:.1f} 分钟")
        print(f"   最终成功率: {result['success_rate']:.1%}")
        print(f"   平均奖励: {result['avg_reward']:.2f}")
        
        # 保存模型
        model_path = "models/visual_sac_3joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")

if __name__ == "__main__":
    main()
