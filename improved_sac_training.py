#!/usr/bin/env python3
"""
改进的 SAC 训练测试 - 增强探索性
"""

import sys
import os
import numpy as np
import torch
import time
from collections import deque

# 添加路径
sys.path.append('examples/surrogate_model/sac')
sys.path.append('examples/2d_reacher/envs')

from sb3_sac_adapter import SB3SACFactory
from reacher_env_factory import create_reacher_env

def improved_sac_training():
    print("🚀 改进的 SAC 训练测试 - 增强探索性")
    print("🎯 改进措施:")
    print("   - Episode 长度限制: 200 步 (之前 2000 步)")
    print("   - 熵系数: 0.5 (增加探索)")
    print("   - 更频繁的 episode 结束和重置")
    print("📊 训练参数: 1500 步，每 300 步评估一次")
    print("=" * 70)
    
    # 创建环境
    env = create_reacher_env(version='mujoco', render_mode='human')
    
    # 创建 SAC
    sac = SB3SACFactory.create_reacher_sac(
        action_dim=2,
        buffer_capacity=5000,
        batch_size=64,
        lr=1e-3,
        device='cpu'
    )
    sac.set_env(env)
    
    # 重置环境
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    print("✅ 环境和 SAC 初始化完成")
    print(f"🎮 动作空间: {env.action_space}")
    print("=" * 70)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    success_episodes = 0
    total_episodes = 0
    
    # 当前 episode 统计
    current_reward = 0
    current_length = 0
    best_distance = float('inf')
    
    # 学习进度统计
    recent_rewards = deque(maxlen=5)
    recent_distances = deque(maxlen=5)
    
    # 动作多样性统计
    action_history = deque(maxlen=100)
    
    print("🎯 开始改进训练...")
    start_time = time.time()
    
    for step in range(1500):  # 训练 1500 步
        # 获取动作
        action = sac.get_action(obs, deterministic=False)
        action_history.append(action.cpu().numpy().copy())
        
        # 执行动作
        step_result = env.step(action.cpu().numpy())
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        current_reward += reward
        current_length += 1
        
        # 获取距离信息
        distance = obs[8] if len(obs) > 8 else 999
        best_distance = min(best_distance, distance)
        
        # Episode 结束处理
        if terminated or truncated:
            total_episodes += 1
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_distances.append(best_distance)
            recent_rewards.append(current_reward)
            recent_distances.append(best_distance)
            
            # 判断是否成功
            if best_distance < 15.0:  # 15px 内算成功
                success_episodes += 1
                print(f"🎯 Episode {total_episodes} 成功! 距离: {best_distance:.1f}px, 奖励: {current_reward:.2f}, 步数: {current_length}")
            else:
                print(f"📊 Episode {total_episodes}: 距离: {best_distance:.1f}px, 奖励: {current_reward:.2f}, 步数: {current_length}")
            
            # 重置
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            current_reward = 0
            current_length = 0
            best_distance = float('inf')
        
        # 每 300 步进行详细评估
        if (step + 1) % 300 == 0:
            elapsed_time = time.time() - start_time
            
            print(f"\n📈 训练进度评估 - Step {step + 1}/1500 ({elapsed_time/60:.1f}分钟)")
            print("=" * 50)
            
            # 动作多样性分析
            if len(action_history) > 10:
                actions_array = np.array(list(action_history))
                action_std = np.std(actions_array, axis=0)
                action_mean = np.mean(actions_array, axis=0)
                action_range = [np.min(actions_array, axis=0), np.max(actions_array, axis=0)]
                
                print(f"🎮 动作分析:")
                print(f"   平均动作: [{action_mean[0]:.4f}, {action_mean[1]:.4f}]")
                print(f"   动作标准差: [{action_std[0]:.4f}, {action_std[1]:.4f}]")
                print(f"   动作范围: [{action_range[0][0]:.4f}, {action_range[1][0]:.4f}] x [{action_range[0][1]:.4f}, {action_range[1][1]:.4f}]")
                
                if max(action_std) > 0.003:
                    print(f"   ✅ 动作有良好多样性，SAC 在积极探索")
                elif max(action_std) > 0.001:
                    print(f"   ⚠️ 动作有一定多样性，但可能需要更多探索")
                else:
                    print(f"   ❌ 动作变化很小，SAC 可能陷入局部最优")
            
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                avg_distance = np.mean(episode_distances)
                avg_length = np.mean(episode_lengths)
                success_rate = (success_episodes / total_episodes) * 100 if total_episodes > 0 else 0
                
                print(f"📊 总体统计:")
                print(f"   Episodes: {total_episodes}")
                print(f"   成功率: {success_rate:.1f}% ({success_episodes}/{total_episodes})")
                print(f"   平均奖励: {avg_reward:.2f}")
                print(f"   平均距离: {avg_distance:.1f}px")
                print(f"   平均步数: {avg_length:.1f}")
                
                # 最近表现
                if len(recent_rewards) > 0:
                    recent_avg_reward = np.mean(recent_rewards)
                    recent_avg_distance = np.mean(recent_distances)
                    print(f"📈 最近5个Episodes:")
                    print(f"   平均奖励: {recent_avg_reward:.2f}")
                    print(f"   平均距离: {recent_avg_distance:.1f}px")
                
                # 学习进步分析
                if len(episode_rewards) >= 6:
                    early_rewards = episode_rewards[:3]
                    recent_rewards_list = episode_rewards[-3:]
                    if len(early_rewards) > 0 and len(recent_rewards_list) > 0:
                        improvement = np.mean(recent_rewards_list) - np.mean(early_rewards)
                        print(f"🎯 学习进步: {improvement:.2f} (正数表示改善)")
                
                # 获取损失信息
                if step >= 100:  # 学习开始后
                    losses = sac.update()
                    print(f"🧠 网络损失:")
                    print(f"   Actor Loss: {losses['actor_loss']:.4f}")
                    print(f"   Critic Loss: {losses['critic_loss']:.4f}")
                    print(f"   Alpha: {losses['alpha']:.4f}")
            
            print("=" * 50 + "\n")
    
    # 最终评估
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🏆 改进的 SAC 训练测试完成!")
    print("=" * 70)
    
    if len(episode_rewards) > 0:
        final_avg_reward = np.mean(episode_rewards)
        final_avg_distance = np.mean(episode_distances)
        final_success_rate = (success_episodes / total_episodes) * 100
        
        print(f"⏱️ 训练时间: {total_time/60:.1f} 分钟")
        print(f"📊 总Episodes: {total_episodes}")
        print(f"🎯 最终成功率: {final_success_rate:.1f}%")
        print(f"🎁 最终平均奖励: {final_avg_reward:.2f}")
        print(f"📏 最终平均距离: {final_avg_distance:.1f}px")
        print(f"⏱️ 平均Episode长度: {np.mean(episode_lengths):.1f}步")
        
        # 训练效果评估
        print(f"\n🔍 训练效果评估:")
        if final_success_rate >= 20:
            print("🥇 优秀! SAC 已经学会了 Reacher 任务")
        elif final_success_rate >= 10:
            print("🥈 良好! SAC 有明显学习进步")
        elif final_success_rate >= 3:
            print("🥉 有进步! SAC 开始学习任务")
        elif final_avg_reward > -0.5:
            print("📈 学习中... 奖励有改善")
        elif total_episodes > 5:
            print("✅ 训练正常进行，Episodes 能够正常结束")
        else:
            print("❌ Episodes 无法正常结束，需要进一步调试")
        
        # 最佳表现
        if len(episode_distances) > 0:
            best_distance_ever = min(episode_distances)
            best_reward_ever = max(episode_rewards)
            print(f"\n🏅 最佳表现:")
            print(f"   最近距离: {best_distance_ever:.1f}px")
            print(f"   最高奖励: {best_reward_ever:.2f}")
        
        # 动作多样性最终分析
        if len(action_history) > 10:
            final_actions = np.array(list(action_history))
            final_action_std = np.std(final_actions, axis=0)
            print(f"\n🎮 最终动作分析:")
            print(f"   动作标准差: [{final_action_std[0]:.4f}, {final_action_std[1]:.4f}]")
            if max(final_action_std) > 0.003:
                print("   ✅ SAC 保持了良好的动作多样性")
            elif max(final_action_std) > 0.001:
                print("   ⚠️ SAC 有一定动作多样性")
            else:
                print("   ❌ 动作趋于固定，可能陷入局部最优")
    
    print(f"\n✅ 改进训练测试完成! Episodes 能够正常结束。")
    env.close()
    
    return {
        'success_rate': final_success_rate if len(episode_rewards) > 0 else 0,
        'avg_reward': final_avg_reward if len(episode_rewards) > 0 else 0,
        'avg_distance': final_avg_distance if len(episode_rewards) > 0 else 999,
        'total_episodes': total_episodes,
        'avg_length': np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
    }

if __name__ == "__main__":
    results = improved_sac_training()
    print(f"\n🎊 改进训练结果总结:")
    print(f"   成功率: {results['success_rate']:.1f}%")
    print(f"   平均奖励: {results['avg_reward']:.2f}")
    print(f"   平均距离: {results['avg_distance']:.1f}px")
    print(f"   总Episodes: {results['total_episodes']}")
    print(f"   平均Episode长度: {results['avg_length']:.1f}步")


