#!/usr/bin/env python3
"""
长期训练 SAC，随机目标位置，直到稳定到达目标
"""

import sys
import os
import numpy as np
import torch
import time
import random
from collections import deque

# 添加路径
sys.path.append('examples/surrogate_model/sac')
sys.path.append('examples/2d_reacher/envs')

from sb3_sac_adapter import SB3SACFactory
from reacher_env_factory import create_reacher_env

def long_term_training():
    print("🚀 长期训练 SAC - 随机目标位置")
    print("🎯 训练目标:")
    print("   - 每次到达目标后随机生成新位置")
    print("   - 训练直到能稳定到达各种目标")
    print("   - 成功标准: 距离 < 20px")
    print("   - 稳定标准: 连续10个目标成功率 > 80%")
    print("=" * 70)
    
    # 创建环境
    env = create_reacher_env(version='mujoco', render_mode='human')
    
    # 创建 SAC
    sac = SB3SACFactory.create_reacher_sac(
        action_dim=2,
        buffer_capacity=20000,  # 增大缓冲区
        batch_size=128,         # 增大批次大小
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
    total_episodes = 0
    successful_episodes = 0
    recent_success_rates = deque(maxlen=10)  # 最近10个目标的成功率
    
    # 当前目标统计
    current_target_attempts = 0
    current_target_successes = 0
    
    # 训练进度
    total_steps = 0
    start_time = time.time()
    
    # 目标位置生成函数
    def generate_random_goal():
        """生成随机目标位置"""
        # 在合理范围内生成目标
        # 基于 MuJoCo Reacher 的工作空间
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(0.05, 0.2)  # MuJoCo 单位是米
        
        # 转换为像素坐标（相对于 anchor_point）
        goal_x = 480 + radius * np.cos(angle) * 600  # 缩放到像素
        goal_y = 620 + radius * np.sin(angle) * 600
        
        # 确保在合理范围内
        goal_x = np.clip(goal_x, 400, 700)
        goal_y = np.clip(goal_y, 450, 750)
        
        return [goal_x, goal_y]
    
    # 设置初始目标
    current_goal = generate_random_goal()
    if hasattr(env, 'set_goal_position'):
        env.set_goal_position(current_goal)
    
    print(f"🎯 开始长期训练...")
    print(f"📍 初始目标位置: [{current_goal[0]:.1f}, {current_goal[1]:.1f}]")
    print("=" * 70)
    
    # 当前 episode 统计
    current_reward = 0
    current_length = 0
    best_distance = float('inf')
    
    while True:  # 持续训练直到达到稳定标准
        # 获取动作
        action = sac.get_action(obs, deterministic=False)
        
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
        total_steps += 1
        
        # 获取距离信息
        distance = obs[8] if len(obs) > 8 else 999
        best_distance = min(best_distance, distance)
        
        # Episode 结束处理
        if terminated or truncated:
            total_episodes += 1
            current_target_attempts += 1
            
            # 判断是否成功
            success = best_distance < 20.0  # 20px 内算成功
            if success:
                successful_episodes += 1
                current_target_successes += 1
                print(f"🎯 Episode {total_episodes} 成功! 距离: {best_distance:.1f}px, 奖励: {current_reward:.2f}, 步数: {current_length}")
            else:
                print(f"❌ Episode {total_episodes} 失败. 距离: {best_distance:.1f}px, 奖励: {current_reward:.2f}, 步数: {current_length}")
            
            # 检查当前目标是否完成训练
            if current_target_attempts >= 10:  # 每个目标尝试10次
                target_success_rate = current_target_successes / current_target_attempts
                recent_success_rates.append(target_success_rate)
                
                print(f"\n📊 目标完成统计:")
                print(f"   目标位置: [{current_goal[0]:.1f}, {current_goal[1]:.1f}]")
                print(f"   成功率: {target_success_rate:.1%} ({current_target_successes}/{current_target_attempts})")
                
                # 生成新目标
                current_goal = generate_random_goal()
                if hasattr(env, 'set_goal_position'):
                    env.set_goal_position(current_goal)
                
                print(f"🎯 新目标位置: [{current_goal[0]:.1f}, {current_goal[1]:.1f}]")
                
                # 重置当前目标统计
                current_target_attempts = 0
                current_target_successes = 0
                
                # 检查是否达到稳定标准
                if len(recent_success_rates) >= 10:
                    avg_success_rate = np.mean(recent_success_rates)
                    print(f"📈 最近10个目标平均成功率: {avg_success_rate:.1%}")
                    
                    if avg_success_rate >= 0.8:  # 80% 成功率
                        print(f"\n🎉 训练完成! 达到稳定标准!")
                        print(f"   最近10个目标平均成功率: {avg_success_rate:.1%}")
                        break
                
                print("=" * 50)
            
            # 重置
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            current_reward = 0
            current_length = 0
            best_distance = float('inf')
        
        # 每1000步进行进度报告
        if total_steps % 1000 == 0:
            elapsed_time = time.time() - start_time
            overall_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
            
            print(f"\n📈 训练进度报告 - Step {total_steps}")
            print(f"⏱️ 训练时间: {elapsed_time/60:.1f} 分钟")
            print(f"📊 总Episodes: {total_episodes}")
            print(f"🎯 总体成功率: {overall_success_rate:.1%} ({successful_episodes}/{total_episodes})")
            print(f"📍 当前目标: [{current_goal[0]:.1f}, {current_goal[1]:.1f}]")
            print(f"🎯 当前目标成功率: {current_target_successes}/{current_target_attempts}")
            
            # 获取损失信息
            if total_steps >= 100:
                losses = sac.update()
                print(f"🧠 网络损失: Actor={losses['actor_loss']:.4f}, Critic={losses['critic_loss']:.4f}")
            
            print("-" * 50)
        
        # 安全检查：如果训练时间过长，提供选项退出
        if total_steps >= 50000:  # 50k 步后询问是否继续
            elapsed_time = time.time() - start_time
            print(f"\n⏰ 训练已进行 {elapsed_time/60:.1f} 分钟，{total_steps} 步")
            print(f"📊 当前整体成功率: {successful_episodes/total_episodes:.1%}")
            if len(recent_success_rates) > 0:
                print(f"📈 最近目标平均成功率: {np.mean(recent_success_rates):.1%}")
            
            # 这里可以添加用户交互，但为了自动化，我们继续训练
            if total_steps >= 100000:  # 100k 步后强制停止
                print("⚠️ 达到最大训练步数限制，停止训练")
                break
    
    # 最终统计
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🏆 长期训练完成!")
    print("=" * 70)
    
    overall_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
    
    print(f"⏱️ 总训练时间: {total_time/60:.1f} 分钟")
    print(f"🔢 总训练步数: {total_steps}")
    print(f"📊 总Episodes: {total_episodes}")
    print(f"🎯 总体成功率: {overall_success_rate:.1%} ({successful_episodes}/{total_episodes})")
    
    if len(recent_success_rates) > 0:
        final_avg_success_rate = np.mean(recent_success_rates)
        print(f"📈 最终稳定成功率: {final_avg_success_rate:.1%}")
        
        if final_avg_success_rate >= 0.8:
            print("🎉 训练成功! Reacher 能够稳定到达各种目标位置!")
        elif final_avg_success_rate >= 0.6:
            print("👍 训练良好! Reacher 有较好的目标到达能力!")
        else:
            print("⚠️ 需要继续训练以提高稳定性")
    
    print(f"\n✅ 长期训练测试完成!")
    env.close()
    
    return {
        'total_steps': total_steps,
        'total_episodes': total_episodes,
        'overall_success_rate': overall_success_rate,
        'final_stable_rate': np.mean(recent_success_rates) if len(recent_success_rates) > 0 else 0,
        'training_time_minutes': total_time / 60
    }

if __name__ == "__main__":
    results = long_term_training()
    print(f"\n📋 最终训练结果:")
    print(f"   训练步数: {results['total_steps']}")
    print(f"   总Episodes: {results['total_episodes']}")
    print(f"   整体成功率: {results['overall_success_rate']:.1%}")
    print(f"   稳定成功率: {results['final_stable_rate']:.1%}")
    print(f"   训练时间: {results['training_time_minutes']:.1f}分钟")
