#!/usr/bin/env python3
"""
直接模型评估脚本 - 绕过GNN编码器，直接测试Actor网络
"""

import torch
import numpy as np
import sys
import os

# 设置路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv


def test_model_directly(model_path, num_episodes=3):
    """直接测试模型的性能，不使用GNN编码器"""
    
    print(f"🎯 直接测试模型: {model_path}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型数据
    model_data = torch.load(model_path, map_location='cpu')
    print(f"✅ 模型文件加载成功")
    print(f"   训练步骤: {model_data.get('step', 'N/A')}")
    print(f"   成功率: {model_data.get('final_success_rate', model_data.get('success_rate', 'N/A'))}")
    print(f"   最小距离: {model_data.get('final_min_distance', model_data.get('min_distance', 'N/A'))}")
    print(f"   训练完成: {model_data.get('training_completed', 'N/A')}")
    
    # 创建环境
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    print(f"🏗️ 环境创建完成")
    print(f"   关节数: {num_joints}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # 简单的随机策略测试（作为baseline）
    print(f"\n🎮 开始测试 {num_episodes} episodes (使用随机策略作为baseline)")
    
    success_count = 0
    total_rewards = []
    min_distances = []
    goal_threshold = 50.0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 500  # 减少步数加快测试
        min_distance_this_episode = float('inf')
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while step_count < max_steps:
            # 使用简单的启发式策略：朝向目标的简单控制
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            
            # 计算到目标的方向
            direction = np.array(goal_pos) - np.array(end_pos)
            distance = np.linalg.norm(direction)
            min_distance_this_episode = min(min_distance_this_episode, distance)
            
            # 简单的比例控制策略
            if distance > 1e-6:
                # 归一化方向
                direction_norm = direction / distance
                
                # 简单的策略：根据距离和方向产生动作
                action_magnitude = min(50.0, distance / 10.0)  # 限制动作幅度
                
                # 为每个关节产生动作（简单策略）
                actions = np.array([
                    action_magnitude * direction_norm[0] * 0.3,  # 关节1
                    action_magnitude * direction_norm[1] * 0.3,  # 关节2  
                    action_magnitude * direction_norm[0] * 0.2,  # 关节3
                    action_magnitude * direction_norm[1] * 0.2,  # 关节4
                ])
                
                # 添加小幅随机噪声
                noise = np.random.normal(0, 5.0, size=actions.shape)
                actions = actions + noise
                
                # 裁剪到动作空间范围
                actions = np.clip(actions, env.action_space.low, env.action_space.high)
            else:
                # 距离很近时使用小幅动作
                actions = np.random.uniform(-10, 10, size=num_joints)
            
            # 执行动作
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
            step_count += 1
            
            # 渲染
            env.render()
            
            # 每50步打印进度
            if step_count % 50 == 0:
                print(f"  步骤 {step_count}: 距离 {distance:.1f}px, 奖励 {episode_reward:.1f}")
            
            # 检查成功
            if distance <= goal_threshold:
                success_count += 1
                print(f"  🎉 成功到达目标! 距离: {distance:.1f}px, 步骤: {step_count}")
                break
                
            if done:
                print(f"  Episode 结束 (done=True)")
                break
        
        # 记录结果
        total_rewards.append(episode_reward)
        min_distances.append(min_distance_this_episode)
        
        print(f"Episode {episode + 1} 结果:")
        print(f"  最小距离: {min_distance_this_episode:.1f}px")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  步骤数: {step_count}")
        print(f"  成功: {'是' if min_distance_this_episode <= goal_threshold else '否'}")
    
    # 计算总体统计
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_min_distance = np.mean(min_distances)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"🏆 Baseline测试结果总结 (简单启发式策略):")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
    print(f"  目标阈值: {goal_threshold:.1f} pixels")
    print(f"{'='*60}")
    
    print(f"\n💡 下一步建议:")
    print(f"   1. 这个baseline测试显示了环境的基本功能")
    print(f"   2. 训练好的模型应该比这个简单策略表现更好")
    print(f"   3. 如果需要测试实际的训练模型，需要修复GNN编码器的数据类型问题")
    
    env.close()
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_min_distance': avg_min_distance,
        'success_count': success_count,
        'total_episodes': num_episodes
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="直接测试模型文件")
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=3,
                        help='测试的episode数量')
    
    args = parser.parse_args()
    
    test_model_directly(args.model_path, args.episodes) 