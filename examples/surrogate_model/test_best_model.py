#!/usr/bin/env python3
"""
测试最佳保存模型的脚本
使用方式: python test_best_model.py --model-path ./trained_models/reacher2d/test/*/best_models/latest_best_model.pth
"""

import sys
import os
import torch
import numpy as np
import argparse

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/attn_model'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/sac'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/utils'))

from reacher2d_env import Reacher2DEnv
from attn_model import AttnModel
from sac_model import AttentionSACWithBuffer
from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder

def test_best_model(model_path, num_episodes=10):
    """测试最佳模型"""
    
    # 环境配置
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    # 创建环境
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    # 创建GNN编码器
    reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
    gnn_embed = reacher2d_encoder.get_gnn_embeds(
        num_links=num_joints, 
        link_lengths=env_params['link_lengths']
    )
    
    # 创建SAC模型
    attn_model = AttnModel(128, 128, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, num_joints, env_type='reacher2d')
    
    # 加载最佳模型
    if os.path.exists(model_path):
        model_data = torch.load(model_path, map_location='cpu')
        sac.actor.load_state_dict(model_data['actor_state_dict'])
        sac.critic1.load_state_dict(model_data['critic1_state_dict'])
        sac.critic2.load_state_dict(model_data['critic2_state_dict'])
        
        print(f"✅ 加载模型成功:")
        print(f"   步骤: {model_data.get('step', 'N/A')}")
        print(f"   成功率: {model_data.get('success_rate', 'N/A')}")
        print(f"   最小距离: {model_data.get('min_distance', 'N/A')}")
        print(f"   时间戳: {model_data.get('timestamp', 'N/A')}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 测试多个episode
    success_count = 0
    total_rewards = []
    min_distances = []
    goal_threshold = 50.0  # 调整目标阈值为50像素，与训练脚本保持一致
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 1000
        min_distance_this_episode = float('inf')
        
        print(f"\n🎮 Episode {episode + 1}/{num_episodes}")
        
        while step_count < max_steps:
            # 获取动作（使用确定性策略）
            action = sac.get_action(
                torch.from_numpy(obs).float(),
                gnn_embed.squeeze(0),
                num_joints=num_joints,
                deterministic=True  # 测试时使用确定性策略
            )
            
            # 执行动作
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode_reward += reward
            step_count += 1
            
            # 检查距离
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
            min_distance_this_episode = min(min_distance_this_episode, distance)
            
            # 渲染
            env.render()
            
            # 检查是否到达目标
            if distance <= goal_threshold:
                success_count += 1
                print(f"  🎉 目标到达! 距离: {distance:.1f} pixels, 步骤: {step_count}")
                break
                
            if done:
                break
        
        total_rewards.append(episode_reward)
        min_distances.append(min_distance_this_episode)
        
        print(f"  Episode {episode + 1} 总结:")
        print(f"    奖励: {episode_reward:.2f}")
        print(f"    最小距离: {min_distance_this_episode:.1f} pixels")
        print(f"    步骤数: {step_count}")
        print(f"    成功: {'是' if min_distance_this_episode <= goal_threshold else '否'}")
    
    # 测试总结
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_min_distance = np.mean(min_distances)
    
    print(f"\n{'='*50}")
    print(f"🏆 测试结果总结:")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_rate:.2%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
    print(f"  目标阈值: {goal_threshold:.1f} pixels")
    print(f"{'='*50}")
    
    env.close()
    return success_rate, avg_reward, avg_min_distance

def main():
    parser = argparse.ArgumentParser(description="测试最佳保存的模型")
    parser.add_argument('--model-path', type=str, required=True, 
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=10,
                        help='测试的episode数量')
    
    args = parser.parse_args()
    
    print(f"🧪 开始测试模型: {args.model_path}")
    test_best_model(args.model_path, args.episodes)

if __name__ == "__main__":
    main() 