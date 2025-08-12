#!/usr/bin/env python3
"""
基于train.py的模型评估脚本
重用训练脚本中已验证的导入和配置
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/train'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/common'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/environments'))
sys.path.append(os.path.join(base_dir, 'examples/rl'))

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import argparse

gym.logger.set_level(40)

# 直接导入，现在environments在路径中
import environments

from arguments import get_parser
from utils import solve_argv_conflict
from common import *

from attn_dataset.sim_data_handler import DataHandler
from attn_model.attn_model import AttnModel
from sac.sac_model import AttentionSACWithBuffer

# 修改第50行的导入
from env_config.env_wrapper import make_reacher2d_vec_envs, make_smart_reacher2d_vec_envs
from reacher2d_env import Reacher2DEnv
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))


def evaluate_model(model_path, num_episodes=5, render=True):
    """评估模型性能"""
    
    print(f"🧪 开始评估模型: {model_path}")
    
    # 环境参数（从train.py复制）
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human' if render else None,
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    print(f"🏗️ 环境配置:")
    print(f"   关节数: {env_params['num_links']}")
    print(f"   连杆长度: {env_params['link_lengths']}")
    print(f"   渲染: {'是' if render else '否'}")
    
    # 创建环境
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    # 创建GNN编码器（从train.py复制）
    sys.path.append(os.path.join(os.path.dirname(__file__), '../2d_reacher/utils'))
    from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
    
    print("🤖 初始化 Reacher2D GNN 编码器...")
    reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
    single_gnn_embed = reacher2d_encoder.get_gnn_embeds(
        num_links=num_joints, 
        link_lengths=env_params['link_lengths']
    )
    print(f"✅ GNN 嵌入生成成功，形状: {single_gnn_embed.shape}")
    
    # 创建SAC模型（从train.py复制）
    action_dim = num_joints
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, 
                                buffer_capacity=10000, batch_size=32,
                                lr=1e-4,
                                env_type='reacher2d')
    
    # 加载模型
    if os.path.exists(model_path):
        model_data = torch.load(model_path, map_location='cpu')
        sac.actor.load_state_dict(model_data['actor_state_dict'])
        if 'critic1_state_dict' in model_data:
            sac.critic1.load_state_dict(model_data['critic1_state_dict'])
        if 'critic2_state_dict' in model_data:
            sac.critic2.load_state_dict(model_data['critic2_state_dict'])
        
        print(f"✅ 模型加载成功!")
        print(f"   训练步骤: {model_data.get('step', 'N/A')}")
        print(f"   成功率: {model_data.get('final_success_rate', model_data.get('success_rate', 'N/A'))}")
        print(f"   最小距离: {model_data.get('final_min_distance', model_data.get('min_distance', 'N/A'))}")
        print(f"   训练完成: {model_data.get('training_completed', 'N/A')}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 测试模型
    print(f"\n🎮 开始测试 {num_episodes} episodes")
    
    success_count = 0
    total_rewards = []
    min_distances = []
    step_counts = []
    goal_threshold = 50.0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 1000
        min_distance_this_episode = float('inf')
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while step_count < max_steps:
            # 获取动作（使用确定性策略进行评估）
            action = sac.get_action(
                torch.from_numpy(obs).float(),
                single_gnn_embed.squeeze(0),
                num_joints=num_joints,
                deterministic=True  # 评估时使用确定性策略
            )
            
            # 执行动作
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode_reward += reward
            step_count += 1
            
            # 计算距离
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
            min_distance_this_episode = min(min_distance_this_episode, distance)
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.02)  # 控制速度
            
            # 每100步打印进度
            if step_count % 100 == 0:
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
        step_counts.append(step_count)
        
        print(f"Episode {episode + 1} 结果:")
        print(f"  最小距离: {min_distance_this_episode:.1f}px")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  步骤数: {step_count}")
        print(f"  成功: {'是' if min_distance_this_episode <= goal_threshold else '否'}")
    
    # 计算总体统计
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_min_distance = np.mean(min_distances)
    avg_steps = np.mean(step_counts)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"🏆 评估结果总结:")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
    print(f"  平均步骤数: {avg_steps:.1f}")
    print(f"  目标阈值: {goal_threshold:.1f} pixels")
    print(f"{'='*60}")
    
    env.close()
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_min_distance': avg_min_distance,
        'avg_steps': avg_steps,
        'success_count': success_count,
        'total_episodes': num_episodes
    }


def main():
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=5,
                        help='测试的episode数量')
    parser.add_argument('--no-render', action='store_true',
                        help='不显示渲染（加快测试）')
    
    args = parser.parse_args()
    
    render = not args.no_render
    evaluate_model(args.model_path, args.episodes, render)


if __name__ == "__main__":
    main() 