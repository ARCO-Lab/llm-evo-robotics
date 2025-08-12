#!/usr/bin/env python3
"""
快速测试脚本 - 简单测试训练好的模型
使用方式: python quick_test.py --model-path path/to/model.pth
"""

import sys
import os
import torch
import numpy as np
import argparse
import time

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


def quick_test(model_path, num_episodes=5, render=True):
    """快速测试模型"""
    
    print(f"🚀 快速测试模型: {model_path}")
    
    # 环境配置
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human' if render else None,
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
    
    # 加载模型
    if os.path.exists(model_path):
        model_data = torch.load(model_path, map_location='cpu')
        sac.actor.load_state_dict(model_data['actor_state_dict'])
        
        print(f"✅ 模型加载成功!")
        print(f"   训练步骤: {model_data.get('step', 'N/A')}")
        print(f"   成功率: {model_data.get('success_rate', 'N/A')}")
        print(f"   最小距离: {model_data.get('min_distance', 'N/A')}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 运行测试episodes
    successes = 0
    goal_threshold = 50.0
    
    print(f"\n🎮 开始测试 {num_episodes} episodes (目标阈值: {goal_threshold}px)")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 1000
        min_distance = float('inf')
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while step_count < max_steps:
            # 获取动作（确定性策略）
            action = sac.get_action(
                torch.from_numpy(obs).float(),
                gnn_embed.squeeze(0),
                num_joints=num_joints,
                deterministic=True
            )
            
            # 执行动作
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode_reward += reward
            step_count += 1
            
            # 计算距离
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
            min_distance = min(min_distance, distance)
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.02)  # 控制速度
            
            # 每100步打印进度
            if step_count % 100 == 0:
                print(f"  步骤 {step_count}: 距离 {distance:.1f}px, 奖励 {episode_reward:.1f}")
            
            # 检查成功
            if distance <= goal_threshold:
                successes += 1
                print(f"  🎉 成功到达目标! 距离: {distance:.1f}px, 步骤: {step_count}")
                break
                
            if done:
                print(f"  Episode 结束 (done=True)")
                break
        
        print(f"Episode {episode + 1} 结果:")
        print(f"  最小距离: {min_distance:.1f}px")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  步骤数: {step_count}")
        print(f"  成功: {'是' if min_distance <= goal_threshold else '否'}")
    
    # 总结
    success_rate = successes / num_episodes
    print(f"\n{'='*50}")
    print(f"🏆 测试总结:")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {successes}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  目标阈值: {goal_threshold}px")
    print(f"{'='*50}")
    
    env.close()
    return success_rate


def main():
    parser = argparse.ArgumentParser(description="快速测试训练好的模型")
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=5,
                        help='测试的episode数量')
    parser.add_argument('--no-render', action='store_true',
                        help='不显示渲染（加快测试）')
    
    args = parser.parse_args()
    
    render = not args.no_render
    quick_test(args.model_path, args.episodes, render)


if __name__ == "__main__":
    main() 