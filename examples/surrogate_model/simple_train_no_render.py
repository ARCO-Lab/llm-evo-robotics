#!/usr/bin/env python3
"""
简化版训练脚本 - 禁用渲染，专注训练稳定性
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

# 基础导入
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/sac'))

from reacher2d_env import Reacher2DEnv
from sac_model import AttentionSACWithBuffer
from gnn_encoder import GNN_Encoder
import numpy as np
import torch
import time

def main():
    print("🚀 启动简化版训练（无渲染）")
    print("="*50)
    
    # 环境参数
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': None,  # 禁用渲染
        'config_path': '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml',
        'debug_level': 'SILENT'
    }
    
    # 创建环境
    print("1️⃣ 创建训练环境...")
    env = Reacher2DEnv(**env_params)
    obs = env.reset()
    
    action_dim = 4  # 4个关节
    obs_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
    
    print(f"   观察维度: {obs_dim}")
    print(f"   动作维度: {action_dim}")
    
    # 创建模型（简化版GNN）
    print("2️⃣ 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")
    
    # 简化的GNN编码器
    class SimpleGNNEncoder:
        def __init__(self, obs_dim):
            self.obs_dim = obs_dim
            
        def encode(self, obs):
            # 简单的观察编码
            joint_features = obs[:12].reshape(4, 3)  # 4个关节，每个3个特征
            vertex_features = obs[12:].reshape(1, -1)  # 其他特征
            
            joint_q = torch.FloatTensor(joint_features).unsqueeze(0).to(device)
            vertex_k = torch.FloatTensor(vertex_features).to(device)
            vertex_v = torch.FloatTensor(vertex_features).to(device)
            vertex_mask = torch.ones(1, 1).bool().to(device)
            
            return joint_q, vertex_k, vertex_v, vertex_mask
    
    encoder = SimpleGNNEncoder(obs_dim)
    
    # 创建SAC模型
    print("3️⃣ 创建SAC模型...")
    sac_model = AttentionSACWithBuffer(
        attn_model=encoder,
        action_dim=action_dim,
        lr=3e-4,
        device=device
    )
    
    # 训练循环
    print("4️⃣ 开始训练...")
    total_steps = 0
    episode_count = 0
    episode_reward = 0
    episode_steps = 0
    
    obs = env.reset()
    
    for step in range(1000):  # 短期测试
        # 获取动作
        joint_q, vertex_k, vertex_v, vertex_mask = encoder.encode(obs)
        action = sac_model.get_action(joint_q, vertex_k, vertex_v, vertex_mask, deterministic=False)
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        total_steps += 1
        
        # 存储经验
        next_joint_q, next_vertex_k, next_vertex_v, next_vertex_mask = encoder.encode(next_obs)
        sac_model.memory.push(
            joint_q.cpu(), vertex_k.cpu(), vertex_v.cpu(),
            torch.FloatTensor(action),
            torch.FloatTensor([reward]),
            next_joint_q.cpu(), next_vertex_k.cpu(), next_vertex_v.cpu(),
            torch.FloatTensor([float(done)]),
            vertex_mask.cpu()
        )
        
        # 更新模型
        if total_steps > 100 and total_steps % 2 == 0:  # 从步数100开始更新
            update_info = sac_model.update()
            if update_info and step % 50 == 0:
                print(f"   Step {step}: Critic Loss = {update_info.get('critic_loss', 'N/A'):.3f}, "
                      f"Actor Loss = {update_info.get('actor_loss', 'N/A'):.3f}")
        
        # Episode结束处理
        if done or episode_steps >= 500:
            episode_count += 1
            print(f"Episode {episode_count}: 奖励 = {episode_reward:.2f}, 步数 = {episode_steps}")
            
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
        else:
            obs = next_obs
            
        # 每100步输出进度
        if step % 100 == 0:
            print(f"Training progress: {step}/1000 steps")
    
    env.close()
    print("\n✅ 简化版训练测试完成！")
    print("   基础奖励系统和SAC模型运行正常")
    print("   可以继续开发完整的训练系统")

if __name__ == "__main__":
    main()
