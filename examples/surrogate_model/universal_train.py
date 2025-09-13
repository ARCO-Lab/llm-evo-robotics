#!/usr/bin/env python3
"""
通用PPO训练脚本 - 支持任意关节数的机器人控制
可以在单次训练中处理2-6关节的机器人，训练出通用模型
"""

import torch
import numpy as np
import os
import sys
import argparse
import random
from collections import deque
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "examples/2d_reacher/envs"))
sys.path.append(os.path.join(base_dir, "examples/2d_reacher/utils"))

from reacher2d_env import Reacher2DEnv
from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
from sac.universal_ppo_model import UniversalPPOWithBuffer


class UniversalRobotTrainer:
    """通用机器人训练器 - 支持多种关节数混合训练"""
    
    def __init__(self, joint_configs, device='cpu'):
        """
        Args:
            joint_configs: List[dict] - 不同关节数的配置
                例: [{'num_joints': 3, 'link_lengths': [60, 60, 60]},
                     {'num_joints': 4, 'link_lengths': [50, 50, 50, 50]}]
        """
        self.joint_configs = joint_configs
        self.device = device
        self.envs = {}
        self.gnn_encoders = {}
        self.gnn_embeds = {}
        
        print(f"🤖 初始化通用机器人训练器")
        print(f"   支持关节数配置: {[cfg['num_joints'] for cfg in joint_configs]}")
        
        # 为每种关节数配置创建环境和编码器
        for cfg in joint_configs:
            num_joints = cfg['num_joints']
            link_lengths = cfg['link_lengths']
            
            # 创建环境
            env = Reacher2DEnv(
                num_links=num_joints,
                link_lengths=link_lengths,
                render_mode=None,
                config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
            )
            self.envs[num_joints] = env
            
            # 创建GNN编码器
            encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
            gnn_embed = encoder.get_gnn_embeds(num_links=num_joints, link_lengths=link_lengths)
            self.gnn_encoders[num_joints] = encoder
            self.gnn_embeds[num_joints] = gnn_embed
            
            print(f"   ✅ {num_joints}关节配置初始化完成，GNN嵌入形状: {gnn_embed.shape}")
        
        # 创建通用PPO模型
        self.ppo = UniversalPPOWithBuffer(
            buffer_size=2048,
            batch_size=64,
            lr=1e-4,
            device=device,
            env_type='reacher2d'
        )
        
        # 训练统计
        self.episode_rewards = {num_joints: deque(maxlen=100) for num_joints in self.envs.keys()}
        self.success_rates = {num_joints: deque(maxlen=100) for num_joints in self.envs.keys()}
        
    def sample_robot_config(self):
        """随机采样一个机器人配置"""
        return random.choice(self.joint_configs)
    
    def train_episode(self, robot_config, max_steps=200):
        """训练一个回合"""
        num_joints = robot_config['num_joints']
        env = self.envs[num_joints]
        gnn_embed = self.gnn_embeds[num_joints]
        
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        success = False
        
        for step in range(max_steps):
            # 获取动作
            action, log_prob, value = self.ppo.get_action(
                torch.tensor(obs, dtype=torch.float32).to(self.device),
                gnn_embed.to(self.device),
                num_joints,
                deterministic=False
            )
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
            
            # 存储经验
            self.ppo.store_experience(
                torch.tensor(obs, dtype=torch.float32),
                gnn_embed,
                action,
                reward,
                done or truncated,
                log_prob,
                value,
                num_joints
            )
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            if info.get('success', False):
                success = True
            
            if done or truncated:
                break
        
        # 记录统计信息
        self.episode_rewards[num_joints].append(episode_reward)
        self.success_rates[num_joints].append(1.0 if success else 0.0)
        
        return episode_reward, episode_steps, success
    
    def train(self, total_episodes=10000, update_frequency=50, save_frequency=1000):
        """主训练循环"""
        print(f"🚀 开始通用PPO训练")
        print(f"   总回合数: {total_episodes}")
        print(f"   更新频率: 每{update_frequency}回合")
        print(f"   保存频率: 每{save_frequency}回合")
        
        best_avg_reward = -float('inf')
        episode_count = 0
        
        for episode in range(total_episodes):
            # 随机选择机器人配置
            robot_config = self.sample_robot_config()
            num_joints = robot_config['num_joints']
            
            # 训练一个回合
            episode_reward, episode_steps, success = self.train_episode(robot_config)
            episode_count += 1
            
            # 定期更新模型
            if episode_count % update_frequency == 0:
                # 使用最后一个环境的状态作为next_state
                last_config = robot_config
                last_env = self.envs[last_config['num_joints']]
                last_gnn_embed = self.gnn_embeds[last_config['num_joints']]
                
                # 获取最后状态
                try:
                    last_obs = last_env.get_observation()
                    metrics = self.ppo.update(
                        torch.tensor(last_obs, dtype=torch.float32),
                        last_gnn_embed,
                        last_config['num_joints'],
                        ppo_epochs=4
                    )
                except:
                    # 如果获取不到最后状态，使用None
                    metrics = self.ppo.update(ppo_epochs=4)
                
                if metrics:
                    print(f"\n📊 Episode {episode+1}/{total_episodes} - 模型更新:")
                    print(f"   Actor Loss: {metrics['actor_loss']:.4f}")
                    print(f"   Critic Loss: {metrics['critic_loss']:.4f}")
                    print(f"   Entropy: {metrics['entropy']:.4f}")
                    print(f"   处理批次: {metrics.get('batches_processed', 'N/A')}")
            
            # 定期打印统计信息
            if (episode + 1) % 100 == 0:
                print(f"\n📈 Episode {episode+1} 统计:")
                
                for joints in self.envs.keys():
                    if len(self.episode_rewards[joints]) > 0:
                        avg_reward = np.mean(self.episode_rewards[joints])
                        avg_success = np.mean(self.success_rates[joints])
                        print(f"   {joints}关节: 平均奖励={avg_reward:.2f}, 成功率={avg_success:.2%}")
                
                # 计算总体平均奖励
                all_rewards = []
                for rewards in self.episode_rewards.values():
                    all_rewards.extend(rewards)
                
                if all_rewards:
                    current_avg_reward = np.mean(all_rewards)
                    print(f"   🎯 总体平均奖励: {current_avg_reward:.2f}")
                    
                    # 保存最佳模型
                    if current_avg_reward > best_avg_reward:
                        best_avg_reward = current_avg_reward
                        self.save_model(f"universal_ppo_best.pth")
                        print(f"   💾 保存最佳模型 (奖励: {best_avg_reward:.2f})")
            
            # 定期保存检查点
            if (episode + 1) % save_frequency == 0:
                self.save_model(f"universal_ppo_episode_{episode+1}.pth")
                print(f"💾 保存检查点: episode_{episode+1}")
        
        print(f"🎉 训练完成！最佳平均奖励: {best_avg_reward:.2f}")
        self.save_model("universal_ppo_final.pth")
    
    def save_model(self, filename):
        """保存模型"""
        os.makedirs("trained_models/universal_ppo", exist_ok=True)
        filepath = os.path.join("trained_models/universal_ppo", filename)
        self.ppo.save_model(filepath)
    
    def test_all_configurations(self, num_episodes=10):
        """测试所有配置的性能"""
        print(f"\n🧪 测试所有配置的性能 ({num_episodes}回合/配置)")
        
        for robot_config in self.joint_configs:
            num_joints = robot_config['num_joints']
            env = self.envs[num_joints]
            gnn_embed = self.gnn_embeds[num_joints]
            
            rewards = []
            successes = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0
                success = False
                
                for step in range(200):
                    action, _, _ = self.ppo.get_action(
                        torch.tensor(obs, dtype=torch.float32).to(self.device),
                        gnn_embed.to(self.device),
                        num_joints,
                        deterministic=True  # 测试时使用确定性策略
                    )
                    
                    obs, reward, done, truncated, info = env.step(action.cpu().numpy())
                    episode_reward += reward
                    
                    if info.get('success', False):
                        success = True
                    
                    if done or truncated:
                        break
                
                rewards.append(episode_reward)
                successes.append(success)
            
            avg_reward = np.mean(rewards)
            success_rate = np.mean(successes)
            
            print(f"   {num_joints}关节: 平均奖励={avg_reward:.2f}, 成功率={success_rate:.2%}")


def main():
    parser = argparse.ArgumentParser(description='通用PPO训练')
    parser.add_argument('--total_episodes', type=int, default=10000, help='总训练回合数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式')
    parser.add_argument('--model_path', type=str, help='模型路径（测试模式）')
    
    args = parser.parse_args()
    
    # 定义不同的机器人配置
    joint_configs = [
        {'num_joints': 3, 'link_lengths': [60, 60, 60]},
        {'num_joints': 4, 'link_lengths': [50, 50, 50, 50]}, 
        {'num_joints': 5, 'link_lengths': [40, 40, 40, 40, 40]},
        {'num_joints': 6, 'link_lengths': [35, 35, 35, 35, 35, 35]}
    ]
    
    # 创建训练器
    trainer = UniversalRobotTrainer(joint_configs, device=args.device)
    
    if args.test_only:
        if args.model_path:
            trainer.ppo.load_model(args.model_path)
            print(f"✅ 加载模型: {args.model_path}")
        trainer.test_all_configurations()
    else:
        # 开始训练
        trainer.train(total_episodes=args.total_episodes)
        
        # 训练完成后测试
        print(f"\n🧪 训练完成，测试最终模型性能...")
        trainer.test_all_configurations()


if __name__ == "__main__":
    main()
