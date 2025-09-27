#!/usr/bin/env python3
"""
简单的MADDPG实现，带有手动loss计算和显示：
1. 每个关节由一个DDPG智能体控制
2. 手动计算和显示Actor/Critic损失
3. 可视化训练过程
4. 详细的训练统计
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
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

class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    """Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class DDPGAgent:
    """单个DDPG智能体"""
    def __init__(self, agent_id, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # 网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 初始化目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Loss记录
        self.actor_losses = []
        self.critic_losses = []
        
        print(f"   🤖 DDPG Agent {agent_id}: 状态维度={state_dim}, 动作维度={action_dim}")
    
    def act(self, state, noise_scale=0.1):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy()[0]
            # 添加噪声
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            return action
    
    def update(self, states, actions, rewards, next_states, dones):
        """更新网络并返回loss"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update()
        
        # 记录loss
        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        self.actor_losses.append(actor_loss_value)
        self.critic_losses.append(critic_loss_value)
        
        return actor_loss_value, critic_loss_value
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class SimpleMADDPG:
    """简单的MADDPG实现"""
    def __init__(self, num_joints=3):
        self.num_joints = num_joints
        
        print(f"🌟 创建简单MADDPG系统: {num_joints}个智能体协作")
        
        # 创建环境
        self.env = create_env(num_joints, render_mode='human', show_position_info=True)
        
        # 获取状态和动作维度
        state_dim = self.env.observation_space.shape[0]
        action_dim = 1  # 每个智能体控制一个关节
        
        print(f"🔧 环境配置: 状态维度={state_dim}, 每个智能体动作维度={action_dim}")
        
        # 创建智能体
        self.agents = []
        for i in range(num_joints):
            agent = DDPGAgent(i, state_dim, action_dim)
            self.agents.append(agent)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        print(f"🌟 简单MADDPG系统初始化完成")
    
    def train(self, total_timesteps=5000, batch_size=64):
        """训练MADDPG"""
        print(f"\n🎯 开始简单MADDPG训练 {total_timesteps} 步...")
        print(f"   每个智能体控制一个关节，显示实时loss信息")
        
        obs, _ = self.env.reset()
        episode_count = 0
        episode_reward = 0
        step_count = 0
        
        # 训练统计
        training_stats = {
            'episode_rewards': [],
            'episode_successes': [],
            'actor_losses': [[] for _ in range(self.num_joints)],
            'critic_losses': [[] for _ in range(self.num_joints)]
        }
        
        start_time = time.time()
        
        try:
            while step_count < total_timesteps:
                # 所有智能体同时决策
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.act(obs, noise_scale=0.1)
                    actions.append(action[0])  # 取出标量
                
                # 执行联合动作
                joint_action = np.array(actions)
                next_obs, reward, terminated, truncated, info = self.env.step(joint_action)
                episode_reward += reward
                step_count += 1
                
                # 存储经验
                self.replay_buffer.push(obs, actions, reward, next_obs, terminated or truncated)
                
                # 训练智能体
                if len(self.replay_buffer) > batch_size and step_count % 10 == 0:
                    self._train_agents(batch_size, training_stats, step_count)
                
                obs = next_obs
                
                # 重置环境
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    episode_count += 1
                    
                    # 记录统计
                    training_stats['episode_rewards'].append(episode_reward)
                    success = info.get('is_success', False) if info else False
                    training_stats['episode_successes'].append(success)
                    
                    if episode_count % 5 == 0:
                        distance = info.get('distance_to_target', 0) if info else 0
                        recent_success_rate = np.mean(training_stats['episode_successes'][-10:]) if len(training_stats['episode_successes']) >= 10 else 0
                        print(f"   🎮 Episode {episode_count}: 奖励={episode_reward:.2f}, 距离={distance:.4f}, 成功={'✅' if success else '❌'}, 近期成功率={recent_success_rate:.1%}")
                    
                    episode_reward = 0
                
                # 定期输出进度（每5000步）
                if step_count % 5000 == 0:
                    elapsed_time = time.time() - start_time
                    fps = step_count / elapsed_time
                    print(f"\n   🚀 步数 {step_count}/{total_timesteps}, FPS: {fps:.1f}, 用时: {elapsed_time/60:.1f}分钟")
                    self._print_loss_summary(training_stats)
        
        except KeyboardInterrupt:
            print(f"\n⚠️ 训练被用户中断")
        
        training_time = time.time() - start_time
        print(f"\n✅ 简单MADDPG训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"🎯 总episodes: {episode_count}")
        print(f"📊 平均FPS: {step_count/training_time:.1f}")
        
        # 最终统计
        self._print_final_summary(training_stats)
        
        return episode_count, training_stats
    
    def _train_agents(self, batch_size, training_stats, step_count):
        """训练所有智能体"""
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 为每个智能体准备数据并训练
        for i, agent in enumerate(self.agents):
            # 提取该智能体的动作
            agent_actions = actions[:, i:i+1]  # 保持二维
            
            # 训练智能体
            actor_loss, critic_loss = agent.update(states, agent_actions, rewards, next_states, dones)
            
            # 记录loss
            training_stats['actor_losses'][i].append(actor_loss)
            training_stats['critic_losses'][i].append(critic_loss)
            
            # 每500步显示一次loss（减少输出频率）
            if step_count % 500 == 0:
                print(f"     🤖 Agent {i}: Actor_Loss={actor_loss:.4f}, Critic_Loss={critic_loss:.4f}")
    
    def _print_loss_summary(self, training_stats):
        """打印loss统计摘要"""
        print(f"     📊 Loss统计摘要 (最近10次训练):")
        for i in range(self.num_joints):
            if training_stats['actor_losses'][i]:
                recent_actor = training_stats['actor_losses'][i][-10:]
                recent_critic = training_stats['critic_losses'][i][-10:]
                avg_actor = np.mean(recent_actor)
                avg_critic = np.mean(recent_critic)
                print(f"       Agent {i}: 平均Actor_Loss={avg_actor:.4f}, 平均Critic_Loss={avg_critic:.4f}")
    
    def _print_final_summary(self, training_stats):
        """打印最终统计"""
        print(f"\n📊 最终训练统计:")
        print("-" * 60)
        
        # Loss统计
        for i in range(self.num_joints):
            if training_stats['actor_losses'][i]:
                actor_losses = training_stats['actor_losses'][i]
                critic_losses = training_stats['critic_losses'][i]
                
                print(f"🤖 Agent {i}:")
                print(f"   Actor Loss: 平均={np.mean(actor_losses):.4f}, 最小={np.min(actor_losses):.4f}, 最大={np.max(actor_losses):.4f}")
                print(f"   Critic Loss: 平均={np.mean(critic_losses):.4f}, 最小={np.min(critic_losses):.4f}, 最大={np.max(critic_losses):.4f}")
        
        # Episode统计
        if training_stats['episode_rewards']:
            print(f"\n🎯 Episode统计:")
            print(f"   总Episodes: {len(training_stats['episode_rewards'])}")
            print(f"   平均奖励: {np.mean(training_stats['episode_rewards']):.2f}")
            print(f"   成功率: {np.mean(training_stats['episode_successes']):.1%}")
    
    def test(self, n_episodes=5):
        """测试训练好的MADDPG"""
        print(f"\n🧪 开始测试简单MADDPG {n_episodes} episodes...")
        
        success_count = 0
        episode_rewards = []
        episode_distances = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            
            print(f"\n   🎮 Episode {episode+1} 开始...")
            
            for step in range(100):
                # 所有智能体协作决策（确定性）
                actions = []
                for agent in self.agents:
                    action = agent.act(obs, noise_scale=0.0)  # 测试时不加噪声
                    actions.append(action[0])
                
                # 执行联合动作
                joint_action = np.array(actions)
                obs, reward, terminated, truncated, info = self.env.step(joint_action)
                episode_reward += reward
                
                # 检查成功和距离
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
            
            print(f"   📊 Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_count / n_episodes
        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances)
        
        print(f"\n🎯 简单MADDPG测试结果:")
        print(f"   🏆 成功率: {success_rate:.1%} ({success_count}/{n_episodes})")
        print(f"   💰 平均奖励: {avg_reward:.2f}")
        print(f"   📏 平均最小距离: {avg_distance:.4f}")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_distance': avg_distance
        }

def main():
    """主函数"""
    print("🌟 简单MADDPG训练系统 (手动Loss计算)")
    print("🤖 策略: 每个关节由独立DDPG智能体控制，多智能体协作学习")
    print("🎯 目标: 3关节Reacher任务")
    print("👀 特点: 实时可视化 + 详细Loss显示")
    print("📊 监控: 手动计算Actor Loss, Critic Loss, Episode奖励, 成功率")
    print()
    
    # 创建MADDPG系统
    maddpg = SimpleMADDPG(num_joints=3)
    
    # 训练
    print("\n" + "="*60)
    episode_count, training_stats = maddpg.train(total_timesteps=30000)
    
    # 测试
    print("\n" + "="*60)
    result = maddpg.test(n_episodes=3)
    
    print(f"\n🎉 简单MADDPG训练完成!")
    print(f"   📈 训练episodes: {episode_count}")
    print(f"   🏆 最终成功率: {result['success_rate']:.1%}")
    print(f"   💰 平均奖励: {result['avg_reward']:.2f}")
    print(f"   📏 平均距离: {result['avg_distance']:.4f}")
    
    print(f"\n🔍 简单MADDPG特点:")
    print(f"   ✅ 手动实现DDPG，完全控制训练过程")
    print(f"   ✅ 实时显示每个智能体的Actor和Critic损失")
    print(f"   ✅ 多智能体协作：每个关节独立学习但共享环境")
    print(f"   ✅ 可视化训练过程，直观观察学习效果")

if __name__ == "__main__":
    main()
