#!/usr/bin/env python3
"""
简单的SAC实现 - 不使用Attention机制
基于标准的MLP网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from torch.distributions import Normal

# 定义经验元组
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])

class ReplayBuffer:
    """简单的经验回放缓冲区"""
    def __init__(self, capacity=100000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """存储一个经验"""
        experience = Experience(
            state=state.cpu() if torch.is_tensor(state) else state,
            action=action.cpu() if torch.is_tensor(action) else action,
            reward=reward.cpu() if torch.is_tensor(reward) else reward,
            next_state=next_state.cpu() if torch.is_tensor(next_state) else next_state,
            done=done.cpu() if torch.is_tensor(done) else done
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """采样一个batch的经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.stack([torch.FloatTensor(e.state) for e in experiences]).to(self.device)
        actions = torch.stack([torch.FloatTensor(e.action) for e in experiences]).to(self.device)
        rewards = torch.stack([torch.FloatTensor([e.reward]) for e in experiences]).to(self.device)
        next_states = torch.stack([torch.FloatTensor(e.next_state) for e in experiences]).to(self.device)
        dones = torch.stack([torch.FloatTensor([e.done]) for e in experiences]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size


class SimpleActor(nn.Module):
    """简单的Actor网络 - 使用MLP"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(SimpleActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """前向传播"""
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        
        # 修正tanh的log_prob
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, mean


class SimpleCritic(nn.Module):
    """简单的Critic网络 - 使用MLP"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SimpleCritic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        """前向传播"""
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SimpleSAC:
    """简单的SAC算法实现"""
    def __init__(self, state_dim, action_dim, 
                 buffer_capacity=100000, batch_size=256, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, device='cpu'):
        
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 创建网络
        self.actor = SimpleActor(state_dim, action_dim).to(device)
        self.critic1 = SimpleCritic(state_dim, action_dim).to(device)
        self.critic2 = SimpleCritic(state_dim, action_dim).to(device)
        
        # Target networks
        self.target_critic1 = SimpleCritic(state_dim, action_dim).to(device)
        self.target_critic2 = SimpleCritic(state_dim, action_dim).to(device)
        
        # 初始化target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 冻结target networks
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        # 自动调整alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_capacity, device)
        
        print(f"🎯 简单SAC初始化完成:")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   学习率: {lr}")
        print(f"   Buffer大小: {buffer_capacity}")
        print(f"   Alpha: {alpha}")
        
    def get_action(self, state, deterministic=False):
        """获取动作"""
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state)
                
        return action.squeeze(0).cpu().numpy()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def soft_update_targets(self):
        """软更新target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self):
        """更新网络"""
        if not self.memory.can_sample(self.batch_size):
            return None
            
        # 从buffer采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # === Critic Update ===
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            max_norm=1.0
        )
        self.critic_optimizer.step()
        
        # === Actor Update ===
        new_actions, log_probs, _ = self.actor.sample(states)
        
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # === Alpha Update ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 软更新target networks
        self.soft_update_targets()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'buffer_size': len(self.memory)
        }


if __name__ == "__main__":
    # 简单测试
    print("测试简单SAC实现...")
    
    state_dim = 20  # 示例状态维度
    action_dim = 6  # 示例动作维度
    
    sac = SimpleSAC(state_dim, action_dim)
    
    # 测试动作采样
    test_state = np.random.randn(state_dim)
    action = sac.get_action(test_state)
    print(f"测试动作形状: {action.shape}")
    print(f"动作范围: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试经验存储
    next_state = np.random.randn(state_dim)
    reward = np.random.randn()
    done = False
    
    sac.store_experience(test_state, action, reward, next_state, done)
    print(f"Buffer大小: {len(sac.memory)}")
    
    print("✅ 简单SAC测试完成！")
