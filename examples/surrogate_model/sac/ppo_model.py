#!/usr/bin/env python3
"""
PPO版本的Attention模型 - 修复版
基于AttnModel的PPO实现，适合维持任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from collections import deque
from torch.distributions import Normal
import copy

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_dataset"))
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_model"))

from data_utils import prepare_joint_q_input, prepare_reacher2d_joint_q_input, prepare_dynamic_vertex_v
from attn_model.attn_model import AttnModel

class AttentionPPOActor(nn.Module):
    """基于Attention的PPO Actor网络"""
    def __init__(self, attn_model, action_dim, log_std_init=-1.0):
        super(AttentionPPOActor, self).__init__()
        self.attn_model = attn_model
        self.action_dim = action_dim
        
        # 🎯 PPO使用固定或学习的log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        """前向传播，输出动作分布参数"""
        # AttnModel输出动作均值
        mean = self.attn_model(joint_q, vertex_k, vertex_v, vertex_mask)  # [B, action_dim]
        
        # 标准差（可学习或固定）
        std = torch.exp(self.log_std.expand_as(mean))
        
        return mean, std
    
    def get_action(self, joint_q, vertex_k, vertex_v, vertex_mask=None, deterministic=False):
        """获取动作和log概率"""
        mean, std = self.forward(joint_q, vertex_k, vertex_v, vertex_mask)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean, std
    
    def evaluate_action(self, joint_q, vertex_k, vertex_v, action, vertex_mask=None):
        """评估给定动作的log概率和熵"""
        mean, std = self.forward(joint_q, vertex_k, vertex_v, vertex_mask)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy

class AttentionPPOCritic(nn.Module):
    """基于Attention的PPO Critic网络 - 修复版"""
    def __init__(self, attn_model, action_dim, hidden_dim=256, device='cpu'):
        super(AttentionPPOCritic, self).__init__()
        self.attn_model = attn_model
        self.action_dim = action_dim  # 🔧 改为使用action_dim而不是joint_embed_dim
        self.device = device
        
        # 🔧 值函数网络 - 直接从AttnModel输出维度（action_dim）到值函数
        self.value_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),  # 🔧 AttnModel输出[B, action_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        """前向传播，输出状态值"""
        # 使用AttnModel编码状态特征
        state_features = self.attn_model(joint_q, vertex_k, vertex_v, vertex_mask)  # [B, action_dim]
        
        # 🔧 直接使用state_features，不需要投影和池化
        value = self.value_net(state_features)  # [B, 1]
        
        return value

class RolloutBuffer:
    """PPO的经验收集缓冲区"""
    def __init__(self, buffer_size, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.joint_q = []
        self.vertex_k = []
        self.vertex_v = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        self.ptr = 0
    
    def store(self, joint_q, vertex_k, vertex_v, action, reward, value, log_prob, done):
        """存储一步经验"""
        self.joint_q.append(joint_q.cpu())
        self.vertex_k.append(vertex_k.cpu())
        self.vertex_v.append(vertex_v.cpu())
        self.actions.append(action.cpu())
        self.rewards.append(reward)
        # 🔧 确保value维度一致，都是标量
        self.values.append(value.cpu().squeeze())  # 🔧 添加squeeze()确保是标量
        self.log_probs.append(log_prob.cpu() if log_prob is not None else torch.tensor(0.0))
        self.dones.append(done)
        self.ptr += 1
    
    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算GAE优势函数"""
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        
        # 🔧 确保last_value也是标量
        last_value_scalar = last_value.cpu().squeeze()
        
        # 🔧 将所有values转换为标量tensor列表，然后拼接
        value_tensors = []
        for v in self.values:
            if v.dim() == 0:  # 已经是标量
                value_tensors.append(v)
            else:  # 需要squeeze
                value_tensors.append(v.squeeze())
        
        # 添加last_value
        value_tensors.append(last_value_scalar)
        
        # 拼接成一维tensor
        values = torch.stack(value_tensors)
        
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size=None):
        """获取批量数据"""
        if batch_size is None:
            batch_size = len(self.joint_q)
        
        indices = torch.randperm(len(self.joint_q))[:batch_size]
        
        batch = {
            'joint_q': torch.stack([self.joint_q[i] for i in indices]).to(self.device),
            'vertex_k': torch.stack([self.vertex_k[i] for i in indices]).to(self.device),
            'vertex_v': torch.stack([self.vertex_v[i] for i in indices]).to(self.device),
            'actions': torch.stack([self.actions[i] for i in indices]).to(self.device),
            'old_log_probs': torch.stack([self.log_probs[i] for i in indices]).to(self.device),
            'advantages': self.advantages[indices].to(self.device),
            'returns': self.returns[indices].to(self.device),
        }
        
        return batch

class AttentionPPOWithBuffer:
    """基于Attention的PPO算法实现 - 修复版"""
    def __init__(self, attn_model, action_dim, buffer_size=2048, batch_size=64, 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, 
                 device='cpu', env_type='reacher2d'):
        
        self.device = device
        self.action_dim = action_dim
        self.env_type = env_type
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # 创建独立的网络
        actor_model = copy.deepcopy(attn_model)
        critic_model = copy.deepcopy(attn_model)
        
        self.actor = AttentionPPOActor(actor_model, action_dim).to(device)
        self.critic = AttentionPPOCritic(critic_model, action_dim, device=device).to(device)  # 🔧 传递action_dim
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(buffer_size, device)
        
        # 统计信息
        self.update_count = 0
    
    def get_action(self, obs, gnn_embeds, num_joints=12, deterministic=False):
        """获取动作"""
        # 准备输入数据
        if self.env_type == 'reacher2d':
            joint_q = prepare_reacher2d_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        else:
            joint_q = prepare_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        
        vertex_k = gnn_embeds.unsqueeze(0)
        vertex_v = prepare_dynamic_vertex_v(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints, self.env_type)
        
        with torch.no_grad():
            action, log_prob, mean, std = self.actor.get_action(
                joint_q, vertex_k, vertex_v, deterministic=deterministic
            )
            value = self.critic(joint_q, vertex_k, vertex_v)
        
        # 🔧 动作缩放到环境范围
        if self.env_type == 'reacher2d':
            action_scale = 100.0  # 缩放到[-100, 100]
            scaled_action = torch.tanh(action.squeeze(0)) * action_scale
            return scaled_action, log_prob.squeeze(0) if log_prob is not None else None, value.squeeze(0)
        else:
            return torch.tanh(action.squeeze(0)), log_prob.squeeze(0) if log_prob is not None else None, value.squeeze(0)
    
    def store_experience(self, obs, gnn_embeds, action, reward, done, log_prob, value, num_joints=12):
        """存储经验"""
        # 准备输入数据
        if self.env_type == 'reacher2d':
            joint_q = prepare_reacher2d_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        else:
            joint_q = prepare_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        
        vertex_k = gnn_embeds.unsqueeze(0)
        vertex_v = prepare_dynamic_vertex_v(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints, self.env_type)
        
        self.buffer.store(joint_q.squeeze(0), vertex_k.squeeze(0), vertex_v.squeeze(0),
                         action, reward, value, log_prob, done)
    
    def update(self, next_obs=None, next_gnn_embeds=None, num_joints=12, ppo_epochs=4):
        """PPO更新"""
        # 计算最后一个状态的值函数（用于GAE）
        if next_obs is not None and next_gnn_embeds is not None:
            if self.env_type == 'reacher2d':
                next_joint_q = prepare_reacher2d_joint_q_input(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints)
            else:
                next_joint_q = prepare_joint_q_input(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints)
            
            next_vertex_k = next_gnn_embeds.unsqueeze(0)
            next_vertex_v = prepare_dynamic_vertex_v(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints, self.env_type)
            
            with torch.no_grad():
                last_value = self.critic(next_joint_q, next_vertex_k, next_vertex_v)
        else:
            last_value = torch.zeros(1, 1)
        
        # 计算优势函数
        self.buffer.compute_advantages(last_value, self.gamma, self.gae_lambda)
        
        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(ppo_epochs):
            batch = self.buffer.get_batch(self.batch_size)
            
            # 计算当前策略的log概率和熵
            log_probs, entropy = self.actor.evaluate_action(
                batch['joint_q'], batch['vertex_k'], batch['vertex_v'], batch['actions']
            )
            
            # 计算当前值函数
            values = self.critic(batch['joint_q'], batch['vertex_k'], batch['vertex_v'])
            
            # PPO Actor损失
            ratio = torch.exp(log_probs - batch['old_log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantages']
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            
            # Critic损失
            critic_loss = F.mse_loss(values.squeeze(), batch['returns'])
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        self.buffer.clear()
        self.update_count += 1
        
        return {
            'actor_loss': total_actor_loss / ppo_epochs,
            'critic_loss': total_critic_loss / ppo_epochs,
            'entropy': total_entropy / ppo_epochs,
            'update_count': self.update_count
        }
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_count': self.update_count
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)

# 测试代码
if __name__ == "__main__":
    print("🚀 测试PPO + Attention模型...")
    
    # 创建模型
    attn_model = AttnModel(128, 128, 130, 4)
    ppo = AttentionPPOWithBuffer(
        attn_model, action_dim=12, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 模拟数据测试
    obs = torch.randn(40)
    gnn_embeds = torch.randn(12, 128)
    
    # 测试获取动作
    action, log_prob, value = ppo.get_action(obs, gnn_embeds, deterministic=False)
    print(f"✅ 动作形状: {action.shape}")
    print(f"✅ Log概率: {log_prob.item() if log_prob is not None else None}")
    print(f"✅ 状态值: {value.item()}")
    
    # 测试存储经验
    ppo.store_experience(obs, gnn_embeds, action, 1.0, False, log_prob, value)
    print(f"✅ 经验存储成功")
    
    print("🎯 PPO模型创建成功！")