#!/usr/bin/env python3
"""
PPO版本的Attention模型 - 修复维度错误版
基于AttnModel的PPO实现，支持2,3,4,5,6...任意关节数
修复Critic Loss过高和维度不匹配问题
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
    def __init__(self, attn_model, action_dim, log_std_init=-1.0, device='cpu'):
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
    
    # def get_action(self, joint_q, vertex_k, vertex_v, vertex_mask=None, deterministic=False):
    #     """获取动作和log概率"""
    #     mean, std = self.forward(joint_q, vertex_k, vertex_v, vertex_mask)
        
    #     if deterministic:
    #         action = mean
    #         log_prob = None
    #     else:
    #         dist = Normal(mean, std)
    #         action = dist.sample()
    #         log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
    #     return action, log_prob, mean, std
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
    """修复版PPO Critic网络 - 解决维度错误"""
    def __init__(self, attn_model, hidden_dim=256, device='cpu'):
        super(AttentionPPOCritic, self).__init__()
        self.attn_model = copy.deepcopy(attn_model)  # 独立的AttnModel副本
        self.device = device
        
        # 🔧 修复: 使用简单稳定的网络结构
        self.value_head = nn.Sequential(
            nn.Linear(130, hidden_dim),  # 直接使用平均池化特征
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 🔧 修复: 值函数输出归一化
        self.value_scale = 200.0  # 增大输出范围，匹配环境奖励
        
    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        """前向传播，输出状态价值 - 简化稳定版"""
        batch_size, num_joints, feature_dim = joint_q.shape
        
        # 🔧 使用简单的平均池化，避免复杂的tensor操作
        avg_features = joint_q.mean(dim=1)  # [B, 130]
        
        # 直接使用平均特征
        value = self.value_head(avg_features)  # [B, 1]
        
        # 🔧 输出缩放
        value = torch.tanh(value) * self.value_scale
        
        return value

class RolloutBuffer:
    """PPO经验回放缓冲区"""
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
        self.values.append(value.cpu().squeeze())  # 确保是标量
        self.log_probs.append(log_prob.cpu() if log_prob is not None else torch.tensor(0.0))
        self.dones.append(done)
        self.ptr += 1
    
    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算优势函数和回报"""
        # 🔧 修复: 改进优势函数计算
        values = torch.stack(self.values + [last_value.cpu().squeeze()])
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        # 🔧 奖励归一化，稳定训练
        if len(rewards) > 1:
            reward_mean = rewards.mean()
            reward_std = rewards.std() + 1e-8
            rewards = (rewards - reward_mean) / reward_std
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
        
        # 🔧 优势函数归一化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, batch_size):
        """获取批量数据"""
        indices = torch.randperm(len(self.joint_q))[:batch_size]
        
        batch = {
            'joint_q': torch.stack([self.joint_q[i] for i in indices]),
            'vertex_k': torch.stack([self.vertex_k[i] for i in indices]),
            'vertex_v': torch.stack([self.vertex_v[i] for i in indices]),
            'actions': torch.stack([self.actions[i] for i in indices]),
            'old_log_probs': torch.stack([self.log_probs[i] for i in indices]),
            'advantages': torch.stack([self.advantages[i] for i in indices]),
            'returns': torch.stack([self.returns[i] for i in indices]),
            'old_values': torch.stack([self.values[i] for i in indices])
        }
        
        return batch

class AttentionPPOWithBuffer:
    """修复版PPO算法实现"""
    def __init__(self, attn_model, action_dim, buffer_size=2048, batch_size=64, 
                 lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, 
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, 
                 device='cpu', env_type='reacher2d'):
        
        self.action_dim = action_dim
        self.device = device
        self.env_type = env_type
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.actor = AttentionPPOActor(attn_model, action_dim, device=device)
        self.critic = AttentionPPOCritic(attn_model, device=device)
        
        # 🔧 修复: 使用不同的学习率
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr*0.5)  # Critic学习率降低
        
        # 🔧 修复: 添加学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)
        
        # 经验缓冲区
        self.buffer = RolloutBuffer(buffer_size, device)
        
        # 统计信息
        self.update_count = 0
        self.recent_losses = deque(maxlen=10)
        
        print(f"🎯 PPO初始化完成:")
        print(f"   Action维度: {action_dim}")
        print(f"   学习率: Actor={lr}, Critic={lr*0.5}")
        print(f"   Buffer大小: {buffer_size}")
        print(f"   Clip epsilon: {clip_epsilon}")
        print(f"   Value Scale: {self.critic.value_scale}")
    
    def get_action(self, obs, gnn_embeds, num_joints, deterministic=False):
        """获取动作"""
        joint_q, vertex_k, vertex_v = self._prepare_inputs(obs, gnn_embeds, num_joints)
        
        with torch.no_grad():
            action, log_prob, _, _ = self.actor.get_action(
                joint_q, vertex_k, vertex_v, deterministic=deterministic
            )
            value = self.critic(joint_q, vertex_k, vertex_v)
        
        # 🔧 添加动作缩放
        if self.env_type == 'reacher2d':
            action_scale = 10.0  # 匹配环境的max_torque
            scaled_action = torch.tanh(action) * action_scale  # 先tanh限制到[-1,1]再缩放
            return scaled_action.squeeze(0), log_prob.squeeze(0) if log_prob is not None else None, value.squeeze(0)
        else:
            return action.squeeze(0), log_prob.squeeze(0) if log_prob is not None else None, value.squeeze(0)
    
    def store_experience(self, obs, gnn_embeds, action, reward, done, 
                        log_prob=None, value=None, num_joints=None):
        """存储经验"""
        joint_q, vertex_k, vertex_v = self._prepare_inputs(obs, gnn_embeds, num_joints)
        
        if value is None:
            with torch.no_grad():
                value = self.critic(joint_q, vertex_k, vertex_v)
        
        self.buffer.store(
            joint_q.squeeze(0), vertex_k.squeeze(0), vertex_v.squeeze(0),
            action, reward, value, log_prob, done
        )
    
    def update(self, next_obs=None, next_gnn_embeds=None, num_joints=None, ppo_epochs=4):
        """PPO更新"""
        if len(self.buffer.joint_q) < self.batch_size:
            return None
        
        # 计算最后状态的价值
        if next_obs is not None and next_gnn_embeds is not None:
            joint_q, vertex_k, vertex_v = self._prepare_inputs(next_obs, next_gnn_embeds, num_joints)
            with torch.no_grad():
                last_value = self.critic(joint_q, vertex_k, vertex_v)
        else:
            last_value = torch.zeros(1)
        
        # 计算优势函数
        self.buffer.compute_advantages(last_value, self.gamma, self.gae_lambda)
        
        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(ppo_epochs):
            batch = self.buffer.get_batch(min(self.batch_size, len(self.buffer.joint_q)))
            
            # 移动到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # 前向传播
            new_log_probs, entropy = self.actor.evaluate_action(
                batch['joint_q'], batch['vertex_k'], batch['vertex_v'], batch['actions']
            )
            new_values = self.critic(batch['joint_q'], batch['vertex_k'], batch['vertex_v']).squeeze(-1)
            
            # Actor loss (PPO clip)
            ratio = torch.exp(new_log_probs.squeeze(-1) - batch['old_log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantages']
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 🔧 修复: 改进Critic Loss计算
            # 使用Huber Loss替代MSE，更稳定
            critic_loss = F.smooth_l1_loss(new_values, batch['returns'])
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # 总损失
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # 🔧 修复: 分别优化Actor和Critic
            # Actor更新
            self.actor_optimizer.zero_grad()
            actor_total_loss = actor_loss + self.entropy_coef * entropy_loss
            actor_total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic更新
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # 学习率调度
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 清空缓冲区
        self.buffer.clear()
        self.update_count += 1
        
        # 计算损失趋势
        current_total_loss = total_actor_loss + total_critic_loss
        self.recent_losses.append(current_total_loss)
        
        loss_trend = "📈 上升"
        if len(self.recent_losses) >= 2:
            if self.recent_losses[-1] < self.recent_losses[-2]:
                loss_trend = "📉 下降"
        
        metrics = {
            'actor_loss': total_actor_loss / ppo_epochs,
            'critic_loss': total_critic_loss / ppo_epochs,
            'total_loss': current_total_loss / ppo_epochs,
            'entropy': total_entropy / ppo_epochs,
            'update_count': self.update_count,
            'loss_trend': loss_trend,
            'learning_rate': self.actor_optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def _prepare_inputs(self, obs, gnn_embeds, num_joints):
        """准备输入数据 - 修复版"""
        # 🔧 关键修复：确保输入有batch维度
        if obs.dim() == 1:
            obs_batch = obs.unsqueeze(0)  # [1, obs_dim]
        else:
            obs_batch = obs
            
        if gnn_embeds.dim() == 2:
            gnn_embeds_batch = gnn_embeds.unsqueeze(0)  # [1, N, 128]
        else:
            gnn_embeds_batch = gnn_embeds
        
        # 🔧 修复：正确调用数据准备函数，传入所有必需参数
        if self.env_type == 'reacher2d':
            joint_q = prepare_reacher2d_joint_q_input(obs_batch, gnn_embeds_batch, num_joints)
            vertex_v = prepare_dynamic_vertex_v(obs_batch, gnn_embeds_batch, num_joints, self.env_type)
        else:
            joint_q = prepare_joint_q_input(obs_batch, gnn_embeds_batch, num_joints)
            vertex_v = prepare_dynamic_vertex_v(obs_batch, gnn_embeds_batch, num_joints, self.env_type)
        
        vertex_k = gnn_embeds_batch  # [B, N, 128]
        
        # 确保在正确设备上
        joint_q = joint_q.to(self.device)
        vertex_k = vertex_k.to(self.device)
        vertex_v = vertex_v.to(self.device)
        
        return joint_q, vertex_k, vertex_v
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'action_dim': self.action_dim
        }, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)