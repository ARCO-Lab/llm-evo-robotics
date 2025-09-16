#!/usr/bin/env python3
"""
通用PPO模型 - 支持任意关节数的机器人控制
基于注意力机制，动态适应不同的关节数量
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


class UniversalAttnModel(nn.Module):
    """通用注意力模型 - 支持动态关节数输出"""
    def __init__(self, vertex_key_size=128, vertex_value_size=130, joint_query_size=130, num_heads=4):
        super(UniversalAttnModel, self).__init__()
        self.vertex_key_size = vertex_key_size
        self.vertex_value_size = vertex_value_size
        self.joint_query_size = joint_query_size
        self.num_heads = num_heads
        
        # 关节查询编码器
        self.joint_q_encoder = nn.Sequential(
            nn.Linear(joint_query_size, 64),
            nn.ReLU(),
            nn.Linear(64, vertex_key_size * num_heads)
        )
        
        # 🎯 关键改进：每个关节独立的输出头
        self.joint_output_layer = nn.Sequential(
            nn.Linear(vertex_value_size * num_heads, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 每个关节输出一个标量
        )

    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        """
        Args:
            joint_q: [B, num_joints, 130] - 动态关节数
            vertex_k: [B, N, 128]
            vertex_v: [B, N, 130]
        Returns:
            output: [B, num_joints] - 动态输出维度
        """
        batch_size, num_joints, _ = joint_q.shape
        
        # 编码关节查询
        joint_q_encoded = self.joint_q_encoder(joint_q)  # [B, num_joints, 512]
        joint_q_encoded = joint_q_encoded.view(batch_size, num_joints, self.num_heads, self.vertex_key_size)
        
        # 注意力计算 - 使用简化版本
        attn_output = self._compute_attention(vertex_k, vertex_v, joint_q_encoded, vertex_mask)
        
        # 展平多头输出
        attn_output = attn_output.view(batch_size, num_joints, self.num_heads * self.vertex_value_size)
        
        # 🎯 每个关节独立输出
        output = self.joint_output_layer(attn_output)  # [B, num_joints, 1]
        output = output.squeeze(-1)  # [B, num_joints]
        
        return output
    
    def _compute_attention(self, vertex_k, vertex_v, joint_q, vertex_mask=None):
        """简化的注意力计算 - 修复维度问题"""
        batch_size, num_joints, num_heads, key_size = joint_q.shape
        _, num_vertices, vertex_k_dim = vertex_k.shape
        _, _, vertex_v_dim = vertex_v.shape
        
        # 🔧 修复：正确处理维度匹配
        # 将joint_q重塑为 [B*J*H, 1, key_size]
        joint_q_flat = joint_q.view(batch_size * num_joints * num_heads, 1, key_size)
        
        # 扩展vertex_k和vertex_v到所有关节和头
        vertex_k_expanded = vertex_k.unsqueeze(1).unsqueeze(1).expand(
            batch_size, num_joints, num_heads, num_vertices, vertex_k_dim
        ).contiguous().view(batch_size * num_joints * num_heads, num_vertices, vertex_k_dim)
        
        vertex_v_expanded = vertex_v.unsqueeze(1).unsqueeze(1).expand(
            batch_size, num_joints, num_heads, num_vertices, vertex_v_dim
        ).contiguous().view(batch_size * num_joints * num_heads, num_vertices, vertex_v_dim)
        
        # 计算注意力分数
        scores = torch.matmul(joint_q_flat, vertex_k_expanded.transpose(-2, -1))  # [B*J*H, 1, N]
        scores = scores / (key_size ** 0.5)
        
        # 应用掩码
        if vertex_mask is not None:
            # 扩展掩码
            mask_expanded = vertex_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(
                batch_size, num_joints, num_heads, 1, num_vertices
            ).contiguous().view(batch_size * num_joints * num_heads, 1, num_vertices)
            scores.masked_fill_(mask_expanded, -1e9)
        
        # 注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # [B*J*H, 1, N]
        output_flat = torch.matmul(attn_weights, vertex_v_expanded)  # [B*J*H, 1, vertex_v_dim]
        
        # 重塑回原始维度
        output = output_flat.view(batch_size, num_joints, num_heads, vertex_v_dim)
        
        return output


class UniversalPPOActor(nn.Module):
    """通用PPO Actor - 支持任意关节数"""
    def __init__(self, attn_model, log_std_init=-1.5, device='cpu'):
        super(UniversalPPOActor, self).__init__()
        self.attn_model = attn_model
        self.device = device
        
        # 🔧 修复：使用更保守的初始值，防止entropy爆炸
        self.log_std_base = nn.Parameter(torch.tensor(log_std_init))
        
    # def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
    #     """前向传播，输出动作分布参数"""
    #     batch_size, num_joints, _ = joint_q.shape
        
    #     # 获取动作均值
    #     mean = self.attn_model(joint_q, vertex_k, vertex_v, vertex_mask)  # [B, num_joints]
        
    #     # 动态生成标准差
    #     std = torch.exp(self.log_std_base).expand_as(mean)  # [B, num_joints]
        
    #     return mean, std

    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        """前向传播，输出动作分布参数"""
        batch_size, num_joints, _ = joint_q.shape
        
        # 获取动作均值
        mean = self.attn_model(joint_q, vertex_k, vertex_v, vertex_mask)  # [B, num_joints]
        
        # 🔧 修复：更严格的标准差限制，防止entropy爆炸
        log_std_clamped = torch.clamp(self.log_std_base, min=-2.3, max=-0.5)
        std = torch.exp(log_std_clamped).expand_as(mean)  # std范围: [0.1, 0.6]
        
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


class UniversalPPOCritic(nn.Module):
    """通用PPO Critic - 支持任意关节数"""
    def __init__(self, attn_model, hidden_dim=256, device='cpu'):
        super(UniversalPPOCritic, self).__init__()
        self.attn_model = copy.deepcopy(attn_model)
        self.device = device
        
        # 🎯 使用注意力池化，而不是简单平均
        self.value_attention = nn.MultiheadAttention(128, 4, batch_first=True)  # 128可以被4整除
        
        # 输入投影层，将130维映射到128维
        self.input_projection = nn.Linear(130, 128)
        
        # 价值网络
        self.value_head = nn.Sequential(
            nn.Linear(128, hidden_dim),  # 修改为128维输入
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.value_scale = 200.0
        
    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        """前向传播，输出状态价值"""
        batch_size, num_joints, feature_dim = joint_q.shape
        
        # 投影到128维
        joint_q_proj = self.input_projection(joint_q)  # [B, num_joints, 128]
        
        # 🎯 使用注意力机制聚合关节信息
        # 创建全局查询
        global_query = joint_q_proj.mean(dim=1, keepdim=True)  # [B, 1, 128]
        
        # 注意力聚合
        attn_output, _ = self.value_attention(
            global_query,    # query: [B, 1, 128]
            joint_q_proj,    # key: [B, num_joints, 128] 
            joint_q_proj     # value: [B, num_joints, 128]
        )  # output: [B, 1, 128]
        
        # 计算价值
        value = self.value_head(attn_output.squeeze(1))  # [B, 1]
        value = torch.tanh(value) * self.value_scale
        
        return value


class UniversalRolloutBuffer:
    """通用经验回放缓冲区 - 支持变长数据"""
    def __init__(self, buffer_size, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.experiences = []  # 存储完整的经验元组
        self.advantages = []
        self.returns = []
        self.ptr = 0
    
    def store(self, joint_q, vertex_k, vertex_v, action, reward, value, log_prob, done, num_joints):
        """存储一步经验 - 包含关节数信息"""
        experience = {
            'joint_q': joint_q.cpu(),
            'vertex_k': vertex_k.cpu(), 
            'vertex_v': vertex_v.cpu(),
            'action': action.cpu(),
            'reward': reward,
            'value': value.cpu().squeeze(),
            'log_prob': log_prob.cpu() if log_prob is not None else torch.tensor(0.0),
            'done': done,
            'num_joints': num_joints
        }
        self.experiences.append(experience)
        self.ptr += 1
    
    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算优势函数和回报"""
        values = [exp['value'] for exp in self.experiences] + [last_value.cpu().squeeze()]
        rewards = [exp['reward'] for exp in self.experiences]
        dones = [exp['done'] for exp in self.experiences]
        
        values = torch.stack(values)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # 奖励归一化
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
        
        # 优势函数归一化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 存储到经验中
        for i, exp in enumerate(self.experiences):
            exp['advantage'] = advantages[i]
            exp['return'] = returns[i]
    
    def get_batch(self, batch_size):
        """获取批量数据 - 按关节数分组"""
        if len(self.experiences) < batch_size:
            indices = list(range(len(self.experiences)))
        else:
            indices = torch.randperm(len(self.experiences))[:batch_size].tolist()
        
        # 按关节数分组
        joint_groups = {}
        for i in indices:
            exp = self.experiences[i]
            num_joints = exp['num_joints']
            if num_joints not in joint_groups:
                joint_groups[num_joints] = []
            joint_groups[num_joints].append(exp)
        
        return joint_groups


class UniversalPPOWithBuffer:
    """通用PPO算法 - 支持任意关节数的机器人"""
    def __init__(self, buffer_size=2048, batch_size=64, lr=1e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01, 
                 value_coef=0.5, max_grad_norm=0.5, device='cpu', env_type='reacher2d'):
        
        self.device = device
        self.env_type = env_type
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        # 🔧 修复：降低熵系数，防止过度探索
        self.entropy_coef = min(entropy_coef, 0.005)
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建通用网络
        self.universal_attn = UniversalAttnModel(128, 130, 130, 4)
        self.actor = UniversalPPOActor(self.universal_attn, device=device)
        self.critic = UniversalPPOCritic(self.universal_attn, device=device)
        
        # 🔧 修复：更保守的学习率和更激进的衰减
        actor_lr = lr * 0.3  # Actor学习率降低
        critic_lr = lr * 0.2  # Critic学习率更低
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 🔧 修复：更激进的学习率衰减
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.9)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.9)
        
        # 经验缓冲区
        self.buffer = UniversalRolloutBuffer(buffer_size, device)
        
        # 统计信息
        self.update_count = 0
        self.recent_losses = deque(maxlen=10)
        
        print(f"🎯 通用PPO初始化完成:")
        print(f"   支持任意关节数: 2-20")
        print(f"   学习率: Actor={lr}, Critic={lr*0.5}")
        print(f"   Buffer大小: {buffer_size}")
        print(f"   环境类型: {env_type}")
    
    def get_action(self, obs, gnn_embeds, num_joints, deterministic=False):
        """获取动作 - 支持任意关节数"""
        joint_q, vertex_k, vertex_v = self._prepare_inputs(obs, gnn_embeds, num_joints)
        
        with torch.no_grad():
            action, log_prob, _, _ = self.actor.get_action(
                joint_q, vertex_k, vertex_v, deterministic=deterministic
            )
            value = self.critic(joint_q, vertex_k, vertex_v)
        
        # 动作缩放
        if self.env_type == 'reacher2d':
            action_scale = 10.0
            scaled_action = torch.tanh(action) * action_scale
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
            action, reward, value, log_prob, done, num_joints
        )
    
    def update(self, next_obs=None, next_gnn_embeds=None, num_joints=None, ppo_epochs=4):
        """PPO更新 - 处理混合关节数数据"""
        if len(self.buffer.experiences) < self.batch_size:
            return None
        
        # 🔧 添加：训练前检查和重置
        with torch.no_grad():
            current_log_std = self.actor.log_std_base.item()
            current_std = torch.exp(torch.clamp(torch.tensor(current_log_std), -2.3, -0.5)).item()
            
            # 如果标准差异常，强制重置
            if current_std > 0.8 or current_log_std > -0.3:
                print(f"⚠️ 检测到异常标准差 {current_std:.4f}，重置参数")
                self.actor.log_std_base.data.fill_(-1.8)  # 重置到安全值
        
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
        total_batches = 0
        
        for epoch in range(ppo_epochs):
            joint_groups = self.buffer.get_batch(min(self.batch_size, len(self.buffer.experiences)))
            
            # 对每个关节数组分别处理
            for num_joints, experiences in joint_groups.items():
                if len(experiences) == 0:
                    continue
                
                # 构建批次数据
                batch = self._build_batch_from_experiences(experiences)
                
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 前向传播
                new_log_probs, entropy = self.actor.evaluate_action(
                    batch['joint_q'], batch['vertex_k'], batch['vertex_v'], batch['actions']
                )
                new_values = self.critic(batch['joint_q'], batch['vertex_k'], batch['vertex_v']).squeeze(-1)
                
                # PPO损失计算
                ratio = torch.exp(new_log_probs.squeeze(-1) - batch['old_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch['advantages']
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.smooth_l1_loss(new_values, batch['returns'])
                entropy_loss = -entropy.mean()
                
                # 分别优化
                self.actor_optimizer.zero_grad()
                actor_total_loss = actor_loss + self.entropy_coef * entropy_loss
                actor_total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_batches += 1
        
        # 学习率调度
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 清空缓冲区
        self.buffer.clear()
        self.update_count += 1
        
        if total_batches == 0:
            return None
        
        metrics = {
            'actor_loss': total_actor_loss / total_batches,
            'critic_loss': total_critic_loss / total_batches,
            'total_loss': (total_actor_loss + total_critic_loss) / total_batches,
            'entropy': total_entropy / total_batches,
            'update_count': self.update_count,
            'learning_rate': self.actor_optimizer.param_groups[0]['lr'],
            'batches_processed': total_batches
        }
        
        # 🔧 添加：更新后检查和紧急处理
        if metrics['entropy'] > 3.0:  # 熵值过高
            print(f"🚨 熵值异常高 {metrics['entropy']:.2f}，降低学习率")
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] *= 0.5
            # 强制重置标准差
            with torch.no_grad():
                self.actor.log_std_base.data.fill_(-2.0)
        
        if metrics['critic_loss'] > 5.0:  # Critic loss过高
            print(f"🚨 Critic loss异常高 {metrics['critic_loss']:.2f}，降低学习率")
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] *= 0.3
        
        return metrics
    
    def _build_batch_from_experiences(self, experiences):
        """从经验列表构建批次数据"""
        batch = {
            'joint_q': torch.stack([exp['joint_q'] for exp in experiences]),
            'vertex_k': torch.stack([exp['vertex_k'] for exp in experiences]),
            'vertex_v': torch.stack([exp['vertex_v'] for exp in experiences]),
            'actions': torch.stack([exp['action'] for exp in experiences]),
            'old_log_probs': torch.stack([exp['log_prob'] for exp in experiences]),
            'advantages': torch.stack([exp['advantage'] for exp in experiences]),
            'returns': torch.stack([exp['return'] for exp in experiences]),
            'old_values': torch.stack([exp['value'] for exp in experiences])
        }
        return batch
    
    def _prepare_inputs(self, obs, gnn_embeds, num_joints):
        """准备输入数据 - 修复维度匹配问题"""
        if obs.dim() == 1:
            obs_batch = obs.unsqueeze(0)
        else:
            obs_batch = obs
            
        if gnn_embeds.dim() == 2:
            gnn_embeds_batch = gnn_embeds.unsqueeze(0)
        else:
            gnn_embeds_batch = gnn_embeds
        
        # 🔧 修复：简化数据准备，避免维度不匹配
        batch_size = obs_batch.size(0)
        
        # 准备joint_q：直接使用关节状态信息
        if self.env_type == 'reacher2d':
            # Reacher2D: [angles, angular_vels, end_effector_pos]
            joint_angles = obs_batch[:, :num_joints]                    # [B, num_joints]
            joint_angular_vels = obs_batch[:, num_joints:2*num_joints]  # [B, num_joints]
        else:
            # 其他环境的处理
            joint_pos_start = 16
            joint_angles = obs_batch[:, joint_pos_start:joint_pos_start + num_joints]
            joint_angular_vels = obs_batch[:, joint_pos_start + num_joints:joint_pos_start + 2*num_joints]
        
        # 获取GNN嵌入的前num_joints个节点
        gnn_embed_joints = gnn_embeds_batch[:, :num_joints, :]  # [B, num_joints, 128]
        
        # 构建joint_q: [position, velocity, gnn_embed] = [1 + 1 + 128] = 130
        joint_q = torch.cat([
            joint_angles.unsqueeze(-1),       # [B, num_joints, 1]
            joint_angular_vels.unsqueeze(-1), # [B, num_joints, 1]
            gnn_embed_joints                  # [B, num_joints, 128]
        ], dim=-1)  # [B, num_joints, 130]
        
        # vertex_k和vertex_v使用相同的数据，但维度对齐
        vertex_k = gnn_embed_joints  # [B, num_joints, 128]
        
        # vertex_v包含更多动态信息
        vertex_v = torch.cat([
            gnn_embed_joints,                 # [B, num_joints, 128] 
            joint_angles.unsqueeze(-1),       # [B, num_joints, 1]
            joint_angular_vels.unsqueeze(-1)  # [B, num_joints, 1]
        ], dim=-1)  # [B, num_joints, 130]
        
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
