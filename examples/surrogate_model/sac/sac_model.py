import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque, namedtuple
import os
import sys

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_dataset"))
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_model"))
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/sac"))
from data_utils import prepare_joint_q_input
# 导入你的组件
from attn_actor import AttentionActor
from attn_critic import AttentionCritic
import copy
        
# 定义经验元组
Experience = namedtuple('Experience', [
    'joint_q', 'vertex_k', 'vertex_v', 'action', 'reward', 
    'next_joint_q', 'next_vertex_k', 'next_vertex_v', 'done', 'vertex_mask'
])

class ReplayBuffer:
    def __init__(self, capacity=100000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, joint_q, vertex_k, vertex_v, action, reward, 
             next_joint_q, next_vertex_k, next_vertex_v, done, vertex_mask=None):
        """存储一个经验"""
        
        # 转换为CPU tensor以节省GPU内存
        experience = Experience(
            joint_q=joint_q.cpu() if torch.is_tensor(joint_q) else joint_q,
            vertex_k=vertex_k.cpu() if torch.is_tensor(vertex_k) else vertex_k,
            vertex_v=vertex_v.cpu() if torch.is_tensor(vertex_v) else vertex_v,
            action=action.cpu() if torch.is_tensor(action) else action,
            reward=reward.cpu() if torch.is_tensor(reward) else reward,
            next_joint_q=next_joint_q.cpu() if torch.is_tensor(next_joint_q) else next_joint_q,
            next_vertex_k=next_vertex_k.cpu() if torch.is_tensor(next_vertex_k) else next_vertex_k,
            next_vertex_v=next_vertex_v.cpu() if torch.is_tensor(next_vertex_v) else next_vertex_v,
            done=done.cpu() if torch.is_tensor(done) else done,
            vertex_mask=vertex_mask.cpu() if vertex_mask is not None and torch.is_tensor(vertex_mask) else vertex_mask
        )
        
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """采样一个batch的经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        experiences = random.sample(self.buffer, batch_size)
        
        # 将经验转换为批量tensor
        joint_q = torch.stack([e.joint_q for e in experiences]).to(self.device)
        vertex_k = torch.stack([e.vertex_k for e in experiences]).to(self.device)
        vertex_v = torch.stack([e.vertex_v for e in experiences]).to(self.device)
        actions = torch.stack([e.action for e in experiences]).to(self.device)
        rewards = torch.stack([e.reward for e in experiences]).to(self.device)
        next_joint_q = torch.stack([e.next_joint_q for e in experiences]).to(self.device)
        next_vertex_k = torch.stack([e.next_vertex_k for e in experiences]).to(self.device)
        next_vertex_v = torch.stack([e.next_vertex_v for e in experiences]).to(self.device)
        dones = torch.stack([e.done for e in experiences]).to(self.device)
        
        # 处理vertex_mask (可能为None)
        vertex_mask = None
        if experiences[0].vertex_mask is not None:
            vertex_mask = torch.stack([e.vertex_mask for e in experiences]).to(self.device)
        
        return (joint_q, vertex_k, vertex_v, actions, rewards, 
                next_joint_q, next_vertex_k, next_vertex_v, dones, vertex_mask)
    
    def __len__(self):
        return len(self.buffer)
    
    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size


class AttentionSACWithBuffer:
    def __init__(self, attn_model, action_dim, joint_embed_dim=128, 
                 buffer_capacity=100000, batch_size=256, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, device='cpu'):
        
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.warmup_steps = 10000
 
        # 创建独立的模型拷贝 - 解决共享问题
        actor_model = copy.deepcopy(attn_model)
        critic1_model = copy.deepcopy(attn_model)
        critic2_model = copy.deepcopy(attn_model)
        
        self.actor = AttentionActor(actor_model, action_dim).to(device)
        self.critic1 = AttentionCritic(critic1_model, joint_embed_dim, device=device).to(device)
        self.critic2 = AttentionCritic(critic2_model, joint_embed_dim, device=device).to(device)
        
        # Target networks - 现在是真正独立的
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # 冻结target networks
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
            
        # Memory Buffer
        self.memory = ReplayBuffer(buffer_capacity, device)
        
        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # 优化器 - 现在完全独立
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        # 自动调整alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
    def store_experience(self, obs, gnn_embeds, action, reward, next_obs, next_gnn_embeds, done, num_joints=12):
        """存储环境交互经验"""
        
        # 准备当前状态
        joint_q = prepare_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
        vertex_k = gnn_embeds
        vertex_v = gnn_embeds
        
        # 准备下一个状态
        next_joint_q = prepare_joint_q_input(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
        next_vertex_k = next_gnn_embeds
        next_vertex_v = next_gnn_embeds
        
        # 转换为适当的tensor格式
        if not torch.is_tensor(reward):
            reward = torch.tensor([reward], dtype=torch.float32)
        if not torch.is_tensor(done):
            done = torch.tensor([done], dtype=torch.float32)
            
        self.memory.push(
            joint_q, vertex_k, vertex_v, action, reward,
            next_joint_q, next_vertex_k, next_vertex_v, done
        )
    
    def get_action(self, obs, gnn_embeds, num_joints=12, deterministic=False):
        """获取动作"""
        joint_q = prepare_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        vertex_k = gnn_embeds.unsqueeze(0)
        vertex_v = gnn_embeds.unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(joint_q, vertex_k, vertex_v)
                return torch.tanh(mean).squeeze(0)
            else:
                action, _, _ = self.actor.sample(joint_q, vertex_k, vertex_v)
                return action.squeeze(0)
    
    def soft_update_targets(self):
        """软更新target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self):
        """从memory buffer采样并更新网络"""
        if not self.memory.can_sample(self.batch_size):
            return None
            
        # 从buffer采样
        batch = self.memory.sample(self.batch_size)
        joint_q, vertex_k, vertex_v, actions, rewards, next_joint_q, next_vertex_k, next_vertex_v, dones, vertex_mask = batch
        
        # === Critic Update ===
        with torch.no_grad():
            # 采样下一个状态的动作
            next_actions, next_log_probs, _ = self.actor.sample(next_joint_q, next_vertex_k, next_vertex_v, vertex_mask)
            
            # 计算target Q值
            target_q1 = self.target_critic1(next_joint_q, next_vertex_k, next_vertex_v, 
                                          vertex_mask=vertex_mask, action=next_actions)
            target_q2 = self.target_critic2(next_joint_q, next_vertex_k, next_vertex_v, 
                                          vertex_mask=vertex_mask, action=next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 当前Q值
        current_q1 = self.critic1(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=actions)
        current_q2 = self.critic2(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=actions)
        
        # Critic损失
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # === Actor Update ===
        # 采样新动作
        new_actions, log_probs, _ = self.actor.sample(joint_q, vertex_k, vertex_v, vertex_mask)
        
        # 计算Q值
        q1 = self.critic1(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=new_actions)
        q2 = self.critic2(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=new_actions)
        q = torch.min(q1, q2)
        
        # Actor损失
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 分析loss组件
        entropy_term = (self.alpha * log_probs).mean()
        q_term = q.mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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
            'buffer_size': len(self.memory),
            # 添加loss组件分析
            'entropy_term': entropy_term.item(),
            'q_term': q_term.item(),
            'log_probs_mean': log_probs.mean().item()
        }


# 训练循环示例
def train_sac_with_buffer():
    """训练SAC的示例"""
    from attn_model import AttnModel
    
    # 初始化
    action_dim = 12
    attn_model = AttnModel(128, 128, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟训练循环
    num_episodes = 1000
    max_steps_per_episode = 200
    update_frequency = 4  # 每4步更新一次
    
    for episode in range(num_episodes):
        # 模拟环境重置
        obs = torch.randn(40)  # [40] 观察空间
        gnn_embeds = torch.randn(12, 128)  # [12, 128] GNN嵌入
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # 获取动作
            action = sac.get_action(obs, gnn_embeds, deterministic=False)
            
            # 模拟环境交互
            next_obs = torch.randn(40)  # 下一个观察
            next_gnn_embeds = torch.randn(12, 128)  # 下一个GNN嵌入
            reward = torch.randn(1).item()  # 随机奖励
            done = step == max_steps_per_episode - 1  # 最后一步结束
            
            # 存储经验
            sac.store_experience(obs, gnn_embeds, action, reward, next_obs, next_gnn_embeds, done)
            
            # 更新网络
            if step % update_frequency == 0 and sac.memory.can_sample(sac.batch_size):
                metrics = sac.update()
                if metrics and step % 20 == 0:
                    print(f"Episode {episode}, Step {step}: {metrics}")
            
            # 准备下一步
            obs = next_obs
            gnn_embeds = next_gnn_embeds
            episode_reward += reward
            
            if done:
                break
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Buffer Size: {len(sac.memory)}")


# 测试代码
if __name__ == "__main__":
    print("Testing SAC with Memory Buffer...")
    
    # 创建SAC实例
    from attn_model import AttnModel
    action_dim = 12
    attn_model = AttnModel(128, 128, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim)
    
    # 模拟一些经验存储
    for i in range(10):
        obs = torch.randn(40)
        gnn_embeds = torch.randn(12, 128)
        action = torch.randn(12)
        reward = torch.randn(1).item()
        next_obs = torch.randn(40)
        next_gnn_embeds = torch.randn(12, 128)
        done = i == 9
        
        sac.store_experience(obs, gnn_embeds, action, reward, next_obs, next_gnn_embeds, done)
    
    print(f"Buffer size: {len(sac.memory)}")
    
    # 测试获取动作
    obs = torch.randn(40)
    gnn_embeds = torch.randn(12, 128)
    action = sac.get_action(obs, gnn_embeds)
    print(f"Action shape: {action.shape}")
    
    # 测试更新（如果buffer足够大）
    if sac.memory.can_sample(sac.batch_size):
        metrics = sac.update()
        print("Update metrics:", metrics)
    else:
        print("Buffer too small for update")
    
    print("Buffer integration successful!")