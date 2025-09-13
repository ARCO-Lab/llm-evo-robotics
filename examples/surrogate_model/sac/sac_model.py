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
from data_utils import prepare_joint_q_input, prepare_reacher2d_joint_q_input, prepare_dynamic_vertex_v
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
    
    def clear(self):
        """清空buffer中的所有经验"""
        self.buffer.clear()
        print(f"🧹 Buffer已清空")


class AttentionSACWithBuffer:
    def __init__(self, attn_model, action_dim, joint_embed_dim=128, 
                 buffer_capacity=10000, batch_size=256, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, device='cpu', env_type='bullet'):
        
        self.device = device
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.warmup_steps = 10000
        self.env_type = env_type
 
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
        # lr = 2e-5  # 🔧 移除硬编码，使用传入的lr参数
        # 优化器 - 现在完全独立
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        # 自动调整alpha
        self.target_entropy = -action_dim * 0.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
    def store_experience(self, obs, gnn_embeds, action, reward, next_obs, next_gnn_embeds, done, num_joints=12):
        """存储环境交互经验"""
        if self.env_type == 'reacher2d':
            joint_q = prepare_reacher2d_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
            next_joint_q = prepare_reacher2d_joint_q_input(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
        else:  # bullet 环境
            joint_q = prepare_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
            next_joint_q = prepare_joint_q_input(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
        # 准备当前状态
        vertex_k = gnn_embeds
        vertex_v = vertex_v = prepare_dynamic_vertex_v(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints, self.env_type).squeeze(0)  # 🎯 动态V
        
        # 准备下一个状态
        # next_joint_q = prepare_joint_q_input(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints).squeeze(0)
        next_vertex_k = next_gnn_embeds
        next_vertex_v = prepare_dynamic_vertex_v(next_obs.unsqueeze(0), next_gnn_embeds.unsqueeze(0), num_joints, self.env_type).squeeze(0)  # 🎯 动态V
        
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
          
        # 🎯 根据环境类型选择数据处理函数
        if self.env_type == 'reacher2d':
            joint_q = prepare_reacher2d_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        else:  # bullet 环境
            joint_q = prepare_joint_q_input(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints)
        vertex_k = gnn_embeds.unsqueeze(0)
        vertex_v = prepare_dynamic_vertex_v(obs.unsqueeze(0), gnn_embeds.unsqueeze(0), num_joints, self.env_type)  # 🎯 动态V
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(joint_q, vertex_k, vertex_v)
                tanh_action = torch.tanh(mean).squeeze(0)
            else:
                tanh_action, _, _ = self.actor.sample(joint_q, vertex_k, vertex_v)
                tanh_action = tanh_action.squeeze(0)
            
            # 🔧 关键修复：Action Scaling！
            # SAC输出[-1,+1]，需要缩放到环境的action space
            if self.env_type == 'reacher2d':
                # 🎯 修复：恢复到环境的完整action range [-100, +100]
                action_scale = 100.0  # 恢复到环境max_torque的完整范围
                scaled_action = tanh_action * action_scale
                return scaled_action
            else:
                # Bullet环境保持原有逻辑
                return tanh_action
    
    def soft_update_targets(self):
        """软更新target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self):
        """从memory buffer采样并更新网络 - 增强数值稳定性"""
        if not self.memory.can_sample(self.batch_size):
            return None
        
        # batch = self.memory.sample(min(32, self.batch_size))  # 小批量快速评估
        # joint_q, vertex_k, vertex_v, actions, rewards, next_joint_q, next_vertex_k, next_vertex_v, dones, vertex_mask = batch
        
        # with torch.no_grad():
        #     current_q1 = self.critic1(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=actions)
        #     current_q2 = self.critic2(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=actions)
        #     quick_loss = (current_q1.std() + current_q2.std()).item()  # 简单评估
        
        # # 动态学习率调整
        # if quick_loss < 1.0:  # 如果稳定
        #     new_lr = 5e-5  # 可以提高学习率
        # else:  # 如果不稳定
        #     new_lr = 2e-5  # 保持低学习率
        
        # # 更新所有优化器的学习率
        # for param_group in self.critic_optimizer.param_groups:
        #     param_group['lr'] = new_lr
        # for param_group in self.actor_optimizer.param_groups:
        #     param_group['lr'] = new_lr
        # for param_group in self.alpha_optimizer.param_groups:
        #     param_group['lr'] = new_lr
            
        # # 从buffer采样
        batch = self.memory.sample(self.batch_size)
        joint_q, vertex_k, vertex_v, actions, rewards, next_joint_q, next_vertex_k, next_vertex_v, dones, vertex_mask = batch

        
        
        # 🛡️ 奖励稳定性检查
        rewards = torch.clamp(rewards, -10.0, 10.0)  # 严格限制奖励范围
        
        # === Critic Update ===
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_joint_q, next_vertex_k, next_vertex_v, vertex_mask)
            
            target_q1 = self.target_critic1(next_joint_q, next_vertex_k, next_vertex_v, 
                                        vertex_mask=vertex_mask, action=next_actions)
            target_q2 = self.target_critic2(next_joint_q, next_vertex_k, next_vertex_v, 
                                        vertex_mask=vertex_mask, action=next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
            # 🛡️ Target Q值稳定性检查
            target_q = torch.clamp(target_q, -50.0, 50.0)
        
        # 当前Q值
        current_q1 = self.critic1(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=actions)
        current_q2 = self.critic2(joint_q, vertex_k, vertex_v, vertex_mask=vertex_mask, action=actions)
        
        # 🛡️ 当前Q值稳定性检查
        current_q1 = torch.clamp(current_q1, -50.0, 50.0)
        current_q2 = torch.clamp(current_q2, -50.0, 50.0)
        
        # 使用Huber Loss代替MSE Loss，更稳定
        critic_loss = nn.SmoothL1Loss()(current_q1, target_q) + nn.SmoothL1Loss()(current_q2, target_q)

        current_lr = self.critic_optimizer.param_groups[0]['lr']

        # 添加恢复逻辑
        if not hasattr(self, 'consecutive_low_loss_count'):
            self.consecutive_low_loss_count = 0

        if critic_loss.item() < 0.5:  # 非常稳定
            self.consecutive_low_loss_count += 1
            if self.consecutive_low_loss_count > 50:  # 连续50次低loss
                new_lr = min(current_lr * 1.2, 5e-5)  # 可以大幅恢复
            else:
                new_lr = min(current_lr * 1.05, 5e-5)  # 小幅提高
        elif critic_loss.item() > 2.0:  # 严重不稳定
            self.consecutive_low_loss_count = 0
            new_lr = max(current_lr * 0.5, 1e-5)  # 大幅降低
        elif critic_loss.item() > 1.0:  # 轻微不稳定
            self.consecutive_low_loss_count = 0
            new_lr = max(current_lr * 0.9, 1e-5)  # 适度降低
        else:
            new_lr = current_lr  # 保持不变
                # 限制调整频率
        if not hasattr(self, 'last_lr_adjust_step'):
            self.last_lr_adjust_step = 0

        # 获取当前步数（可以从外部传入或使用计数器）
        if not hasattr(self, 'update_counter'):
            self.update_counter = 0
        self.update_counter += 1

        # 至少100次update才调整一次学习率
        if self.update_counter - self.last_lr_adjust_step > 100:
            # 只在有显著变化时更新学习率
            if abs(current_lr - new_lr) > 1e-7:
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = new_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = new_lr
                for param_group in self.alpha_optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"📈 学习率调整 (第{self.update_counter}次更新): {current_lr:.1e} → {new_lr:.1e} (critic_loss: {critic_loss.item():.3f})")
                self.last_lr_adjust_step = self.update_counter

        
        # 🛡️ Loss稳定性检查
        if critic_loss > 25.0:
            print(f"⚠️ 大Critic Loss: {critic_loss:.3f}, 跳过此次更新")
            return None
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            max_norm=0.5  # 更严格的梯度裁剪
        )
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
        torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=1.0
            )
        self.actor_optimizer.step()
        
        # === Alpha Update ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        # 🔒 Alpha下限约束
        # min_alpha = getattr(self, 'min_alpha', 0.01)
        # if self.alpha < min_alpha:
        #     self.alpha = torch.tensor(min_alpha, device=self.alpha.device)  # 保持tensor类型
        #     self.log_alpha.data.fill_(torch.log(torch.tensor(min_alpha)).item())
        #     print(f"⚠️ Alpha达到下限 {min_alpha}，已限制")
        # 软更新target networks
        self.soft_update_targets()

        def detect_q_divergence(current_q1, current_q2, target_q, step_info=""):
            """检测Q值是否发散"""
            divergence_signals = []
            
            # 1. 绝对值检测
            q1_max = current_q1.max().item()
            q1_min = current_q1.min().item()
            q2_max = current_q2.max().item()
            q2_min = current_q2.min().item()
            target_max = target_q.max().item()
            target_min = target_q.min().item()
            
            # 检查绝对值过大
            if max(abs(q1_max), abs(q1_min), abs(q2_max), abs(q2_min)) > 100.0:
                divergence_signals.append(f"Q值绝对值过大: Q1[{q1_min:.2f}, {q1_max:.2f}], Q2[{q2_min:.2f}, {q2_max:.2f}]")
            
            # 2. Q值方差检测
            q1_std = current_q1.std().item()
            q2_std = current_q2.std().item()
            if q1_std > 50.0 or q2_std > 50.0:
                divergence_signals.append(f"Q值方差过大: Q1_std={q1_std:.2f}, Q2_std={q2_std:.2f}")
            
            # 3. Q值差异检测
            q_diff = torch.abs(current_q1 - current_q2).mean().item()
            if q_diff > 20.0:
                divergence_signals.append(f"Q1和Q2差异过大: mean_diff={q_diff:.2f}")
            
            # 4. Target-Current差异检测
            target_diff_1 = torch.abs(current_q1 - target_q).mean().item()
            target_diff_2 = torch.abs(current_q2 - target_q).mean().item()
            if target_diff_1 > 30.0 or target_diff_2 > 30.0:
                divergence_signals.append(f"Q值与目标差异过大: diff1={target_diff_1:.2f}, diff2={target_diff_2:.2f}")
            
            return divergence_signals
        
        # 在update方法中调用检测
        divergence_signals = detect_q_divergence(current_q1, current_q2, target_q)
        if divergence_signals:
            print(f"\n🔥 Q值发散警报:")
            for signal in divergence_signals:
                print(f"   {signal}")
            print(f"   建议: 降低学习率、增加正则化或重启训练")

            
        return {
            'lr': new_lr,
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'q1_std': current_q1.std().item(),        # 新增
            'q2_std': current_q2.std().item(),        # 新增
            'q1_max': current_q1.max().item(),        # 新增
            'q1_min': current_q1.min().item(),        # 新增
            'q2_max': current_q2.max().item(),        # 新增
            'q2_min': current_q2.min().item(),        # 新增
            'q_diff_mean': torch.abs(current_q1 - current_q2).mean().item(),  # 新增
            'target_q_mean': target_q.mean().item(),  # 新增
            'buffer_size': len(self.memory),
            'entropy_term': entropy_term.item(),
            'q_term': q_term.item(),
            'log_probs_mean': log_probs.mean().item()
        }
    
    def clear_buffer(self):
        """清空经验回放缓冲区"""
        self.memory.clear()
        print(f"🧹 SAC模型buffer已清空 (容量: {self.memory.capacity})")
    
    def reset_for_new_reward_function(self):
        """为新奖励函数重置训练状态"""
        self.clear_buffer()
        print(f"🔄 模型已重置以适应新奖励函数")
        print(f"   建议进行新的warmup期: {self.warmup_steps}步")


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