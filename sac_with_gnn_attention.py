#!/usr/bin/env python3
"""
SAC + GNN + 简化注意力机制
支持任意数量关节的 Reacher 任务
为多关节 Reacher 做准备
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, List, Tuple, Type, Union, Optional
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

class JointGraphBuilder(nn.Module):
    """
    关节图构建器 - 将关节信息转换为图结构
    支持任意数量的关节
    """
    def __init__(self, max_joints: int = 10):
        super(JointGraphBuilder, self).__init__()
        self.max_joints = max_joints
        
        print(f"🔗 JointGraphBuilder 初始化: 最大关节数={max_joints}")
    
    def build_chain_graph(self, num_joints: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建链式关节图 (适用于 Reacher)
        每个关节连接到下一个关节
        """
        # 边索引: 每个关节连接到下一个关节
        edge_list = []
        for i in range(num_joints - 1):
            edge_list.append([i, i + 1])  # 前向连接
            edge_list.append([i + 1, i])  # 反向连接 (无向图)
        
        if len(edge_list) == 0:
            # 单关节情况
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # 边权重 (暂时设为1)
        edge_attr = torch.ones(edge_index.size(1), 1)
        
        return edge_index, edge_attr
    
    def extract_joint_features(self, obs: torch.Tensor, num_joints: int) -> torch.Tensor:
        """
        从观察中提取关节特征
        obs: [batch_size, obs_dim]
        return: [batch_size, num_joints, joint_feature_dim]
        """
        batch_size = obs.size(0)
        
        if num_joints == 2:
            # MuJoCo Reacher-v5 (2关节)
            # [0:2] - cos/sin of joint angles
            # [2:4] - joint velocities
            joint_angles = obs[:, :2]  # [batch_size, 2]
            joint_velocities = obs[:, 2:4]  # [batch_size, 2]
            
            # 组合关节特征
            joint_features = torch.stack([
                torch.cat([joint_angles[:, 0:1], joint_velocities[:, 0:1]], dim=1),  # 关节1
                torch.cat([joint_angles[:, 1:2], joint_velocities[:, 1:2]], dim=1)   # 关节2
            ], dim=1)  # [batch_size, 2, 2]
            
        else:
            # 通用情况 (为未来多关节做准备)
            # 假设观察格式: [joint_angles, joint_velocities, ...]
            joint_dim = num_joints
            joint_angles = obs[:, :joint_dim]
            joint_velocities = obs[:, joint_dim:2*joint_dim]
            
            joint_features = torch.stack([
                torch.cat([joint_angles[:, i:i+1], joint_velocities[:, i:i+1]], dim=1)
                for i in range(num_joints)
            ], dim=1)  # [batch_size, num_joints, 2]
        
        return joint_features

class GNNLayer(nn.Module):
    """
    图神经网络层 - 处理关节图
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, gnn_type: str = "GCN"):
        super(GNNLayer, self).__init__()
        self.gnn_type = gnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if gnn_type == "GCN":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif gnn_type == "GAT":
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1)
            self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.1)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        print(f"🧠 GNNLayer 初始化: {gnn_type}, {input_dim}→{hidden_dim}→{output_dim}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        x: [num_nodes, input_dim] 节点特征
        edge_index: [2, num_edges] 边索引
        batch: [num_nodes] 批次索引 (用于批处理)
        """
        # 第一层 GNN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层 GNN
        x = self.conv2(x, edge_index)
        x = self.layer_norm(x)
        
        return x

class AdaptiveJointAttention(nn.Module):
    """
    自适应关节注意力机制
    可以处理任意数量的关节
    """
    def __init__(self, joint_feature_dim: int = 32, attention_dim: int = 64):
        super(AdaptiveJointAttention, self).__init__()
        self.joint_feature_dim = joint_feature_dim
        self.attention_dim = attention_dim
        
        # 注意力计算
        self.attention_net = nn.Sequential(
            nn.Linear(joint_feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        
        print(f"🎯 AdaptiveJointAttention 初始化: joint_dim={joint_feature_dim}, attention_dim={attention_dim}")
    
    def forward(self, joint_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算关节注意力权重
        joint_features: [batch_size, num_joints, joint_feature_dim]
        return: (weighted_features, attention_weights)
        """
        batch_size, num_joints, feature_dim = joint_features.shape
        
        # 计算注意力分数
        attention_scores = self.attention_net(joint_features)  # [batch_size, num_joints, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_joints, 1]
        
        # 应用注意力权重
        weighted_features = joint_features * attention_weights  # [batch_size, num_joints, feature_dim]
        
        return weighted_features, attention_weights.squeeze(-1)

class GNNAttentionExtractor(BaseFeaturesExtractor):
    """
    GNN + 注意力特征提取器
    支持任意数量关节的 Reacher 任务
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, num_joints: int = 2, gnn_type: str = "GCN"):
        super(GNNAttentionExtractor, self).__init__(observation_space, features_dim)
        
        self.obs_dim = observation_space.shape[0]
        self.num_joints = num_joints
        self.gnn_type = gnn_type
        
        print(f"🔍 GNNAttentionExtractor 初始化:")
        print(f"   观察空间维度: {self.obs_dim}")
        print(f"   关节数量: {num_joints}")
        print(f"   输出特征维度: {features_dim}")
        print(f"   GNN 类型: {gnn_type}")
        
        # 组件初始化
        self.graph_builder = JointGraphBuilder(max_joints=10)
        
        # 关节特征处理
        joint_input_dim = 2  # 每个关节: [angle, velocity]
        joint_hidden_dim = 32
        joint_output_dim = 32
        
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_input_dim, joint_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(joint_hidden_dim),
            nn.Linear(joint_hidden_dim, joint_output_dim)
        )
        
        # GNN 层
        self.gnn = GNNLayer(
            input_dim=joint_output_dim,
            hidden_dim=64,
            output_dim=joint_output_dim,
            gnn_type=gnn_type
        )
        
        # 自适应注意力
        self.joint_attention = AdaptiveJointAttention(
            joint_feature_dim=joint_output_dim,
            attention_dim=64
        )
        
        # 全局特征处理 (位置、目标等非关节信息)
        if num_joints == 2:
            # MuJoCo Reacher-v5: [4:6] end effector, [6:8] target, [8:10] vector
            global_feature_dim = 6  # end_effector(2) + target(2) + vector(2)
        else:
            # 通用情况
            global_feature_dim = max(0, self.obs_dim - 2 * num_joints)
        
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64)
        ) if global_feature_dim > 0 else nn.Identity()
        
        # 特征融合
        fusion_input_dim = joint_output_dim + (64 if global_feature_dim > 0 else 0)
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        # 预构建图结构 (对于固定关节数)
        self.edge_index, self.edge_attr = self.graph_builder.build_chain_graph(num_joints)
        
        print(f"✅ GNNAttentionExtractor 构建完成")
        print(f"   关节特征维度: {joint_input_dim}→{joint_output_dim}")
        print(f"   全局特征维度: {global_feature_dim}")
        print(f"   融合输入维度: {fusion_input_dim}")
        print(f"   图边数: {self.edge_index.size(1)}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        observations: [batch_size, obs_dim]
        return: [batch_size, features_dim]
        """
        batch_size = observations.size(0)
        device = observations.device
        
        # 移动图结构到正确设备
        edge_index = self.edge_index.to(device)
        
        # 1. 提取关节特征
        joint_features = self.graph_builder.extract_joint_features(observations, self.num_joints)
        # joint_features: [batch_size, num_joints, 2]
        
        # 2. 编码关节特征
        joint_encoded = self.joint_encoder(joint_features)
        # joint_encoded: [batch_size, num_joints, joint_output_dim]
        
        # 3. 准备 GNN 输入
        # 重塑为 [batch_size * num_joints, joint_output_dim]
        joint_flat = joint_encoded.view(-1, joint_encoded.size(-1))
        
        # 创建批次索引
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(self.num_joints)
        
        # 调整边索引以适应批处理
        edge_index_batch = edge_index.clone()
        for i in range(1, batch_size):
            batch_edges = edge_index + i * self.num_joints
            edge_index_batch = torch.cat([edge_index_batch, batch_edges], dim=1)
        
        # 4. GNN 处理
        joint_gnn_output = self.gnn(joint_flat, edge_index_batch, batch_idx)
        # joint_gnn_output: [batch_size * num_joints, joint_output_dim]
        
        # 重塑回 [batch_size, num_joints, joint_output_dim]
        joint_gnn_reshaped = joint_gnn_output.view(batch_size, self.num_joints, -1)
        
        # 5. 关节注意力
        joint_attended, attention_weights = self.joint_attention(joint_gnn_reshaped)
        # joint_attended: [batch_size, num_joints, joint_output_dim]
        
        # 6. 全局池化关节特征
        joint_global = torch.mean(joint_attended, dim=1)  # [batch_size, joint_output_dim]
        
        # 7. 处理全局特征 (非关节信息)
        if self.num_joints == 2:
            # MuJoCo Reacher-v5
            global_features = observations[:, 4:]  # [batch_size, 6]
            global_encoded = self.global_encoder(global_features)  # [batch_size, 64]
        else:
            # 通用情况
            if self.obs_dim > 2 * self.num_joints:
                global_features = observations[:, 2*self.num_joints:]
                global_encoded = self.global_encoder(global_features)
            else:
                global_encoded = torch.zeros(batch_size, 0, device=device)
        
        # 8. 特征融合
        if global_encoded.size(1) > 0:
            fused_features = torch.cat([joint_global, global_encoded], dim=1)
        else:
            fused_features = joint_global
        
        output = self.fusion_net(fused_features)
        
        return output

class AdaptiveActionHead(nn.Module):
    """
    自适应动作头 - 可以生成任意数量关节的动作
    """
    def __init__(self, input_dim: int, num_joints: int, hidden_dim: int = 128):
        super(AdaptiveActionHead, self).__init__()
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 共享特征处理
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 每个关节的动作头
        self.joint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # 每个关节输出1个动作
            ) for _ in range(num_joints)
        ])
        
        print(f"🎮 AdaptiveActionHead 初始化: {input_dim}→{hidden_dim}, {num_joints}个关节")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        生成关节动作
        features: [batch_size, input_dim]
        return: [batch_size, num_joints]
        """
        shared_features = self.shared_net(features)
        
        joint_actions = []
        for joint_head in self.joint_heads:
            action = joint_head(shared_features)  # [batch_size, 1]
            joint_actions.append(action)
        
        actions = torch.cat(joint_actions, dim=1)  # [batch_size, num_joints]
        return actions

def sac_with_gnn_attention_training(num_joints: int = 2, gnn_type: str = "GCN"):
    print("🚀 SAC + GNN + 注意力机制训练")
    print(f"🔗 关节数量: {num_joints}")
    print(f"🧠 GNN 类型: {gnn_type}")
    print("🎯 支持任意数量关节，为多关节 Reacher 做准备")
    print("=" * 70)
    
    # 创建原生 MuJoCo Reacher 环境
    print("🏭 创建 MuJoCo Reacher-v5 环境...")
    env = gym.make('Reacher-v5', render_mode='human')
    env = Monitor(env)
    
    print(f"✅ 环境创建完成")
    print(f"🎮 动作空间: {env.action_space}")
    print(f"👁️ 观察空间: {env.observation_space}")
    print(f"📏 观察维度: {env.observation_space.shape}")
    print(f"🔗 实际关节数: {env.action_space.shape[0]} (环境固定)")
    print(f"🧠 模型关节数: {num_joints} (模型设计)")
    
    # 创建评估环境
    eval_env = gym.make('Reacher-v5')
    eval_env = Monitor(eval_env)
    
    print("=" * 70)
    
    # 创建 GNN + 注意力 SAC 模型
    print("🤖 创建 SAC + GNN + 注意力模型...")
    
    # 定义策略参数
    policy_kwargs = {
        "features_extractor_class": GNNAttentionExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "num_joints": num_joints,
            "gnn_type": gnn_type
        },
        "net_arch": [256, 256],  # Actor 和 Critic 网络架构
        "activation_fn": torch.nn.ReLU,
    }
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,          # 与之前相同
        buffer_size=1000000,         # 与之前相同
        learning_starts=100,         # 与之前相同
        batch_size=256,              # 与之前相同
        tau=0.005,                   # 与之前相同
        gamma=0.99,                  # 与之前相同
        train_freq=1,                # 与之前相同
        gradient_steps=1,            # 与之前相同
        ent_coef='auto',             # 与之前相同
        target_update_interval=1,    # 与之前相同
        use_sde=False,               # 与之前相同
        policy_kwargs=policy_kwargs, # GNN + 注意力机制
        verbose=1,
        device='cpu'
    )
    
    print("✅ SAC + GNN + 注意力模型创建完成")
    print(f"📊 模型参数:")
    print(f"   策略: MlpPolicy + GNNAttentionExtractor")
    print(f"   GNN 类型: {gnn_type}")
    print(f"   关节数量: {num_joints}")
    print(f"   特征维度: 128")
    print(f"   网络架构: [256, 256]")
    print(f"   支持任意关节数量")
    
    print("=" * 70)
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./sac_gnn_attention_{gnn_type}_{num_joints}joints_best/',
        log_path=f'./sac_gnn_attention_{gnn_type}_{num_joints}joints_logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("🎯 开始训练...")
    print("📊 训练配置:")
    print("   总步数: 50,000")
    print("   评估频率: 每 5,000 步")
    print("   预期: GNN 增强关节间协调，注意力提升特征选择")
    print("=" * 70)
    
    start_time = time.time()
    
    # 训练模型
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏆 训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model.save(f"sac_gnn_attention_{gnn_type}_{num_joints}joints_final")
    print(f"💾 模型已保存为: sac_gnn_attention_{gnn_type}_{num_joints}joints_final.zip")
    
    # 最终评估
    print("\n🔍 最终评估 (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    print(f"📊 最终评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 与之前结果对比
    baseline_reward = -4.86
    simple_attention_reward = -4.69
    
    improvement_vs_baseline = mean_reward - baseline_reward
    improvement_vs_simple = mean_reward - simple_attention_reward
    
    print(f"\n📈 性能对比:")
    print(f"   Baseline SAC: {baseline_reward:.2f}")
    print(f"   简化注意力: {simple_attention_reward:.2f}")
    print(f"   GNN + 注意力: {mean_reward:.2f}")
    print(f"   vs Baseline: {improvement_vs_baseline:+.2f}")
    print(f"   vs 简化注意力: {improvement_vs_simple:+.2f}")
    
    if improvement_vs_baseline > 0.3 and improvement_vs_simple > 0.1:
        print("   🎉 GNN + 注意力效果最佳!")
    elif improvement_vs_baseline > 0.1:
        print("   👍 GNN + 注意力有效改进!")
    elif improvement_vs_baseline > -0.1:
        print("   ⚖️ GNN + 注意力效果相当")
    else:
        print("   ⚠️ GNN + 注意力需要进一步优化")
    
    # 演示训练好的模型
    print("\n🎮 演示训练好的模型 (10 episodes)...")
    demo_env = gym.make('Reacher-v5', render_mode='human')
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(10):
        obs, info = demo_env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 判断成功
        if episode_reward > -5:
            success_count += 1
            print(f"🎯 Episode {episode+1}: 成功! 奖励={episode_reward:.2f}, 长度={episode_length}")
        else:
            print(f"📊 Episode {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")
    
    demo_env.close()
    
    # 演示统计
    demo_success_rate = success_count / 10
    demo_avg_reward = np.mean(episode_rewards)
    
    print("\n" + "=" * 70)
    print("📊 演示统计:")
    print(f"   成功率: {demo_success_rate:.1%} ({success_count}/10)")
    print(f"   平均奖励: {demo_avg_reward:.2f}")
    print(f"   平均长度: {np.mean(episode_lengths):.1f}")
    print(f"   奖励标准差: {np.std(episode_rewards):.2f}")
    
    # 与之前演示对比
    baseline_demo_success = 0.9
    simple_demo_success = 0.7
    baseline_demo_reward = -4.82
    simple_demo_reward = -3.91
    
    print(f"\n📈 演示效果对比:")
    print(f"   Baseline 成功率: {baseline_demo_success:.1%}")
    print(f"   简化注意力成功率: {simple_demo_success:.1%}")
    print(f"   GNN + 注意力成功率: {demo_success_rate:.1%}")
    print(f"   ")
    print(f"   Baseline 平均奖励: {baseline_demo_reward:.2f}")
    print(f"   简化注意力平均奖励: {simple_demo_reward:.2f}")
    print(f"   GNN + 注意力平均奖励: {demo_avg_reward:.2f}")
    
    # 训练时间对比
    baseline_time = 14.3
    simple_time = 16.4
    time_vs_baseline = training_time/60 - baseline_time
    time_vs_simple = training_time/60 - simple_time
    
    print(f"\n⏱️ 训练时间对比:")
    print(f"   Baseline: {baseline_time:.1f} 分钟")
    print(f"   简化注意力: {simple_time:.1f} 分钟")
    print(f"   GNN + 注意力: {training_time/60:.1f} 分钟")
    print(f"   vs Baseline: {time_vs_baseline:+.1f} 分钟")
    print(f"   vs 简化注意力: {time_vs_simple:+.1f} 分钟")
    
    if abs(time_vs_simple) < 3:
        print("   ✅ 训练时间增加可接受，GNN 开销合理")
    elif time_vs_simple > 5:
        print("   ⚠️ 训练时间显著增加，GNN 计算开销较大")
    
    print("\n✅ SAC + GNN + 注意力训练完成!")
    print("🔗 模型已准备好处理多关节 Reacher 任务!")
    
    # 清理
    env.close()
    eval_env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'demo_success_rate': demo_success_rate,
        'demo_avg_reward': demo_avg_reward,
        'improvement_vs_baseline': improvement_vs_baseline,
        'improvement_vs_simple': improvement_vs_simple,
        'time_vs_baseline': time_vs_baseline,
        'time_vs_simple': time_vs_simple
    }

if __name__ == "__main__":
    print("🔥 开始 SAC + GNN + 注意力机制训练")
    print("🔗 支持任意数量关节的 Reacher 任务")
    print("🎯 为多关节 Reacher 做准备")
    print()
    
    # 测试不同配置
    configs = [
        {"num_joints": 2, "gnn_type": "GCN"},
        # {"num_joints": 2, "gnn_type": "GAT"},  # 可选: 测试不同 GNN 类型
    ]
    
    for config in configs[:1]:  # 先测试一个配置
        print(f"\n{'='*60}")
        print(f"🧠 测试配置: {config['num_joints']} 关节, {config['gnn_type']} GNN")
        print(f"{'='*60}")
        
        try:
            results = sac_with_gnn_attention_training(**config)
            
            print(f"\n🎊 {config} 训练结果总结:")
            print(f"   最终评估奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   训练时间: {results['training_time']/60:.1f} 分钟")
            print(f"   演示成功率: {results['demo_success_rate']:.1%}")
            print(f"   演示平均奖励: {results['demo_avg_reward']:.2f}")
            print(f"   vs Baseline 改进: {results['improvement_vs_baseline']:+.2f}")
            print(f"   vs 简化注意力改进: {results['improvement_vs_simple']:+.2f}")
            print(f"   训练时间 vs Baseline: {results['time_vs_baseline']:+.1f} 分钟")
            print(f"   训练时间 vs 简化注意力: {results['time_vs_simple']:+.1f} 分钟")
            
            # 总体评估
            if (results['improvement_vs_baseline'] > 0.3 and 
                results['demo_success_rate'] > 0.8 and 
                results['time_vs_simple'] < 5):
                print(f"\n🏆 GNN + 注意力机制表现优秀!")
                print("   性能提升 + 高成功率 + 合理训练时间")
            elif results['improvement_vs_baseline'] > 0.1:
                print(f"\n👍 GNN + 注意力机制有效!")
            else:
                print(f"\n⚠️ GNN + 注意力机制需要进一步优化")
            
            print(f"\n🔗 模型已准备好扩展到多关节 Reacher!")
            
        except Exception as e:
            print(f"❌ {config} 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
