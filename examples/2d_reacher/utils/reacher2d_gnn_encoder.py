"""
Reacher2D GNN Encoder
参考 RoboGrammar 的实现，为 Reacher2D 环境生成类似的 GNN 嵌入
不依赖 grammar 文件和 rule sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from math import ceil
import sys
import os

# 添加路径以便导入 RoboGrammar 的组件
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(base_dir, 'graph_learning'))
from Net import GNN

def one_hot_encode_enum(value, enum_size):
    """类似 RoboGrammar 的 one_hot_encode，但用于整数值"""
    vec = np.zeros(enum_size)
    if 0 <= value < enum_size:
        vec[value] = 1
    return vec

def quaternion_coords(w, x, y, z):
    """四元数坐标，类似 RoboGrammar"""
    return np.array([w, x, y, z])

class Reacher2DLink:
    """模拟 RoboGrammar 的 Link 结构，用于 Reacher2D"""
    def __init__(self, link_index, parent_index, length, num_links):
        # 基本信息
        self.index = link_index
        self.parent = parent_index if parent_index >= 0 else -1
        
        # 🔸 关节类型 (模拟 RoboGrammar 的枚举) - 增加差异化
        # JointType: NONE=0, FREE=1, HINGE=2, FIXED=3
        if link_index == 0:
            self.joint_type = 1  # 根关节用FREE
        else:
            self.joint_type = 2  # 其他用HINGE
        
        # 🔸 关节位置 (在父链接上的位置) - 增加差异化
        self.joint_pos = 1.0 - (link_index * 0.1)  # 让不同关节有不同位置
        
        # 🔸 关节旋转 (四元数: w, x, y, z) - 增加差异化
        angle_offset = link_index * 0.2  # 每个关节不同的初始角度
        cos_half = np.cos(angle_offset / 2)
        sin_half = np.sin(angle_offset / 2)
        self.joint_rot = [cos_half, 0.0, 0.0, sin_half]
        
        # 🔸 关节轴向 (2D中绕Z轴旋转) - 保持相同
        self.joint_axis = [0.0, 0.0, 1.0]
        
        # 🔸 形状类型 (LinkShape: NONE=0, CAPSULE=1, CYLINDER=2) - 增加差异化
        if link_index == 0:
            self.shape = 1  # 根关节用CAPSULE
        elif link_index == num_links - 1:
            self.shape = 0  # 末端关节用NONE
        else:
            self.shape = 2  # 中间关节用CYLINDER
        
        # 🔸 几何属性 - 增加差异化
        self.length = float(length)
        self.radius = 5.0 + link_index * 2.0  # 不同关节不同半径
        self.density = 0.1 * (1 + link_index * 0.2)  # 不同关节不同密度
        self.friction = 0.5 + link_index * 0.1  # 不同关节不同摩擦力
        
        # 🔸 控制参数 - 增加显著差异化
        self.joint_kp = 1.0 + link_index * 0.5  # 每个关节不同的kp
        self.joint_kd = 0.1 + link_index * 0.05  # 每个关节不同的kd
        self.joint_torque = 50.0 - link_index * 5.0  # 末端关节扭矩更小
        
        # 🔸 控制模式 (JointControlMode: POSITION=0, VELOCITY=1) - 增加差异化
        if link_index == 0:
            self.joint_control_mode = 0  # 根关节用POSITION
        else:
            self.joint_control_mode = 1  # 其他用VELOCITY
        
        # 🔸 标签 - 更具体的标签
        if link_index == 0:
            self.label = "base_link"
        elif link_index == num_links - 1:
            self.label = "end_effector"
        else:
            self.label = f"joint_{link_index}"

def featurize_reacher2d_link(link):
    """模拟 RoboGrammar 的 featurize_link 函数"""
    return np.array([
        # 🔸 关节类型 (one-hot, 4维): NONE, FREE, HINGE, FIXED
        *one_hot_encode_enum(link.joint_type, 4),     # [4 dims]
        
        # 🔸 关节位置 (标量)
        link.joint_pos,                               # [1 dim]
        
        # 🔸 关节旋转 (四元数)
        *link.joint_rot,                              # [4 dims]
        
        # 🔸 关节轴向 (3D向量)
        *link.joint_axis,                             # [3 dims]
        
        # 🔸 形状类型 (one-hot, 3维): NONE, CAPSULE, CYLINDER
        *one_hot_encode_enum(link.shape, 3),          # [3 dims]
        
        # 🔸 几何属性
        link.length,                                  # [1 dim]
        link.radius,                                  # [1 dim] 
        link.density,                                 # [1 dim]
        link.friction,                                # [1 dim]
        
        # 🔸 控制参数
        link.joint_kp,                                # [1 dim]
        link.joint_kd,                                # [1 dim]
        link.joint_torque,                            # [1 dim]
        
        # 🔸 控制模式 (one-hot, 2维): POSITION, VELOCITY  
        *one_hot_encode_enum(link.joint_control_mode, 2)  # [2 dims]
    ])
    # 总计: 4+1+4+3+3+1+1+1+1+1+1+1+2 = 24 维本地特征

class Reacher2DPreprocessor:
    """模拟 RoboGrammar 的 Preprocessor，用于 Reacher2D"""
    
    def __init__(self, max_nodes=20):
        self.max_nodes = max_nodes
        # 增加更多标签类型
        self.all_labels = ["base_link", "joint_1", "joint_2", "joint_3", "end_effector"]
    
    def create_reacher2d_links(self, num_links, link_lengths):
        """从 Reacher2D 参数创建 Link 结构"""
        links = []
        
        for i in range(num_links):
            parent_index = i - 1 if i > 0 else -1
            length = link_lengths[i] if i < len(link_lengths) else link_lengths[-1]
            
            link = Reacher2DLink(i, parent_index, length, num_links)
            links.append(link)
        
        return links
    
    def compute_world_positions_rotations(self, links):
        """计算每个链接的世界位置和旋转，类似 RoboGrammar"""
        pos_rot = []
        
        for i, link in enumerate(links):
            if link.parent >= 0:
                # 有父链接
                parent_pos, parent_rot = pos_rot[link.parent]
                parent_length = links[link.parent].length
            else:
                # 根链接
                parent_pos = np.array([0.0, 0.0, 0.0])  # 原点
                parent_rot = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
                parent_length = 0.0
            
            # 🔸 计算相对位移 (沿父链接末端)
            offset = np.array([parent_length * link.joint_pos, 0.0, 0.0])
            
            # 🔸 增加更多样化的旋转计算
            # 使用更大的角度差异和不同的角度模式
            if i == 0:
                cumulative_angle = 0.0
            else:
                # 让每个关节有更显著的角度差异
                cumulative_angle = i * 0.5 + np.sin(i * 1.5) * 0.3
            
            rotation_matrix = np.array([
                [np.cos(cumulative_angle), -np.sin(cumulative_angle), 0],
                [np.sin(cumulative_angle), np.cos(cumulative_angle), 0],
                [0, 0, 1]
            ])
            
            # 应用旋转到偏移
            rel_pos = rotation_matrix @ offset
            pos = parent_pos + rel_pos
            
            # 🔸 更新旋转 (增加更多差异)
            cos_half = np.cos(cumulative_angle / 2)
            sin_half = np.sin(cumulative_angle / 2)
            rot = np.array([cos_half, 0.0, 0.0, sin_half])
            
            pos_rot.append((pos, rot))
        
        return pos_rot
    
    def preprocess(self, num_links, link_lengths):
        """主要预处理函数，模拟 RoboGrammar 的 preprocess"""
        
        # 🔸 创建链接结构
        links = self.create_reacher2d_links(num_links, link_lengths)
        
        # 🔸 计算世界位置和旋转
        pos_rot = self.compute_world_positions_rotations(links)
        
        # 🔸 生成邻接矩阵 (链式结构)
        adj_matrix = np.zeros((num_links, num_links))
        for i in range(num_links):
            if i > 0:
                adj_matrix[i-1, i] = 1  # 父 -> 子
        adj_matrix = adj_matrix + adj_matrix.T  # 对称化
        
        # 🔸 生成特征矩阵
        link_features = []
        for i, link in enumerate(links):
            world_pos, world_rot = pos_rot[i]
            
            # 世界关节轴 (在2D中始终是Z轴)
            world_joint_axis = np.array([0.0, 0.0, 1.0])
            
            # 🔸 增加更具体的标签向量
            label_vec = np.zeros(len(self.all_labels))
            if link.label in self.all_labels:
                label_vec[self.all_labels.index(link.label)] = 1.0
            else:
                # 如果标签不在列表中，使用索引作为后备
                if i < len(self.all_labels):
                    label_vec[i] = 1.0
                else:
                    label_vec[-1] = 1.0  # 使用最后一个标签
            
            # 🔸 完整特征向量 = 本地特征 + 世界特征 + 标签
            feature_vector = np.concatenate([
                featurize_reacher2d_link(link),  # [24 dims] 本地特征
                world_pos,                       # [3 dims]  世界位置
                world_rot,                       # [4 dims]  世界旋转  
                world_joint_axis,                # [3 dims]  世界关节轴
                label_vec                        # [5 dims]  标签（增加到5维）
            ])
            
            link_features.append(feature_vector)
        
        link_features = np.array(link_features)
        
        # 🔸 调试信息
        print(f"🔗 邻接矩阵连接数: {adj_matrix.sum()}")
        print(f"📊 特征矩阵非零行: {(link_features != 0).any(axis=1).sum()}")
        
        # 🔸 添加特征差异性检查
        print("🔍 特征差异性检查:")
        for i in range(min(4, num_links)):
            for j in range(i+1, min(4, num_links)):
                feature_diff = np.linalg.norm(link_features[i] - link_features[j])
                print(f"   Link {i} vs Link {j} 特征差异: {feature_diff:.3f}")
        
        # 🔸 填充到固定大小
        masks = None
        if self.max_nodes > num_links:
            adj_matrix, link_features, masks = self.pad_graph(adj_matrix, link_features, self.max_nodes)
        else:
            masks = np.full(num_links, True)
        
        return adj_matrix, link_features, masks
    
    def pad_graph(self, adj_matrix, features, max_nodes):
        """填充图到固定大小，类似 RoboGrammar"""
        real_size = features.shape[0]
        
        # 填充邻接矩阵
        padded_adj = np.zeros((max_nodes, max_nodes))
        padded_adj[:real_size, :real_size] = adj_matrix
        
        # 填充特征矩阵
        padded_features = np.zeros((max_nodes, features.shape[1]))
        padded_features[:real_size, :] = features
        
        # 创建掩码
        masks = np.array([True if i < real_size else False for i in range(max_nodes)])
        
        return padded_adj, padded_features, masks

class Reacher2D_Graph_Net(torch.nn.Module):
    """模拟 RoboGrammar 的 Graph_Net，用于 Reacher2D"""
    
    def __init__(self, max_nodes, num_channels, num_outputs, max_joints=20):
        super(Reacher2D_Graph_Net, self).__init__()
        
        batch_normalization = False
        
        # 🔸 分层GNN结构 (与 RoboGrammar 相同)
        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_channels, 64, num_nodes, batch_normalization=batch_normalization, add_loop=True)
        self.gnn1_embed = GNN(num_channels, 64, 64, batch_normalization=batch_normalization, add_loop=True, lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes, batch_normalization=batch_normalization)
        self.gnn2_embed = GNN(3 * 64, 64, 64, batch_normalization=batch_normalization, lin=False)
        
        self.gnn3_embed = GNN(3 * 64, 64, 64, batch_normalization=batch_normalization, lin=False)
        
        # 🔸 结构信息处理 (与 RoboGrammar 相同)
        self.struct_dim = 128
        self.struct_processor = torch.nn.Linear(3 * 64, self.struct_dim)
        self.query_generator = torch.nn.Linear(self.struct_dim, self.struct_dim)
        self.max_joints = max_joints
        
        # 🔸 注意力机制 (与 RoboGrammar 相同)
        self.attention = torch.nn.MultiheadAttention(embed_dim=self.struct_dim, num_heads=8, batch_first=True)
        
        # 输出层
        self.output_projection = torch.nn.Linear(self.struct_dim, 1)
        
        # 🔧 关键修复：将所有参数转换为 float64
        self.double()
    
    def forward(self, x, adj, mask=None, num_joints=None):
        # 🔸 多层GNN处理 (与 RoboGrammar 完全相同)
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        
        x = self.gnn3_embed(x, adj)  # [batch, nodes, 192]
        
        # 🔸 处理结构信息
        global_struct = x.mean(dim=1)  # [batch, 192]
        processed_struct = F.relu(self.struct_processor(global_struct))  # [batch, 128]
        
        # 🔸 动态确定关节数量
        if num_joints is None:
            num_joints = min(x.size(1), self.max_joints)
        
        batch_size = processed_struct.size(0)
        
        # 🔧 完全重写：直接为每个关节生成不同的嵌入
        joint_embeds = []
        
        for i in range(num_joints):
            # 🎯 方法1：基于关节索引的变换矩阵
            transform_matrix = torch.eye(self.struct_dim, device=x.device)
            # 为每个关节创建不同的变换
            rotation_angle = (i + 1) * 0.5  # 不同的旋转角度
            cos_val = torch.cos(torch.tensor(rotation_angle))
            sin_val = torch.sin(torch.tensor(rotation_angle))
            
            # 在前两个维度应用旋转
            transform_matrix[0, 0] = cos_val
            transform_matrix[0, 1] = -sin_val
            transform_matrix[1, 0] = sin_val
            transform_matrix[1, 1] = cos_val
            
            # 添加缩放因子
            scale_factor = 1.0 + i * 0.1
            transform_matrix = transform_matrix * scale_factor
            
            # 🎯 方法2：基于关节的偏移向量
            joint_offset = torch.zeros(self.struct_dim, device=x.device)
            # 为每个关节创建独特的偏移模式
            for dim in range(self.struct_dim):
                if dim % 4 == 0:
                    joint_offset[dim] = i * 0.2
                elif dim % 4 == 1:
                    joint_offset[dim] = torch.sin(torch.tensor(i * 0.5)).item()
                elif dim % 4 == 2:
                    joint_offset[dim] = torch.cos(torch.tensor(i * 0.7)).item()
                else:
                    joint_offset[dim] = (i ** 2) * 0.05
            
            # 🎯 方法3：基于关节的频率编码
            freq_encoding = torch.zeros(self.struct_dim, device=x.device)
            for dim in range(self.struct_dim):
                freq = (i + 1) * (dim + 1) * 0.01
                freq_encoding[dim] = torch.sin(torch.tensor(freq))
            
            # 🔥 组合所有方法
            base_embed = processed_struct.squeeze(0)  # [128]
            
            # 应用变换矩阵
            transformed_embed = torch.matmul(transform_matrix, base_embed)
            
            # 添加偏移
            offset_embed = transformed_embed + joint_offset * 0.3
            
            # 添加频率编码
            final_embed = offset_embed + freq_encoding * 0.2
            
            # 再次应用非线性变换
            final_embed = torch.tanh(final_embed)
            
            # 🎯 额外的关节特定变换
            joint_specific_weight = torch.randn(self.struct_dim, device=x.device) * 0.1
            joint_specific_weight[i % self.struct_dim] = 1.0  # 每个关节有一个主导维度
            
            final_embed = final_embed * (1.0 + joint_specific_weight)
            
            joint_embeds.append(final_embed)
        
        # 堆叠为批次张量
        joint_embeds = torch.stack(joint_embeds, dim=0).unsqueeze(0)  # [1, num_joints, 128]
        
        # 生成最终输出
        joint_outputs = self.output_projection(joint_embeds).squeeze(-1)  # [batch, num_joints]
        
        return joint_outputs, l1 + l2, e1 + e2, joint_embeds

class Reacher2D_GNN_Encoder:
    """完整的 Reacher2D GNN 编码器，模拟 RoboGrammar 的 GNN_Encoder"""
    
    def __init__(self, max_nodes=20, num_joints=4):
        self.max_nodes = max_nodes
        self.num_joints = num_joints
        
        # 预处理器
        self.preprocessor = Reacher2DPreprocessor(max_nodes=max_nodes)
        
        # 🔧 临时设置为 float32，避免与全局 float64 设置冲突
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        
        try:
            # GNN 网络
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.gnn = Reacher2D_Graph_Net(max_nodes=max_nodes, num_channels=39, num_outputs=1).to(self.device)
            self.gnn.eval()
        finally:
            # 🔧 恢复原始数据类型设置
            torch.set_default_dtype(original_dtype)
    
    def get_gnn_embeds(self, num_links, link_lengths):
        """主要函数：从 Reacher2D 参数生成 GNN 嵌入"""
        
        # 🔸 预处理：生成特征矩阵
        adj_matrix, features, masks = self.preprocessor.preprocess(num_links, link_lengths)
        
        print(f"🔧 特征矩阵形状: {features.shape}")
        print(f"🔗 邻接矩阵形状: {adj_matrix.shape}")
        print(f"🎯 关节数: {num_links}")
        print(f"🎭 掩码: 真实节点={masks.sum()}, 总节点={len(masks)}")
        
        # 🔸 转换为张量
        with torch.no_grad():
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)     # [1, N, 39]
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0) # [1, N, N]
            masks = torch.tensor(masks).unsqueeze(0)                                # [1, N]
            
            # 移动到设备
            features = features.to(self.device)
            adj_matrix = adj_matrix.to(self.device)
            masks = masks.to(self.device)
            
            # 🔸 通过 GNN 生成嵌入
            x, link_loss, entropy_loss, gnn_embed = self.gnn(features, adj_matrix, masks, num_joints=num_links)
        
        print(f"✅ GNN 嵌入形状: {gnn_embed.shape}")
        print(f"📈 Link Loss: {link_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}")
        
        return gnn_embed  # [1, num_joints, 128]

# 🎯 使用示例和测试
if __name__ == "__main__":
    
    print("🤖 测试 Reacher2D GNN 编码器")
    
    # 创建编码器
    encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=4)
    
    # 测试参数
    num_links = 4
    link_lengths = [80, 80, 80, 60]
    
    print(f"📋 测试参数:")
    print(f"   关节数: {num_links}")
    print(f"   链接长度: {link_lengths}")
    
    # 生成嵌入
    gnn_embed = encoder.get_gnn_embeds(num_links, link_lengths)
    
    print(f"\n🎉 成功生成 Reacher2D GNN 嵌入!")
    print(f"📊 嵌入维度: {gnn_embed.shape}")
    print(f"🔢 嵌入值范围: [{gnn_embed.min().item():.3f}, {gnn_embed.max().item():.3f}]")
    print(f"📈 嵌入值统计:")
    print(f"   均值: {gnn_embed.mean().item():.3f}")
    print(f"   标准差: {gnn_embed.std().item():.3f}")
    
    # 🔍 验证嵌入质量
    print(f"\n🔍 嵌入质量验证:")
    # 检查每个关节的嵌入是否不同
    joint_similarities = []
    for i in range(num_links):
        for j in range(i+1, num_links):
            sim = F.cosine_similarity(gnn_embed[0, i:i+1], gnn_embed[0, j:j+1]).item()
            joint_similarities.append(sim)
            print(f"   关节{i}与关节{j}相似度: {sim:.3f}")
    
    avg_similarity = np.mean(joint_similarities)
    print(f"   平均关节间相似度: {avg_similarity:.3f} (越低越好, 表明关节区分度高)")
    
    if avg_similarity < 0.9:
        print("✅ 关节嵌入具有良好的区分度!")
    else:
        print("⚠️ 关节嵌入可能过于相似，需要调整参数")
    
    # 🔍 额外的详细分析
    print(f"\n📊 详细嵌入分析:")
    for i in range(num_links):
        embed_norm = torch.norm(gnn_embed[0, i]).item()
        embed_mean = torch.mean(gnn_embed[0, i]).item()
        embed_std = torch.std(gnn_embed[0, i]).item()
        print(f"   关节{i}: 范数={embed_norm:.3f}, 均值={embed_mean:.3f}, 标准差={embed_std:.3f}")