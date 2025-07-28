import torch
import torch.nn.functional as F
def prepare_joint_q_input(obs_batch, gnn_embeds, num_joints):
    """
    obs_batch: Tensor of shape [B, 40]  → B 个机器人个体
    gnn_embeds: Tensor of shape [B, 12, D] → 每个个体的每个关节的 GNN embedding
    Returns:
        joint_q_input: Tensor of shape [B, 12, 2 + D]
    """
    joint_q_input = []

    for b in range(obs_batch.size(0)):
        obs = obs_batch[b]
        joint_pos_start = 16
        joint_pos_end = joint_pos_start + num_joints
        joint_vel_start = joint_pos_end  
        joint_vel_end = joint_vel_start + num_joints
        
        joint_pos = obs[joint_pos_start:joint_pos_end]  # [num_joints]
        joint_vel = obs[joint_vel_start:joint_vel_end]  # [num_joints]
        gnn_embed = gnn_embeds[b]  # [num_joints, D]

        # Expand joint_pos and joint_vel to [12, 1]
        pos = joint_pos.unsqueeze(1)
        vel = joint_vel.unsqueeze(1)
        # print(pos.shape, vel.shape, gnn_embed.shape)
        # Concatenate: [12, 1] + [12, 1] + [12, D] → [12, 2 + D]
        q_input = torch.cat([pos, vel, gnn_embed], dim=1)
        joint_q_input.append(q_input)

    joint_q_input = torch.stack(joint_q_input, dim=0)  # [B, 12, 2 + D]
    return joint_q_input

def vertex_attention(vertex_k, vertex_v, joint_q, mask=None):
    """
    vertex_k: [B, N, D_k]          
    vertex_v: [B, N, D_v]          
    joint_q : [B, J, H, D_k]       
    """
    B, N, D_k = vertex_k.shape
    _, J, H, _ = joint_q.shape
    _, _, D_v = vertex_v.shape

    # 重塑为便于计算的形状
    k = vertex_k.view(B, 1, 1, N, D_k).expand(B, J, H, N, D_k)
    v = vertex_v.view(B, 1, 1, N, D_v).expand(B, J, H, N, D_v)
    q = joint_q.view(B, J, H, 1, D_k).expand(B, J, H, N, D_k)

    # 计算注意力分数
    attn_scores = torch.sum(q * k, dim=-1)  # [B, J, H, N]
    
    if mask is not None:
        mask_expanded = mask.view(B, 1, 1, N)
        attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))

    # 计算注意力权重并应用
    attn_weights = F.softmax(attn_scores, dim=-1)  # [B, J, H, N]
    attn_weights_expanded = attn_weights.view(B, J, H, N, 1)  # [B, J, H, N, 1]
    weighted_values = attn_weights_expanded * v  # [B, J, H, N, D_v]
    output = torch.sum(weighted_values, dim=3)   # [B, J, H, D_v]
    
    return output

def create_vertex_mask(vertex_lengths, max_vertices):
    """
    vertex_lengths: List[int] of length B (真实的节点数量)
    max_vertices: int，最大节点数量，用于padding
    Returns:
        mask: Tensor of shape [B, 1, 1, N]，bool 类型，True 表示位置需要mask掉
    """
    batch_size = len(vertex_lengths)
    mask = torch.zeros(batch_size, max_vertices, dtype=torch.bool)
    
    for i, l in enumerate(vertex_lengths):
        if l < max_vertices:
            mask[i, l:] = True  # 将 padding 的部分标记为 True（即被 mask 掉）

    # reshape 成为 [B, 1, 1, N]，方便广播到 multi-head attention
    return mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
