import sys
import os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "surrogate_model"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_dataset.data_utils import vertex_attention



class AttnModel(nn.Module):
    def __init__(self, vertex_key_size=128, vertex_value_size=128, joint_query_size=130, num_heads=4):  # 改为128
        super(AttnModel, self).__init__()
        self.vertex_key_size = vertex_key_size
        self.vertex_value_size = vertex_value_size  # 现在是128
        self.joint_query_size = joint_query_size
        self.num_heads = num_heads
        
        self.joint_q_encoder = nn.Sequential(
            nn.Linear(joint_query_size, 64),
            nn.ReLU(),
            nn.Linear(64, vertex_key_size * num_heads)  # 128*4=512
        )

        self.output_layer = nn.Sequential(
            nn.Linear(vertex_value_size * num_heads, 64),  # 128*4=512
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()            
        )

    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        batch_size, num_joints, _ = joint_q.shape
        
        joint_q = self.joint_q_encoder(joint_q)
        joint_q = joint_q.view(batch_size, num_joints, self.num_heads, self.vertex_key_size)
        
        attn_output = vertex_attention(vertex_k, vertex_v, joint_q, vertex_mask)
        # print(f"attn_output shape: {attn_output.shape}")  # 验证实际形状
        
        # 展平多头输出
        attn_output = attn_output.view(batch_size, num_joints, self.num_heads * self.vertex_value_size)
        # print(f"after view shape: {attn_output.shape}")  # 应该是 [32, 12, 512]
        
        output = self.output_layer(attn_output)  # [32, 12, 1]
        output = output.squeeze(-1)              # [32, 12]
        
        return output