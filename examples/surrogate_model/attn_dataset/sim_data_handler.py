import torch
from torch.utils.data import DataLoader
from attn_dataset.robot_attn_dataset import RobotDataset, collate_fn
from attn_dataset.data_utils import prepare_joint_q_input, prepare_reacher2d_joint_q_input, prepare_dynamic_vertex_v


class DataHandler:
    def __init__(self, num_joints, env_type):
        self.dataset = RobotDataset()
        self.num_joints = num_joints
        self.env_type = env_type
        print(f"🤖 初始化 DataHandler，环境类型: {env_type}，关节数: {num_joints}")

    def save_data(self, obs, action, reward, gnn_embeds, done):
           
        if self.env_type == 'reacher2d':
            joint_q_input = prepare_reacher2d_joint_q_input(obs, gnn_embeds, self.num_joints)
        else:
            joint_q_input = prepare_joint_q_input(obs, gnn_embeds, self.num_joints)

        vertex_k = gnn_embeds  # [B, N, 128] - K保持静态结构信息
        
        # 🎯 关键改进：使用动态增强的V向量
        vertex_v = prepare_dynamic_vertex_v(obs, gnn_embeds, self.num_joints, self.env_type)
        print(f"🔍 Enhanced vertex_v shape: {vertex_v.shape}")  # 调试信息

        for i in zip(joint_q_input, vertex_k, vertex_v):
            sample = {
                "joint_q": i[0],
                "vertex_k": i[1], 
                "vertex_v": i[2],  # 现在包含动态信息
                "vertex_len": i[1].shape[0],
            }
            self.dataset.add_data(sample)

    def get_data_loader(self):
        return DataLoader(self.dataset, collate_fn=collate_fn, batch_size=32, shuffle=True)
    
    def get_data_length(self):
        return len(self.dataset)