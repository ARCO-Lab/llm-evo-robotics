import torch
from torch.utils.data import DataLoader
from attn_dataset.robot_attn_dataset import RobotDataset, collate_fn
from attn_dataset.data_utils import prepare_joint_q_input


class DataHandler:
    def __init__(self,num_joints):
        self.dataset = RobotDataset()
        self.num_joints = num_joints
      

    def save_data(self, obs, action, reward, gnn_embeds, done):
           
        joint_q_input = prepare_joint_q_input(obs, gnn_embeds, self.num_joints)

        vertex_k = gnn_embeds  # [B, N, 128]
        joint_vel_start = 16 + self.num_joints  # 位置信息后面是速度信息
        joint_vel_end = joint_vel_start + self.num_joints
        joint_vel = obs[:, joint_vel_start:joint_vel_end]  # [B, num_joints]
        vertex_v = gnn_embeds  # [B, N, 128] - 直接使用相同的嵌入

        for i in zip(joint_q_input, vertex_k, vertex_v):
            sample = {
                "joint_q": i[0],
                "vertex_k": i[1], 
                "vertex_v": i[2],
                "vertex_len": i[1].shape[0],
            }
            self.dataset.add_data(sample)

    def get_data_loader(self):
        return  DataLoader(self.dataset, collate_fn=collate_fn, batch_size=32, shuffle=True)
    def get_data_length(self):
        return len(self.dataset)