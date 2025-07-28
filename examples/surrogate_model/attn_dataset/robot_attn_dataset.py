import torch
from torch.utils.data import Dataset
import random

class RobotDataset(Dataset):
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def add_data(self, obs):


        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)

        self.buffer.append(obs)


    def sample_batch(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        return torch.stack([self.buffer[i] for i in indices])
    

def collate_fn(batch):
    batch_size = len(batch)
    # TODO: make this dynamic
    max_vertices = 70
    D_k = batch[0]["vertex_k"].shape[-1]
    D_v = batch[0]["vertex_v"].shape[-1]

    obs_batch = torch.stack([d["joint_q"] for d in batch], dim=0)

    vertex_k = torch.zeros(batch_size, max_vertices, D_k)
    vertex_v = torch.zeros(batch_size, max_vertices, D_v)
    mask = torch.zeros(batch_size, 1, 1, max_vertices, dtype=torch.bool)

    for i, d in enumerate(batch):
        L = d["vertex_len"]
        vertex_k[i, :L, :] = d["vertex_k"]
        vertex_v[i, :L, :] = d["vertex_v"]
        if L < max_vertices:
            mask[i, 0, 0, L:] = True

    return {
        "joint_q": obs_batch,
        "vertex_k": vertex_k,
        "vertex_v": vertex_v,
        "vertex_len": L,
        "vertex_mask": mask,
    }
