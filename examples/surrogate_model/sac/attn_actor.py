import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
class AttentionActor(nn.Module):
    def __init__(self, attn_model, action_dim, log_std_min=-20, log_std_max=2):
        super(AttentionActor, self).__init__()
        self.attn_model = attn_model
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max


        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        mean = self.attn_model(joint_q, vertex_k, vertex_v, vertex_mask)

        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
    

    def sample(self, joint_q, vertex_k, vertex_v, vertex_mask=None):
        mean, log_std = self.forward(joint_q, vertex_k, vertex_v, vertex_mask)
        std = log_std.exp()

        normal = Normal(mean, std)    
        x_t = normal.rsample()

        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action , log_prob, mean