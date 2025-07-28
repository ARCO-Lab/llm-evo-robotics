import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_actor import AttentionActor
from attn_critic import AttentionCritic
class SAC(nn.Module):

    def __init__(self, attn_model, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):

        super(SAC, self).__init__()

        self.actor = AttentionActor(attn_model, action_dim)
        self.critic1 = AttentionCritic(attn_model, state_dim, action_dim)
        self.critic2 = AttentionCritic(attn_model, state_dim, action_dim)

        self.target_critic1 = AttentionCritic(attn_model, state_dim, action_dim)
        self.target_critic2 = AttentionCritic(attn_model, state_dim, action_dim)
        
        
        

