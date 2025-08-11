import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
base_dir = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_dataset"))
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_model"))
print(sys.path)
from data_utils import prepare_joint_q_input
from attn_model.attn_model import AttnModel
class AttentionCritic(nn.Module):
    def __init__(self, attn_model, joint_embed_dim, hidden_dim=256, device='cpu'):
        super(AttentionCritic, self).__init__()

        self.attn_model = attn_model  # 你自定义的多输入 attention 模型
        self.joint_embed_dim = joint_embed_dim
        # 聚合后传入 Q 网络的 MLP
        self.q_net = nn.Sequential(
            nn.Linear(joint_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.proj_dict = nn.ModuleDict()
        self.device = device

    def forward(self, joint_q, vertex_k, vertex_v, vertex_mask=None, action=None):
        """
        joint_q:     [B, N, Dq]，每个关节 query（例如 position/velocity）
        vertex_k/v:  [B, N, Dk] / [B, N, Dv]，图结构中的 key / value
        vertex_mask: [B, N]，True 表示该节点是 padding，需要 mask 掉
        action:      [B, N, Da]，每个关节对应的动作（需拼接到每个 state 上）
        """

        # 1. 使用你的 attention 模型编码结构状态
        state_features = self.attn_model(joint_q, vertex_k, vertex_v, vertex_mask)  # [B, N, D]

        # 2. 拼接动作到每个节点表示（假设 action 与 joint 对齐）
        if action is not None:
            x = torch.cat([state_features, action], dim=-1)  # [B, N, D + Da]
        else:
            x = state_features  # 允许无动作输入

        # 3. 线性映射到统一维度

        proj = self.get_proj(x.size(-1))
        x = proj(x)
        # x = x.unsqueeze(-1)
        # 4. 处理 padding mask：用 masked mean pooling 聚合节点特征
        # if vertex_mask is not None:
        #     # vertex_mask: True = padding，需要忽略
        #     mask_float = (~vertex_mask).unsqueeze(-1).float()  # [B, N, 1]
        #     x_sum = (x * mask_float).sum(dim=1)                 # [B, D]
        #     valid_count = mask_float.sum(dim=1) + 1e-6          # [B, 1]，避免除以0
        #     x_pooled = x_sum / valid_count                      # [B, D]
        # else:
        #     x_pooled = x.mean(dim=1)  # 没有 mask，就简单平均

        # 5. 送入 Q 网络输出标量 Q 值
        q_value = self.q_net(x)  # [B, 1]

        return q_value
    
    def get_proj(self, input_dim):
        key = str(input_dim)
        if key not in self.proj_dict:
            self.proj_dict[key] = nn.Linear(input_dim, self.joint_embed_dim).to(self.device)
        return self.proj_dict[key]


if __name__ == "__main__":

    B = 8
    J = 6
    D = 32
    N_max = 60

    test_obs = torch.tensor([[ 1.5792e+00,  1.6710e+00,  1.9350e+00,  1.6399e+00, -1.5213e+00,
         -1.5991e+00, -2.1151e+00, -4.6594e-01,  6.1969e-01, -5.4217e-01,
         -4.8190e-01, -1.3664e+00, -1.8061e+00, -1.9311e+00, -9.4792e-01,
          7.7807e-01, -1.8289e+00,  1.4493e+00,  2.3937e+00,  1.4891e+00,
         -1.7868e+00,  1.2160e+00, -1.4543e+00, -6.6117e-01, -2.2635e-01,
         -1.0822e+00,  1.1626e+00,  1.2674e+00, -1.8546e+00,  1.0580e+00,
          2.2741e+00,  1.4726e+00, -1.8210e+00,  1.3237e+00, -1.4984e+00,
         -7.1052e-01, -2.0631e-01, -1.0944e+00,  1.5697e+00,  9.9483e-01],
        [-5.3190e-01, -2.3999e-01, -9.7030e-01, -2.1623e-01,  5.1954e-01,
         -1.8349e+00,  1.0001e+00, -1.7703e+00,  5.3180e-01, -4.7350e-01,
         -2.2072e+00,  1.0164e+00,  2.3688e-01, -5.3535e-01, -7.3144e-01,
         -2.0535e+00, -2.7801e-02,  1.1331e+00, -1.6204e+00, -4.7636e-01,
         -9.1435e-02,  1.3800e+00, -1.6355e+00, -2.0422e+00,  1.1430e-01,
          1.7407e+00,  1.6848e+00,  2.5316e-01, -1.1471e-02,  1.2381e+00,
         -1.7352e+00, -4.4865e-01, -6.9775e-02,  1.3298e+00, -1.5231e+00,
         -2.0845e+00,  1.4675e-01,  1.7235e+00,  1.4620e+00,  9.1657e-02],
        [ 1.0653e+00, -2.1166e+00, -6.3734e-02, -2.1238e+00, -1.1302e+00,
          6.8247e-01,  8.4890e-02,  9.6695e-01, -3.8633e-01, -1.5040e-01,
          1.6941e-01, -1.1963e-01,  1.5749e+00,  1.0833e+00, -4.8101e-01,
          1.9020e-01,  2.3683e+00, -4.9519e-01, -1.6132e+00,  7.3752e-01,
         -1.7595e+00, -4.9084e-03,  1.2917e+00,  9.0742e-01,  1.2229e-01,
          4.2009e-01, -1.9650e+00,  7.7563e-01,  2.3548e+00, -8.5385e-02,
         -1.6093e+00,  6.3169e-01, -1.7349e+00,  5.3448e-04,  1.6892e+00,
          7.4927e-01,  1.7760e-01,  4.4231e-01, -2.0409e+00,  1.2620e+00],
        [-3.3281e-01, -7.4954e-01, -1.4622e+00, -7.1293e-01,  4.2784e-01,
         -2.0396e+00,  1.5792e+00, -1.6890e+00,  1.0945e+00, -1.5526e+00,
         -5.9893e-01,  2.2768e+00,  8.1526e-01,  1.3457e+00, -1.4010e+00,
         -1.8664e+00,  7.4176e-01, -4.6633e-01,  6.1337e-01, -1.4477e+00,
          1.9017e+00,  1.3788e+00, -8.8177e-01, -5.7803e-01, -6.5382e-02,
          3.0847e+00,  2.2385e+00, -6.3671e-01,  7.4607e-01, -1.4345e+00,
          6.4308e-01, -1.5544e+00,  1.8392e+00,  1.3128e+00, -8.9914e-01,
         -6.2796e-01, -7.0117e-02,  3.0818e+00,  1.9543e+00, -6.4790e-01],
        [ 1.2532e+00,  1.5626e+00, -5.6045e-01,  1.5656e+00, -1.3005e+00,
         -2.3065e-01,  4.5846e-01, -6.9249e-01, -1.2369e-01, -8.6954e-01,
          1.1168e-01,  7.0666e-02, -1.7956e+00, -2.0827e+00, -9.0418e-01,
         -8.3449e-01, -1.2593e+00,  6.8642e-01, -9.4674e-01, -6.7277e-01,
          9.0638e-01,  1.3686e+00, -8.9712e-01, -3.5714e-01, -2.2843e+00,
         -8.7351e-01, -7.2166e-01,  2.0406e+00, -1.2701e+00,  1.0573e+00,
         -9.1799e-01, -6.1053e-01,  8.7397e-01,  1.3232e+00, -9.2751e-01,
         -4.3622e-01, -2.2872e+00, -8.4480e-01, -9.9729e-01,  1.9762e+00],
        [ 2.4665e+00,  2.0524e+00, -1.0578e+00,  2.0728e+00, -2.5415e+00,
         -1.1510e+00,  7.1224e-01, -2.0219e+00,  4.0974e-01, -1.1189e+00,
         -2.0873e+00, -2.4082e-01, -2.0956e+00, -1.0573e+00, -1.4331e+00,
          2.0534e+00, -1.8583e+00, -2.3148e+00, -1.8011e-01, -1.9859e+00,
         -1.6667e+00,  4.2188e-01,  2.2482e+00,  2.5093e+00, -2.2643e+00,
         -6.2435e-01,  5.4961e-01,  2.0506e+00, -1.8468e+00, -1.6428e+00,
         -1.0639e-01, -1.8384e+00, -1.7758e+00,  3.9066e-01,  2.0925e+00,
          2.7420e+00, -2.2550e+00, -6.6147e-01,  5.0261e-01,  2.1435e+00],
        [ 3.3980e-01,  7.1580e-01, -2.2118e+00,  7.2259e-01,  3.9917e-02,
          4.8438e-01,  2.2112e+00, -3.1109e-01,  1.6503e+00, -1.5973e+00,
         -1.2344e+00,  2.3581e+00, -5.4388e-01, -3.8175e-01, -1.1391e+00,
         -1.1465e+00, -7.3005e-01, -1.7941e+00, -1.6513e+00, -1.3584e+00,
          8.1753e-01, -2.2825e+00,  5.4717e-01,  2.8406e-01,  1.2007e+00,
         -6.9984e-02,  4.8855e-01, -1.6927e+00, -7.1616e-01, -2.1844e+00,
         -1.6913e+00, -1.2478e+00,  7.8660e-01, -2.3910e+00,  5.0469e-01,
          1.6151e-01,  1.2092e+00, -1.2239e-01,  4.6819e-01, -1.4744e+00],
        [-6.0278e-01, -3.9434e-01,  1.2594e+00, -3.9809e-01,  6.0811e-01,
          9.0704e-01, -1.2064e+00,  9.1328e-01, -2.1171e-01, -9.4515e-01,
          1.0861e+00, -1.1136e+00,  8.1851e-03, -9.2969e-01, -7.6695e-01,
         -3.4889e-01,  6.8036e-01,  1.7602e+00,  2.7090e-01,  2.0829e+00,
          1.3426e+00,  9.8267e-01, -1.4143e+00, -1.8268e+00,  1.9668e+00,
         -7.0366e-01, -1.0873e+00,  2.3249e-02,  6.8005e-01,  1.6365e+00,
          3.0697e-01,  2.3014e+00,  1.3062e+00,  9.4530e-01, -1.3501e+00,
         -1.4542e+00,  1.9766e+00, -7.6521e-01, -1.2677e+00, -4.2978e-02]],
       dtype=torch.float32)
    
    gnn_embeds_test = torch.randn(B, 12, 128)
    joint_q_input = prepare_joint_q_input(test_obs, gnn_embeds_test, 12)

    vertex_k_test = gnn_embeds_test
    joint_vel = test_obs[:, 28:40]
    vertex_v_test = gnn_embeds_test
    test_action = torch.randn(B, 12)
    attn_model = AttnModel(128, 130, 130, 4)
    critic = AttentionCritic(attn_model, 128, 256)
    q_value = critic(joint_q_input, vertex_k_test, vertex_v_test, action = test_action)
    print(q_value)