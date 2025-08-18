#!/usr/bin/env python3
"""
调试checkpoint的脚本
"""
import torch
import sys
import os

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model'))

from attn_model.attn_model import AttnModel
from sac.sac_model import AttentionSACWithBuffer

def debug_checkpoint(checkpoint_path):
    print(f"🔍 调试checkpoint: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\n📋 Checkpoint内容:")
    for key in checkpoint.keys():
        if '_state_dict' in key:
            print(f"  {key}:")
            state_dict = checkpoint[key]
            for param_name in sorted(state_dict.keys()):
                print(f"    {param_name}: {state_dict[param_name].shape}")
    
    print(f"\n🤖 当前模型结构:")
    
    # 创建当前模型
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, 4, 
                                buffer_capacity=10000, batch_size=64,
                                lr=1e-5, env_type='reacher2d')
    
    print(f"  Current Critic1 parameters:")
    for name, param in sac.critic1.named_parameters():
        print(f"    {name}: {param.shape}")
    
    print(f"\n🔍 比较 Critic1:")
    current_keys = set(sac.critic1.state_dict().keys())
    checkpoint_keys = set(checkpoint['critic1_state_dict'].keys())
    
    missing_in_current = checkpoint_keys - current_keys
    extra_in_current = current_keys - checkpoint_keys
    
    if missing_in_current:
        print(f"  ❌ 当前模型缺少的键: {missing_in_current}")
    if extra_in_current:
        print(f"  ➕ 当前模型新增的键: {extra_in_current}")
    if not missing_in_current and not extra_in_current:
        print(f"  ✅ 结构完全匹配!")

if __name__ == "__main__":
    checkpoint_path = "/home/xli149/Documents/repos/RoboGrammar/trained_models/reacher2d/enhanced_test/08-17-2025-19-31-52/best_models/checkpoint_step_80000.pth"
    debug_checkpoint(checkpoint_path)