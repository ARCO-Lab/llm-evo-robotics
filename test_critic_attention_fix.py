#!/usr/bin/env python3
"""
简单测试脚本：验证修复后的Critic attention网络是否被正确使用
"""

import sys
import os
import torch
import torch.nn as nn

# 添加路径
sys.path.append('.')
sys.path.append('./sac')

def test_critic_attention_usage():
    """测试Critic是否正确使用了主要attention网络"""
    print("🔍 测试Critic attention网络使用情况...")
    
    try:
        from sac.universal_ppo_model import UniversalAttnModel, UniversalPPOCritic
        
        # 创建模型
        attn_model = UniversalAttnModel(128, 130, 130, 4)
        critic = UniversalPPOCritic(attn_model, device='cpu')
        
        # 创建测试数据
        batch_size = 2
        num_joints = 3
        joint_q = torch.randn(batch_size, num_joints, 130)
        vertex_k = torch.randn(batch_size, num_joints, 128)
        vertex_v = torch.randn(batch_size, num_joints, 130)
        
        print(f"📊 测试数据形状:")
        print(f"   joint_q: {joint_q.shape}")
        print(f"   vertex_k: {vertex_k.shape}")
        print(f"   vertex_v: {vertex_v.shape}")
        
        # 前向传播
        print("\n🚀 执行前向传播...")
        with torch.no_grad():
            output = critic(joint_q, vertex_k, vertex_v)
            print(f"✅ Critic输出: {output.shape} = {output}")
        
        # 检查梯度计算
        print("\n🔍 测试梯度计算...")
        output = critic(joint_q, vertex_k, vertex_v)
        loss = output.sum()
        loss.backward()
        
        # 检查主要attention网络的梯度
        main_attn_grad_count = 0
        main_attn_grad_norm = 0.0
        for name, param in critic.attn_model.named_parameters():
            if param.grad is not None:
                main_attn_grad_count += 1
                main_attn_grad_norm += param.grad.data.norm(2).item() ** 2
        
        # 检查value attention的梯度
        value_attn_grad_count = 0
        value_attn_grad_norm = 0.0
        for name, param in critic.value_attention.named_parameters():
            if param.grad is not None:
                value_attn_grad_count += 1
                value_attn_grad_norm += param.grad.data.norm(2).item() ** 2
        
        main_attn_grad_norm = main_attn_grad_norm ** 0.5 if main_attn_grad_count > 0 else 0.0
        value_attn_grad_norm = value_attn_grad_norm ** 0.5 if value_attn_grad_count > 0 else 0.0
        
        print(f"\n📊 梯度检查结果:")
        print(f"   主要attention网络:")
        print(f"     - 有梯度的参数数量: {main_attn_grad_count}")
        print(f"     - 梯度范数: {main_attn_grad_norm:.6f}")
        print(f"   内部value attention:")
        print(f"     - 有梯度的参数数量: {value_attn_grad_count}")
        print(f"     - 梯度范数: {value_attn_grad_norm:.6f}")
        
        # 判断修复是否成功
        if main_attn_grad_count > 0 and main_attn_grad_norm > 0:
            print(f"\n🎉 修复成功！主要attention网络现在有梯度了！")
            print(f"   主要attention梯度范数: {main_attn_grad_norm:.6f}")
            return True
        else:
            print(f"\n❌ 修复失败！主要attention网络仍然没有梯度。")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_critic_attention_usage()
    if success:
        print("\n✅ 测试通过：Critic attention网络修复成功！")
    else:
        print("\n❌ 测试失败：需要进一步调试。")
