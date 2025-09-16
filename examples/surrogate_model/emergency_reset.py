#!/usr/bin/env python3
"""
紧急修复脚本 - 重置PPO模型参数
当entropy爆炸时使用此脚本重置模型
"""

import torch
import sys
import os

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)

from sac.universal_ppo_model import UniversalPPOWithBuffer

def emergency_reset_ppo_model(model_path=None):
    """紧急重置PPO模型参数"""
    print("🚨 开始紧急修复PPO模型...")
    
    # 创建新的PPO模型
    ppo = UniversalPPOWithBuffer(
        buffer_size=2048,
        batch_size=64,
        lr=5e-5,  # 更低的学习率
        device='cpu',
        env_type='reacher2d'
    )
    
    print(f"✅ 创建新PPO模型完成")
    
    # 如果提供了模型路径，尝试加载并修复
    if model_path and os.path.exists(model_path):
        try:
            print(f"🔄 尝试加载并修复模型: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 只加载网络权重，跳过优化器状态
            if 'actor_state_dict' in checkpoint:
                ppo.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                print("✅ Actor权重加载成功")
            
            if 'critic_state_dict' in checkpoint:
                ppo.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
                print("✅ Critic权重加载成功")
                
        except Exception as e:
            print(f"⚠️ 加载模型失败，使用全新模型: {e}")
    
    # 强制重置关键参数
    with torch.no_grad():
        # 重置标准差参数
        ppo.actor.log_std_base.data.fill_(-1.8)  # 对应std ≈ 0.17
        print(f"🔧 重置log_std_base到-1.8 (std ≈ 0.17)")
        
        # 重置学习率到安全值
        for param_group in ppo.actor_optimizer.param_groups:
            param_group['lr'] = 3e-5
        for param_group in ppo.critic_optimizer.param_groups:
            param_group['lr'] = 2e-5
        print(f"🔧 重置学习率: Actor=3e-5, Critic=2e-5")
    
    # 保存修复后的模型
    save_path = "emergency_fixed_ppo_model.pth"
    ppo.save_model(save_path)
    print(f"💾 修复后模型已保存: {save_path}")
    
    # 验证修复结果
    with torch.no_grad():
        current_log_std = ppo.actor.log_std_base.item()
        current_std = torch.exp(torch.clamp(torch.tensor(current_log_std), -2.3, -0.5)).item()
        actor_lr = ppo.actor_optimizer.param_groups[0]['lr']
        critic_lr = ppo.critic_optimizer.param_groups[0]['lr']
        
        print(f"\n🔍 修复验证:")
        print(f"   log_std_base: {current_log_std:.4f}")
        print(f"   std: {current_std:.4f}")
        print(f"   Actor学习率: {actor_lr:.2e}")
        print(f"   Critic学习率: {critic_lr:.2e}")
        print(f"   熵系数: {ppo.entropy_coef:.4f}")
        
        if current_std < 0.3 and actor_lr < 5e-5:
            print(f"   ✅ 修复成功！参数已恢复到安全范围")
        else:
            print(f"   ⚠️ 可能需要进一步调整")
    
    print(f"\n🎯 使用建议:")
    print(f"   1. 停止当前训练")
    print(f"   2. 使用修复后的模型: {save_path}")
    print(f"   3. 重新开始训练，预期entropy在0.5-1.5范围内")
    print(f"   4. 如果问题再次出现，进一步降低学习率")
    
    return save_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='紧急修复PPO模型')
    parser.add_argument('--model-path', type=str, help='要修复的模型路径（可选）')
    args = parser.parse_args()
    
    # 查找最新的问题模型
    if not args.model_path:
        possible_paths = [
            "./trained_models/reacher2d/enhanced_test/*/best_models/latest_best_model.pth",
            "./trained_models/reacher2d/enhanced_test/*/best_models/final_*_model_*.pth"
        ]
        
        import glob
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                args.model_path = sorted(matches)[-1]  # 最新的
                print(f"🔍 自动找到模型: {args.model_path}")
                break
    
    emergency_reset_ppo_model(args.model_path)
