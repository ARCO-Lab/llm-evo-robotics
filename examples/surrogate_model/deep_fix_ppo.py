#!/usr/bin/env python3
"""
深度修复PPO - 解决持续的Critic Loss过高问题
"""

import torch
import sys
import os

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)

from sac.universal_ppo_model import UniversalPPOWithBuffer

def deep_fix_ppo_model():
    """深度修复PPO模型 - 解决Critic Loss过高问题"""
    print("🔧 开始深度修复PPO模型...")
    
    # 使用更极端的保守参数
    ppo = UniversalPPOWithBuffer(
        buffer_size=2048,
        batch_size=32,      # 减小批次大小
        lr=1e-5,           # 更低的基础学习率
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.1,   # 更小的clip范围
        entropy_coef=0.001, # 更低的熵系数
        value_coef=1.0,     # 提高value loss权重
        device='cpu',
        env_type='reacher2d'
    )
    
    print("✅ 创建超保守PPO模型完成")
    
    # 强制重置所有关键参数
    with torch.no_grad():
        # 1. 重置Actor的log_std到极低值
        ppo.actor.log_std_base.data.fill_(-2.5)  # std ≈ 0.08
        print(f"🔧 强制重置log_std_base到-2.5 (std ≈ 0.08)")
        
        # 2. 重新初始化Critic网络权重
        def reinit_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # 小增益
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
        
        ppo.critic.apply(reinit_weights)
        print("🔧 重新初始化Critic网络权重 (小增益)")
        
        # 3. 设置极低的学习率
        for param_group in ppo.actor_optimizer.param_groups:
            param_group['lr'] = 5e-6  # 极低Actor学习率
        for param_group in ppo.critic_optimizer.param_groups:
            param_group['lr'] = 1e-5  # 极低Critic学习率
        print("🔧 设置极低学习率: Actor=5e-6, Critic=1e-5")
        
        # 4. 调整优化器的动量参数
        ppo.actor_optimizer.param_groups[0]['betas'] = (0.9, 0.999)
        ppo.critic_optimizer.param_groups[0]['betas'] = (0.9, 0.999)
        print("🔧 调整Adam优化器参数")
    
    # 保存深度修复后的模型
    save_path = "deep_fixed_ppo_model.pth"
    ppo.save_model(save_path)
    print(f"💾 深度修复模型已保存: {save_path}")
    
    # 验证修复结果
    with torch.no_grad():
        current_log_std = ppo.actor.log_std_base.item()
        current_std = torch.exp(torch.clamp(torch.tensor(current_log_std), -2.3, -0.5)).item()
        actor_lr = ppo.actor_optimizer.param_groups[0]['lr']
        critic_lr = ppo.critic_optimizer.param_groups[0]['lr']
        
        print(f"\n🔍 深度修复验证:")
        print(f"   log_std_base: {current_log_std:.4f}")
        print(f"   std: {current_std:.4f}")
        print(f"   Actor学习率: {actor_lr:.2e}")
        print(f"   Critic学习率: {critic_lr:.2e}")
        print(f"   批次大小: 32")
        print(f"   Clip范围: 0.1")
        print(f"   熵系数: 0.001")
        print(f"   Value系数: 1.0")
        
        print(f"\n🎯 深度修复目标:")
        print(f"   ✅ Entropy: 预期 0.1-0.8 (极低探索)")
        print(f"   ✅ Critic Loss: 预期 < 5.0 (稳定)")
        print(f"   ✅ Actor Loss: 预期 0.1-0.5 (稳定)")
        print(f"   ✅ 整体稳定性: 无异常波动")
    
    print(f"\n🚀 使用建议:")
    print(f"   python enhanced_train.py --env-name reacher2d \\")
    print(f"       --lr 1e-5 --entropy-coef 0.001 --batch-size 32 \\")
    print(f"       --resume-checkpoint {save_path}")
    
    return save_path

if __name__ == "__main__":
    deep_fix_ppo_model()







