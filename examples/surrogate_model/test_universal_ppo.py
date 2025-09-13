#!/usr/bin/env python3
"""
通用PPO模型测试脚本
验证模型是否能正确处理不同关节数的机器人
"""

import torch
import numpy as np
import os
import sys

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "examples/2d_reacher/envs"))
sys.path.append(os.path.join(base_dir, "examples/2d_reacher/utils"))

from reacher2d_env import Reacher2DEnv
from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
from sac.universal_ppo_model import UniversalPPOWithBuffer


def test_universal_ppo():
    """测试通用PPO模型的基本功能"""
    print("🧪 测试通用PPO模型")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   使用设备: {device}")
    
    # 创建通用PPO模型
    ppo = UniversalPPOWithBuffer(device=device)
    
    # 测试不同关节数的配置
    test_configs = [
        {'num_joints': 3, 'link_lengths': [60, 60, 60]},
        {'num_joints': 4, 'link_lengths': [50, 50, 50, 50]},
        {'num_joints': 5, 'link_lengths': [40, 40, 40, 40, 40]},
    ]
    
    print(f"\n📋 测试配置: {[cfg['num_joints'] for cfg in test_configs]}关节")
    
    for i, config in enumerate(test_configs):
        num_joints = config['num_joints']
        link_lengths = config['link_lengths']
        
        print(f"\n🤖 测试 {num_joints} 关节配置...")
        
        try:
            # 创建环境
            env = Reacher2DEnv(
                num_links=num_joints,
                link_lengths=link_lengths,
                render_mode=None,
                config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
            )
            
            # 创建GNN编码器
            encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
            gnn_embed = encoder.get_gnn_embeds(num_links=num_joints, link_lengths=link_lengths)
            
            print(f"   ✅ 环境和编码器创建成功")
            print(f"   GNN嵌入形状: {gnn_embed.shape}")
            
            # 重置环境
            obs, info = env.reset()
            print(f"   观测形状: {obs.shape}")
            
            # 测试动作生成
            action, log_prob, value = ppo.get_action(
                torch.tensor(obs, dtype=torch.float32).to(device),
                gnn_embed.to(device),
                num_joints,
                deterministic=False
            )
            
            print(f"   ✅ 动作生成成功")
            print(f"   动作形状: {action.shape}")
            print(f"   动作范围: [{action.min():.2f}, {action.max():.2f}]")
            print(f"   价值估计: {value.item():.2f}")
            
            # 测试环境交互
            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
            print(f"   ✅ 环境交互成功")
            print(f"   奖励: {reward:.4f}")
            
            # 测试经验存储
            ppo.store_experience(
                torch.tensor(obs, dtype=torch.float32),
                gnn_embed,
                action,
                reward,
                done or truncated,
                log_prob,
                value,
                num_joints
            )
            print(f"   ✅ 经验存储成功")
            
            # 进行几步交互以积累经验
            for step in range(10):
                action, log_prob, value = ppo.get_action(
                    torch.tensor(next_obs, dtype=torch.float32).to(device),
                    gnn_embed.to(device),
                    num_joints
                )
                
                obs = next_obs
                next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
                
                ppo.store_experience(
                    torch.tensor(obs, dtype=torch.float32),
                    gnn_embed,
                    action,
                    reward,
                    done or truncated,
                    log_prob,
                    value,
                    num_joints
                )
                
                if done or truncated:
                    break
            
            print(f"   ✅ {num_joints}关节测试完成")
            
        except Exception as e:
            print(f"   ❌ {num_joints}关节测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 测试模型更新
    print(f"\n🔄 测试模型更新...")
    try:
        # 需要足够的经验才能更新
        if len(ppo.buffer.experiences) >= 10:
            metrics = ppo.update(ppo_epochs=2)
            if metrics:
                print(f"   ✅ 模型更新成功")
                print(f"   Actor Loss: {metrics['actor_loss']:.4f}")
                print(f"   Critic Loss: {metrics['critic_loss']:.4f}")
                print(f"   处理批次: {metrics['batches_processed']}")
            else:
                print(f"   ⚠️ 模型更新返回None（经验不足）")
        else:
            print(f"   ⚠️ 经验数量不足，跳过更新测试")
    except Exception as e:
        print(f"   ❌ 模型更新失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试模型保存和加载
    print(f"\n💾 测试模型保存和加载...")
    try:
        save_path = "test_universal_ppo.pth"
        ppo.save_model(save_path)
        print(f"   ✅ 模型保存成功: {save_path}")
        
        # 创建新模型并加载
        new_ppo = UniversalPPOWithBuffer(device=device)
        new_ppo.load_model(save_path)
        print(f"   ✅ 模型加载成功")
        
        # 清理测试文件
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"   🧹 清理测试文件")
            
    except Exception as e:
        print(f"   ❌ 模型保存/加载失败: {e}")
        return False
    
    print(f"\n🎉 通用PPO模型测试全部通过！")
    return True


def test_model_compatibility():
    """测试模型与不同关节数的兼容性"""
    print(f"\n🔄 测试模型兼容性...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppo = UniversalPPOWithBuffer(device=device)
    
    # 测试连续切换不同关节数
    joint_numbers = [3, 5, 4, 6, 3, 4]  # 随机顺序
    
    for num_joints in joint_numbers:
        try:
            # 创建随机观测和GNN嵌入
            obs_dim = 2 * num_joints + 2  # Reacher2D观测维度
            obs = torch.randn(obs_dim).to(device)
            gnn_embed = torch.randn(1, 20, 128).to(device)  # 假设最大20个节点
            
            # 测试动作生成
            action, log_prob, value = ppo.get_action(obs, gnn_embed, num_joints)
            
            # 验证输出维度
            assert action.shape[0] == num_joints, f"动作维度错误: {action.shape} vs {num_joints}"
            
            print(f"   ✅ {num_joints}关节 - 动作形状: {action.shape}")
            
        except Exception as e:
            print(f"   ❌ {num_joints}关节测试失败: {e}")
            return False
    
    print(f"   🎉 模型兼容性测试通过！")
    return True


if __name__ == "__main__":
    print("🚀 开始通用PPO模型测试")
    
    success = True
    
    # 基本功能测试
    if not test_universal_ppo():
        success = False
    
    # 兼容性测试
    if not test_model_compatibility():
        success = False
    
    if success:
        print(f"\n🎊 所有测试通过！通用PPO模型可以正常工作")
        print(f"💡 现在可以使用 python universal_train.py 开始训练")
    else:
        print(f"\n❌ 部分测试失败，请检查模型实现")
        sys.exit(1)
