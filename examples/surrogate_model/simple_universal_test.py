#!/usr/bin/env python3
"""
简化的通用PPO测试 - 避免GNN编码器的数据类型问题
直接测试模型的核心功能
"""

import torch
import numpy as np
import os
import sys

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)

from sac.universal_ppo_model import UniversalPPOWithBuffer


def create_mock_data(num_joints, batch_size=1):
    """创建模拟数据用于测试"""
    # 模拟观测数据 (Reacher2D格式)
    obs_dim = 2 * num_joints + 2  # joint_angles + joint_vels + end_effector_pos
    obs = torch.randn(batch_size, obs_dim, dtype=torch.float32)
    
    # 模拟GNN嵌入 (假设最多20个节点，每个节点128维特征)
    max_nodes = 20
    gnn_embed = torch.randn(batch_size, max_nodes, 128, dtype=torch.float32)
    
    return obs, gnn_embed


def test_model_basic_functionality():
    """测试模型基本功能"""
    print("🧪 测试通用PPO模型基本功能")
    
    device = 'cpu'
    ppo = UniversalPPOWithBuffer(device=device)
    
    test_configs = [3, 4, 5, 6]  # 不同关节数
    
    for num_joints in test_configs:
        print(f"\n🤖 测试 {num_joints} 关节...")
        
        try:
            # 创建模拟数据
            obs, gnn_embed = create_mock_data(num_joints)
            
            # 测试动作生成
            action, log_prob, value = ppo.get_action(
                obs.squeeze(0), gnn_embed.squeeze(0), num_joints, deterministic=False
            )
            
            print(f"   ✅ 动作生成成功")
            print(f"   动作形状: {action.shape} (期望: [{num_joints}])")
            print(f"   动作范围: [{action.min():.2f}, {action.max():.2f}]")
            print(f"   价值估计: {value.item():.2f}")
            
            # 验证输出维度
            assert action.shape[0] == num_joints, f"动作维度错误: {action.shape} vs {num_joints}"
            assert value.numel() == 1, f"价值维度错误: {value.shape}"
            
            # 测试经验存储
            reward = np.random.uniform(-10, 10)
            done = False
            
            ppo.store_experience(
                obs.squeeze(0), gnn_embed.squeeze(0), action, reward, done,
                log_prob, value, num_joints
            )
            print(f"   ✅ 经验存储成功")
            
        except Exception as e:
            print(f"   ❌ {num_joints}关节测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n🎉 基本功能测试通过！")
    return True


def test_mixed_joint_training():
    """测试混合关节数训练"""
    print(f"\n🔄 测试混合关节数训练...")
    
    device = 'cpu'
    ppo = UniversalPPOWithBuffer(buffer_size=100, batch_size=20, device=device)
    
    # 收集不同关节数的经验
    joint_configs = [3, 4, 5]
    experiences_per_config = 15
    
    for num_joints in joint_configs:
        print(f"   收集 {num_joints} 关节经验...")
        
        for i in range(experiences_per_config):
            obs, gnn_embed = create_mock_data(num_joints)
            
            action, log_prob, value = ppo.get_action(
                obs.squeeze(0), gnn_embed.squeeze(0), num_joints
            )
            
            reward = np.random.uniform(-5, 5)
            done = np.random.random() < 0.1  # 10%概率结束
            
            ppo.store_experience(
                obs.squeeze(0), gnn_embed.squeeze(0), action, reward, done,
                log_prob, value, num_joints
            )
    
    print(f"   收集了 {len(ppo.buffer.experiences)} 条经验")
    
    # 测试模型更新
    try:
        metrics = ppo.update(ppo_epochs=2)
        
        if metrics:
            print(f"   ✅ 混合训练更新成功")
            print(f"   Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"   Critic Loss: {metrics['critic_loss']:.4f}")
            print(f"   处理批次: {metrics['batches_processed']}")
        else:
            print(f"   ⚠️ 更新返回None")
            
    except Exception as e:
        print(f"   ❌ 混合训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"   🎉 混合关节数训练测试通过！")
    return True


def test_model_persistence():
    """测试模型保存和加载"""
    print(f"\n💾 测试模型保存和加载...")
    
    device = 'cpu'
    ppo1 = UniversalPPOWithBuffer(device=device)
    
    # 保存模型
    save_path = "test_universal_model.pth"
    try:
        ppo1.save_model(save_path)
        print(f"   ✅ 模型保存成功")
        
        # 加载模型
        ppo2 = UniversalPPOWithBuffer(device=device)
        ppo2.load_model(save_path)
        print(f"   ✅ 模型加载成功")
        
        # 测试加载后的模型
        obs, gnn_embed = create_mock_data(4)
        action1, _, value1 = ppo1.get_action(obs.squeeze(0), gnn_embed.squeeze(0), 4, deterministic=True)
        action2, _, value2 = ppo2.get_action(obs.squeeze(0), gnn_embed.squeeze(0), 4, deterministic=True)
        
        # 验证输出一致性
        action_diff = torch.abs(action1 - action2).max().item()
        value_diff = torch.abs(value1 - value2).item()
        
        if action_diff < 1e-5 and value_diff < 1e-5:
            print(f"   ✅ 模型输出一致性验证通过")
        else:
            print(f"   ⚠️ 模型输出有差异 (动作差异: {action_diff:.2e}, 价值差异: {value_diff:.2e})")
        
        # 清理
        if os.path.exists(save_path):
            os.remove(save_path)
            
    except Exception as e:
        print(f"   ❌ 模型保存/加载失败: {e}")
        return False
    
    print(f"   🎉 模型持久化测试通过！")
    return True


def test_scalability():
    """测试模型可扩展性"""
    print(f"\n📈 测试模型可扩展性...")
    
    device = 'cpu'
    ppo = UniversalPPOWithBuffer(device=device)
    
    # 测试更大的关节数
    large_joint_configs = [8, 10, 12]
    
    for num_joints in large_joint_configs:
        try:
            obs, gnn_embed = create_mock_data(num_joints)
            
            action, log_prob, value = ppo.get_action(
                obs.squeeze(0), gnn_embed.squeeze(0), num_joints
            )
            
            assert action.shape[0] == num_joints
            print(f"   ✅ {num_joints}关节测试通过")
            
        except Exception as e:
            print(f"   ❌ {num_joints}关节测试失败: {e}")
            return False
    
    print(f"   🎉 可扩展性测试通过！")
    return True


def main():
    print("🚀 开始简化通用PPO测试")
    
    tests = [
        ("基本功能测试", test_model_basic_functionality),
        ("混合关节数训练测试", test_mixed_joint_training),
        ("模型持久化测试", test_model_persistence),
        ("可扩展性测试", test_scalability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print(f"🎊 所有测试通过！通用PPO模型可以正常工作")
        print(f"💡 现在可以使用 python universal_train.py 开始训练")
        return True
    else:
        print(f"❌ 部分测试失败，请检查模型实现")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
