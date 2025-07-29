import torch
import torch.nn as nn
import numpy as np
import os
import sys
# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_dataset"))
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_model"))

import torch


from attn_model import AttnModel
from sac_model import AttentionSACWithBuffer

def test_buffer_update_functionality():
    """完整测试SAC buffer更新功能"""
    
    print("="*60)
    print("Testing SAC Buffer Update Functionality")
    print("="*60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建模型
    action_dim = 12
    attn_model = AttnModel(128, 128, 130, 4)
    sac = AttentionSACWithBuffer(
        attn_model, 
        action_dim, 
        buffer_capacity=1000,  # 较小的buffer便于测试
        batch_size=32,         # 较小的batch size
        device=device
    )
    
    print(f"✓ SAC模型创建成功")
    print(f"  - Buffer capacity: {sac.memory.capacity}")
    print(f"  - Batch size: {sac.batch_size}")
    print(f"  - Action dim: {action_dim}")
    
    # 测试1: Buffer存储功能
    print("\n" + "-"*40)
    print("Test 1: Buffer Storage")
    print("-"*40)
    
    num_experiences = 100
    for i in range(num_experiences):
        obs = torch.randn(40)
        gnn_embeds = torch.randn(12, 128)
        action = torch.randn(12)
        reward = np.random.normal(0, 1)  # 随机奖励
        next_obs = torch.randn(40)
        next_gnn_embeds = torch.randn(12, 128)
        done = (i % 50 == 49)  # 每50步结束一个episode
        
        sac.store_experience(obs, gnn_embeds, action, reward, next_obs, next_gnn_embeds, done)
        
        if (i + 1) % 20 == 0:
            print(f"  Stored {i+1} experiences, buffer size: {len(sac.memory)}")
    
    print(f"✓ Buffer存储测试完成，最终buffer大小: {len(sac.memory)}")
    
    # 测试2: 网络参数更新前后对比
    print("\n" + "-"*40)
    print("Test 2: Parameter Updates")
    print("-"*40)
    
    # 记录更新前的参数
    def get_model_params(model):
        return {name: param.clone().detach() for name, param in model.named_parameters()}
    
    actor_params_before = get_model_params(sac.actor)
    critic1_params_before = get_model_params(sac.critic1)
    critic2_params_before = get_model_params(sac.critic2)
    
    print("✓ 记录更新前参数")
    
    # 执行多次更新
    update_count = 0
    loss_history = {'critic': [], 'actor': [], 'alpha': []}
    
    if sac.memory.can_sample(sac.batch_size):
        for i in range(10):  # 执行10次更新
            metrics = sac.update()
            if metrics:
                update_count += 1
                loss_history['critic'].append(metrics['critic_loss'])
                loss_history['actor'].append(metrics['actor_loss'])
                loss_history['alpha'].append(metrics['alpha_loss'])
                
                if i % 2 == 0:
                    print(f"  Update {i+1}: Critic Loss: {metrics['critic_loss']:.4f}, "
                          f"Actor Loss: {metrics['actor_loss']:.4f}, "
                          f"Alpha: {metrics['alpha']:.4f}")
    
    print(f"✓ 完成 {update_count} 次参数更新")
    
    # 测试3: 验证参数确实发生了变化
    print("\n" + "-"*40)
    print("Test 3: Parameter Change Verification")
    print("-"*40)
    
    def compare_params(params_before, model, model_name):
        total_params = 0
        changed_params = 0
        max_change = 0
        
        for name, param in model.named_parameters():
            if name in params_before:
                diff = torch.abs(param - params_before[name])
                max_diff = diff.max().item()
                total_params += 1
                
                if max_diff > 1e-8:  # 参数变化阈值
                    changed_params += 1
                    max_change = max(max_change, max_diff)
        
        change_ratio = changed_params / total_params if total_params > 0 else 0
        print(f"  {model_name}:")
        print(f"    - Total parameters: {total_params}")
        print(f"    - Changed parameters: {changed_params}")
        print(f"    - Change ratio: {change_ratio:.2%}")
        print(f"    - Max parameter change: {max_change:.6f}")
        
        return change_ratio > 0.5  # 至少50%的参数发生变化
    
    actor_changed = compare_params(actor_params_before, sac.actor, "Actor")
    critic1_changed = compare_params(critic1_params_before, sac.critic1, "Critic1")
    critic2_changed = compare_params(critic2_params_before, sac.critic2, "Critic2")
    
    if actor_changed and critic1_changed and critic2_changed:
        print("✓ 所有网络参数都发生了显著变化")
    else:
        print("✗ 部分网络参数未发生预期变化")
    
    # 测试4: 动作输出一致性测试
    print("\n" + "-"*40)
    print("Test 4: Action Output Consistency")
    print("-"*40)
    
    test_obs = torch.randn(40)
    test_gnn_embeds = torch.randn(12, 128)
    
    # 测试确定性动作
    action1 = sac.get_action(test_obs, test_gnn_embeds, deterministic=True)
    action2 = sac.get_action(test_obs, test_gnn_embeds, deterministic=True)
    
    deterministic_diff = torch.abs(action1 - action2).max().item()
    print(f"  确定性动作差异: {deterministic_diff:.8f}")
    
    if deterministic_diff < 1e-6:
        print("✓ 确定性动作输出一致")
    else:
        print("✗ 确定性动作输出不一致")
    
    # 测试随机动作
    stochastic_actions = []
    for _ in range(5):
        action = sac.get_action(test_obs, test_gnn_embeds, deterministic=False)
        stochastic_actions.append(action)
    
    # 计算随机动作的方差
    stochastic_actions = torch.stack(stochastic_actions)
    action_variance = torch.var(stochastic_actions, dim=0).mean().item()
    print(f"  随机动作方差: {action_variance:.6f}")
    
    if action_variance > 1e-4:
        print("✓ 随机动作具有适当的随机性")
    else:
        print("✗ 随机动作可能过于确定")
    
    # 测试5: Buffer采样一致性
    print("\n" + "-"*40)
    print("Test 5: Buffer Sampling Consistency")
    print("-"*40)
    
    if sac.memory.can_sample(sac.batch_size):
        # 采样两次，检查是否不同
        batch1 = sac.memory.sample(16)  # 较小的batch便于比较
        batch2 = sac.memory.sample(16)
        
        # 比较joint_q的第一个样本
        diff = torch.abs(batch1[0][0] - batch2[0][0]).sum().item()
        print(f"  两次采样的差异: {diff:.6f}")
        
        if diff > 1e-6:
            print("✓ Buffer采样具有随机性")
        else:
            print("! Buffer采样可能过于相似（如果buffer很小这是正常的）")
    
    # 测试6: 损失趋势分析
    print("\n" + "-"*40)
    print("Test 6: Loss Trend Analysis")
    print("-"*40)
    
    if len(loss_history['critic']) > 5:
        # 简单的趋势分析
        critic_trend = np.mean(loss_history['critic'][-3:]) - np.mean(loss_history['critic'][:3])
        actor_trend = np.mean(loss_history['actor'][-3:]) - np.mean(loss_history['actor'][:3])
        
        print(f"  Critic loss趋势: {critic_trend:+.6f}")
        print(f"  Actor loss趋势: {actor_trend:+.6f}")
        print(f"  最终Alpha值: {loss_history['alpha'][-1]:.4f}")
        
        # 可视化损失（如果需要）
        # plot_loss_curves(loss_history)
    
    # 测试7: Target网络更新测试
    print("\n" + "-"*40)
    print("Test 7: Target Network Update")
    print("-"*40)
    
    # 记录target网络参数
    target_params_before = get_model_params(sac.target_critic1)
    
    # 执行一次更新
    if sac.memory.can_sample(sac.batch_size):
        sac.update()
    
    # 检查target网络参数变化
    target_changed = compare_params(target_params_before, sac.target_critic1, "Target Critic1")
    
    if target_changed:
        print("✓ Target网络参数正确更新")
    else:
        print("✗ Target网络参数未正确更新")
    
    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✓ Buffer容量: {len(sac.memory)}/{sac.memory.capacity}")
    print(f"✓ 执行更新次数: {update_count}")
    print(f"✓ 参数更新状态: Actor({actor_changed}), Critic1({critic1_changed}), Critic2({critic2_changed})")
    print(f"✓ 最终losses: Critic:{loss_history['critic'][-1]:.4f}, Actor:{loss_history['actor'][-1]:.4f}")
    
    return True

def plot_loss_curves(loss_history):
    """可视化损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(loss_history['critic'])
    axes[0].set_title('Critic Loss')
    axes[0].set_xlabel('Update Step')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(loss_history['actor'])
    axes[1].set_title('Actor Loss')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    
    axes[2].plot(loss_history['alpha'])
    axes[2].set_title('Alpha Loss')
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('sac_training_curves.png')
    plt.show()

def test_memory_efficiency():
    """测试内存使用效率"""
    print("\n" + "="*60)
    print("Memory Efficiency Test")
    print("="*60)
    
    import psutil
    import gc
    
    # 获取初始内存使用
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建SAC
    action_dim = 12
    attn_model = AttnModel(128, 128, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, buffer_capacity=10000)
    
    model_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
    print(f"模型创建后内存使用: {model_memory:.2f} MB")
    
    # 填充buffer
    for i in range(5000):
        obs = torch.randn(40)
        gnn_embeds = torch.randn(12, 128)
        action = torch.randn(12)
        reward = np.random.normal(0, 1)
        next_obs = torch.randn(40)
        next_gnn_embeds = torch.randn(12, 128)
        done = (i % 100 == 99)
        
        sac.store_experience(obs, gnn_embeds, action, reward, next_obs, next_gnn_embeds, done)
    
    buffer_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
    print(f"Buffer填充后内存使用: {buffer_memory:.2f} MB")
    
    # 执行训练
    for i in range(100):
        if sac.memory.can_sample(sac.batch_size):
            sac.update()
    
    training_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
    print(f"训练后内存使用: {training_memory:.2f} MB")
    
    # 清理
    del sac
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024 - initial_memory
    print(f"清理后内存使用: {final_memory:.2f} MB")

if __name__ == "__main__":
    try:
        # 运行主要功能测试
        success = test_buffer_update_functionality()
        
        # 运行内存效率测试
        test_memory_efficiency()
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()