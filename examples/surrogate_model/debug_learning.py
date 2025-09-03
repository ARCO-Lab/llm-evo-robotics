#!/usr/bin/env python3
"""
SAC学习能力诊断脚本
一步步验证模型是否能学到正确策略
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/sac"))
sys.path.insert(0, os.path.join(base_dir, "examples/surrogate_model/attn_model"))
sys.path.insert(0, os.path.join(base_dir, "examples/2d_reacher/envs"))

from sac_model import AttentionSACWithBuffer
from attn_model import AttnModel
from reacher2d_env import Reacher2DEnv

class LearningDiagnostics:
    def __init__(self):
        print("🔬 初始化学习诊断系统...")
        
        # 创建简单的测试环境
        self.env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 60, 60],
            render_mode=None,  # 不渲染，提高速度
            config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        # 创建SAC模型
        self.attn_model = AttnModel(128, 128, 130, 4)
        self.sac = AttentionSACWithBuffer(
            self.attn_model, 
            action_dim=3,  # 3个关节
            lr=1e-4,  # 提高学习率
            batch_size=64,  # 小批量，快速测试
            env_type='reacher2d'
        )
        
        print("✅ 初始化完成")
    
    def step1_basic_functionality_test(self):
        """步骤1: 测试基础功能是否正常"""
        print("\n" + "="*60)
        print("🧪 步骤1: 基础功能测试")
        print("="*60)
        
        try:
            # 1.1 环境重置测试
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            print(f"✅ 环境重置成功: obs shape = {obs_tensor.shape}")
            
            # 1.2 GNN编码测试
            gnn_embeds = torch.randn(3, 128)  # 3个关节的GNN嵌入
            print(f"✅ GNN嵌入创建成功: {gnn_embeds.shape}")
            
            # 1.3 动作生成测试
            action = self.sac.get_action(obs_tensor, gnn_embeds, num_joints=3)
            print(f"✅ 动作生成成功: {action.shape}, 范围: [{action.min():.2f}, {action.max():.2f}]")
            
            # 1.4 环境交互测试
            next_obs, reward, done, info = self.env.step(action.numpy())
            print(f"✅ 环境交互成功: reward = {reward:.3f}, done = {done}")
            
            return True
            
        except Exception as e:
            print(f"❌ 基础功能测试失败: {e}")
            return False
    
    def step2_reward_signal_test(self):
        """步骤2: 验证奖励信号是否合理"""
        print("\n" + "="*60)
        print("🎯 步骤2: 奖励信号测试")
        print("="*60)
        
        rewards = []
        distances = []
        
        # 收集不同状态下的奖励
        for i in range(20):
            obs = self.env.reset()
            
            # 随机动作，观察奖励变化
            for step in range(50):
                action = np.random.uniform(-50, 50, 3)  # 中等强度动作
                next_obs, reward, done, info = self.env.step(action)
                
                # 获取距离信息
                end_pos = self.env._get_end_effector_position()
                distance = np.linalg.norm(np.array(end_pos) - self.env.goal_pos)
                
                rewards.append(reward)
                distances.append(distance)
                
                if done:
                    break
        
        # 分析奖励信号
        rewards = np.array(rewards)
        distances = np.array(distances)
        
        print(f"📊 奖励统计:")
        print(f"   范围: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"   平均: {rewards.mean():.3f}, 标准差: {rewards.std():.3f}")
        print(f"   距离范围: [{distances.min():.1f}, {distances.max():.1f}]")
        
        # 检查奖励与距离的相关性
        correlation = np.corrcoef(rewards, distances)[0, 1]
        print(f"   奖励-距离相关性: {correlation:.3f} (应该是负值)")
        
        if correlation < -0.1:
            print("✅ 奖励信号合理：距离越近，奖励越高")
            return True
        else:
            print("❌ 奖励信号可能有问题：缺乏明确的距离导向")
            return False
    
    def step3_action_effect_test(self):
        """步骤3: 测试动作是否对环境产生预期效果"""
        print("\n" + "="*60)
        print("🎮 步骤3: 动作效果测试")
        print("="*60)
        
        obs = self.env.reset()
        initial_pos = self.env._get_end_effector_position()
        
        # 测试不同强度的动作
        action_effects = []
        
        for action_scale in [0, 25, 50, 100]:
            self.env.reset()  # 重置到相同初始状态
            
            # 应用固定动作
            action = np.array([action_scale, 0, 0])  # 只动第一个关节
            
            positions = []
            for step in range(10):
                next_obs, reward, done, info = self.env.step(action)
                pos = self.env._get_end_effector_position()
                positions.append(pos)
            
            # 计算移动距离
            total_movement = 0
            for i in range(1, len(positions)):
                move = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                total_movement += move
            
            action_effects.append(total_movement)
            print(f"   动作强度 {action_scale:3d}: 总移动距离 {total_movement:.1f}")
        
        # 检查动作效果是否递增
        is_increasing = all(action_effects[i] <= action_effects[i+1] for i in range(len(action_effects)-1))
        
        if is_increasing:
            print("✅ 动作效果正常：更大的动作产生更大的移动")
            return True
        else:
            print("❌ 动作效果异常：动作强度与移动距离不成正比")
            return False
    
    def step4_q_value_test(self):
        """步骤4: 测试Q值估计是否合理"""
        print("\n" + "="*60)
        print("🧠 步骤4: Q值估计测试")
        print("="*60)
        
        # 收集一些经验
        print("📚 收集训练数据...")
        for episode in range(5):
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            gnn_embeds = torch.randn(3, 128)
            
            for step in range(20):
                action = self.sac.get_action(obs_tensor, gnn_embeds, num_joints=3)
                next_obs, reward, done, info = self.env.step(action.numpy())
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                next_gnn_embeds = torch.randn(3, 128)
                
                # 存储经验
                self.sac.store_experience(
                    obs_tensor, gnn_embeds, action, reward,
                    next_obs_tensor, next_gnn_embeds, done, num_joints=3
                )
                
                obs_tensor = next_obs_tensor
                gnn_embeds = next_gnn_embeds
                
                if done:
                    break
        
        print(f"📊 Buffer大小: {len(self.sac.memory)}")
        
        # 测试Q值训练
        if len(self.sac.memory) >= self.sac.batch_size:
            print("🔄 开始Q值训练测试...")
            
            initial_losses = []
            final_losses = []
            
            # 记录初始损失
            for _ in range(5):
                metrics = self.sac.update()
                if metrics:
                    initial_losses.append(metrics['critic_loss'])
            
            # 多次更新
            for _ in range(100):
                metrics = self.sac.update()
            
            # 记录最终损失
            for _ in range(5):
                metrics = self.sac.update()
                if metrics:
                    final_losses.append(metrics['critic_loss'])
            
            if initial_losses and final_losses:
                initial_avg = np.mean(initial_losses)
                final_avg = np.mean(final_losses)
                
                print(f"📈 Critic Loss变化:")
                print(f"   初始: {initial_avg:.3f}")
                print(f"   最终: {final_avg:.3f}")
                print(f"   改善: {((initial_avg - final_avg) / initial_avg * 100):.1f}%")
                
                if final_avg < initial_avg:
                    print("✅ Q值学习正常：损失在下降")
                    return True
                else:
                    print("❌ Q值学习异常：损失没有下降")
                    return False
            else:
                print("❌ 无法获取损失数据")
                return False
        else:
            print("❌ Buffer数据不足，无法测试")
            return False
    
    def step5_policy_improvement_test(self):
        """步骤5: 测试策略是否在改进"""
        print("\n" + "="*60)
        print("📈 步骤5: 策略改进测试")
        print("="*60)
        
        # 测试初始策略性能
        initial_rewards = []
        for episode in range(5):
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            gnn_embeds = torch.randn(3, 128)
            episode_reward = 0
            
            for step in range(50):
                action = self.sac.get_action(obs_tensor, gnn_embeds, num_joints=3, deterministic=True)
                next_obs, reward, done, info = self.env.step(action.numpy())
                episode_reward += reward
                
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                gnn_embeds = torch.randn(3, 128)
                
                if done:
                    break
            
            initial_rewards.append(episode_reward)
        
        print(f"🎯 初始策略性能: {np.mean(initial_rewards):.2f} ± {np.std(initial_rewards):.2f}")
        
        # 简短训练
        print("🏃‍♂️ 进行简短训练...")
        for episode in range(20):
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            gnn_embeds = torch.randn(3, 128)
            
            for step in range(30):
                action = self.sac.get_action(obs_tensor, gnn_embeds, num_joints=3)
                next_obs, reward, done, info = self.env.step(action.numpy())
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                next_gnn_embeds = torch.randn(3, 128)
                
                self.sac.store_experience(
                    obs_tensor, gnn_embeds, action, reward,
                    next_obs_tensor, next_gnn_embeds, done, num_joints=3
                )
                
                # 每几步更新一次
                if step % 4 == 0 and len(self.sac.memory) >= self.sac.batch_size:
                    self.sac.update()
                
                obs_tensor = next_obs_tensor
                gnn_embeds = next_gnn_embeds
                
                if done:
                    break
        
        # 测试训练后策略性能
        final_rewards = []
        for episode in range(5):
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            gnn_embeds = torch.randn(3, 128)
            episode_reward = 0
            
            for step in range(50):
                action = self.sac.get_action(obs_tensor, gnn_embeds, num_joints=3, deterministic=True)
                next_obs, reward, done, info = self.env.step(action.numpy())
                episode_reward += reward
                
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                gnn_embeds = torch.randn(3, 128)
                
                if done:
                    break
            
            final_rewards.append(episode_reward)
        
        print(f"🎯 训练后策略性能: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
        
        improvement = np.mean(final_rewards) - np.mean(initial_rewards)
        print(f"📊 性能改善: {improvement:.2f}")
        
        if improvement > 0:
            print("✅ 策略正在改进")
            return True
        else:
            print("❌ 策略没有明显改进")
            return False
    
    def run_full_diagnostic(self):
        """运行完整诊断"""
        print("🔬 开始SAC学习能力全面诊断")
        print("="*80)
        
        test_results = {}
        
        # 依次运行所有测试
        test_results['basic'] = self.step1_basic_functionality_test()
        test_results['reward'] = self.step2_reward_signal_test()
        test_results['action'] = self.step3_action_effect_test()
        test_results['qvalue'] = self.step4_q_value_test()
        test_results['policy'] = self.step5_policy_improvement_test()
        
        # 生成诊断报告
        print("\n" + "="*80)
        print("📋 诊断报告")
        print("="*80)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:10s}: {status}")
        
        print(f"\n📊 总体评估: {passed}/{total} 项测试通过")
        
        if passed == total:
            print("🎉 恭喜！你的模型学习能力正常")
        elif passed >= total * 0.6:
            print("⚠️  模型有学习能力，但存在一些问题需要调优")
        else:
            print("🚨 模型学习能力存在严重问题，需要检查架构或训练方法")
        
        return test_results

def main():
    """主函数"""
    diagnostics = LearningDiagnostics()
    results = diagnostics.run_full_diagnostic()
    
    print("\n🔧 建议的后续行动:")
    if not results['basic']:
        print("   1. 检查环境和模型的基本配置")
    if not results['reward']:
        print("   2. 调整奖励函数，确保信号明确")
    if not results['action']:
        print("   3. 检查动作缩放和环境交互")
    if not results['qvalue']:
        print("   4. 调整学习率或网络架构")
    if not results['policy']:
        print("   5. 增加训练时间或调整探索策略")

if __name__ == "__main__":
    main()
