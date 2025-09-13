#!/usr/bin/env python3
"""
🔍 Enhanced Train 深度分析工具 - 完全修复版
分析模型学习状态、瓶颈和优化建议
"""
import numpy as np
import torch
import sys
import os
import time
from collections import deque, defaultdict
import json

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.extend([
    base_dir,
    os.path.join(base_dir, 'examples/2d_reacher/envs'),
    os.path.join(base_dir, 'examples/2d_reacher/utils'),
    os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'),
    os.path.join(base_dir, 'examples/rl/train'),
    os.path.join(base_dir, 'examples/rl/common'),
    os.path.join(base_dir, 'examples/rl/environments'),
    os.path.join(base_dir, 'examples/rl')
])

from reacher2d_env import Reacher2DEnv
from attn_model.attn_model import AttnModel
from sac.sac_model import AttentionSACWithBuffer

class TrainingAnalyzer:
    """训练过程深度分析器 - 完全修复版"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.analysis_results = {}
        
    def analyze_model_capacity(self):
        """分析模型容量是否合适"""
        print("🔍 === 模型容量分析 ===")
        
        # 创建模型
        attn_model = AttnModel(128, 130, 130, 4)
        sac = AttentionSACWithBuffer(
            attn_model, 3,
            buffer_capacity=1000,
            batch_size=32,
            lr=1e-4,
            env_type='reacher2d'
        )
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in attn_model.parameters())
        trainable_params = sum(p.numel() for p in attn_model.parameters() if p.requires_grad)
        
        actor_params = sum(p.numel() for p in sac.actor.parameters())
        critic_params = sum(p.numel() for p in sac.critic1.parameters())
        
        print(f"📊 模型参数统计:")
        print(f"   AttnModel总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   Actor参数: {actor_params:,}")
        print(f"   Critic参数: {critic_params:,}")
        print(f"   总SAC参数: {actor_params + critic_params * 2:,}")
        
        # 分析复杂度
        task_complexity = 3  # 3关节任务
        param_per_joint = total_params / task_complexity
        
        print(f"\n🎯 复杂度分析:")
        print(f"   任务复杂度: {task_complexity}关节")
        print(f"   每关节参数: {param_per_joint:,.0f}")
        
        if param_per_joint > 50000:
            print(f"   ⚠️  模型可能过于复杂 (建议<50k参数/关节)")
            complexity_verdict = "过于复杂"
        elif param_per_joint < 5000:
            print(f"   ⚠️  模型可能过于简单 (建议>5k参数/关节)")
            complexity_verdict = "过于简单"
        else:
            print(f"   ✅ 模型复杂度合适")
            complexity_verdict = "合适"
            
        self.analysis_results['model_complexity'] = {
            'total_params': total_params,
            'param_per_joint': param_per_joint,
            'verdict': complexity_verdict
        }
        
        return complexity_verdict
    
    def analyze_gradient_flow_fixed(self, steps=200):
        """完全修复版梯度流动分析"""
        print(f"\n🔍 === 梯度流动分析 ({steps}步) ===")
        
        # 创建环境和模型
        env = Reacher2DEnv(num_links=3, link_lengths=[90, 90, 90], config_path=None, render_mode=None)
        attn_model = AttnModel(128, 130, 130, 4)
        sac = AttentionSACWithBuffer(attn_model, 3, buffer_capacity=5000, batch_size=32, lr=1e-4, env_type='reacher2d')
        
        # 🔧 关键修复1: 正确的GNN嵌入维度
        # reacher2d需要 [3, 128] 而不是 [1, 20, 128]
        gnn_embed = torch.randn(3, 128)  # 3个关节，每个128维
        
        # 收集梯度统计
        gradient_norms = {'actor': [], 'critic1': [], 'critic2': []}
        loss_history = {'actor': [], 'critic': [], 'alpha': []}
        reward_history = []
        distance_history = []
        
        obs = env.reset()
        episode_reward = 0
        
        print("🎯 开始梯度分析训练...")
        
        for step in range(steps):
            # 🔧 关键修复2: 正确的观察空间处理
            # reacher2d的观察空间是 [angles(3) + angular_vels(3) + end_pos(2)] = 8维
            obs_tensor = torch.from_numpy(obs).float()
            
            # 获取动作
            if step < 20:  # 短暂warmup
                action = env.action_space.sample()
            else:
                try:
                    action = sac.get_action(obs_tensor, gnn_embed, num_joints=3, deterministic=False)
                    action = action.detach().cpu().numpy()
                except Exception as e:
                    print(f"   ⚠️  动作生成失败，使用随机动作: {e}")
                    action = env.action_space.sample()
            
            # 执行步骤
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # 记录距离
            end_pos = env._get_end_effector_position()
            distance = np.linalg.norm(np.array(end_pos) - env.goal_pos)
            distance_history.append(distance)
            
            # 存储经验
            if step >= 10:  # 减少warmup时间
                next_obs_tensor = torch.from_numpy(next_obs).float()
                action_tensor = torch.from_numpy(action).float()
                
                try:
                    sac.store_experience(
                        obs_tensor, gnn_embed, action_tensor, reward,
                        next_obs_tensor, gnn_embed, done, num_joints=3
                    )
                except Exception as e:
                    print(f"   ⚠️  经验存储失败: {e}")
            
            # 更新和分析梯度
            if step > 20 and step % 4 == 0 and sac.memory.can_sample(sac.batch_size):
                try:
                    metrics = sac.update()
                    
                    if metrics:
                        loss_history['actor'].append(metrics.get('actor_loss', 0))
                        loss_history['critic'].append(metrics.get('critic_loss', 0))
                        loss_history['alpha'].append(metrics.get('alpha_loss', 0))
                        
                        # 简化梯度分析
                        actor_grad_norm = 0
                        for param in sac.actor.parameters():
                            if param.grad is not None:
                                actor_grad_norm += param.grad.norm().item() ** 2
                        
                        if actor_grad_norm > 0:
                            gradient_norms['actor'].append(np.sqrt(actor_grad_norm))
                
                except Exception as e:
                    print(f"   ⚠️  模型更新失败: {e}")
            
            obs = next_obs
            
            if done:
                reward_history.append(episode_reward)
                episode_reward = 0
                obs = env.reset()
            
            if step % 50 == 0:  # 减少打印频率
                print(f"   步骤 {step}: 距离={distance:.1f}px, 缓冲区={len(sac.memory)}")
        
        # 分析结果
        self._analyze_gradient_statistics(gradient_norms, loss_history, distance_history, reward_history)
        env.close()
    
    def _analyze_gradient_statistics(self, gradient_norms, loss_history, distance_history, reward_history):
        """分析梯度统计数据"""
        print(f"\n📊 梯度分析结果:")
        
        # 梯度范数分析
        for network, norms in gradient_norms.items():
            if norms:
                avg_norm = np.mean(norms)
                std_norm = np.std(norms)
                print(f"   {network} 平均梯度范数: {avg_norm:.6f} ± {std_norm:.6f}")
                
                if avg_norm < 1e-6:
                    print(f"   ⚠️  {network} 梯度过小，可能学习停滞")
                elif avg_norm > 1.0:
                    print(f"   ⚠️  {network} 梯度过大，可能不稳定")
                else:
                    print(f"   ✅ {network} 梯度范数正常")
        
        # 损失分析
        print(f"\n📈 损失趋势分析:")
        for loss_type, losses in loss_history.items():
            if losses and len(losses) > 5:
                recent_losses = losses[-10:] if len(losses) > 10 else losses
                if len(recent_losses) > 1:
                    trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                    
                    if abs(trend) < 1e-6:
                        trend_desc = "平稳"
                    elif trend < 0:
                        trend_desc = "下降 ✅"
                    else:
                        trend_desc = "上升 ⚠️"
                    
                    print(f"   {loss_type}_loss: 最近趋势 {trend_desc} (斜率: {trend:.6f})")
        
        # 性能分析
        if distance_history and len(distance_history) > 20:
            initial_dist = np.mean(distance_history[:10])
            final_dist = np.mean(distance_history[-10:])
            improvement = initial_dist - final_dist
            
            print(f"\n🎯 性能改善分析:")
            print(f"   初始距离: {initial_dist:.1f}px")
            print(f"   最终距离: {final_dist:.1f}px")
            print(f"   改善程度: {improvement:.1f}px")
            
            if improvement > 20:
                print(f"   ✅ 显著改善")
                learning_verdict = "正常学习"
            elif improvement > 5:
                print(f"   ⚠️  轻微改善")
                learning_verdict = "学习缓慢"
            else:
                print(f"   ❌ 无明显改善")
                learning_verdict = "学习停滞"
        else:
            print(f"   ⚠️  数据不足，无法分析")
            learning_verdict = "数据不足"
            improvement = 0
        
        self.analysis_results['learning_progress'] = {
            'distance_improvement': improvement,
            'verdict': learning_verdict
        }
    
    def analyze_action_diversity_fixed(self):
        """完全修复版动作多样性分析"""
        print(f"\n🔍 === 动作多样性分析 ===")
        
        try:
            # 创建模型
            attn_model = AttnModel(128, 130, 130, 4)
            sac = AttentionSACWithBuffer(attn_model, 3, buffer_capacity=1000, batch_size=32, lr=1e-4, env_type='reacher2d')
            
            # 🔧 关键修复3: 正确的GNN嵌入和观察空间
            gnn_embed = torch.randn(3, 128)  # 3关节，每个128维
            
            # 收集动作样本
            actions = []
            dummy_obs = torch.zeros(8)  # reacher2d观察空间是8维
            
            for i in range(100):  # 减少样本数量以加快测试
                try:
                    action = sac.get_action(dummy_obs, gnn_embed, num_joints=3, deterministic=False)
                    actions.append(action.detach().cpu().numpy())
                except Exception as e:
                    if i == 0:  # 只打印第一次错误
                        print(f"   ⚠️  动作生成失败: {e}")
                    # 使用随机动作作为替代
                    action = np.random.uniform(-100, 100, 3)
                    actions.append(action)
            
            actions = np.array(actions)
            
            # 分析多样性
            action_std = np.std(actions, axis=0)
            action_range = np.max(actions, axis=0) - np.min(actions, axis=0)
            action_mean = np.mean(actions, axis=0)
            
            print(f"📊 动作统计:")
            for i in range(3):
                print(f"   关节{i+1}: 均值={action_mean[i]:+.1f}, 标准差={action_std[i]:.1f}, 范围={action_range[i]:.1f}")
            
            # 多样性评估
            avg_std = np.mean(action_std)
            if avg_std < 5:
                print(f"   ⚠️  动作多样性不足 (平均标准差: {avg_std:.1f})")
                diversity_verdict = "多样性不足"
            elif avg_std > 50:
                print(f"   ⚠️  动作过于随机 (平均标准差: {avg_std:.1f})")
                diversity_verdict = "过于随机"
            else:
                print(f"   ✅ 动作多样性合适 (平均标准差: {avg_std:.1f})")
                diversity_verdict = "合适"
            
            self.analysis_results['action_diversity'] = {
                'avg_std': avg_std,
                'verdict': diversity_verdict
            }
            
        except Exception as e:
            print(f"   ❌ 动作分析失败: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_results['action_diversity'] = {
                'avg_std': 0,
                'verdict': "分析失败"
            }
    
    def analyze_learning_bottlenecks(self):
        """分析学习瓶颈"""
        print(f"\n🔍 === 学习瓶颈分析 ===")
        
        bottlenecks = []
        
        # 检查模型复杂度瓶颈
        if self.analysis_results.get('model_complexity', {}).get('verdict') == '过于复杂':
            bottlenecks.append("模型过于复杂，可能过拟合")
        elif self.analysis_results.get('model_complexity', {}).get('verdict') == '过于简单':
            bottlenecks.append("模型过于简单，可能欠拟合")
        
        # 检查学习进度瓶颈
        learning_verdict = self.analysis_results.get('learning_progress', {}).get('verdict')
        if learning_verdict == '学习停滞':
            bottlenecks.append("学习停滞，可能需要调整学习率或探索策略")
        elif learning_verdict == '学习缓慢':
            bottlenecks.append("学习缓慢，建议增加学习率或减少批次大小")
        
        # 检查动作多样性瓶颈
        diversity_verdict = self.analysis_results.get('action_diversity', {}).get('verdict')
        if diversity_verdict == '多样性不足':
            bottlenecks.append("探索不足，建议增加alpha值或减少warmup步数")
        elif diversity_verdict == '过于随机':
            bottlenecks.append("探索过度，建议减少alpha值")
        
        if bottlenecks:
            print("❌ 发现的瓶颈:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"   {i}. {bottleneck}")
        else:
            print("✅ 未发现明显瓶颈")
        
        return bottlenecks
    
    def generate_optimization_recommendations(self):
        """生成优化建议"""
        print(f"\n💡 === 优化建议 ===")
        
        recommendations = []
        
        # 🔧 添加维度修复建议
        recommendations.append("🔧 修复维度问题: 确保GNN嵌入为[3, 128]而不是[1, 20, 128]")
        recommendations.append("🔧 修复观察空间: 确保obs为8维而不是其他维度")
        
        # 基于分析结果生成建议
        model_complexity = self.analysis_results.get('model_complexity', {}).get('verdict')
        if model_complexity == '过于复杂':
            recommendations.append("减少模型复杂度: AttnModel(64, 66, 66, 2)")
        elif model_complexity == '过于简单':
            recommendations.append("增加模型复杂度: AttnModel(256, 260, 260, 8)")
        
        learning_progress = self.analysis_results.get('learning_progress', {}).get('verdict')
        if learning_progress == '学习停滞':
            recommendations.append("增加学习率: --lr 5e-4")
            recommendations.append("减少批次大小: --batch-size 16")
            recommendations.append("增加探索: --alpha 0.4")
        elif learning_progress == '学习缓慢':
            recommendations.append("适度增加学习率: --lr 2e-4")
            recommendations.append("增加探索: --alpha 0.25")
        
        action_diversity = self.analysis_results.get('action_diversity', {}).get('verdict')
        if action_diversity == '多样性不足':
            recommendations.append("增加探索: --alpha 0.4")
            recommendations.append("减少warmup: --warmup-steps 1000")
        elif action_diversity == '过于随机':
            recommendations.append("减少探索: --alpha 0.15")
            recommendations.append("增加warmup: --warmup-steps 3000")
        
        # 通用建议
        recommendations.extend([
            "使用可达配置: 修改enhanced_train.py第54行为reacher_easy.yaml",
            "监控训练: 每1000步检查距离变化",
            "早期停止: 连续5000步无改善时停止"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return recommendations
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🚀 开始深度训练分析...")
        print("="*60)
        
        try:
            # 1. 模型容量分析
            self.analyze_model_capacity()
            
            # 2. 修复版梯度流动分析
            self.analyze_gradient_flow_fixed(steps=100)  # 进一步减少步数
            
            # 3. 修复版动作多样性分析
            self.analyze_action_diversity_fixed()
            
            # 4. 瓶颈分析
            bottlenecks = self.analyze_learning_bottlenecks()
            
            # 5. 优化建议
            recommendations = self.generate_optimization_recommendations()
            
            print("\n" + "="*60)
            print("📋 === 分析总结 ===")
            print(f"模型复杂度: {self.analysis_results.get('model_complexity', {}).get('verdict', '未知')}")
            print(f"学习进度: {self.analysis_results.get('learning_progress', {}).get('verdict', '未知')}")
            print(f"动作多样性: {self.analysis_results.get('action_diversity', {}).get('verdict', '未知')}")
            print(f"发现瓶颈: {len(bottlenecks)}个")
            print(f"优化建议: {len(recommendations)}条")
            
            return {
                'bottlenecks': bottlenecks,
                'recommendations': recommendations,
                'analysis_results': self.analysis_results
            }
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                'bottlenecks': ["分析失败"],
                'recommendations': ["修复维度不匹配问题"],
                'analysis_results': self.analysis_results
            }

def main():
    analyzer = TrainingAnalyzer()
    results = analyzer.run_full_analysis()
    
    # 保存分析结果
    try:
        with open('training_analysis_results_fixed.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 分析结果已保存到: training_analysis_results_fixed.json")
    except Exception as e:
        print(f"⚠️  保存结果失败: {e}")

if __name__ == "__main__":
    main()