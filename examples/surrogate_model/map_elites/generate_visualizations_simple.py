#!/usr/bin/env python3
"""
简单的MAP-Elites可视化生成器（不依赖pandas）
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from collections import defaultdict

def read_csv_data(csv_file: str) -> dict:
    """
    读取CSV文件并返回数据字典
    """
    data = defaultdict(list)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    # 尝试转换为数字
                    try:
                        if '.' in value:
                            data[key].append(float(value))
                        else:
                            data[key].append(int(value))
                    except (ValueError, TypeError):
                        data[key].append(value)
        
        print(f"📊 成功读取CSV文件: {len(data[list(data.keys())[0]])} 条记录")
        print(f"📊 列名: {list(data.keys())}")
        return data
        
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return {}

def create_map_elites_heatmap(data: dict, output_path: str) -> bool:
    """
    创建MAP-Elites热力图
    """
    try:
        # 检查必需的列
        required_cols = ['num_links', 'total_length', 'fitness']
        if not all(col in data for col in required_cols):
            print(f"⚠️ 缺少必需的列: {required_cols}")
            return False
        
        num_links = np.array(data['num_links'])
        total_length = np.array(data['total_length'])
        fitness = np.array(data['fitness'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'MAP-Elites 训练结果热力图分析\\n个体数: {len(num_links)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 关节数 vs 总长度 散点图
        scatter1 = ax1.scatter(num_links, total_length, c=fitness, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='white')
        ax1.set_xlabel('关节数')
        ax1.set_ylabel('总长度 (px)')
        ax1.set_title('关节数 vs 总长度')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='适应度')
        
        # 2. 学习率分析（如果有的话）
        if 'lr' in data and 'alpha' in data:
            lr = np.array(data['lr'])
            alpha = np.array(data['alpha'])
            scatter2 = ax2.scatter(lr, alpha, c=fitness, 
                                 cmap='plasma', s=100, alpha=0.7, edgecolors='white')
            ax2.set_xlabel('学习率')
            ax2.set_ylabel('Alpha')
            ax2.set_title('学习率 vs Alpha')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='适应度')
        else:
            ax2.text(0.5, 0.5, '学习率数据不可用', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        
        # 3. 适应度分布
        ax3.hist(fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('适应度')
        ax3.set_ylabel('个体数量')
        ax3.set_title('适应度分布')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_fitness = np.mean(fitness)
        max_fitness = np.max(fitness)
        ax3.axvline(mean_fitness, color='red', linestyle='--', 
                   label=f'平均值: {mean_fitness:.3f}')
        ax3.axvline(max_fitness, color='green', linestyle='--', 
                   label=f'最大值: {max_fitness:.3f}')
        ax3.legend()
        
        # 4. 关节数分布
        unique_links, counts = np.unique(num_links, return_counts=True)
        ax4.bar(unique_links, counts, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('关节数')
        ax4.set_ylabel('个体数量')
        ax4.set_title('关节数分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MAP-Elites热力图已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成MAP-Elites热力图失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_metrics_plot(data: dict, output_path: str) -> bool:
    """
    创建训练指标可视化
    """
    try:
        if 'fitness' not in data:
            return False
        
        fitness = np.array(data['fitness'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'训练指标分析\\n总个体数: {len(fitness)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 适应度vs成功率
        if 'success_rate' in data:
            success_rate = np.array(data['success_rate'])
            ax1.scatter(fitness, success_rate, s=100, alpha=0.7, color='blue', edgecolors='white')
            ax1.set_xlabel('适应度')
            ax1.set_ylabel('成功率')
            ax1.set_title('适应度 vs 成功率')
            ax1.grid(True, alpha=0.3)
            
            # 成功率分布
            ax2.hist(success_rate, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('成功率')
            ax2.set_ylabel('个体数量')
            ax2.set_title('成功率分布')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, '成功率数据不可用', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, '成功率数据不可用', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. 奖励分析
        if 'avg_reward' in data:
            avg_reward = np.array(data['avg_reward'])
            ax3.scatter(fitness, avg_reward, s=100, alpha=0.7, color='red', edgecolors='white')
            ax3.set_xlabel('适应度')
            ax3.set_ylabel('平均奖励')
            ax3.set_title('适应度 vs 平均奖励')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '奖励数据不可用', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 代际进化（如果有的话）
        if 'generation' in data:
            generation = np.array(data['generation'])
            
            # 按代数分组计算统计
            generations = {}
            for i, gen in enumerate(generation):
                if gen not in generations:
                    generations[gen] = []
                generations[gen].append(fitness[i])
            
            gen_nums = sorted(generations.keys())
            gen_max = [max(generations[gen]) for gen in gen_nums]
            gen_mean = [np.mean(generations[gen]) for gen in gen_nums]
            
            ax4.plot(gen_nums, gen_max, 'b-o', label='最佳适应度', linewidth=2, markersize=6)
            ax4.plot(gen_nums, gen_mean, 'r--s', label='平均适应度', linewidth=2, markersize=6)
            ax4.set_xlabel('代数')
            ax4.set_ylabel('适应度')
            ax4.set_title('代际进化趋势')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, '代数数据不可用', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练指标可视化已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成训练指标可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_neural_network_loss_simulation(data: dict, output_path: str) -> bool:
    """
    创建模拟的神经网络Loss可视化
    """
    try:
        if 'fitness' not in data:
            return False
        
        fitness = np.array(data['fitness'])
        n_individuals = len(fitness)
        
        # 模拟不同网络组件的Loss变化
        np.random.seed(42)  # 确保结果可重现
        
        # 为每个个体模拟训练过程
        training_steps = 100
        steps = np.arange(training_steps)
        
        # 模拟不同网络的loss
        actor_losses = []
        critic_losses = []
        attention_losses = []
        gnn_losses = []
        
        for i in range(n_individuals):
            # 基于fitness生成不同的loss曲线
            base_fitness = fitness[i]
            
            # Actor loss: 从高开始，逐渐下降
            actor_loss = 10.0 * (1 - base_fitness) * np.exp(-steps / 50) + np.random.normal(0, 0.1, training_steps)
            actor_losses.append(actor_loss)
            
            # Critic loss: 类似但稍微不同的模式
            critic_loss = 8.0 * (1 - base_fitness) * np.exp(-steps / 40) + np.random.normal(0, 0.15, training_steps)
            critic_losses.append(critic_loss)
            
            # Attention loss: 更快收敛
            attention_loss = 5.0 * (1 - base_fitness) * np.exp(-steps / 30) + np.random.normal(0, 0.08, training_steps)
            attention_losses.append(attention_loss)
            
            # GNN loss: 中等收敛速度
            gnn_loss = 6.0 * (1 - base_fitness) * np.exp(-steps / 45) + np.random.normal(0, 0.12, training_steps)
            gnn_losses.append(gnn_loss)
        
        # 计算平均loss曲线
        avg_actor_loss = np.mean(actor_losses, axis=0)
        avg_critic_loss = np.mean(critic_losses, axis=0)
        avg_attention_loss = np.mean(attention_losses, axis=0)
        avg_gnn_loss = np.mean(gnn_losses, axis=0)
        
        # 创建可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'神经网络Loss变化分析（基于{n_individuals}个个体模拟）', 
                    fontsize=16, fontweight='bold')
        
        # 1. 主要网络Loss曲线
        ax1.plot(steps, avg_actor_loss, 'r-', linewidth=2, label='Actor Loss', alpha=0.8)
        ax1.plot(steps, avg_critic_loss, 'b-', linewidth=2, label='Critic Loss', alpha=0.8)
        ax1.plot(steps, avg_attention_loss, 'g-', linewidth=2, label='Attention Loss', alpha=0.8)
        ax1.plot(steps, avg_gnn_loss, 'm-', linewidth=2, label='GNN Loss', alpha=0.8)
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('Loss值')
        ax1.set_title('主要网络Loss变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Loss收敛分析
        final_losses = [avg_actor_loss[-1], avg_critic_loss[-1], 
                       avg_attention_loss[-1], avg_gnn_loss[-1]]
        network_names = ['Actor', 'Critic', 'Attention', 'GNN']
        colors = ['red', 'blue', 'green', 'magenta']
        
        bars = ax2.bar(network_names, final_losses, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('最终Loss值')
        ax2.set_title('各网络最终Loss对比')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        # 3. Loss下降速度分析
        # 计算每个网络达到50%初始loss的步数
        convergence_steps = []
        initial_losses = [avg_actor_loss[0], avg_critic_loss[0], 
                         avg_attention_loss[0], avg_gnn_loss[0]]
        all_losses = [avg_actor_loss, avg_critic_loss, avg_attention_loss, avg_gnn_loss]
        
        for i, loss_curve in enumerate(all_losses):
            target = initial_losses[i] * 0.5
            convergence_step = np.argmax(loss_curve <= target)
            convergence_steps.append(convergence_step if convergence_step > 0 else training_steps)
        
        bars = ax3.bar(network_names, convergence_steps, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('收敛步数')
        ax3.set_title('收敛速度对比（达到50%初始Loss）')
        ax3.grid(True, alpha=0.3)
        
        for bar, steps in zip(bars, convergence_steps):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{steps}', ha='center', va='bottom')
        
        # 4. Loss稳定性分析
        # 计算最后20步的标准差
        stability_scores = []
        for loss_curve in all_losses:
            last_20_steps = loss_curve[-20:]
            stability = np.std(last_20_steps)
            stability_scores.append(stability)
        
        bars = ax4.bar(network_names, stability_scores, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Loss稳定性 (标准差)')
        ax4.set_title('训练稳定性分析（最后20步）')
        ax4.grid(True, alpha=0.3)
        
        for bar, stability in zip(bars, stability_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{stability:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 神经网络Loss可视化已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成神经网络Loss可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comprehensive_report(data: dict, output_path: str) -> bool:
    """
    创建综合统计报告
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MAP-Elites 综合训练结果报告\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"总个体数: {len(data[list(data.keys())[0]])}\\n\\n")
            
            # 基础统计
            f.write("📊 基础统计信息:\\n")
            f.write("-" * 30 + "\\n")
            
            if 'fitness' in data:
                fitness = np.array(data['fitness'])
                f.write(f"适应度统计:\\n")
                f.write(f"  平均值: {np.mean(fitness):.4f}\\n")
                f.write(f"  最大值: {np.max(fitness):.4f}\\n")
                f.write(f"  最小值: {np.min(fitness):.4f}\\n")
                f.write(f"  标准差: {np.std(fitness):.4f}\\n\\n")
            
            if 'success_rate' in data:
                success_rate = np.array(data['success_rate'])
                f.write(f"成功率统计:\\n")
                f.write(f"  平均成功率: {np.mean(success_rate):.4f}\\n")
                f.write(f"  最高成功率: {np.max(success_rate):.4f}\\n")
                f.write(f"  成功率>0.5的个体: {np.sum(success_rate > 0.5)}个\\n\\n")
            
            if 'num_links' in data:
                num_links = np.array(data['num_links'])
                f.write(f"机器人结构统计:\\n")
                f.write(f"  关节数范围: {np.min(num_links)}-{np.max(num_links)}\\n")
                unique_links, counts = np.unique(num_links, return_counts=True)
                for links, count in zip(unique_links, counts):
                    f.write(f"  {int(links)}关节: {count}个 ({count/len(num_links)*100:.1f}%)\\n")
                f.write("\\n")
            
            if 'total_length' in data:
                total_length = np.array(data['total_length'])
                f.write(f"机器人长度统计:\\n")
                f.write(f"  平均长度: {np.mean(total_length):.1f}px\\n")
                f.write(f"  长度范围: {np.min(total_length):.1f}-{np.max(total_length):.1f}px\\n\\n")
            
            # 代际分析
            if 'generation' in data:
                generation = np.array(data['generation'])
                fitness = np.array(data['fitness'])
                
                f.write("🧬 代际进化分析:\\n")
                f.write("-" * 30 + "\\n")
                
                generations = {}
                for i, gen in enumerate(generation):
                    if gen not in generations:
                        generations[gen] = []
                    generations[gen].append(fitness[i])
                
                for gen in sorted(generations.keys()):
                    gen_fitness = generations[gen]
                    f.write(f"第{gen}代: {len(gen_fitness)}个个体, ")
                    f.write(f"最佳适应度={max(gen_fitness):.4f}, ")
                    f.write(f"平均适应度={np.mean(gen_fitness):.4f}\\n")
                f.write("\\n")
            
            # 最佳个体信息
            if 'fitness' in data:
                fitness = np.array(data['fitness'])
                best_idx = np.argmax(fitness)
                f.write("🏆 最佳个体信息:\\n")
                f.write("-" * 30 + "\\n")
                f.write(f"适应度: {fitness[best_idx]:.4f}\\n")
                
                if 'num_links' in data:
                    f.write(f"关节数: {data['num_links'][best_idx]}\\n")
                if 'total_length' in data:
                    f.write(f"总长度: {data['total_length'][best_idx]:.1f}px\\n")
                if 'success_rate' in data:
                    f.write(f"成功率: {data['success_rate'][best_idx]:.4f}\\n")
                if 'lr' in data:
                    f.write(f"学习率: {data['lr'][best_idx]:.2e}\\n")
                if 'alpha' in data:
                    f.write(f"Alpha: {data['alpha'][best_idx]:.4f}\\n")
        
        print(f"✅ 综合报告已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成综合报告失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成MAP-Elites简单可视化")
    parser.add_argument("--csv-file", type=str, 
                       default="./experiment_results/session_20250917_160838/results.csv",
                       help="CSV结果文件")
    parser.add_argument("--output-dir", type=str, 
                       default="./map_elites_shared_ppo_results/visualizations",
                       help="输出目录")
    args = parser.parse_args()
    
    print("🎨 生成MAP-Elites简单可视化")
    print("=" * 60)
    
    if not os.path.exists(args.csv_file):
        print(f"❌ CSV文件不存在: {args.csv_file}")
        return
    
    # 读取数据
    data = read_csv_data(args.csv_file)
    if not data:
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    
    # 1. 生成MAP-Elites热力图
    print("\\n📊 生成MAP-Elites热力图...")
    heatmap_path = os.path.join(args.output_dir, "map_elites_heatmap.png")
    if create_map_elites_heatmap(data, heatmap_path):
        success_count += 1
    
    # 2. 生成训练指标可视化
    print("\\n📈 生成训练指标可视化...")
    metrics_path = os.path.join(args.output_dir, "training_metrics.png")
    if create_training_metrics_plot(data, metrics_path):
        success_count += 1
    
    # 3. 生成神经网络Loss可视化（模拟）
    print("\\n🧠 生成神经网络Loss可视化...")
    loss_path = os.path.join(args.output_dir, "neural_network_losses.png")
    if create_neural_network_loss_simulation(data, loss_path):
        success_count += 1
    
    # 4. 生成综合报告
    print("\\n📝 生成综合报告...")
    report_path = os.path.join(args.output_dir, "comprehensive_report.txt")
    if create_comprehensive_report(data, report_path):
        success_count += 1
    
    print("\\n" + "=" * 60)
    print(f"🎉 可视化生成完成! 成功: {success_count}/4")
    print(f"📁 所有文件保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

