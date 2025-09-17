#!/usr/bin/env python3
"""
MAP-Elites和神经网络训练可视化演示脚本
功能：
1. 演示MAP-Elites热力图生成
2. 演示神经网络loss可视化
3. 集成训练过程中的实时可视化
4. 生成综合分析报告
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

# 导入可视化工具
from map_elites_visualizer import MAPElitesVisualizer
from network_loss_visualizer import NetworkLossVisualizer

# 导入MAP-Elites核心组件
from map_elites_core import MAPElitesArchive, Individual, RobotGenotype, RobotPhenotype
from training_adapter import MAPElitesTrainingAdapter

class VisualizationDemo:
    """可视化演示类"""
    
    def __init__(self, output_dir: str = "./demo_visualizations"):
        """
        初始化演示
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化可视化工具
        self.map_elites_viz = MAPElitesVisualizer(output_dir=output_dir)
        self.loss_viz = NetworkLossVisualizer(output_dir=output_dir)
        
        print(f"🎨 可视化演示初始化完成")
        print(f"   输出目录: {output_dir}")
    
    def create_demo_map_elites_data(self) -> str:
        """创建演示用的MAP-Elites数据"""
        print("🎲 正在创建演示用的MAP-Elites数据...")
        
        # 创建存档
        archive = MAPElitesArchive()
        archive.generation = 5
        archive.total_evaluations = 50
        
        # 生成演示个体
        np.random.seed(42)  # 确保可重现性
        
        for i in range(50):
            # 随机生成基因型
            num_links = np.random.randint(2, 8)
            link_lengths = np.random.uniform(20, 100, num_links).tolist()
            lr = np.random.uniform(1e-5, 1e-2)
            alpha = np.random.uniform(0.1, 0.5)
            
            genotype = RobotGenotype(
                num_links=num_links,
                link_lengths=link_lengths,
                lr=lr,
                alpha=alpha
            )
            
            # 创建表现型
            phenotype = RobotPhenotype()
            phenotype.avg_reward = np.random.uniform(-100, 50)
            phenotype.success_rate = np.random.uniform(0, 1)
            phenotype.min_distance = np.random.uniform(10, 200)
            
            # 创建个体
            individual = Individual(
                individual_id=f"demo_{i}",
                genotype=genotype,
                phenotype=phenotype,
                generation=np.random.randint(0, 6),
                parent_id=f"demo_{max(0, i-1)}" if i > 0 else None
            )
            
            # 计算适应度（使用简化的公式）
            total_length = sum(link_lengths)
            if total_length < 100:
                # 长度不足
                individual.fitness = 0.1 + np.random.uniform(0, 0.3)
                individual.fitness_details = {
                    'category': 'insufficient_for_direct',
                    'strategy': 'length_optimization',
                    'reason': '总长度不足以直接到达目标'
                }
            elif total_length < 200:
                # 中等长度
                individual.fitness = 0.4 + np.random.uniform(0, 0.4)
                individual.fitness_details = {
                    'category': 'insufficient_for_path',
                    'strategy': 'hybrid_optimization',
                    'reason': '需要优化路径规划'
                }
            else:
                # 充足长度
                individual.fitness = 0.7 + np.random.uniform(0, 0.3)
                individual.fitness_details = {
                    'category': 'sufficient_length',
                    'strategy': 'performance_optimization',
                    'reason': '专注于训练性能优化'
                }
            
            # 添加到存档
            archive.add_individual(individual)
        
        # 保存存档
        archive_path = os.path.join(self.output_dir, 'demo_archive.pkl')
        archive.save_archive(archive_path)
        
        print(f"✅ 演示存档已创建: {archive_path}")
        print(f"   个体数量: {len(archive.archive)}")
        print(f"   代数: {archive.generation}")
        
        return archive_path
    
    def create_demo_training_logs(self) -> str:
        """创建演示用的训练日志"""
        print("📊 正在创建演示用的训练日志...")
        
        # 创建日志目录
        log_dir = os.path.join(self.output_dir, 'demo_training_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成模拟训练数据
        np.random.seed(42)
        n_steps = 1000
        
        # 模拟各种loss
        actor_losses = []
        critic_losses = []
        attention_losses = []
        gnn_losses = []
        alpha_values = []
        learning_rates = []
        entropies = []
        
        # 初始值
        actor_loss = 5.0
        critic_loss = 10.0
        attention_loss = 3.0
        gnn_loss = 2.0
        alpha = 0.2
        lr = 3e-4
        entropy = 1.5
        
        for step in range(n_steps):
            # 模拟loss下降（带噪声）
            actor_loss *= (0.999 + np.random.normal(0, 0.001))
            critic_loss *= (0.9995 + np.random.normal(0, 0.002))
            attention_loss *= (0.999 + np.random.normal(0, 0.0015))
            gnn_loss *= (0.9998 + np.random.normal(0, 0.001))
            
            # 模拟alpha和学习率的变化
            alpha += np.random.normal(0, 0.01)
            alpha = np.clip(alpha, 0.05, 0.5)
            
            lr *= (0.9999 + np.random.normal(0, 0.0001))
            lr = max(lr, 1e-6)
            
            entropy += np.random.normal(0, 0.05)
            entropy = max(entropy, 0.1)
            
            # 记录数据
            actor_losses.append(max(0.01, actor_loss))
            critic_losses.append(max(0.01, critic_loss))
            attention_losses.append(max(0.01, attention_loss))
            gnn_losses.append(max(0.01, gnn_loss))
            alpha_values.append(alpha)
            learning_rates.append(lr)
            entropies.append(entropy)
        
        # 创建训练指标数据
        training_data = []
        for i in range(n_steps):
            metrics = {
                'step': i,
                'actor_loss': actor_losses[i],
                'critic_loss': critic_losses[i],
                'attention_loss': attention_losses[i],
                'gnn_loss': gnn_losses[i],
                'total_loss': actor_losses[i] + critic_losses[i],
                'alpha': alpha_values[i],
                'learning_rate': learning_rates[i],
                'entropy': entropies[i],
                'buffer_size': min(10000, i * 10),
                'update_count': i // 10
            }
            training_data.append(metrics)
        
        # 保存为JSON格式
        import json
        
        # 保存指标数据
        metrics_path = os.path.join(log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # 保存loss数据
        loss_data = {
            'actor_loss': actor_losses,
            'critic_loss': critic_losses,
            'attention_loss': attention_losses,
            'gnn_loss': gnn_losses,
            'total_loss': [a + c for a, c in zip(actor_losses, critic_losses)]
        }
        
        losses_path = os.path.join(log_dir, 'losses.json')
        with open(losses_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        # 保存配置信息
        config = {
            'experiment_name': 'demo_training',
            'start_time': datetime.now().isoformat(),
            'hyperparams': {
                'lr': 3e-4,
                'alpha': 0.2,
                'batch_size': 64,
                'buffer_capacity': 10000
            },
            'env_config': {
                'env_name': 'reacher2d',
                'num_links': 3,
                'link_lengths': [60, 40, 30]
            }
        }
        
        config_path = os.path.join(log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 演示训练日志已创建: {log_dir}")
        print(f"   训练步数: {n_steps}")
        print(f"   网络类型: Actor, Critic, Attention, GNN")
        
        return log_dir
    
    def demo_map_elites_visualization(self):
        """演示MAP-Elites可视化"""
        print("\n" + "="*60)
        print("🗺️  MAP-ELITES 可视化演示")
        print("="*60)
        
        # 创建演示数据
        archive_path = self.create_demo_map_elites_data()
        
        # 加载数据到可视化器
        self.map_elites_viz.load_archive(archive_path)
        
        # 生成各种可视化
        print("\n📊 正在生成MAP-Elites可视化...")
        
        # 1. 适应度热力图
        heatmap_path = self.map_elites_viz.create_fitness_heatmap()
        if heatmap_path:
            print(f"✅ 适应度热力图: {heatmap_path}")
        
        # 2. 进化分析
        evolution_path = self.map_elites_viz.create_evolution_analysis()
        if evolution_path:
            print(f"✅ 进化分析图: {evolution_path}")
        
        # 3. 精英个体展示
        elite_path = self.map_elites_viz.create_elite_showcase(top_n=8)
        if elite_path:
            print(f"✅ 精英个体展示: {elite_path}")
        
        # 4. 综合报告
        report_dir = self.map_elites_viz.generate_comprehensive_report()
        if report_dir:
            print(f"✅ 综合报告: {report_dir}")
        
        return True
    
    def demo_network_loss_visualization(self):
        """演示神经网络loss可视化"""
        print("\n" + "="*60)
        print("🧠 神经网络 LOSS 可视化演示")
        print("="*60)
        
        # 创建演示训练日志
        log_dir = self.create_demo_training_logs()
        
        # 加载数据到可视化器
        if not self.loss_viz.load_training_logs(log_dir):
            print("❌ 无法加载演示训练日志")
            return False
        
        # 生成各种可视化
        print("\n📈 正在生成神经网络Loss可视化...")
        
        # 1. Loss曲线图
        curves_path = self.loss_viz.create_loss_curves()
        if curves_path:
            print(f"✅ Loss曲线图: {curves_path}")
        
        # 2. 网络对比图
        comparison_path = self.loss_viz.create_network_comparison()
        if comparison_path:
            print(f"✅ 网络对比图: {comparison_path}")
        
        # 3. 综合报告
        report_dir = self.loss_viz.generate_comprehensive_loss_report()
        if report_dir:
            print(f"✅ Loss分析报告: {report_dir}")
        
        return True
    
    def demo_integrated_visualization(self):
        """演示集成可视化"""
        print("\n" + "="*60)
        print("🎨 集成可视化演示")
        print("="*60)
        
        # 创建集成报告目录
        integrated_dir = os.path.join(self.output_dir, f'integrated_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(integrated_dir, exist_ok=True)
        
        print(f"📁 集成报告目录: {integrated_dir}")
        
        # 1. MAP-Elites可视化
        print("\n🗺️  生成MAP-Elites可视化...")
        archive_path = self.create_demo_map_elites_data()
        self.map_elites_viz.load_archive(archive_path)
        
        map_elites_heatmap = self.map_elites_viz.create_fitness_heatmap(
            save_path=os.path.join(integrated_dir, 'map_elites_heatmap.png')
        )
        
        # 2. 神经网络Loss可视化
        print("\n🧠 生成神经网络Loss可视化...")
        log_dir = self.create_demo_training_logs()
        self.loss_viz.load_training_logs(log_dir)
        
        loss_curves = self.loss_viz.create_loss_curves(
            save_path=os.path.join(integrated_dir, 'network_loss_curves.png')
        )
        
        # 3. 创建综合仪表板
        self._create_integrated_dashboard(integrated_dir)
        
        print(f"✅ 集成可视化完成: {integrated_dir}")
        return integrated_dir
    
    def _create_integrated_dashboard(self, output_dir: str):
        """创建综合仪表板"""
        print("📊 正在创建综合仪表板...")
        
        # 创建一个综合的仪表板图
        fig = plt.figure(figsize=(20, 15))
        
        # 设置整体标题
        fig.suptitle('MAP-Elites + 神经网络训练 综合仪表板', fontsize=20, fontweight='bold')
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. MAP-Elites概览 (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_map_elites_overview(ax1)
        
        # 2. 网络Loss概览 (右上)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_network_loss_overview(ax2)
        
        # 3. 适应度分布 (中左)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_fitness_distribution(ax3)
        
        # 4. Loss趋势 (中右)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_loss_trends_overview(ax4)
        
        # 5. 训练进度指标 (底部)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_training_progress(ax5)
        
        # 保存仪表板
        dashboard_path = os.path.join(output_dir, 'integrated_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 综合仪表板已保存: {dashboard_path}")
    
    def _plot_map_elites_overview(self, ax):
        """绘制MAP-Elites概览"""
        # 模拟数据
        generations = list(range(6))
        best_fitness = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85]
        archive_size = [10, 18, 25, 35, 42, 50]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(generations, best_fitness, 'o-', color='green', linewidth=3, label='最佳适应度')
        line2 = ax2.plot(generations, archive_size, 's-', color='blue', linewidth=3, label='存档大小')
        
        ax.set_xlabel('代数')
        ax.set_ylabel('最佳适应度', color='green')
        ax2.set_ylabel('存档大小', color='blue')
        ax.set_title('MAP-Elites 进化概览')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_network_loss_overview(self, ax):
        """绘制网络Loss概览"""
        # 模拟数据
        steps = np.arange(0, 1000, 10)
        actor_loss = 5.0 * np.exp(-steps/300) + 0.1 + 0.1 * np.random.random(len(steps))
        critic_loss = 10.0 * np.exp(-steps/250) + 0.1 + 0.2 * np.random.random(len(steps))
        
        ax.plot(steps, actor_loss, label='Actor Loss', color='#FF6B6B', alpha=0.8)
        ax.plot(steps, critic_loss, label='Critic Loss', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('Loss值')
        ax.set_title('神经网络训练Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_fitness_distribution(self, ax):
        """绘制适应度分布"""
        # 模拟适应度数据
        np.random.seed(42)
        fitness_values = np.concatenate([
            np.random.beta(2, 5, 20) * 0.4,  # 低适应度
            np.random.beta(3, 3, 20) * 0.6 + 0.2,  # 中等适应度
            np.random.beta(5, 2, 10) * 0.3 + 0.7   # 高适应度
        ])
        
        ax.hist(fitness_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(fitness_values), color='red', linestyle='--', 
                  label=f'平均值: {np.mean(fitness_values):.3f}')
        ax.axvline(np.max(fitness_values), color='green', linestyle='--', 
                  label=f'最大值: {np.max(fitness_values):.3f}')
        
        ax.set_xlabel('适应度')
        ax.set_ylabel('个体数量')
        ax.set_title('适应度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_loss_trends_overview(self, ax):
        """绘制Loss趋势概览"""
        # 模拟多个网络的loss趋势
        steps = np.arange(0, 1000, 5)
        
        networks = {
            'Actor': {'color': '#FF6B6B', 'decay': 300},
            'Critic': {'color': '#4ECDC4', 'decay': 250},
            'Attention': {'color': '#45B7D1', 'decay': 350},
            'GNN': {'color': '#96CEB4', 'decay': 400}
        }
        
        for name, config in networks.items():
            initial_loss = np.random.uniform(3, 8)
            loss = initial_loss * np.exp(-steps/config['decay']) + 0.05
            # 添加噪声
            loss += 0.1 * np.random.random(len(steps))
            
            ax.plot(steps, loss, label=name, color=config['color'], alpha=0.8)
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('Loss值')
        ax.set_title('各网络Loss趋势对比')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_progress(self, ax):
        """绘制训练进度"""
        # 模拟训练指标
        metrics = ['成功率', '平均奖励', '探索效率', '收敛速度', '稳定性']
        current_values = [0.75, 0.68, 0.82, 0.71, 0.79]
        target_values = [0.9, 0.8, 0.9, 0.85, 0.9]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, current_values, width, label='当前值', 
                      color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, target_values, width, label='目标值', 
                      color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('训练指标')
        ax.set_ylabel('指标值')
        ax.set_title('训练进度概览')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
    
    def run_full_demo(self):
        """运行完整演示"""
        print("🎬 启动完整可视化演示")
        print("="*60)
        
        start_time = time.time()
        
        # 1. MAP-Elites可视化演示
        success1 = self.demo_map_elites_visualization()
        
        # 2. 神经网络Loss可视化演示
        success2 = self.demo_network_loss_visualization()
        
        # 3. 集成可视化演示
        integrated_dir = self.demo_integrated_visualization()
        
        end_time = time.time()
        
        # 总结
        print("\n" + "="*60)
        print("🎉 可视化演示完成!")
        print("="*60)
        print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
        print(f"📁 输出目录: {self.output_dir}")
        
        if success1 and success2 and integrated_dir:
            print("✅ 所有可视化功能正常工作")
            print("\n📋 生成的可视化包括:")
            print("   🗺️  MAP-Elites热力图和进化分析")
            print("   🧠 神经网络训练Loss曲线")
            print("   📊 综合仪表板和分析报告")
            print("   📈 实时监控功能")
            
            print(f"\n💡 使用提示:")
            print(f"   查看所有文件: ls -la {self.output_dir}")
            print(f"   查看图片: 使用图片查看器打开 *.png 文件")
            print(f"   查看报告: 阅读 *_report.txt 文件")
            
            return True
        else:
            print("⚠️  部分可视化功能可能存在问题")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAP-Elites和神经网络可视化演示')
    parser.add_argument('--output', type=str, default='./demo_visualizations', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['map-elites', 'networks', 'integrated', 'all'], 
                       default='all', help='演示模式')
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = VisualizationDemo(args.output)
    
    # 根据模式运行演示
    if args.mode == 'map-elites':
        demo.demo_map_elites_visualization()
    elif args.mode == 'networks':
        demo.demo_network_loss_visualization()
    elif args.mode == 'integrated':
        demo.demo_integrated_visualization()
    elif args.mode == 'all':
        demo.run_full_demo()
    
    print("\n🎯 演示完成! 现在你可以:")
    print("1. 在真实训练中使用这些可视化工具")
    print("2. 修改 map_elites_trainer.py 集成可视化")
    print("3. 使用实时监控功能跟踪训练进度")


if __name__ == "__main__":
    main()
