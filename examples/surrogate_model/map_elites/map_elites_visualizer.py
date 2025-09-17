#!/usr/bin/env python3
"""
MAP-Elites可视化工具
功能：
1. 生成MAP-Elites热力图
2. 显示个体分布和适应度
3. 分析进化过程
4. 生成多维度可视化
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
import pickle
from typing import Dict, List, Optional, Tuple, Any
import argparse
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入MAP-Elites相关类
from map_elites_core import MAPElitesArchive, Individual, RobotGenotype, RobotPhenotype

class MAPElitesVisualizer:
    """MAP-Elites可视化器"""
    
    def __init__(self, archive_path: Optional[str] = None, output_dir: str = "./visualizations"):
        """
        初始化可视化器
        
        Args:
            archive_path: 存档文件路径
            output_dir: 输出目录
        """
        self.archive_path = archive_path
        self.output_dir = output_dir
        self.archive = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置matplotlib样式
        try:
            if SEABORN_AVAILABLE:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            else:
                plt.style.use('default')
        except:
            plt.style.use('default')
        
        if archive_path and os.path.exists(archive_path):
            self.load_archive(archive_path)
    
    def load_archive(self, archive_path: str):
        """加载MAP-Elites存档"""
        try:
            with open(archive_path, 'rb') as f:
                self.archive = pickle.load(f)
            print(f"✅ 成功加载存档: {archive_path}")
            print(f"   存档大小: {len(self.archive.archive)}")
            print(f"   总评估次数: {self.archive.total_evaluations}")
            print(f"   当前代数: {getattr(self.archive, 'generation', 'N/A')}")
        except Exception as e:
            print(f"❌ 加载存档失败: {e}")
            self.archive = None
    
    def create_fitness_heatmap(self, save_path: Optional[str] = None, 
                              figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        创建适应度热力图
        
        Args:
            save_path: 保存路径
            figsize: 图片大小
            
        Returns:
            保存的文件路径
        """
        if not self.archive or not self.archive.archive:
            print("❌ 没有可用的存档数据")
            return None
        
        # 收集数据
        individuals = list(self.archive.archive.values())
        
        # 提取维度数据
        num_links = [ind.genotype.num_links for ind in individuals]
        total_lengths = [sum(ind.genotype.link_lengths) for ind in individuals]
        fitness_values = [ind.fitness for ind in individuals]
        
        # 创建网格
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'MAP-Elites 适应度热力图分析\n存档大小: {len(individuals)}, 代数: {getattr(self.archive, "generation", "N/A")}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 关节数 vs 总长度的适应度热力图
        self._plot_2d_heatmap(ax1, num_links, total_lengths, fitness_values,
                             '关节数', '总长度 (px)', '关节数 vs 总长度')
        
        # 2. 学习率分析
        learning_rates = [ind.genotype.lr for ind in individuals]
        alphas = [ind.genotype.alpha for ind in individuals]
        self._plot_2d_heatmap(ax2, learning_rates, alphas, fitness_values,
                             '学习率', 'Alpha', '学习率 vs Alpha', log_x=True)
        
        # 3. 适应度分布直方图
        ax3.hist(fitness_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('适应度')
        ax3.set_ylabel('个体数量')
        ax3.set_title('适应度分布')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        max_fitness = np.max(fitness_values)
        ax3.axvline(mean_fitness, color='red', linestyle='--', label=f'平均: {mean_fitness:.3f}')
        ax3.axvline(max_fitness, color='green', linestyle='--', label=f'最大: {max_fitness:.3f}')
        ax3.legend()
        
        # 4. 形态多样性分析
        joint_counts = {}
        for num in num_links:
            joint_counts[num] = joint_counts.get(num, 0) + 1
        
        joints = list(joint_counts.keys())
        counts = list(joint_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(joints)))
        
        ax4.pie(counts, labels=[f'{j}关节' for j in joints], autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax4.set_title('关节数分布')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'fitness_heatmap_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 适应度热力图已保存: {save_path}")
        
        return save_path
    
    def _plot_2d_heatmap(self, ax, x_data, y_data, fitness_data, 
                        xlabel, ylabel, title, log_x=False, log_y=False):
        """绘制2D热力图"""
        # 创建散点图，颜色表示适应度
        scatter = ax.scatter(x_data, y_data, c=fitness_data, 
                           cmap='viridis', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('适应度', rotation=270, labelpad=15)
    
    def create_evolution_analysis(self, save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (16, 12)) -> str:
        """
        创建进化过程分析图
        
        Args:
            save_path: 保存路径
            figsize: 图片大小
            
        Returns:
            保存的文件路径
        """
        if not self.archive or not self.archive.archive:
            print("❌ 没有可用的存档数据")
            return None
        
        individuals = list(self.archive.archive.values())
        
        # 按代数分组
        generation_data = {}
        for ind in individuals:
            gen = ind.generation
            if gen not in generation_data:
                generation_data[gen] = []
            generation_data[gen].append(ind)
        
        generations = sorted(generation_data.keys())
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('MAP-Elites 进化过程分析', fontsize=16, fontweight='bold')
        
        # 1. 每代最佳适应度趋势
        best_fitness_per_gen = []
        avg_fitness_per_gen = []
        std_fitness_per_gen = []
        
        for gen in generations:
            gen_individuals = generation_data[gen]
            fitness_values = [ind.fitness for ind in gen_individuals]
            best_fitness_per_gen.append(max(fitness_values))
            avg_fitness_per_gen.append(np.mean(fitness_values))
            std_fitness_per_gen.append(np.std(fitness_values))
        
        ax1.plot(generations, best_fitness_per_gen, 'o-', label='最佳适应度', color='green', linewidth=2)
        ax1.plot(generations, avg_fitness_per_gen, 's-', label='平均适应度', color='blue', linewidth=2)
        ax1.fill_between(generations, 
                        np.array(avg_fitness_per_gen) - np.array(std_fitness_per_gen),
                        np.array(avg_fitness_per_gen) + np.array(std_fitness_per_gen),
                        alpha=0.3, color='blue')
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度进化趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 每代个体数量
        individuals_per_gen = [len(generation_data[gen]) for gen in generations]
        ax2.bar(generations, individuals_per_gen, alpha=0.7, color='orange')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('个体数量')
        ax2.set_title('每代个体数量')
        ax2.grid(True, alpha=0.3)
        
        # 3. 形态多样性进化
        diversity_per_gen = []
        for gen in generations:
            gen_individuals = generation_data[gen]
            # 计算关节数的方差作为多样性指标
            joint_counts = [ind.genotype.num_links for ind in gen_individuals]
            diversity = np.std(joint_counts) if len(joint_counts) > 1 else 0
            diversity_per_gen.append(diversity)
        
        ax3.plot(generations, diversity_per_gen, 'o-', color='purple', linewidth=2)
        ax3.set_xlabel('代数')
        ax3.set_ylabel('关节数标准差')
        ax3.set_title('形态多样性进化')
        ax3.grid(True, alpha=0.3)
        
        # 4. 适应度类别分析（如果有fitness_details）
        if hasattr(individuals[0], 'fitness_details') and individuals[0].fitness_details:
            category_counts = {}
            for ind in individuals:
                if hasattr(ind, 'fitness_details') and ind.fitness_details:
                    category = ind.fitness_details.get('category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
            
            ax4.pie(counts, labels=categories, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax4.set_title('适应度类别分布')
        else:
            # 如果没有fitness_details，显示总长度分布
            total_lengths = [sum(ind.genotype.link_lengths) for ind in individuals]
            ax4.hist(total_lengths, bins=20, alpha=0.7, color='lightcoral')
            ax4.set_xlabel('总长度 (px)')
            ax4.set_ylabel('个体数量')
            ax4.set_title('总长度分布')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'evolution_analysis_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 进化分析图已保存: {save_path}")
        
        return save_path
    
    def create_elite_showcase(self, top_n: int = 10, save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (20, 12)) -> str:
        """
        创建精英个体展示
        
        Args:
            top_n: 显示前N个精英个体
            save_path: 保存路径
            figsize: 图片大小
            
        Returns:
            保存的文件路径
        """
        if not self.archive or not self.archive.archive:
            print("❌ 没有可用的存档数据")
            return None
        
        individuals = list(self.archive.archive.values())
        
        # 按适应度排序，取前N个
        top_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)[:top_n]
        
        # 创建图表
        fig, axes = plt.subplots(2, 5, figsize=figsize)
        fig.suptitle(f'前{top_n}名精英个体展示', fontsize=16, fontweight='bold')
        
        for i, ind in enumerate(top_individuals):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            # 绘制机器人形态（简化版）
            self._draw_robot_morphology(ax, ind)
            
            # 添加信息
            info_text = f"#{i+1}\n"
            info_text += f"适应度: {ind.fitness:.3f}\n"
            info_text += f"关节数: {ind.genotype.num_links}\n"
            info_text += f"总长度: {sum(ind.genotype.link_lengths):.0f}px\n"
            info_text += f"学习率: {ind.genotype.lr:.2e}\n"
            info_text += f"代数: {ind.generation}"
            
            if hasattr(ind, 'fitness_details') and ind.fitness_details:
                info_text += f"\n类别: {ind.fitness_details.get('category', 'N/A')}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(top_n, 10):
            row = i // 5
            col = i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'elite_showcase_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 精英个体展示已保存: {save_path}")
        
        return save_path
    
    def _draw_robot_morphology(self, ax, individual):
        """绘制机器人形态（简化版）"""
        link_lengths = individual.genotype.link_lengths
        num_links = individual.genotype.num_links
        
        # 设置坐标系
        ax.set_xlim(-sum(link_lengths), sum(link_lengths))
        ax.set_ylim(-sum(link_lengths), sum(link_lengths))
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制基座
        base_size = 10
        base = patches.Circle((0, 0), base_size, color='red', alpha=0.7)
        ax.add_patch(base)
        
        # 绘制链节（假设初始角度都为0，即水平伸展）
        x, y = 0, 0
        colors = plt.cm.rainbow(np.linspace(0, 1, num_links))
        
        for i, (length, color) in enumerate(zip(link_lengths, colors)):
            # 绘制链节
            line = patches.Rectangle((x, y-2), length, 4, 
                                   color=color, alpha=0.8, edgecolor='black')
            ax.add_patch(line)
            
            # 绘制关节
            joint = patches.Circle((x + length, y), 3, color='black', alpha=0.9)
            ax.add_patch(joint)
            
            x += length
        
        # 添加末端执行器
        end_effector = patches.Circle((x, y), 5, color='blue', alpha=0.8)
        ax.add_patch(end_effector)
    
    def generate_comprehensive_report(self, report_path: Optional[str] = None) -> str:
        """
        生成综合报告
        
        Args:
            report_path: 报告保存路径
            
        Returns:
            报告文件路径
        """
        if not self.archive or not self.archive.archive:
            print("❌ 没有可用的存档数据")
            return None
        
        # 生成所有可视化
        print("🎨 正在生成MAP-Elites综合可视化报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f'map_elites_report_{timestamp}')
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. 适应度热力图
        heatmap_path = self.create_fitness_heatmap(
            save_path=os.path.join(report_dir, 'fitness_heatmap.png')
        )
        
        # 2. 进化分析
        evolution_path = self.create_evolution_analysis(
            save_path=os.path.join(report_dir, 'evolution_analysis.png')
        )
        
        # 3. 精英个体展示
        elite_path = self.create_elite_showcase(
            save_path=os.path.join(report_dir, 'elite_showcase.png')
        )
        
        # 4. 生成文本报告
        text_report_path = os.path.join(report_dir, 'analysis_report.txt')
        self._generate_text_report(text_report_path)
        
        print(f"✅ 综合报告已生成: {report_dir}")
        return report_dir
    
    def _generate_text_report(self, report_path: str):
        """生成文本分析报告"""
        individuals = list(self.archive.archive.values())
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MAP-Elites 训练分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计
            f.write("📊 基本统计信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"存档大小: {len(individuals)}\n")
            f.write(f"总评估次数: {self.archive.total_evaluations}\n")
            f.write(f"当前代数: {getattr(self.archive, 'generation', 'N/A')}\n\n")
            
            # 适应度统计
            fitness_values = [ind.fitness for ind in individuals]
            f.write("🎯 适应度统计\n")
            f.write("-" * 20 + "\n")
            f.write(f"最大适应度: {max(fitness_values):.6f}\n")
            f.write(f"平均适应度: {np.mean(fitness_values):.6f}\n")
            f.write(f"适应度标准差: {np.std(fitness_values):.6f}\n")
            f.write(f"适应度中位数: {np.median(fitness_values):.6f}\n\n")
            
            # 形态多样性
            num_links = [ind.genotype.num_links for ind in individuals]
            total_lengths = [sum(ind.genotype.link_lengths) for ind in individuals]
            
            f.write("🤖 形态多样性\n")
            f.write("-" * 20 + "\n")
            f.write(f"关节数范围: {min(num_links)} - {max(num_links)}\n")
            f.write(f"总长度范围: {min(total_lengths):.1f} - {max(total_lengths):.1f} px\n")
            f.write(f"平均关节数: {np.mean(num_links):.2f}\n")
            f.write(f"平均总长度: {np.mean(total_lengths):.1f} px\n\n")
            
            # 前10名个体
            top_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)[:10]
            f.write("🏆 前10名个体\n")
            f.write("-" * 20 + "\n")
            for i, ind in enumerate(top_individuals):
                f.write(f"#{i+1}: 适应度={ind.fitness:.6f}, "
                       f"关节数={ind.genotype.num_links}, "
                       f"总长度={sum(ind.genotype.link_lengths):.1f}px, "
                       f"代数={ind.generation}\n")
            
            f.write("\n报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"✅ 文本报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAP-Elites可视化工具')
    parser.add_argument('--archive', type=str, help='存档文件路径')
    parser.add_argument('--output', type=str, default='./visualizations', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['heatmap', 'evolution', 'elite', 'all'], 
                       default='all', help='可视化模式')
    
    args = parser.parse_args()
    
    # 如果没有指定存档，尝试查找最新的存档
    if not args.archive:
        possible_paths = [
            './map_elites_archive.pkl',
            './map_elites_training_results/map_elites_archive.pkl',
            '../map_elites_archive.pkl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.archive = path
                print(f"🔍 找到存档文件: {path}")
                break
        
        if not args.archive:
            print("❌ 未找到存档文件，请使用 --archive 指定路径")
            return
    
    # 创建可视化器
    visualizer = MAPElitesVisualizer(args.archive, args.output)
    
    # 根据模式生成可视化
    if args.mode == 'heatmap':
        visualizer.create_fitness_heatmap()
    elif args.mode == 'evolution':
        visualizer.create_evolution_analysis()
    elif args.mode == 'elite':
        visualizer.create_elite_showcase()
    elif args.mode == 'all':
        visualizer.generate_comprehensive_report()
    
    print("🎉 可视化完成!")


if __name__ == "__main__":
    main()
