#!/usr/bin/env python3
"""
综合可视化生成脚本
专门为MAP-Elites训练结果生成热力图和神经网络Loss可视化
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def create_map_elites_heatmap_from_results(results_dir: str, save_path: str) -> bool:
    """
    从实验结果创建MAP-Elites热力图
    """
    try:
        # 读取results.json
        results_file = os.path.join(results_dir, 'results.json')
        if not os.path.exists(results_file):
            print(f"⚠️ 结果文件不存在: {results_file}")
            return False
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        if not results_data:
            print("⚠️ 结果数据为空")
            return False
        
        # 提取数据
        num_links = []
        total_lengths = []
        fitness_values = []
        learning_rates = []
        alphas = []
        
        for result in results_data:
            if isinstance(result, dict):
                genotype = result.get('genotype', {})
                if genotype:
                    num_links.append(genotype.get('num_links', 3))
                    link_lengths = genotype.get('link_lengths', [50, 50, 50])
                    total_lengths.append(sum(link_lengths))
                    learning_rates.append(genotype.get('lr', 3e-4))
                    alphas.append(genotype.get('alpha', 0.1))
                    fitness_values.append(result.get('fitness', 0.0))
        
        if not num_links:
            print("⚠️ 没有找到有效的基因型数据")
            return False
        
        # 创建热力图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'MAP-Elites 训练结果热力图分析\n个体数: {len(num_links)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 关节数 vs 总长度 散点图
        scatter1 = ax1.scatter(num_links, total_lengths, c=fitness_values, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='white')
        ax1.set_xlabel('关节数')
        ax1.set_ylabel('总长度 (px)')
        ax1.set_title('关节数 vs 总长度')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='适应度')
        
        # 2. 学习率 vs Alpha
        scatter2 = ax2.scatter(learning_rates, alphas, c=fitness_values, 
                              cmap='plasma', s=100, alpha=0.7, edgecolors='white')
        ax2.set_xlabel('学习率')
        ax2.set_ylabel('Alpha')
        ax2.set_title('学习率 vs Alpha')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='适应度')
        
        # 3. 适应度分布
        ax3.hist(fitness_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('适应度')
        ax3.set_ylabel('个体数量')
        ax3.set_title('适应度分布')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        ax3.axvline(mean_fitness, color='red', linestyle='--', label=f'平均值: {mean_fitness:.3f}')
        ax3.axvline(max_fitness, color='green', linestyle='--', label=f'最大值: {max_fitness:.3f}')
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MAP-Elites热力图已保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成MAP-Elites热力图失败: {e}")
        return False

def create_training_loss_visualization(results_dir: str, save_path: str) -> bool:
    """
    从实验结果创建训练Loss可视化
    """
    try:
        # 读取results.json
        results_file = os.path.join(results_dir, 'results.json')
        if not os.path.exists(results_file):
            print(f"⚠️ 结果文件不存在: {results_file}")
            return False
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        if not results_data:
            print("⚠️ 结果数据为空")
            return False
        
        # 提取训练数据
        episodes = []
        fitness_values = []
        success_rates = []
        avg_rewards = []
        min_distances = []
        generations = []
        
        for i, result in enumerate(results_data):
            if isinstance(result, dict):
                episodes.append(i + 1)
                fitness_values.append(result.get('fitness', 0.0))
                
                # 从训练结果中提取指标
                training_results = result.get('training_results', {})
                success_rates.append(training_results.get('success_rate', 0.0))
                avg_rewards.append(training_results.get('avg_reward', -500.0))
                min_distances.append(training_results.get('min_distance', 500.0))
                
                # 从ID中提取代数信息
                individual_id = result.get('individual_id', '')
                if 'gen_' in individual_id:
                    try:
                        gen_num = int(individual_id.split('_')[1])
                        generations.append(gen_num)
                    except:
                        generations.append(0)
                else:
                    generations.append(0)
        
        if not episodes:
            print("⚠️ 没有找到有效的训练数据")
            return False
        
        # 创建Loss可视化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'神经网络训练Loss分析\n总训练轮次: {len(episodes)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 适应度变化
        ax1.plot(episodes, fitness_values, 'b-o', alpha=0.7, linewidth=2, markersize=4)
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度变化趋势')
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(episodes) > 1:
            z = np.polyfit(episodes, fitness_values, 1)
            p = np.poly1d(z)
            ax1.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'趋势线 (斜率: {z[0]:.4f})')
            ax1.legend()
        
        # 2. 成功率变化
        ax2.plot(episodes, success_rates, 'g-s', alpha=0.7, linewidth=2, markersize=4)
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('成功率')
        ax2.set_title('成功率变化趋势')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. 平均奖励变化
        ax3.plot(episodes, avg_rewards, 'r-^', alpha=0.7, linewidth=2, markersize=4)
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('平均奖励')
        ax3.set_title('平均奖励变化趋势')
        ax3.grid(True, alpha=0.3)
        
        # 4. 最小距离变化
        ax4.plot(episodes, min_distances, 'm-d', alpha=0.7, linewidth=2, markersize=4)
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('最小距离 (px)')
        ax4.set_title('最小距离变化趋势')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 训练Loss可视化已保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成训练Loss可视化失败: {e}")
        return False

def create_generation_analysis(results_dir: str, save_path: str) -> bool:
    """
    创建代际分析可视化
    """
    try:
        # 读取results.json
        results_file = os.path.join(results_dir, 'results.json')
        if not os.path.exists(results_file):
            return False
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # 按代数分组
        generations = {}
        for result in results_data:
            if isinstance(result, dict):
                individual_id = result.get('individual_id', '')
                if 'gen_' in individual_id:
                    try:
                        gen_num = int(individual_id.split('_')[1])
                        if gen_num not in generations:
                            generations[gen_num] = []
                        generations[gen_num].append(result)
                    except:
                        continue
        
        if not generations:
            return False
        
        # 计算每代统计
        gen_numbers = sorted(generations.keys())
        gen_best_fitness = []
        gen_avg_fitness = []
        gen_size = []
        
        for gen_num in gen_numbers:
            gen_data = generations[gen_num]
            fitness_values = [r.get('fitness', 0.0) for r in gen_data]
            gen_best_fitness.append(max(fitness_values))
            gen_avg_fitness.append(np.mean(fitness_values))
            gen_size.append(len(gen_data))
        
        # 创建代际分析图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'MAP-Elites 代际进化分析\n总代数: {len(gen_numbers)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 最佳适应度进化
        ax1.plot(gen_numbers, gen_best_fitness, 'b-o', linewidth=3, markersize=8, label='最佳适应度')
        ax1.plot(gen_numbers, gen_avg_fitness, 'r--s', linewidth=2, markersize=6, label='平均适应度')
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度进化趋势')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 种群大小变化
        ax2.bar(gen_numbers, gen_size, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('个体数量')
        ax2.set_title('每代个体数量')
        ax2.grid(True, alpha=0.3)
        
        # 3. 适应度改善率
        improvement_rates = []
        for i in range(1, len(gen_best_fitness)):
            if gen_best_fitness[i-1] > 0:
                rate = (gen_best_fitness[i] - gen_best_fitness[i-1]) / gen_best_fitness[i-1]
                improvement_rates.append(rate * 100)
            else:
                improvement_rates.append(0)
        
        if improvement_rates:
            ax3.bar(gen_numbers[1:], improvement_rates, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('代数')
            ax3.set_ylabel('改善率 (%)')
            ax3.set_title('适应度改善率')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. 多样性分析（关节数分布）
        all_num_links = []
        for gen_data in generations.values():
            for result in gen_data:
                genotype = result.get('genotype', {})
                if genotype:
                    all_num_links.append(genotype.get('num_links', 3))
        
        if all_num_links:
            unique_links, counts = np.unique(all_num_links, return_counts=True)
            ax4.pie(counts, labels=[f'{int(link)}关节' for link in unique_links], 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('关节数分布')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 代际分析可视化已保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 生成代际分析可视化失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成MAP-Elites综合可视化")
    parser.add_argument("--results-dir", type=str, 
                       default="./experiment_results/session_20250917_160838",
                       help="实验结果目录")
    parser.add_argument("--output-dir", type=str, 
                       default="./map_elites_shared_ppo_results/visualizations",
                       help="输出目录")
    args = parser.parse_args()
    
    print("🎨 开始生成MAP-Elites综合可视化")
    print("=" * 60)
    
    # 检查结果目录
    if not os.path.exists(args.results_dir):
        print(f"❌ 结果目录不存在: {args.results_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    
    # 1. 生成MAP-Elites热力图
    print("\n📊 生成MAP-Elites热力图...")
    heatmap_path = os.path.join(args.output_dir, "map_elites_comprehensive_heatmap.png")
    if create_map_elites_heatmap_from_results(args.results_dir, heatmap_path):
        success_count += 1
    
    # 2. 生成训练Loss可视化
    print("\n🧠 生成训练Loss可视化...")
    loss_path = os.path.join(args.output_dir, "training_loss_comprehensive.png")
    if create_training_loss_visualization(args.results_dir, loss_path):
        success_count += 1
    
    # 3. 生成代际分析
    print("\n🧬 生成代际进化分析...")
    generation_path = os.path.join(args.output_dir, "generation_analysis.png")
    if create_generation_analysis(args.results_dir, generation_path):
        success_count += 1
    
    # 4. 生成总结报告
    print("\n📝 生成总结报告...")
    report_path = os.path.join(args.output_dir, "visualization_summary.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MAP-Elites 可视化总结报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据源: {args.results_dir}\n")
            f.write(f"输出目录: {args.output_dir}\n\n")
            f.write("生成的可视化文件:\n")
            f.write("1. map_elites_comprehensive_heatmap.png - MAP-Elites热力图分析\n")
            f.write("2. training_loss_comprehensive.png - 训练Loss变化分析\n")
            f.write("3. generation_analysis.png - 代际进化分析\n\n")
            f.write(f"成功生成: {success_count}/3 个可视化文件\n")
        
        print(f"✅ 总结报告已保存: {report_path}")
        success_count += 1
    except Exception as e:
        print(f"❌ 生成总结报告失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 可视化生成完成! 成功: {success_count}/4")
    print(f"📁 所有文件保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

