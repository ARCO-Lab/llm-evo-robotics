#!/usr/bin/env python3
"""
从CSV文件生成MAP-Elites可视化
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def create_visualizations_from_csv(csv_file: str, output_dir: str) -> int:
    """
    从CSV文件创建可视化
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"📊 读取CSV文件: {len(df)} 条记录")
        print(f"📊 列名: {list(df.columns)}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        
        # 1. MAP-Elites热力图
        if all(col in df.columns for col in ['num_links', 'total_length', 'fitness']):
            print("📊 生成MAP-Elites热力图...")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'MAP-Elites 训练结果分析\\n个体数: {len(df)}', 
                        fontsize=16, fontweight='bold')
            
            # 散点图：关节数 vs 总长度
            scatter1 = ax1.scatter(df['num_links'], df['total_length'], 
                                 c=df['fitness'], cmap='viridis', 
                                 s=100, alpha=0.7, edgecolors='white')
            ax1.set_xlabel('关节数')
            ax1.set_ylabel('总长度 (px)')
            ax1.set_title('关节数 vs 总长度')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='适应度')
            
            # 学习率分析（如果有的话）
            if 'lr' in df.columns and 'alpha' in df.columns:
                scatter2 = ax2.scatter(df['lr'], df['alpha'], 
                                     c=df['fitness'], cmap='plasma', 
                                     s=100, alpha=0.7, edgecolors='white')
                ax2.set_xlabel('学习率')
                ax2.set_ylabel('Alpha')
                ax2.set_title('学习率 vs Alpha')
                ax2.set_xscale('log')
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter2, ax=ax2, label='适应度')
            else:
                ax2.text(0.5, 0.5, '学习率数据不可用', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            # 适应度分布
            ax3.hist(df['fitness'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('适应度')
            ax3.set_ylabel('个体数量')
            ax3.set_title('适应度分布')
            ax3.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_fitness = df['fitness'].mean()
            max_fitness = df['fitness'].max()
            ax3.axvline(mean_fitness, color='red', linestyle='--', 
                       label=f'平均值: {mean_fitness:.3f}')
            ax3.axvline(max_fitness, color='green', linestyle='--', 
                       label=f'最大值: {max_fitness:.3f}')
            ax3.legend()
            
            # 关节数分布
            unique_links, counts = np.unique(df['num_links'], return_counts=True)
            ax4.bar(unique_links, counts, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.set_xlabel('关节数')
            ax4.set_ylabel('个体数量')
            ax4.set_title('关节数分布')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            heatmap_path = os.path.join(output_dir, "map_elites_heatmap_from_csv.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ MAP-Elites热力图已保存: {heatmap_path}")
            success_count += 1
        
        # 2. 训练指标可视化
        if 'success_rate' in df.columns:
            print("🧠 生成训练指标可视化...")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'训练指标分析\\n总个体数: {len(df)}', 
                        fontsize=16, fontweight='bold')
            
            # 适应度vs成功率
            scatter1 = ax1.scatter(df['fitness'], df['success_rate'], 
                                 s=100, alpha=0.7, color='blue', edgecolors='white')
            ax1.set_xlabel('适应度')
            ax1.set_ylabel('成功率')
            ax1.set_title('适应度 vs 成功率')
            ax1.grid(True, alpha=0.3)
            
            # 成功率分布
            ax2.hist(df['success_rate'], bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('成功率')
            ax2.set_ylabel('个体数量')
            ax2.set_title('成功率分布')
            ax2.grid(True, alpha=0.3)
            
            # 如果有奖励数据
            if 'avg_reward' in df.columns:
                ax3.scatter(df['fitness'], df['avg_reward'], 
                           s=100, alpha=0.7, color='red', edgecolors='white')
                ax3.set_xlabel('适应度')
                ax3.set_ylabel('平均奖励')
                ax3.set_title('适应度 vs 平均奖励')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '奖励数据不可用', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # 代数分析（如果有的话）
            if 'generation' in df.columns:
                gen_stats = df.groupby('generation')['fitness'].agg(['mean', 'max', 'count'])
                ax4.plot(gen_stats.index, gen_stats['max'], 'b-o', label='最佳适应度', linewidth=2)
                ax4.plot(gen_stats.index, gen_stats['mean'], 'r--s', label='平均适应度', linewidth=2)
                ax4.set_xlabel('代数')
                ax4.set_ylabel('适应度')
                ax4.set_title('代际进化趋势')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, '代数数据不可用', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            
            metrics_path = os.path.join(output_dir, "training_metrics_from_csv.png")
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 训练指标可视化已保存: {metrics_path}")
            success_count += 1
        
        # 3. 详细统计报告
        print("📝 生成统计报告...")
        report_path = os.path.join(output_dir, "detailed_statistics_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MAP-Elites 详细统计报告\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"数据源: {csv_file}\\n")
            f.write(f"总个体数: {len(df)}\\n\\n")
            
            # 基础统计
            f.write("基础统计信息:\\n")
            f.write("-" * 30 + "\\n")
            if 'fitness' in df.columns:
                f.write(f"适应度 - 平均: {df['fitness'].mean():.4f}, 最大: {df['fitness'].max():.4f}, 最小: {df['fitness'].min():.4f}\\n")
            if 'success_rate' in df.columns:
                f.write(f"成功率 - 平均: {df['success_rate'].mean():.4f}, 最大: {df['success_rate'].max():.4f}\\n")
            if 'num_links' in df.columns:
                f.write(f"关节数 - 范围: {df['num_links'].min()}-{df['num_links'].max()}\\n")
            if 'total_length' in df.columns:
                f.write(f"总长度 - 平均: {df['total_length'].mean():.1f}px, 范围: {df['total_length'].min():.1f}-{df['total_length'].max():.1f}px\\n")
            
            # 代数统计（如果有的话）
            if 'generation' in df.columns:
                f.write("\\n代数统计:\\n")
                f.write("-" * 30 + "\\n")
                gen_stats = df.groupby('generation').agg({
                    'fitness': ['count', 'mean', 'max'],
                    'success_rate': 'mean' if 'success_rate' in df.columns else 'count'
                })
                f.write(gen_stats.to_string())
                f.write("\\n")
            
            # 关节数分布
            if 'num_links' in df.columns:
                f.write("\\n关节数分布:\\n")
                f.write("-" * 30 + "\\n")
                link_dist = df['num_links'].value_counts().sort_index()
                for links, count in link_dist.items():
                    f.write(f"{int(links)}关节: {count}个 ({count/len(df)*100:.1f}%)\\n")
        
        print(f"✅ 统计报告已保存: {report_path}")
        success_count += 1
        
        return success_count
        
    except Exception as e:
        print(f"❌ 处理CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从CSV生成MAP-Elites可视化")
    parser.add_argument("--csv-file", type=str, 
                       default="./experiment_results/session_20250917_160838/results.csv",
                       help="CSV结果文件")
    parser.add_argument("--output-dir", type=str, 
                       default="./map_elites_shared_ppo_results/visualizations",
                       help="输出目录")
    args = parser.parse_args()
    
    print("🎨 从CSV文件生成MAP-Elites可视化")
    print("=" * 60)
    
    if not os.path.exists(args.csv_file):
        print(f"❌ CSV文件不存在: {args.csv_file}")
        return
    
    success_count = create_visualizations_from_csv(args.csv_file, args.output_dir)
    
    print("\\n" + "=" * 60)
    print(f"🎉 可视化生成完成! 成功: {success_count}/3")
    print(f"📁 所有文件保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

