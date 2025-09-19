#!/usr/bin/env python3
"""
Individual Reacher成功率分析器
分析训练过程中的成功率变化和性能改善
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def analyze_success_rate(experiment_name):
    """分析individual reacher的成功率"""
    
    performance_file = f"enhanced_multi_network_logs/{experiment_name}_multi_network_loss/performance_losses.csv"
    
    if not os.path.exists(performance_file):
        print(f"❌ 性能文件不存在: {performance_file}")
        return
    
    print(f"📊 分析Individual Reacher成功率 - {experiment_name}")
    print("=" * 60)
    
    # 读取数据
    try:
        df = pd.read_csv(performance_file)
        print(f"✅ 读取到 {len(df)} 条性能记录")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return
    
    if len(df) == 0:
        print("⚠️ 没有性能数据")
        return
    
    # 基础统计
    print(f"\n📈 成功率统计:")
    success_rates = df['success_rate'].dropna()
    if len(success_rates) > 0:
        print(f"   平均成功率: {success_rates.mean():.1f}%")
        print(f"   最高成功率: {success_rates.max():.1f}%")
        print(f"   最低成功率: {success_rates.min():.1f}%")
        
        # 找到成功率突破点
        breakthrough_points = df[df['success_rate'] > 50]
        if len(breakthrough_points) > 0:
            first_success = breakthrough_points.iloc[0]
            print(f"   🎯 首次成功突破: Step {first_success['step']}, 成功率 {first_success['success_rate']:.1f}%")
        
        # 计算成功率改善速度
        if len(success_rates) > 1:
            final_rate = success_rates.iloc[-1]
            initial_rate = success_rates.iloc[0]
            improvement = final_rate - initial_rate
            print(f"   📈 总体改善: {improvement:.1f}% (从 {initial_rate:.1f}% 到 {final_rate:.1f}%)")
    
    # 距离统计
    print(f"\n📏 距离优化统计:")
    distances = df['best_distance'].dropna()
    if len(distances) > 0:
        print(f"   平均最佳距离: {distances.mean():.1f}px")
        print(f"   最短距离: {distances.min():.1f}px")
        print(f"   初始距离: {distances.iloc[0]:.1f}px")
        
        distance_improvement = distances.iloc[0] - distances.min()
        print(f"   📉 距离改善: {distance_improvement:.1f}px")
    
    # 训练效率分析
    print(f"\n⚡ 训练效率分析:")
    if len(df) > 1:
        total_steps = df['step'].max() - df['step'].min()
        print(f"   总训练步数: {total_steps:,} 步")
        
        # 计算到达成功的步数
        success_data = df[df['success_rate'] >= 100]
        if len(success_data) > 0:
            steps_to_success = success_data.iloc[0]['step']
            print(f"   达到100%成功率用时: {steps_to_success:,} 步")
            efficiency = (steps_to_success / total_steps) * 100 if total_steps > 0 else 0
            print(f"   训练效率: {efficiency:.1f}% (达到成功所需步数比例)")
    
    # 生成可视化图表
    try:
        create_success_rate_visualization(df, experiment_name)
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")
    
    print(f"\n✅ 成功率分析完成")

def create_success_rate_visualization(df, experiment_name):
    """创建成功率可视化图表"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 成功率曲线
    ax1.plot(df['step'], df['success_rate'], 'g-', linewidth=2, label='Success Rate', marker='o', markersize=3)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title(f'Individual Reacher Success Rate - {experiment_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加成功率突破点标注
    breakthrough_points = df[df['success_rate'] > 50]
    if len(breakthrough_points) > 0:
        first_success = breakthrough_points.iloc[0]
        ax1.annotate(f'Breakthrough!\nStep {first_success["step"]}\n{first_success["success_rate"]:.1f}%',
                    xy=(first_success['step'], first_success['success_rate']),
                    xytext=(first_success['step'], first_success['success_rate'] + 20),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 距离优化曲线
    ax2.plot(df['step'], df['best_distance'], 'b-', linewidth=2, label='Best Distance', marker='s', markersize=3)
    ax2.plot(df['step'], df['episode_best_distance'], 'r--', linewidth=1, label='Episode Best Distance', alpha=0.7)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Distance (pixels)')
    ax2.set_title(f'Distance Optimization - {experiment_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加成功阈值线（假设20px为成功）
    ax2.axhline(y=20, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Success Threshold (20px)')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = f"enhanced_multi_network_logs/{experiment_name}_multi_network_loss"
    plot_path = os.path.join(output_dir, "success_rate_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 成功率分析图表已保存: {plot_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = "large_scale_with_success_rate"
    
    analyze_success_rate(experiment_name)

