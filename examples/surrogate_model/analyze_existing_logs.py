#!/usr/bin/env python3
"""
分析现有训练日志脚本
从训练的控制台输出中提取损失信息并生成可视化图表
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
import argparse

def parse_training_log(log_file_path):
    """从训练日志文件中解析损失信息"""
    
    print(f"📖 解析训练日志: {log_file_path}")
    
    # 用于存储解析的数据
    training_data = {
        'steps': [],
        'critic_loss': [],
        'actor_loss': [],
        'alpha_loss': [],
        'alpha': [],
        'q1_mean': [],
        'q2_mean': [],
        'buffer_size': [],
        'entropy_term': [],
        'q_term': [],
        'log_probs_mean': []
    }
    
    # 正则表达式模式
    patterns = {
        'step_pattern': r'Step (\d+).*?Critic Loss: ([\d\.-]+), Actor Loss: ([\d\.-]+), Alpha: ([\d\.-]+), Buffer Size: (\d+)',
        'entropy_pattern': r'Entropy Term \(α\*log_π\): ([\d\.-]+)',
        'q_term_pattern': r'Q Term \(Q值\): ([\d\.-]+)',
        'q_means_pattern': r'q1_mean.*?: ([\d\.-]+).*?q2_mean.*?: ([\d\.-]+)'
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取主要损失信息
        step_matches = re.findall(patterns['step_pattern'], content)
        
        for match in step_matches:
            step, critic_loss, actor_loss, alpha, buffer_size = match
            training_data['steps'].append(int(step))
            training_data['critic_loss'].append(float(critic_loss))
            training_data['actor_loss'].append(float(actor_loss))
            training_data['alpha'].append(float(alpha))
            training_data['buffer_size'].append(int(buffer_size))
        
        # 提取熵项信息
        entropy_matches = re.findall(patterns['entropy_pattern'], content)
        training_data['entropy_term'] = [float(match) for match in entropy_matches]
        
        # 提取Q项信息
        q_term_matches = re.findall(patterns['q_term_pattern'], content)
        training_data['q_term'] = [float(match) for match in q_term_matches]
        
        print(f"✅ 解析完成:")
        print(f"   找到 {len(training_data['steps'])} 个训练步骤记录")
        print(f"   步骤范围: {min(training_data['steps']) if training_data['steps'] else 0} - {max(training_data['steps']) if training_data['steps'] else 0}")
        
        return training_data
        
    except Exception as e:
        print(f"❌ 解析日志失败: {e}")
        return None


def plot_training_analysis(training_data, save_dir):
    """生成训练分析图表"""
    
    if not training_data or not training_data['steps']:
        print("❌ 没有数据可以绘制")
        return
    
    steps = training_data['steps']
    
    # 创建多子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. 主要损失曲线
    ax = axes[0]
    if training_data['critic_loss']:
        ax.plot(steps, training_data['critic_loss'], 'b-', label='Critic Loss', linewidth=2)
    if training_data['actor_loss']:
        ax.plot(steps, training_data['actor_loss'], 'r-', label='Actor Loss', linewidth=2)
    if training_data.get('alpha_loss'):
        ax.plot(steps, training_data['alpha_loss'], 'g-', label='Alpha Loss', linewidth=2)
    
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Alpha 变化
    ax = axes[1]
    if training_data['alpha']:
        ax.plot(steps, training_data['alpha'], 'purple', linewidth=2)
        ax.set_title('Alpha (Temperature) Evolution', fontsize=14)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Alpha Value')
        ax.grid(True, alpha=0.3)
    
    # 3. Buffer Size
    ax = axes[2]
    if training_data['buffer_size']:
        ax.plot(steps, training_data['buffer_size'], 'orange', linewidth=2)
        ax.set_title('Replay Buffer Size', fontsize=14)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Buffer Size')
        ax.grid(True, alpha=0.3)
    
    # 4. Actor Loss 组件分析
    ax = axes[3]
    if training_data['entropy_term'] and training_data['q_term']:
        # 匹配长度
        min_len = min(len(training_data['entropy_term']), len(training_data['q_term']), len(steps))
        entropy_steps = steps[:min_len]
        
        ax.plot(entropy_steps, training_data['entropy_term'][:min_len], 'cyan', 
               label='Entropy Term (α*log_π)', linewidth=2)
        ax.plot(entropy_steps, training_data['q_term'][:min_len], 'magenta', 
               label='Q Term', linewidth=2)
        
        ax.set_title('Actor Loss Components', fontsize=14)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. 损失趋势分析（移动平均）
    ax = axes[4]
    if len(training_data['critic_loss']) > 10:
        # 计算移动平均
        window = min(50, len(training_data['critic_loss']) // 10)
        critic_smooth = np.convolve(training_data['critic_loss'], np.ones(window)/window, mode='valid')
        actor_smooth = np.convolve(training_data['actor_loss'], np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        
        ax.plot(smooth_steps, critic_smooth, 'b-', label=f'Critic Loss (MA{window})', linewidth=3)
        ax.plot(smooth_steps, actor_smooth, 'r-', label=f'Actor Loss (MA{window})', linewidth=3)
        
        ax.set_title('Loss Trends (Moving Average)', fontsize=14)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. 学习进度总结
    ax = axes[5]
    if training_data['critic_loss'] and training_data['actor_loss']:
        # 计算学习进度指标
        early_critic = np.mean(training_data['critic_loss'][:min(len(training_data['critic_loss'])//4, 10)])
        late_critic = np.mean(training_data['critic_loss'][-min(len(training_data['critic_loss'])//4, 10):])
        
        early_actor = np.mean(training_data['actor_loss'][:min(len(training_data['actor_loss'])//4, 10)])
        late_actor = np.mean(training_data['actor_loss'][-min(len(training_data['actor_loss'])//4, 10):])
        
        metrics = ['Early Critic', 'Late Critic', 'Early Actor', 'Late Actor']
        values = [early_critic, late_critic, early_actor, late_actor]
        colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
        
        bars = ax.bar(metrics, values, color=colors)
        ax.set_title('Learning Progress Summary', fontsize=14)
        ax.set_ylabel('Average Loss')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Training Loss Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'training_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 训练分析图表已保存到: {save_path}")
    
    plt.show()
    
    return save_path


def generate_summary_report(training_data, save_dir):
    """生成训练总结报告"""
    
    if not training_data or not training_data['steps']:
        print("❌ 没有数据生成报告")
        return
    
    report_path = os.path.join(save_dir, 'training_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Training Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 基本信息
        f.write("基本训练信息:\n")
        f.write(f"  总训练步数: {training_data['steps'][-1] if training_data['steps'] else 0}\n")
        f.write(f"  记录的步骤数: {len(training_data['steps'])}\n")
        f.write(f"  步骤范围: {min(training_data['steps'])} - {max(training_data['steps'])}\n\n")
        
        # 损失统计
        for metric_name in ['critic_loss', 'actor_loss', 'alpha']:
            if metric_name in training_data and training_data[metric_name]:
                values = training_data[metric_name]
                f.write(f"{metric_name} 统计:\n")
                f.write(f"  最终值: {values[-1]:.6f}\n")
                f.write(f"  平均值: {np.mean(values):.6f}\n")
                f.write(f"  标准差: {np.std(values):.6f}\n")
                f.write(f"  最小值: {np.min(values):.6f}\n")
                f.write(f"  最大值: {np.max(values):.6f}\n")
                
                # 计算改进情况
                if len(values) > 10:
                    early_avg = np.mean(values[:len(values)//4])
                    late_avg = np.mean(values[-len(values)//4:])
                    improvement = (early_avg - late_avg) / early_avg * 100
                    f.write(f"  改进程度: {improvement:.2f}%\n")
                f.write("\n")
        
        # Buffer使用情况
        if training_data['buffer_size']:
            f.write("Replay Buffer 使用情况:\n")
            f.write(f"  最终Buffer大小: {training_data['buffer_size'][-1]}\n")
            f.write(f"  平均Buffer大小: {np.mean(training_data['buffer_size']):.0f}\n")
            f.write(f"  Buffer增长率: {(training_data['buffer_size'][-1] - training_data['buffer_size'][0]) / len(training_data['buffer_size']):.1f} per step\n\n")
        
        # 学习稳定性分析
        if len(training_data['critic_loss']) > 20:
            recent_critic = training_data['critic_loss'][-10:]
            critic_stability = np.std(recent_critic) / np.mean(recent_critic)
            f.write("学习稳定性分析:\n")
            f.write(f"  最近10步Critic Loss变异系数: {critic_stability:.4f}\n")
            if critic_stability < 0.1:
                f.write("  -> 训练已收敛，损失稳定\n")
            elif critic_stability < 0.3:
                f.write("  -> 训练基本稳定，仍有小幅波动\n")
            else:
                f.write("  -> 训练尚不稳定，损失波动较大\n")
    
    print(f"📋 训练分析报告已保存到: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="分析训练日志并生成图表")
    parser.add_argument('--log-file', type=str, required=True,
                        help='训练日志文件路径')
    parser.add_argument('--output-dir', type=str, default='./log_analysis_output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析日志
    training_data = parse_training_log(args.log_file)
    
    if training_data:
        # 生成图表
        plot_training_analysis(training_data, args.output_dir)
        
        # 生成报告
        generate_summary_report(training_data, args.output_dir)
        
        # 保存解析的数据
        data_path = os.path.join(args.output_dir, 'parsed_training_data.json')
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"💾 解析的数据已保存到: {data_path}")
        
        print(f"\n✅ 分析完成，所有结果已保存到: {args.output_dir}")
    else:
        print("❌ 日志解析失败，无法生成分析结果")


if __name__ == "__main__":
    main() 