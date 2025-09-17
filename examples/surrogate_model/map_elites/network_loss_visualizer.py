#!/usr/bin/env python3
"""
神经网络训练Loss可视化工具
功能：
1. 可视化各个网络的loss变化（Attention、GNN、PPO、SAC等）
2. 实时监控训练过程
3. 生成综合loss分析报告
4. 支持多网络对比分析
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
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
from typing import Dict, List, Optional, Tuple, Any, Union
import argparse
from datetime import datetime
import glob
import pickle
from collections import defaultdict, deque
import threading
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class NetworkLossVisualizer:
    """神经网络训练Loss可视化器"""
    
    def __init__(self, log_dir: str = "./training_logs", output_dir: str = "./loss_visualizations"):
        """
        初始化可视化器
        
        Args:
            log_dir: 训练日志目录
            output_dir: 输出目录
        """
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.training_data = {}
        self.network_types = ['actor', 'critic', 'attention', 'gnn', 'ppo', 'sac', 'alpha']
        
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
        
        # 颜色映射
        self.color_map = {
            'actor_loss': '#FF6B6B',      # 红色
            'critic_loss': '#4ECDC4',     # 青色
            'attention_loss': '#45B7D1',  # 蓝色
            'gnn_loss': '#96CEB4',        # 绿色
            'ppo_loss': '#FFEAA7',        # 黄色
            'sac_loss': '#DDA0DD',        # 紫色
            'alpha_loss': '#FFA07A',      # 橙色
            'total_loss': '#2C3E50',      # 深灰色
            'entropy': '#E17055',         # 深橙色
            'learning_rate': '#00B894'    # 深绿色
        }
    
    def load_training_logs(self, experiment_path: Optional[str] = None) -> bool:
        """
        加载训练日志数据
        
        Args:
            experiment_path: 实验路径，如果为None则自动查找最新实验
            
        Returns:
            是否成功加载
        """
        if experiment_path is None:
            experiment_path = self._find_latest_experiment()
        
        if not experiment_path:
            print("❌ 未找到训练日志")
            return False
        
        print(f"📊 正在加载训练日志: {experiment_path}")
        
        try:
            # 查找所有可能的日志文件
            log_files = {
                'metrics': os.path.join(experiment_path, 'metrics.json'),
                'losses': os.path.join(experiment_path, 'losses.json'),
                'training_log': os.path.join(experiment_path, 'training_log.json'),
                'config': os.path.join(experiment_path, 'config.json')
            }
            
            # 加载每个文件
            for log_type, log_file in log_files.items():
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        self.training_data[log_type] = json.load(f)
                    print(f"✅ 加载 {log_type}: {len(self.training_data[log_type])} 条记录")
                else:
                    print(f"⚠️  未找到 {log_type} 文件: {log_file}")
            
            # 尝试从pickle文件加载（如果存在）
            pickle_files = glob.glob(os.path.join(experiment_path, '*.pkl'))
            for pickle_file in pickle_files:
                try:
                    with open(pickle_file, 'rb') as f:
                        data = pickle.load(f)
                        filename = os.path.basename(pickle_file).replace('.pkl', '')
                        self.training_data[filename] = data
                        print(f"✅ 加载 {filename} (pickle)")
                except Exception as e:
                    print(f"⚠️  无法加载 {pickle_file}: {e}")
            
            return len(self.training_data) > 0
            
        except Exception as e:
            print(f"❌ 加载训练日志失败: {e}")
            return False
    
    def _find_latest_experiment(self) -> Optional[str]:
        """查找最新的实验目录"""
        if not os.path.exists(self.log_dir):
            print(f"❌ 训练日志目录不存在: {self.log_dir}")
            return None
        
        # 查找所有实验目录
        experiment_dirs = []
        for item in os.listdir(self.log_dir):
            item_path = os.path.join(self.log_dir, item)
            if os.path.isdir(item_path):
                experiment_dirs.append(item_path)
        
        if not experiment_dirs:
            print("❌ 未找到实验目录")
            return None
        
        # 按修改时间排序，返回最新的
        latest_dir = max(experiment_dirs, key=os.path.getmtime)
        print(f"🔍 找到最新实验: {latest_dir}")
        return latest_dir
    
    def create_loss_curves(self, save_path: Optional[str] = None, 
                          figsize: Tuple[int, int] = (16, 12),
                          networks: Optional[List[str]] = None) -> str:
        """
        创建损失曲线图
        
        Args:
            save_path: 保存路径
            figsize: 图片大小
            networks: 要显示的网络类型，如果为None则显示所有
            
        Returns:
            保存的文件路径
        """
        if not self.training_data:
            print("❌ 没有可用的训练数据")
            return None
        
        # 解析训练数据
        parsed_data = self._parse_training_data()
        if not parsed_data:
            print("❌ 无法解析训练数据")
            return None
        
        networks = networks or self.network_types
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('神经网络训练Loss分析', fontsize=16, fontweight='bold')
        
        # 1. 主要Loss曲线
        ax1 = axes[0, 0]
        self._plot_main_losses(ax1, parsed_data, networks)
        
        # 2. 学习率和Alpha变化
        ax2 = axes[0, 1]
        self._plot_hyperparameters(ax2, parsed_data)
        
        # 3. Loss趋势分析
        ax3 = axes[1, 0]
        self._plot_loss_trends(ax3, parsed_data)
        
        # 4. 网络性能指标
        ax4 = axes[1, 1]
        self._plot_performance_metrics(ax4, parsed_data)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'loss_curves_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Loss曲线图已保存: {save_path}")
        
        return save_path
    
    def _parse_training_data(self) -> Dict[str, Any]:
        """解析训练数据"""
        parsed = defaultdict(list)
        
        # 从不同的数据源解析
        for data_type, data in self.training_data.items():
            if data_type == 'metrics' and isinstance(data, list):
                for record in data:
                    if isinstance(record, dict):
                        for key, value in record.items():
                            if isinstance(value, (int, float)):
                                parsed[key].append(value)
            
            elif data_type == 'losses' and isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        parsed[key].extend(values)
            
            elif data_type == 'training_log' and isinstance(data, list):
                for record in data:
                    if isinstance(record, dict) and 'metrics' in record:
                        metrics = record['metrics']
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                parsed[key].append(value)
        
        # 确保所有列表长度一致
        max_length = max(len(values) for values in parsed.values()) if parsed else 0
        for key in list(parsed.keys()):
            if len(parsed[key]) < max_length:
                # 用最后一个值填充
                last_value = parsed[key][-1] if parsed[key] else 0
                parsed[key].extend([last_value] * (max_length - len(parsed[key])))
        
        return dict(parsed)
    
    def _plot_main_losses(self, ax, parsed_data, networks):
        """绘制主要Loss曲线"""
        loss_keys = [key for key in parsed_data.keys() if 'loss' in key.lower()]
        
        if not loss_keys:
            ax.text(0.5, 0.5, '未找到Loss数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('主要Loss曲线')
            return
        
        for loss_key in loss_keys:
            if any(net in loss_key.lower() for net in networks):
                values = parsed_data[loss_key]
                if values:
                    color = self.color_map.get(loss_key, None)
                    steps = range(len(values))
                    ax.plot(steps, values, label=loss_key, color=color, alpha=0.8)
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('Loss值')
        ax.set_title('主要Loss曲线')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 使用对数坐标，更好地显示loss变化
    
    def _plot_hyperparameters(self, ax, parsed_data):
        """绘制超参数变化"""
        param_keys = ['learning_rate', 'alpha', 'entropy']
        
        for param_key in param_keys:
            if param_key in parsed_data:
                values = parsed_data[param_key]
                if values:
                    color = self.color_map.get(param_key, None)
                    steps = range(len(values))
                    ax.plot(steps, values, label=param_key, color=color, alpha=0.8)
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('参数值')
        ax.set_title('超参数变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_loss_trends(self, ax, parsed_data):
        """绘制Loss趋势分析"""
        # 计算滑动平均
        window_size = 100
        trend_data = {}
        
        loss_keys = [key for key in parsed_data.keys() if 'loss' in key.lower()]
        
        for loss_key in loss_keys:
            values = parsed_data[loss_key]
            if len(values) > window_size:
                # 计算滑动平均
                moving_avg = []
                for i in range(window_size, len(values)):
                    avg = np.mean(values[i-window_size:i])
                    moving_avg.append(avg)
                trend_data[loss_key] = moving_avg
        
        if trend_data:
            for loss_key, trend in trend_data.items():
                color = self.color_map.get(loss_key, None)
                steps = range(window_size, window_size + len(trend))
                ax.plot(steps, trend, label=f'{loss_key} (滑动平均)', color=color, alpha=0.8)
        
        ax.set_xlabel('训练步数')
        ax.set_ylabel('Loss值 (滑动平均)')
        ax.set_title(f'Loss趋势分析 (窗口大小: {window_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_performance_metrics(self, ax, parsed_data):
        """绘制性能指标"""
        perf_keys = ['buffer_size', 'update_count', 'grad_norm', 'clip_fraction']
        
        found_metrics = []
        for key in parsed_data.keys():
            if any(perf_key in key.lower() for perf_key in perf_keys):
                found_metrics.append(key)
        
        if not found_metrics:
            # 如果没有找到性能指标，显示loss分布
            loss_keys = [key for key in parsed_data.keys() if 'loss' in key.lower()]
            if loss_keys:
                loss_values = []
                labels = []
                for loss_key in loss_keys[:5]:  # 只显示前5个
                    values = parsed_data[loss_key]
                    if values:
                        loss_values.append(values)
                        labels.append(loss_key)
                
                if loss_values:
                    ax.boxplot(loss_values, labels=labels)
                    ax.set_ylabel('Loss值')
                    ax.set_title('Loss分布')
                    ax.tick_params(axis='x', rotation=45)
        else:
            # 显示性能指标
            for metric_key in found_metrics:
                values = parsed_data[metric_key]
                if values:
                    steps = range(len(values))
                    ax.plot(steps, values, label=metric_key, alpha=0.8)
            
            ax.set_xlabel('训练步数')
            ax.set_ylabel('指标值')
            ax.set_title('性能指标')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def create_network_comparison(self, save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (20, 10)) -> str:
        """
        创建网络对比分析
        
        Args:
            save_path: 保存路径
            figsize: 图片大小
            
        Returns:
            保存的文件路径
        """
        if not self.training_data:
            print("❌ 没有可用的训练数据")
            return None
        
        parsed_data = self._parse_training_data()
        if not parsed_data:
            print("❌ 无法解析训练数据")
            return None
        
        # 找到所有loss相关的键
        loss_keys = [key for key in parsed_data.keys() if 'loss' in key.lower()]
        
        if not loss_keys:
            print("❌ 未找到loss数据")
            return None
        
        # 创建子图 - 每个网络一个子图
        n_networks = len(loss_keys)
        cols = 3
        rows = (n_networks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_networks == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('各网络Loss详细对比', fontsize=16, fontweight='bold')
        
        for i, loss_key in enumerate(loss_keys):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            values = parsed_data[loss_key]
            steps = range(len(values))
            
            # 绘制原始曲线
            color = self.color_map.get(loss_key, f'C{i}')
            ax.plot(steps, values, alpha=0.3, color=color, linewidth=0.5, label='原始数据')
            
            # 绘制滑动平均
            if len(values) > 50:
                window_size = min(50, len(values) // 10)
                moving_avg = []
                for j in range(window_size, len(values)):
                    avg = np.mean(values[j-window_size:j])
                    moving_avg.append(avg)
                
                avg_steps = range(window_size, len(values))
                ax.plot(avg_steps, moving_avg, color=color, linewidth=2, label='滑动平均')
            
            ax.set_title(f'{loss_key}')
            ax.set_xlabel('训练步数')
            ax.set_ylabel('Loss值')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加统计信息
            if values:
                min_val = min(values)
                max_val = max(values)
                final_val = values[-1]
                ax.text(0.05, 0.95, f'最小: {min_val:.4f}\n最大: {max_val:.4f}\n最终: {final_val:.4f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(n_networks, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'network_comparison_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 网络对比图已保存: {save_path}")
        
        return save_path
    
    def create_real_time_monitor(self, experiment_path: str, update_interval: int = 5):
        """
        创建实时监控（实验性功能）
        
        Args:
            experiment_path: 实验路径
            update_interval: 更新间隔（秒）
        """
        print(f"🔄 启动实时监控: {experiment_path}")
        print(f"   更新间隔: {update_interval}秒")
        print("   按 Ctrl+C 停止监控")
        
        # 创建实时图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('实时训练监控', fontsize=16, fontweight='bold')
        
        # 数据存储
        real_time_data = defaultdict(list)
        
        def update_plots():
            """更新图表"""
            # 重新加载数据
            self.load_training_logs(experiment_path)
            parsed_data = self._parse_training_data()
            
            if not parsed_data:
                return
            
            # 清空所有子图
            for ax in axes.flat:
                ax.clear()
            
            # 重新绘制
            self._plot_main_losses(axes[0, 0], parsed_data, self.network_types)
            self._plot_hyperparameters(axes[0, 1], parsed_data)
            self._plot_loss_trends(axes[1, 0], parsed_data)
            self._plot_performance_metrics(axes[1, 1], parsed_data)
            
            plt.tight_layout()
            plt.draw()
        
        # 设置定时更新
        def monitor_loop():
            while True:
                try:
                    update_plots()
                    time.sleep(update_interval)
                except KeyboardInterrupt:
                    print("\n⚠️  实时监控已停止")
                    break
                except Exception as e:
                    print(f"⚠️  更新图表时出错: {e}")
                    time.sleep(update_interval)
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        # 显示图表
        plt.show()
    
    def generate_comprehensive_loss_report(self, report_path: Optional[str] = None) -> str:
        """
        生成综合loss分析报告
        
        Args:
            report_path: 报告保存路径
            
        Returns:
            报告目录路径
        """
        if not self.training_data:
            print("❌ 没有可用的训练数据")
            return None
        
        print("📊 正在生成综合Loss分析报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f'loss_report_{timestamp}')
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Loss曲线图
        curves_path = self.create_loss_curves(
            save_path=os.path.join(report_dir, 'loss_curves.png')
        )
        
        # 2. 网络对比图
        comparison_path = self.create_network_comparison(
            save_path=os.path.join(report_dir, 'network_comparison.png')
        )
        
        # 3. 生成文本报告
        text_report_path = os.path.join(report_dir, 'loss_analysis_report.txt')
        self._generate_loss_text_report(text_report_path)
        
        # 4. 导出原始数据
        data_path = os.path.join(report_dir, 'training_data.json')
        parsed_data = self._parse_training_data()
        with open(data_path, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_data = {}
            for key, values in parsed_data.items():
                if isinstance(values, np.ndarray):
                    serializable_data[key] = values.tolist()
                elif isinstance(values, list):
                    serializable_data[key] = values
                else:
                    serializable_data[key] = str(values)
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ 综合Loss报告已生成: {report_dir}")
        return report_dir
    
    def _generate_loss_text_report(self, report_path: str):
        """生成loss分析文本报告"""
        parsed_data = self._parse_training_data()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("神经网络训练Loss分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本信息
            f.write("📊 训练数据概览\n")
            f.write("-" * 20 + "\n")
            f.write(f"数据源数量: {len(self.training_data)}\n")
            f.write(f"解析指标数量: {len(parsed_data)}\n\n")
            
            # Loss统计分析
            loss_keys = [key for key in parsed_data.keys() if 'loss' in key.lower()]
            
            if loss_keys:
                f.write("🔥 Loss统计分析\n")
                f.write("-" * 20 + "\n")
                
                for loss_key in loss_keys:
                    values = parsed_data[loss_key]
                    if values:
                        f.write(f"\n{loss_key}:\n")
                        f.write(f"  训练步数: {len(values)}\n")
                        f.write(f"  最小值: {min(values):.6f}\n")
                        f.write(f"  最大值: {max(values):.6f}\n")
                        f.write(f"  平均值: {np.mean(values):.6f}\n")
                        f.write(f"  标准差: {np.std(values):.6f}\n")
                        f.write(f"  最终值: {values[-1]:.6f}\n")
                        
                        # 趋势分析
                        if len(values) > 100:
                            first_100 = np.mean(values[:100])
                            last_100 = np.mean(values[-100:])
                            improvement = (first_100 - last_100) / first_100 * 100
                            f.write(f"  改善程度: {improvement:.2f}%\n")
            
            # 超参数分析
            param_keys = ['learning_rate', 'alpha', 'entropy']
            param_data = {key: parsed_data[key] for key in param_keys if key in parsed_data}
            
            if param_data:
                f.write(f"\n📈 超参数变化分析\n")
                f.write("-" * 20 + "\n")
                
                for param_key, values in param_data.items():
                    if values:
                        f.write(f"\n{param_key}:\n")
                        f.write(f"  初始值: {values[0]:.6f}\n")
                        f.write(f"  最终值: {values[-1]:.6f}\n")
                        f.write(f"  平均值: {np.mean(values):.6f}\n")
                        f.write(f"  变化范围: {min(values):.6f} - {max(values):.6f}\n")
            
            f.write(f"\n\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"✅ Loss分析文本报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='神经网络训练Loss可视化工具')
    parser.add_argument('--log-dir', type=str, default='./training_logs', help='训练日志目录')
    parser.add_argument('--output', type=str, default='./loss_visualizations', help='输出目录')
    parser.add_argument('--experiment', type=str, help='指定实验路径')
    parser.add_argument('--mode', type=str, choices=['curves', 'comparison', 'monitor', 'all'], 
                       default='all', help='可视化模式')
    parser.add_argument('--networks', nargs='+', help='指定要分析的网络类型')
    parser.add_argument('--real-time', action='store_true', help='启用实时监控')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = NetworkLossVisualizer(args.log_dir, args.output)
    
    # 加载训练日志
    if not visualizer.load_training_logs(args.experiment):
        print("❌ 无法加载训练日志，退出")
        return
    
    # 根据模式生成可视化
    if args.real_time and args.experiment:
        visualizer.create_real_time_monitor(args.experiment)
    elif args.mode == 'curves':
        visualizer.create_loss_curves(networks=args.networks)
    elif args.mode == 'comparison':
        visualizer.create_network_comparison()
    elif args.mode == 'monitor':
        if args.experiment:
            visualizer.create_real_time_monitor(args.experiment)
        else:
            print("❌ 实时监控需要指定实验路径")
    elif args.mode == 'all':
        visualizer.generate_comprehensive_loss_report()
    
    print("🎉 Loss可视化完成!")


if __name__ == "__main__":
    main()
