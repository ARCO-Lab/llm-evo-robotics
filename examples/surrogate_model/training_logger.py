#!/usr/bin/env python3
"""
训练损失记录和监控系统
功能：
- 实时记录各网络的训练损失
- 生成损失曲线图表
- 提供损失统计分析
- 支持多种保存格式(CSV, JSON, PNG)
- 支持实时监控和预警
"""

import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, deque
from datetime import datetime
import pickle
import sys # Added for sys.version_info


class TrainingLogger:
    """训练损失记录器"""
    
    def __init__(self, log_dir="training_logs", experiment_name=None, hyperparams=None, env_config=None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        
        # 创建目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 损失历史记录
        self.loss_history = defaultdict(list)
        self.step_history = []
        self.time_history = []
        self.episode_history = []
        
        # 实时统计
        self.recent_losses = defaultdict(list)
        self.max_recent_size = 100
        self.start_time = time.time()
        
        # 🚀 NEW: 增强的配置信息
        self.config = {
            'experiment_info': {
                'experiment_name': self.experiment_name,
                'start_time': datetime.now().isoformat(),
                'log_dir': self.experiment_dir,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': os.name
            },
            'hyperparams': hyperparams or {},
            'env_config': env_config or {},
            'training_metrics': {
                'total_steps': 0,
                'total_episodes': 0,
                'training_time_hours': 0,
                'avg_steps_per_second': 0
            }
        }
        
        print(f"📊 TrainingLogger 初始化完成")
        print(f"   实验名称: {self.experiment_name}")
        print(f"   日志目录: {self.experiment_dir}")
        
        # 保存配置
        self.save_config()
    
    def update_hyperparams(self, hyperparams):
        """更新超参数信息"""
        self.config['hyperparams'].update(hyperparams)
        self.save_config()
    
    def update_env_config(self, env_config):
        """更新环境配置信息"""
        self.config['env_config'].update(env_config)
        self.save_config()
    
    def update_training_metrics(self, metrics):
        """更新训练指标"""
        self.config['training_metrics'].update(metrics)
        self.save_config()
    
    def log_step(self, step, metrics, episode=None):
        """记录一个训练步骤的损失"""
        current_time = time.time() - self.start_time
        
        # 记录基本信息
        self.step_history.append(step)
        self.time_history.append(current_time)
        if episode is not None:
            self.episode_history.append(episode)
        
        # 记录所有指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.loss_history[key].append(float(value))
                self.recent_losses[key].append(float(value))
                
                # 🔧 手动管理recent_losses大小
                if len(self.recent_losses[key]) > self.max_recent_size:
                    self.recent_losses[key] = self.recent_losses[key][-self.max_recent_size:]
        
        # 定期保存
        if step % 100 == 0:
            self.save_logs()
    
    def log_episode(self, episode, episode_metrics):
        """记录episode级别的指标"""
        for key, value in episode_metrics.items():
            episode_key = f"episode_{key}"
            if isinstance(value, (int, float)):
                self.loss_history[episode_key].append(float(value))
    
    def get_recent_stats(self, metric_name, window=50):
        """获取最近指标的统计信息"""
        if metric_name not in self.recent_losses:
            return None
        
        recent_values = list(self.recent_losses[metric_name])[-window:]
        if not recent_values:
            return None
        
        return {
            'mean': np.mean(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'count': len(recent_values),
            'trend': self._calculate_trend(recent_values)
        }
    
    def _calculate_trend(self, values, window=20):
        """计算趋势（上升/下降/稳定）"""
        if len(values) < window:
            return 'insufficient_data'
        
        recent = values[-window:]
        earlier = values[-2*window:-window] if len(values) >= 2*window else values[:-window]
        
        if len(earlier) == 0:
            return 'insufficient_data'
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        
        diff_ratio = (recent_mean - earlier_mean) / abs(earlier_mean) if earlier_mean != 0 else 0
        
        if diff_ratio > 0.05:
            return 'increasing'
        elif diff_ratio < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def print_current_stats(self, step, detailed=False):
        """打印当前的统计信息"""
        print(f"\n📊 Step {step} 训练统计 (实验: {self.experiment_name})")
        print(f"   运行时间: {self.time_history[-1]/3600:.2f} 小时")
        
        # 主要损失指标
        main_metrics = ['critic_loss', 'actor_loss', 'alpha_loss', 'alpha', 'lr']
        for metric in main_metrics:
            stats = self.get_recent_stats(metric)
            if stats:
                trend_emoji = {'increasing': '📈', 'decreasing': '📉', 'stable': '➡️'}.get(stats['trend'], '❓')
                print(f"   {metric:12}: {stats['mean']:8.4f} ± {stats['std']:6.4f} {trend_emoji}")
        
        if detailed:
            # 详细指标
            other_metrics = [k for k in self.loss_history.keys() if k not in main_metrics]
            if other_metrics:
                print(f"   详细指标:")
                for metric in other_metrics:
                    stats = self.get_recent_stats(metric)
                    if stats:
                        print(f"     {metric:15}: {stats['mean']:8.4f} ± {stats['std']:6.4f}")
    
    def plot_losses(self, save_path=None, show=True, recent_steps=None):
        """绘制损失曲线"""
        if not self.step_history:
            print("❌ 没有数据可以绘制")
            return
        
        # 确定要显示的步数范围
        if recent_steps:
            start_idx = max(0, len(self.step_history) - recent_steps)
            steps = self.step_history[start_idx:]
            plot_data = {k: v[start_idx:] for k, v in self.loss_history.items()}
        else:
            steps = self.step_history
            plot_data = self.loss_history
        
        # 分组绘制
        loss_groups = {
            'SAC Losses': ['critic_loss', 'actor_loss', 'alpha_loss'],
            'Q Values': ['q1_mean', 'q2_mean'],
            'Policy Metrics': ['alpha', 'entropy_term', 'q_term'],
            'Episode Metrics': [k for k in plot_data.keys() if k.startswith('episode_')]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for idx, (group_name, metrics) in enumerate(loss_groups.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            plotted_any = False
            
            for i, metric in enumerate(metrics):
                if metric in plot_data and len(plot_data[metric]) > 0:
                    # 确保steps和data长度匹配
                    data = plot_data[metric]
                    if len(steps) != len(data):
                        # 取较短的长度
                        min_len = min(len(steps), len(data))
                        steps_to_use = steps[:min_len]
                        data_to_use = data[:min_len]
                    else:
                        steps_to_use = steps
                        data_to_use = data
                    
                    if len(steps_to_use) > 0 and len(data_to_use) > 0:
                        ax.plot(steps_to_use, data_to_use, 
                               label=metric, color=colors[i % len(colors)], linewidth=1.5)
                        plotted_any = True
            
            if plotted_any:
                ax.set_title(group_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Value')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # 添加趋势线（对主要损失）
                if group_name == 'SAC Losses':
                    for metric in ['critic_loss', 'actor_loss']:
                        if metric in plot_data and len(plot_data[metric]) > 10:
                            # 简单移动平均
                            values = plot_data[metric]
                            window = min(50, len(values) // 10)
                            if window > 1:
                                smooth = np.convolve(values, np.ones(window)/window, mode='valid')
                                smooth_steps = steps[window-1:]
                                
                                # 确保smooth_steps和smooth长度匹配
                                if len(smooth_steps) != len(smooth):
                                    min_len = min(len(smooth_steps), len(smooth))
                                    smooth_steps = smooth_steps[:min_len]
                                    smooth = smooth[:min_len]
                                
                                if len(smooth_steps) > 0 and len(smooth) > 0:
                                    ax.plot(smooth_steps, smooth, '--', alpha=0.7, linewidth=2)
            else:
                ax.text(0.5, 0.5, f'No data for\n{group_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(group_name)
        
        plt.suptitle(f'Training Progress - {self.experiment_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.experiment_dir, f'training_curves_step_{steps[-1]}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 损失曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def save_logs(self):
        """保存日志到文件"""
        # 保存为CSV
        csv_path = os.path.join(self.experiment_dir, 'training_log.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = ['step', 'time'] + list(self.loss_history.keys())
            if self.episode_history:
                headers.insert(2, 'episode')
            writer.writerow(headers)
            
            # 写入数据
            for i, step in enumerate(self.step_history):
                row = [step, self.time_history[i]]
                if self.episode_history and i < len(self.episode_history):
                    row.append(self.episode_history[i])
                elif self.episode_history:
                    row.append('')
                
                for metric in self.loss_history.keys():
                    if i < len(self.loss_history[metric]):
                        row.append(self.loss_history[metric][i])
                    else:
                        row.append('')
                writer.writerow(row)
        
        # 保存为JSON
        json_path = os.path.join(self.experiment_dir, 'training_log.json')
        log_data = {
            'config': self.config,
            'steps': self.step_history,
            'times': self.time_history,
            'episodes': self.episode_history,
            'metrics': dict(self.loss_history)
        }
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # 保存为Pickle（完整对象）
        pickle_path = os.path.join(self.experiment_dir, 'training_logger.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
    
    def save_config(self):
        """保存配置信息"""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def generate_report(self):
        """生成详细的训练报告"""
        if not self.step_history:
            print("❌ 没有数据生成报告")
            return
        
        report_path = os.path.join(self.experiment_dir, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Training Report - {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # 🚀 NEW: 实验基本信息
            f.write("📋 实验基本信息:\n")
            exp_info = self.config.get('experiment_info', {})
            f.write(f"  实验名称: {exp_info.get('experiment_name', 'N/A')}\n")
            f.write(f"  开始时间: {exp_info.get('start_time', 'N/A')}\n")
            f.write(f"  结束时间: {datetime.now().isoformat()}\n")
            f.write(f"  Python版本: {exp_info.get('python_version', 'N/A')}\n")
            f.write(f"  平台: {exp_info.get('platform', 'N/A')}\n")
            f.write(f"  日志目录: {exp_info.get('log_dir', 'N/A')}\n\n")
            
            # 🚀 NEW: 训练配置和超参数
            f.write("⚙️ 训练超参数:\n")
            hyperparams = self.config.get('hyperparams', {})
            if hyperparams:
                # SAC相关参数
                sac_params = {k: v for k, v in hyperparams.items() if k.startswith('sac_') or k in ['lr', 'alpha', 'batch_size', 'buffer_capacity']}
                if sac_params:
                    f.write("  SAC算法参数:\n")
                    for key, value in sac_params.items():
                        f.write(f"    {key}: {value}\n")
                
                # 训练相关参数
                train_params = {k: v for k, v in hyperparams.items() if k in ['warmup_steps', 'update_frequency', 'num_processes', 'seed']}
                if train_params:
                    f.write("  训练流程参数:\n")
                    for key, value in train_params.items():
                        f.write(f"    {key}: {value}\n")
                
                # 其他参数
                other_params = {k: v for k, v in hyperparams.items() if k not in sac_params and k not in train_params}
                if other_params:
                    f.write("  其他参数:\n")
                    for key, value in other_params.items():
                        f.write(f"    {key}: {value}\n")
            else:
                f.write("  未记录超参数信息\n")
            f.write("\n")
            
            # 🚀 NEW: 环境配置
            f.write("🌍 环境配置:\n")
            env_config = self.config.get('env_config', {})
            if env_config:
                for key, value in env_config.items():
                    if isinstance(value, (list, tuple)):
                        f.write(f"  {key}: {list(value)}\n")
                    elif isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"    {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                f.write("  未记录环境配置信息\n")
            f.write("\n")
            
            # 🚀 NEW: 训练统计概览
            f.write("📊 训练统计概览:\n")
            total_time_hours = self.time_history[-1] / 3600 if self.time_history else 0
            f.write(f"  总训练步数: {self.step_history[-1] if self.step_history else 0}\n")
            f.write(f"  总训练时间: {total_time_hours:.2f} 小时\n")
            f.write(f"  平均训练速度: {self.step_history[-1]/self.time_history[-1]:.2f} 步/秒\n")
            
            if self.episode_history:
                f.write(f"  总Episode数: {max(self.episode_history) if self.episode_history else 0}\n")
                f.write(f"  平均Episode长度: {self.step_history[-1]/max(self.episode_history):.1f} 步\n")
            f.write("\n")
            
            # 🚀 ENHANCED: 损失统计分析
            f.write("📈 训练损失分析:\n")
            loss_metrics = ['critic_loss', 'actor_loss', 'alpha_loss', 'alpha']
            
            for metric_name in loss_metrics:
                if metric_name in self.loss_history:
                    values = self.loss_history[metric_name]
                    f.write(f"  {metric_name}:\n")
                    f.write(f"    最终值: {values[-1]:.6f}\n")
                    f.write(f"    平均值: {np.mean(values):.6f}\n")
                    f.write(f"    标准差: {np.std(values):.6f}\n")
                    f.write(f"    最小值: {np.min(values):.6f}\n")
                    f.write(f"    最大值: {np.max(values):.6f}\n")
                    
                    # 趋势分析
                    if len(values) > 100:
                        recent_values = values[-100:]
                        early_values = values[:100]
                        trend = np.mean(recent_values) - np.mean(early_values)
                        f.write(f"    趋势(近期vs早期): {trend:+.6f}\n")
                    f.write("\n")
            
            # 🚀 NEW: Q值统计
            q_metrics = ['q1_mean', 'q2_mean', 'buffer_size', 'learning_rate']
            f.write("🎯 训练细节统计:\n")
            for metric_name in q_metrics:
                if metric_name in self.loss_history:
                    values = self.loss_history[metric_name]
                    f.write(f"  {metric_name}:\n")
                    f.write(f"    最终值: {values[-1]:.6f}\n")
                    f.write(f"    平均值: {np.mean(values):.6f}\n")
                    if metric_name in ['q1_mean', 'q2_mean']:
                        f.write(f"    最小值: {np.min(values):.6f}\n")
                        f.write(f"    最大值: {np.max(values):.6f}\n")
                    f.write("\n")
            
            # 🚀 NEW: 训练建议
            f.write("💡 训练分析建议:\n")
            if 'critic_loss' in self.loss_history:
                final_critic_loss = self.loss_history['critic_loss'][-1]
                if final_critic_loss < 1.0:
                    f.write("  ✅ Critic Loss表现优秀 (< 1.0)\n")
                elif final_critic_loss < 2.0:
                    f.write("  👍 Critic Loss表现良好 (< 2.0)\n")
                elif final_critic_loss < 5.0:
                    f.write("  ⚠️ Critic Loss较高 (< 5.0)，建议检查奖励函数\n")
                else:
                    f.write("  ❌ Critic Loss过高 (>= 5.0)，需要调整超参数或奖励函数\n")
            
            if 'actor_loss' in self.loss_history:
                final_actor_loss = self.loss_history['actor_loss'][-1]
                if final_actor_loss < -2.0:
                    f.write("  ✅ Actor Loss表现优秀 (< -2.0)\n")
                elif final_actor_loss < 0:
                    f.write("  👍 Actor Loss表现良好 (< 0)\n")
                else:
                    f.write("  ⚠️ Actor Loss为正值，策略可能需要改进\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
        
        print(f"📋 详细训练报告已保存到: {report_path}")
        return report_path
    
    @classmethod
    def load_logger(cls, experiment_dir):
        """从保存的文件加载logger"""
        pickle_path = os.path.join(experiment_dir, 'training_logger.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"❌ 未找到保存的logger: {pickle_path}")
            return None


class RealTimeMonitor:
    """实时监控类"""
    
    def __init__(self, logger, alert_thresholds=None):
        self.logger = logger
        self.alert_thresholds = alert_thresholds or {
            'critic_loss': {'max': 10.0, 'nan_check': True},
            'actor_loss': {'max': 5.0, 'nan_check': True},
            'alpha_loss': {'max': 2.0, 'nan_check': True},
        }
        self.alert_history = []
    
    def check_alerts(self, step, metrics):
        """检查是否有异常情况需要警报"""
        alerts = []
        
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # 检查NaN
                if thresholds.get('nan_check', False) and (np.isnan(value) or np.isinf(value)):
                    alerts.append(f"⚠️ {metric_name} 出现 NaN/Inf: {value}")
                
                # 检查超出阈值
                if 'max' in thresholds and value > thresholds['max']:
                    alerts.append(f"⚠️ {metric_name} 超出最大阈值: {value:.4f} > {thresholds['max']}")
                
                if 'min' in thresholds and value < thresholds['min']:
                    alerts.append(f"⚠️ {metric_name} 低于最小阈值: {value:.4f} < {thresholds['min']}")
        
        # 检查趋势异常
        for metric_name in ['critic_loss', 'actor_loss']:
            stats = self.logger.get_recent_stats(metric_name, window=50)
            if stats and stats['trend'] == 'increasing' and stats['mean'] > 1.0:
                alerts.append(f"📈 {metric_name} 持续上升趋势，当前均值: {stats['mean']:.4f}")
        
        # 记录和显示警报
        if alerts:
            self.alert_history.extend([(step, alert) for alert in alerts])
            print(f"\n🚨 Step {step} 监控警报:")
            for alert in alerts:
                print(f"   {alert}")
        
        return alerts


# 示例用法函数
def demo_usage():
    """演示如何使用TrainingLogger"""
    
    # 创建logger
    logger = TrainingLogger(experiment_name="sac_reacher2d_demo")
    monitor = RealTimeMonitor(logger)
    
    # 模拟训练过程
    print("🚀 开始模拟训练...")
    
    for step in range(1000):
        # 模拟损失值
        metrics = {
            'critic_loss': 1.0 + 0.5 * np.exp(-step/200) + np.random.normal(0, 0.1),
            'actor_loss': 0.5 + 0.3 * np.exp(-step/300) + np.random.normal(0, 0.05),
            'alpha_loss': 0.2 + np.random.normal(0, 0.02),
            'alpha': 0.2 + 0.1 * np.sin(step/100),
            'q1_mean': 2.0 + np.random.normal(0, 0.2),
            'q2_mean': 2.1 + np.random.normal(0, 0.2),
            'buffer_size': min(10000, step * 10)
        }
        
        # 记录
        episode = step // 50  # 假设每50步一个episode
        logger.log_step(step, metrics, episode)
        
        # 监控
        alerts = monitor.check_alerts(step, metrics)
        
        # 定期打印统计
        # if step % 200 == 0 and step > 0:
        #     logger.print_current_stats(step, detailed=True)
        
        # 定期保存和绘图
        if step % 500 == 0 and step > 0:
            logger.plot_losses(recent_steps=500, show=False)
    
    # 最终报告
    logger.generate_report()
    logger.plot_losses(show=False)
    
    print(f"\n✅ 演示完成，查看结果: {logger.experiment_dir}")


if __name__ == "__main__":
    demo_usage() 