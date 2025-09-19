#!/usr/bin/env python3
"""
网络损失记录器 - 独立进程版本
功能：
- 独立进程记录attention、GNN、PPO网络的每步损失
- 实时生成损失曲线图表
- 支持多网络同时监控
- 提供损失统计分析和预警
- 与MAP-Elites训练系统集成
"""

import os
import json
import time
import queue
import threading
import multiprocessing as mp
from multiprocessing import Queue, Process, Event, Manager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict, deque
from datetime import datetime
import csv
import pickle
import signal
import sys

# 设置matplotlib后端，避免GUI问题
plt.switch_backend('Agg')

class NetworkLossCollector:
    """单个网络的损失收集器"""
    
    def __init__(self, network_name, max_history=50000):
        self.network_name = network_name
        self.max_history = max_history
        
        # 损失历史
        self.loss_history = defaultdict(deque)
        self.step_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        
        # 统计信息
        self.stats = {
            'total_updates': 0,
            'avg_loss': 0.0,
            'min_loss': float('inf'),
            'max_loss': float('-inf'),
            'recent_trend': 'stable',  # 'increasing', 'decreasing', 'stable'
            'last_update': time.time()
        }
        
    def add_loss(self, step, timestamp, loss_dict):
        """添加损失数据"""
        self.step_history.append(step)
        self.timestamp_history.append(timestamp)
        
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, (int, float)) and not np.isnan(loss_value):
                self.loss_history[loss_name].append(loss_value)
                
                # 维护最大历史长度
                if len(self.loss_history[loss_name]) > self.max_history:
                    self.loss_history[loss_name].popleft()
        
        self.stats['total_updates'] += 1
        self.stats['last_update'] = timestamp
        self._update_stats()
    
    def _update_stats(self):
        """更新统计信息"""
        if not self.loss_history:
            return
            
        # 计算主要损失的统计信息（假设第一个损失是主要的）
        main_loss_name = list(self.loss_history.keys())[0]
        main_losses = list(self.loss_history[main_loss_name])
        
        if main_losses:
            self.stats['avg_loss'] = np.mean(main_losses)
            self.stats['min_loss'] = min(self.stats['min_loss'], min(main_losses))
            self.stats['max_loss'] = max(self.stats['max_loss'], max(main_losses))
            
            # 计算趋势（最近20个点的斜率）
            if len(main_losses) >= 20:
                recent_losses = main_losses[-20:]
                x = np.arange(len(recent_losses))
                slope = np.polyfit(x, recent_losses, 1)[0]
                
                if slope > 0.001:
                    self.stats['recent_trend'] = 'increasing'
                elif slope < -0.001:
                    self.stats['recent_trend'] = 'decreasing'
                else:
                    self.stats['recent_trend'] = 'stable'


class NetworkLossLogger:
    """网络损失记录器主类"""
    
    def __init__(self, log_dir="network_loss_logs", experiment_name=None, 
                 networks=['attention', 'ppo', 'gnn'], update_interval=10.0):
        """
        初始化网络损失记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            networks: 要监控的网络列表
            update_interval: 图表更新间隔（秒）
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"network_loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        self.networks = networks
        self.update_interval = update_interval
        
        # 创建目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 进程间通信
        self.loss_queue = Queue(maxsize=50000)
        self.control_queue = Queue()
        self.stop_event = Event()
        
        # 记录进程
        self.logger_process = None
        
        # 配置信息
        self.config = {
            'experiment_info': {
                'experiment_name': self.experiment_name,
                'start_time': datetime.now().isoformat(),
                'log_dir': self.experiment_dir,
                'monitored_networks': networks,
                'update_interval': update_interval
            }
        }
        
        print(f"🚀 网络损失记录器初始化完成")
        print(f"   实验名称: {self.experiment_name}")
        print(f"   监控网络: {', '.join(networks)}")
        print(f"   日志目录: {self.experiment_dir}")
        
        # 保存配置
        self._save_config()
        
    def start_logging(self):
        """启动记录进程"""
        if self.logger_process is not None and self.logger_process.is_alive():
            print("⚠️  记录进程已在运行")
            return
            
        self.stop_event.clear()
        self.logger_process = Process(
            target=self._logging_worker, 
            args=(self.loss_queue, self.control_queue, self.stop_event, 
                  self.experiment_dir, self.networks, self.update_interval)
        )
        self.logger_process.daemon = True
        self.logger_process.start()
        
        print(f"✅ 记录进程已启动 (PID: {self.logger_process.pid})")
        
    def stop_logging(self):
        """停止记录进程"""
        if self.logger_process is None:
            print("⚠️  记录进程未在运行")
            return
            
        print("🛑 正在停止记录进程...")
        self.stop_event.set()
        
        # 等待进程结束
        self.logger_process.join(timeout=15)
        if self.logger_process.is_alive():
            print("⚠️  强制终止记录进程")
            self.logger_process.terminate()
            self.logger_process.join()
            
        self.logger_process = None
        print("✅ 记录进程已停止")
        
    def log_loss(self, network_name, step, loss_dict, timestamp=None):
        """记录损失数据"""
        if network_name not in self.networks:
            print(f"⚠️  未知网络: {network_name}, 支持的网络: {self.networks}")
            return
            
        timestamp = timestamp or time.time()
        
        try:
            # 发送到记录进程
            loss_data = {
                'network': network_name,
                'step': step,
                'timestamp': timestamp,
                'losses': loss_dict
            }
            
            # 非阻塞发送，如果队列满了就跳过旧数据
            if self.loss_queue.full():
                # 清理一些旧数据
                for _ in range(100):
                    try:
                        self.loss_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            self.loss_queue.put_nowait(loss_data)
                
        except Exception as e:
            print(f"❌ 记录损失失败: {e}")
            
    def is_alive(self):
        """检查记录进程是否还在运行"""
        return self.logger_process is not None and self.logger_process.is_alive()
        
    def _save_config(self):
        """保存配置文件"""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            
    @staticmethod
    def _logging_worker(loss_queue, control_queue, stop_event, experiment_dir, 
                       networks, update_interval):
        """记录进程工作函数"""
        print(f"📊 损失记录进程启动 (PID: {os.getpid()})")
        
        # 初始化收集器
        collectors = {network: NetworkLossCollector(network) for network in networks}
        
        # 设置信号处理
        def signal_handler(signum, frame):
            print(f"📊 记录进程接收到信号 {signum}")
            stop_event.set()
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # 🆕 启动实时损失收集器
        try:
            from loss_communication import RealTimeLossCollector
            experiment_name = os.path.basename(experiment_dir).replace('_loss_log', '')
            real_time_collector = RealTimeLossCollector(experiment_name, None)
            
            # 在单独线程中运行实时收集器
            collect_thread = threading.Thread(
                target=real_time_collector.start_collecting,
                daemon=True
            )
            collect_thread.start()
            print("🔄 实时损失收集器已启动")
            
        except ImportError as e:
            print(f"⚠️ 实时损失收集器不可用: {e}")
            real_time_collector = None
        
        # 创建图表更新线程
        plot_thread = threading.Thread(
            target=NetworkLossLogger._plot_worker,
            args=(collectors, experiment_dir, stop_event, update_interval),
            daemon=True
        )
        plot_thread.start()
        
        # 主循环
        last_save_time = time.time()
        save_interval = 30.0  # 每30秒保存一次数据
        
        try:
            while not stop_event.is_set():
                try:
                    # 处理损失数据
                    loss_data = loss_queue.get(timeout=1.0)
                    
                    network = loss_data['network']
                    step = loss_data['step']
                    timestamp = loss_data['timestamp']
                    losses = loss_data['losses']
                    
                    # 添加到对应的收集器
                    if network in collectors:
                        collectors[network].add_loss(step, timestamp, losses)
                        
                    # 定期保存数据
                    current_time = time.time()
                    if current_time - last_save_time > save_interval:
                        NetworkLossLogger._save_data(collectors, experiment_dir)
                        last_save_time = current_time
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ 记录进程错误: {e}")
                    
        except KeyboardInterrupt:
            print("📊 记录进程被中断")
        finally:
            # 停止实时收集器
            if real_time_collector:
                real_time_collector.stop_collecting()
            
            # 最终保存数据
            NetworkLossLogger._save_data(collectors, experiment_dir)
            stop_event.set()  # 确保图表线程也停止
            plot_thread.join(timeout=5)
            print("📊 记录进程结束")
            
    @staticmethod
    def _plot_worker(collectors, experiment_dir, stop_event, update_interval):
        """图表更新工作线程"""
        print(f"📈 图表更新线程启动")
        
        plt.style.use('default')
        
        while not stop_event.is_set():
            try:
                NetworkLossLogger._generate_plots(collectors, experiment_dir)
                time.sleep(update_interval)
            except Exception as e:
                print(f"❌ 图表生成错误: {e}")
                time.sleep(update_interval)
                
        print("📈 图表更新线程结束")
        
    @staticmethod
    def _generate_plots(collectors, experiment_dir):
        """生成损失曲线图"""
        if not collectors:
            return
            
        # 创建子图
        n_networks = len(collectors)
        fig, axes = plt.subplots(n_networks, 1, figsize=(15, 5*n_networks))
        if n_networks == 1:
            axes = [axes]
            
        fig.suptitle(f'Network Loss Real-time Monitor - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                     fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for idx, (network_name, collector) in enumerate(collectors.items()):
            ax = axes[idx]
            
            if not collector.loss_history:
                ax.text(0.5, 0.5, f'{network_name.upper()}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{network_name.upper()} Network Loss')
                continue
                
            # 绘制每种损失
            steps = list(collector.step_history)
            
            for loss_idx, (loss_name, loss_values) in enumerate(collector.loss_history.items()):
                if len(loss_values) == len(steps):
                    color = colors[loss_idx % len(colors)]
                    ax.plot(steps, list(loss_values), label=loss_name, 
                           color=color, linewidth=1.5, alpha=0.8)
                    
                    # 添加最近的趋势线
                    if len(loss_values) > 20:
                        recent_steps = steps[-20:]
                        recent_losses = list(loss_values)[-20:]
                        z = np.polyfit(range(len(recent_losses)), recent_losses, 1)
                        trend_line = np.poly1d(z)(range(len(recent_losses)))
                        ax.plot(recent_steps, trend_line, '--', color=color, alpha=0.5, linewidth=1)
            
            # 设置标题和标签
            ax.set_title(f'{network_name.upper()} Network Loss (Updates: {collector.stats["total_updates"]})')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss Value')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息文本
            last_update_str = datetime.fromtimestamp(collector.stats['last_update']).strftime('%H:%M:%S')
            stats_text = f'Trend: {collector.stats["recent_trend"]}\n'
            stats_text += f'Average: {collector.stats["avg_loss"]:.4f}\n'
            stats_text += f'Range: [{collector.stats["min_loss"]:.4f}, {collector.stats["max_loss"]:.4f}]\n'
            stats_text += f'Last Update: {last_update_str}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(experiment_dir, 'network_loss_curves_realtime.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存带时间戳的版本（每小时保存一次）
        current_hour = datetime.now().strftime("%Y%m%d_%H")
        timestamp_plot_path = os.path.join(experiment_dir, f'network_loss_curves_{current_hour}00.png')
        if not os.path.exists(timestamp_plot_path):
            plt.savefig(timestamp_plot_path, dpi=150, bbox_inches='tight')
        
    @staticmethod
    def _save_data(collectors, experiment_dir):
        """保存损失数据"""
        for network_name, collector in collectors.items():
            # 保存CSV格式
            csv_path = os.path.join(experiment_dir, f'{network_name}_losses.csv')
            
            if collector.step_history and collector.loss_history:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    # 获取所有损失类型
                    loss_names = list(collector.loss_history.keys())
                    fieldnames = ['step', 'timestamp'] + loss_names
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    steps = list(collector.step_history)
                    timestamps = list(collector.timestamp_history)
                    
                    for i in range(len(steps)):
                        row = {
                            'step': steps[i],
                            'timestamp': timestamps[i] if i < len(timestamps) else ''
                        }
                        
                        # 添加损失值
                        for loss_name in loss_names:
                            loss_values = list(collector.loss_history[loss_name])
                            if i < len(loss_values):
                                row[loss_name] = loss_values[i]
                            else:
                                row[loss_name] = ''
                                
                        writer.writerow(row)
            
            # 保存统计信息
            stats_path = os.path.join(experiment_dir, f'{network_name}_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(collector.stats, f, indent=2, ensure_ascii=False)


# 全局实例
_global_logger = None

def init_network_loss_logger(experiment_name=None, networks=['attention', 'ppo', 'gnn'], 
                            log_dir="network_loss_logs", update_interval=10.0):
    """初始化全局网络损失记录器"""
    global _global_logger
    
    if _global_logger is not None:
        print("⚠️  网络损失记录器已初始化")
        return _global_logger
        
    _global_logger = NetworkLossLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        networks=networks,
        update_interval=update_interval
    )
    
    _global_logger.start_logging()
    
    # 注册退出时的清理函数
    import atexit
    atexit.register(cleanup_network_loss_logger)
    
    return _global_logger

def get_network_loss_logger():
    """获取全局网络损失记录器"""
    return _global_logger

def log_network_loss(network_name, step, loss_dict, timestamp=None):
    """记录网络损失的便捷函数"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.log_loss(network_name, step, loss_dict, timestamp)
    else:
        print("⚠️  网络损失记录器未初始化，请先调用 init_network_loss_logger()")

def cleanup_network_loss_logger():
    """清理网络损失记录器"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.stop_logging()
        _global_logger = None
        print("🧹 网络损失记录器已清理")


# 测试代码
if __name__ == "__main__":
    print("🧪 测试网络损失记录器")
    
    # 初始化记录器
    logger = init_network_loss_logger(
        experiment_name="test_network_loss_logger",
        networks=['attention', 'ppo', 'gnn'],
        update_interval=5.0
    )
    
    try:
        # 模拟训练过程
        print("🚀 开始模拟训练...")
        for step in range(1000):
            # 模拟attention网络损失
            attention_loss = {
                'attention_loss': max(0.1, 2.0 - step*0.001 + np.random.normal(0, 0.1)),
                'attention_accuracy': min(1.0, 0.5 + step * 0.0005 + np.random.normal(0, 0.02))
            }
            log_network_loss('attention', step, attention_loss)
            
            # 模拟PPO网络损失
            ppo_loss = {
                'actor_loss': max(0.01, 1.5 - step*0.0008 + np.random.normal(0, 0.08)),
                'critic_loss': max(0.01, 1.2 - step*0.0006 + np.random.normal(0, 0.06)),
                'entropy': max(0.001, 0.8 - step*0.0003 + np.random.normal(0, 0.02))
            }
            log_network_loss('ppo', step, ppo_loss)
            
            # 模拟GNN网络损失
            gnn_loss = {
                'gnn_loss': max(0.1, 3.0 - step*0.0012 + np.random.normal(0, 0.15)),
                'node_accuracy': min(1.0, 0.3 + step * 0.0007 + np.random.normal(0, 0.01))
            }
            log_network_loss('gnn', step, gnn_loss)
            
            # 打印进度
            if step % 100 == 0:
                print(f"📊 训练步数: {step}")
                print(f"   日志目录: {logger.experiment_dir}")
            
            time.sleep(0.01)  # 模拟训练间隔
            
    except KeyboardInterrupt:
        print("🛑 测试被中断")
    finally:
        print("✅ 测试完成")
