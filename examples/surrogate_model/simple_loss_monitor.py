#!/usr/bin/env python3
"""
简化的损失监控器 - 不依赖复杂的多进程
使用简单的文件监控和日志记录
"""

import os
import json
import time
import threading
from datetime import datetime
from collections import defaultdict, deque
import signal
import sys

class SimpleLossMonitor:
    """简化的损失监控器"""
    
    def __init__(self, experiment_name, log_dir="simple_loss_logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_loss_log")
        
        # 创建目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 损失数据存储
        self.loss_data = defaultdict(list)
        self.running = False
        self.monitor_thread = None
        
        # 通信文件路径
        self.comm_file = f"/tmp/simple_loss_{experiment_name}.json"
        
        print(f"📊 简化损失监控器初始化")
        print(f"   实验名称: {experiment_name}")
        print(f"   日志目录: {self.experiment_dir}")
        print(f"   通信文件: {self.comm_file}")
        
    def start_monitoring(self):
        """启动监控"""
        if self.running:
            print("⚠️ 监控已在运行")
            return
            
        self.running = True
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("✅ 损失监控已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # 最终保存数据
        self._save_all_data()
        
        # 清理通信文件
        try:
            if os.path.exists(self.comm_file):
                os.remove(self.comm_file)
        except:
            pass
            
        print("🛑 损失监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        print("🔄 开始监控损失数据...")
        last_save_time = time.time()
        
        while self.running:
            try:
                # 检查通信文件
                if os.path.exists(self.comm_file):
                    with open(self.comm_file, 'r') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                for loss_entry in data:
                                    self._process_loss_entry(loss_entry)
                                
                                # 清空文件
                                open(self.comm_file, 'w').close()
                                
                                print(f"📊 处理了 {len(data)} 条损失数据")
                                
                        except json.JSONDecodeError:
                            pass
                
                # 定期保存数据
                current_time = time.time()
                if current_time - last_save_time > 30:
                    self._save_all_data()
                    last_save_time = current_time
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                print(f"❌ 监控循环错误: {e}")
                time.sleep(5)
                
    def _process_loss_entry(self, loss_entry):
        """处理单个损失条目"""
        try:
            network = loss_entry.get('network', 'unknown')
            step = loss_entry.get('step', 0)
            timestamp = loss_entry.get('timestamp', time.time())
            losses = loss_entry.get('losses', {})
            
            # 存储数据
            entry = {
                'step': step,
                'timestamp': timestamp,
                **losses
            }
            
            self.loss_data[network].append(entry)
            
        except Exception as e:
            print(f"❌ 处理损失条目失败: {e}")
            
    def _save_all_data(self):
        """保存所有数据"""
        try:
            for network, data in self.loss_data.items():
                if data:
                    # 保存CSV文件
                    csv_path = os.path.join(self.experiment_dir, f"{network}_losses.csv")
                    
                    with open(csv_path, 'w') as f:
                        if data:
                            # 写入头部
                            first_entry = data[0]
                            headers = list(first_entry.keys())
                            f.write(','.join(headers) + '\n')
                            
                            # 写入数据
                            for entry in data:
                                values = [str(entry.get(h, '')) for h in headers]
                                f.write(','.join(values) + '\n')
                    
                    print(f"💾 保存 {network} 损失数据: {len(data)} 条记录")
                    
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            
    def send_loss(self, network, step, losses):
        """发送损失数据（从训练进程调用）"""
        loss_entry = {
            'network': network,
            'step': step,
            'timestamp': time.time(),
            'losses': losses
        }
        
        try:
            # 读取现有数据
            existing_data = []
            if os.path.exists(self.comm_file):
                try:
                    with open(self.comm_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []
            
            # 添加新数据
            if not isinstance(existing_data, list):
                existing_data = []
            existing_data.append(loss_entry)
            
            # 写回文件
            with open(self.comm_file, 'w') as f:
                json.dump(existing_data, f)
                
        except Exception as e:
            print(f"❌ 发送损失数据失败: {e}")


# 全局监控器实例
_global_monitor = None

def start_simple_loss_monitor(experiment_name):
    """启动简化损失监控器"""
    global _global_monitor
    
    if _global_monitor is not None:
        print("⚠️ 监控器已启动")
        return _global_monitor
        
    _global_monitor = SimpleLossMonitor(experiment_name)
    _global_monitor.start_monitoring()
    
    # 注册清理函数
    import atexit
    atexit.register(stop_simple_loss_monitor)
    
    return _global_monitor

def stop_simple_loss_monitor():
    """停止简化损失监控器"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor = None

def send_simple_loss(experiment_name, network, step, losses):
    """发送损失数据的简化函数"""
    comm_file = f"/tmp/simple_loss_{experiment_name}.json"
    
    loss_entry = {
        'network': network,
        'step': step,
        'timestamp': time.time(),
        'losses': losses
    }
    
    try:
        # 读取现有数据
        existing_data = []
        if os.path.exists(comm_file):
            try:
                with open(comm_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # 添加新数据
        if not isinstance(existing_data, list):
            existing_data = []
        existing_data.append(loss_entry)
        
        # 限制文件大小，只保留最近1000条记录
        if len(existing_data) > 1000:
            existing_data = existing_data[-1000:]
        
        # 写回文件
        with open(comm_file, 'w') as f:
            json.dump(existing_data, f)
            
    except Exception as e:
        print(f"❌ 发送损失数据失败: {e}")


if __name__ == "__main__":
    # 测试简化监控器
    print("🧪 测试简化损失监控器")
    
    monitor = start_simple_loss_monitor("test_simple")
    
    try:
        # 模拟发送数据
        for i in range(20):
            send_simple_loss("test_simple", "ppo", i, {
                'actor_loss': 1.0 - i*0.02,
                'critic_loss': 0.8 - i*0.015
            })
            send_simple_loss("test_simple", "attention", i, {
                'attention_loss': 2.0 - i*0.03
            })
            time.sleep(0.5)
            
            if i % 5 == 0:
                print(f"📊 发送第 {i} 步数据")
        
        # 等待处理
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("🛑 测试被中断")
    finally:
        stop_simple_loss_monitor()
        print("✅ 测试完成")


